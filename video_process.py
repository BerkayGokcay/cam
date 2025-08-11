# =============================
# File: video_process.py (DeepStream 7.1 → RMQ+MinIO, track-based sampling, NVMM ROI encode)
# =============================
#!/usr/bin/env python3
import os, sys, time, json, ctypes
from io import BytesIO
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds
import cv2

import pika
from minio import Minio

# ---------------------- Konfig ----------------------------
PGIE = os.getenv("PGIE_CONFIG", "/opt/models/yolov10_face/pgie_yolov10_face.txt")
INFERENCE_INTERVAL = int(os.getenv("INFERENCE_INTERVAL", "0"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "face_detections")
FACES_DIR = os.path.join(OUTPUT_DIR, "faces")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
RAW_CROPS_DIR = os.path.join(OUTPUT_DIR, "raw_crops")  # fixed typo
FACE_CLASS_ID = int(os.getenv("FACE_CLASS_ID", "0"))

SAVE_INTERVAL_SECONDS = float(os.getenv("SAVE_INTERVAL_SECONDS", "1.0"))  # 0.7–1.0 öneri
MAX_TRACKING_AGE_SECONDS = float(os.getenv("MAX_TRACKING_AGE_SECONDS", "30"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# MinIO
MINIO_HOST = os.getenv("MINIO_HOST", "minio")
MINIO_PORT = os.getenv("MINIO_PORT", "9000")
MINIO_USER = os.getenv("MINIO_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_PASS", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "videos-face-detection-v2")
RAW_PREFIX = os.getenv("RAW_PREFIX", "raw-crops/")

# RabbitMQ
RMQ_URL = os.getenv("RMQ_URL", "amqp://guest:guest@rabbitmq:5672/%2F")
RMQ_QUEUE = os.getenv("RMQ_QUEUE", "faces")
RMQ_PREFETCH = int(os.getenv("RMQ_PREFETCH", "32"))

# ---------------------- Hazırlık ----------------------------
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(RAW_CROPS_DIR, exist_ok=True)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# MinIO client
minio_client = None
try:
    minio_client = Minio(f"{MINIO_HOST}:{MINIO_PORT}", access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
except Exception as e:
    print(f"[UYARI] MinIO başlatılamadı: {e}")
    minio_client = None

# RabbitMQ channel (tek bağlantı)
def create_rmq_channel():
    params = pika.URLParameters(RMQ_URL)
    conn = pika.BlockingConnection(params)
    chan = conn.channel()
    chan.queue_declare(queue=RMQ_QUEUE, durable=False)
    chan.basic_qos(prefetch_count=RMQ_PREFETCH)
    return conn, chan

rmq_conn, rmq_chan = create_rmq_channel()

# ---------------------- Tracker state ----------------------
face_tracker = {}
tracker_lock = threading.Lock()

stats = {"frames_processed": 0, "faces_published": 0}

# ---------------------- Yardımcılar ------------------------

def make(factory: str, name: str | None = None, **props):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"Eleman oluşturulamadı: {factory}")
    for k, v in props.items():
        el.set_property(k, v)
    return el

# optional: NVMM ROI encoder (DeepStream obj enc)
try:
    from pyds import nvds_obj_enc_process, NvDsObjEncUsrArgs, NvOSD_Bbox_Coordinates
    HAS_NVOBJENC = True
    enc_args = NvDsObjEncUsrArgs()
    enc_args.saveImg = 0
    enc_args.attachUsrMeta = 0
    enc_args.quality = 80
    enc_args.isFrame = 0
except Exception:
    HAS_NVOBJENC = False
    enc_args = None


def should_save_for_track(track_id: int, now: float) -> bool:
    with tracker_lock:
        rec = face_tracker.get(track_id)
        if rec is None:
            face_tracker[track_id] = {"last_saved": 0.0, "last_seen": now}
            return True
        rec["last_seen"] = now
        return (now - rec["last_saved"]) >= SAVE_INTERVAL_SECONDS


def mark_saved(track_id: int, now: float):
    with tracker_lock:
        if track_id in face_tracker:
            face_tracker[track_id]["last_saved"] = now


def gc_tracks(now: float):
    with tracker_lock:
        dead = [tid for tid, r in face_tracker.items() if (now - r["last_seen"]) > MAX_TRACKING_AGE_SECONDS]
        for tid in dead:
            del face_tracker[tid]


def frame_to_numpy(buf, frame_meta):
    surface = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
    if surface is None:
        return None
    arr = np.array(surface, copy=True, order="C")
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def encode_roi_cpu(frame_bgr: np.ndarray, rect) -> bytes | None:
    # CPU fallback: margin + kare boyutlandırma + JPEG
    x, y, w, h = int(rect.left), int(rect.top), int(rect.width), int(rect.height)
    mx, my = int(0.15 * w), int(0.15 * h)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, frame_bgr.shape[1]), min(y + h + my, frame_bgr.shape[0])
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    target = 288
    crop = cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buf.tobytes() if ok else None


def encode_roi_nvmm(gst_buf, frame_meta, obj_meta) -> bytes | None:
    if not HAS_NVOBJENC:
        return None
    try:
        r = obj_meta.rect_params
        bbox = NvOSD_Bbox_Coordinates(int(r.left), int(r.top), int(r.width), int(r.height))
        # margin
        mx, my = int(0.15*bbox.width), int(0.15*bbox.height)
        bbox.left = max(bbox.left - mx, 0)
        bbox.top  = max(bbox.top  - my, 0)
        bbox.width  += 2*mx
        bbox.height += 2*my
        ok, jpeg_bytes = nvds_obj_enc_process(hash(gst_buf), frame_meta, obj_meta, enc_args)
        if ok and jpeg_bytes:
            return bytes(jpeg_bytes)
    except Exception as e:
        print(f"[enc] NVMM ROI encode hata: {e}")
    return None


def publish_face(jpeg_bytes: bytes, src_id: int, track_id: int, frame_idx: int):
    # 1) MinIO PUT (tamamlanınca)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    object_name = f"{RAW_PREFIX}face_s{src_id}_t{track_id}_f{frame_idx}_{ts}.jpg"
    if minio_client:
        minio_client.put_object(MINIO_BUCKET, object_name, BytesIO(jpeg_bytes), len(jpeg_bytes), content_type="image/jpeg")
    # 2) RMQ publish (meta)
    msg = {
        "object_name": object_name,
        "src_id": int(src_id),
        "track_id": int(track_id),
        "frame_idx": int(frame_idx),
        "ts_pub_ms": int(time.time()*1000),
    }
    rmq_chan.basic_publish(exchange="", routing_key=RMQ_QUEUE, body=json.dumps(msg))


# ---------------------- Probe -------------------------------

def tracker_probe(_pad, info, _):
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    now = time.time()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        stats["frames_processed"] += 1

        # CPU fallback için frame
        frame_bgr = None
        if not HAS_NVOBJENC:
            frame_bgr = frame_to_numpy(buf, frame_meta)
            if frame_bgr is None:
                l_frame = l_frame.next
                continue

        l_obj = frame_meta.obj_meta_list
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            if obj_meta.class_id != FACE_CLASS_ID:
                l_obj = l_obj.next
                continue

            track_id = int(obj_meta.object_id)
            if should_save_for_track(track_id, now):
                # NVMM ROI encode → yoksa CPU fallback
                jpeg_bytes = encode_roi_nvmm(buf, frame_meta, obj_meta)
                if jpeg_bytes is None and frame_bgr is not None:
                    jpeg_bytes = encode_roi_cpu(frame_bgr, obj_meta.rect_params)

                if jpeg_bytes:
                    executor.submit(publish_face, jpeg_bytes, int(frame_meta.source_id), track_id, int(frame_meta.frame_num))
                    stats["faces_published"] += 1
                    mark_saved(track_id, now)

            l_obj = l_obj.next

        gc_tracks(now)
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


# ---------------------- Pipeline ----------------------------

def build_video(video_path: str, width: int = 1920, height: int = 1080) -> Gst.Pipeline:
    Gst.init(None)
    p = Gst.Pipeline.new("face-pipeline")

    src = make("filesrc", location=video_path)
    decodebin = make("decodebin")
    mux = make("nvstreammux", batch_size=1, width=width, height=height, live_source=False, nvbuf_memory_type=0)
    conv1 = make("nvvideoconvert")
    caps = make("capsfilter", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
    pgie = make("nvinfer", config_file_path=PGIE, interval=INFERENCE_INTERVAL)
    tracker = make("nvtracker",
                   ll_lib_file="/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                   ll_config_file="/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/tracker_config/config_tracker_NvDCF_perf.yml",
                   tracker_width=960, tracker_height=540, gpu_id=0)
    osd = make("nvdsosd")
    conv2 = make("nvvideoconvert")
    sink = make("fakesink", sync=False)  # ekran çıkışı gerekmez

    for el in [src, decodebin, mux, conv1, caps, pgie, tracker, osd, conv2, sink]:
        p.add(el)

    def on_pad_added(element, pad):
        caps = pad.get_current_caps()
        if caps is not None and caps.to_string().startswith("video/"):
            sinkpad = mux.get_request_pad("sink_0")
            if not sinkpad.is_linked():
                pad.link(sinkpad)

    decodebin.connect("pad-added", on_pad_added)
    src.link(decodebin)
    mux.link(conv1)
    conv1.link(caps)
    caps.link(pgie)
    pgie.link(tracker)
    tracker.link(osd)
    osd.link(conv2)
    conv2.link(sink)

    osd_sink_pad = osd.get_static_pad("sink")
    if osd_sink_pad:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tracker_probe, None)
    else:
        sys.exit("HATA: OSD sink pad alınamadı.")

    return p


def main():
    if len(sys.argv) < 2:
        print(f"Kullanım: {sys.argv[0]} <video_dosyası>")
        return 1
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"[HATA] Video dosyası bulunamadı: {video_path}")
        return 1

    pipeline = build_video(video_path)
    loop = GLib.MainLoop()
    bus = pipeline.get_bus(); bus.add_signal_watch()

    def on_message(_bus, msg):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error(); print(f"[GStreamer HATA] {err.message}"); print(dbg or "")
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("EOS"); loop.quit()

    bus.connect("message", on_message)
    print("Pipeline PLAYING…")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("[HATA] Pipeline başlatılamadı.")
        return 1

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
        executor.shutdown(wait=True)
        try:
            rmq_conn.close()
        except Exception:
            pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
