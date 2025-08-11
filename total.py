# =============================
# File: docker-compose.yml
# =============================
version: "3.9"
services:
  rabbitmq:
    image: rabbitmq:3-management
    container_name: rabbitmq
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest

  minio:
    image: quay.io/minio/minio:latest
    container_name: minio
    command: server /data --console-address :9001
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data

volumes:
  minio-data:


# =============================
# File: .env (örnek)
# =============================
# DeepStream çevresi bu dosyayı okumak zorunda değil; sadece kolaylık için.
PGIE_CONFIG=/opt/models/yolov10_face/pgie_yolov10_face.txt
INFERENCE_INTERVAL=0
FACE_CLASS_ID=0
SAVE_INTERVAL_SECONDS=1.0
MAX_TRACKING_AGE_SECONDS=30
MAX_WORKERS=8

# MinIO
MINIO_HOST=127.0.0.1
MINIO_PORT=9000
MINIO_USER=minioadmin
MINIO_PASS=minioadmin
MINIO_BUCKET=videos-face-detection-v2
RAW_PREFIX=raw-crops/
PROCESSED_PREFIX=processed/

# RabbitMQ
RMQ_URL=amqp://guest:guest@127.0.0.1:5672/%2F
RMQ_QUEUE=faces
RMQ_PREFETCH=32

# Worker
BATCH=16
MAX_WAIT_MS=8
GPU_ID=0
ARCFACE_MODEL=/opt/models/recognition/w600k_r50.onnx


# =============================
# File: requirements.txt
# =============================
# DeepStream Python binding'leri (pyds) NVIDIA imajıyla gelir.
# Aşağıdakiler DS konteynerinin içinde de kurulabilir.
numpy
opencv-python
pika
minio
onnxruntime-gpu
face-alignment
torch
torchvision


# =============================
# File: video_process.py (DeepStream 7.1 → RMQ+MinIO, track-based sampling; CPU ROI fallback)
# =============================
#!/usr/bin/env python3
import os, sys, time, json
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
FACE_CLASS_ID = int(os.getenv("FACE_CLASS_ID", "0"))

SAVE_INTERVAL_SECONDS = float(os.getenv("SAVE_INTERVAL_SECONDS", "1.0"))
MAX_TRACKING_AGE_SECONDS = float(os.getenv("MAX_TRACKING_AGE_SECONDS", "30"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "8"))

# MinIO
MINIO_HOST = os.getenv("MINIO_HOST", "127.0.0.1")
MINIO_PORT = os.getenv("MINIO_PORT", "9000")
MINIO_USER = os.getenv("MINIO_USER", "minioadmin")
MINIO_PASS = os.getenv("MINIO_PASS", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "videos-face-detection-v2")
RAW_PREFIX = os.getenv("RAW_PREFIX", "raw-crops/")

# RabbitMQ
RMQ_URL = os.getenv("RMQ_URL", "amqp://guest:guest@127.0.0.1:5672/%2F")
RMQ_QUEUE = os.getenv("RMQ_QUEUE", "faces")
RMQ_PREFETCH = int(os.getenv("RMQ_PREFETCH", "32"))

# ---------------------- Hazırlık ----------------------------
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

# RabbitMQ channel
import pika
params = pika.URLParameters(RMQ_URL)
rmq_conn = pika.BlockingConnection(params)
rmq_chan = rmq_conn.channel()
rmq_chan.queue_declare(queue=RMQ_QUEUE, durable=False)
rmq_chan.basic_qos(prefetch_count=RMQ_PREFETCH)

face_tracker = {}
tracker_lock = threading.Lock()


def make(factory: str, name: str | None = None, **props):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"Eleman oluşturulamadı: {factory}")
    for k, v in props.items():
        el.set_property(k, v)
    return el


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


def publish_face(jpeg_bytes: bytes, src_id: int, track_id: int, frame_idx: int):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    object_name = f"{RAW_PREFIX}face_s{src_id}_t{track_id}_f{frame_idx}_{ts}.jpg"
    if minio_client:
        minio_client.put_object(MINIO_BUCKET, object_name, BytesIO(jpeg_bytes), len(jpeg_bytes), content_type="image/jpeg")
    msg = {
        "object_name": object_name,
        "src_id": int(src_id),
        "track_id": int(track_id),
        "frame_idx": int(frame_idx),
        "ts_pub_ms": int(time.time()*1000),
    }
    rmq_chan.basic_publish(exchange="", routing_key=RMQ_QUEUE, body=json.dumps(msg))


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
                jpeg_bytes = encode_roi_cpu(frame_bgr, obj_meta.rect_params)
                if jpeg_bytes:
                    # I/O'yu thread'e ver (publish, MinIO PUT, RMQ send)
                    threading.Thread(target=publish_face, args=(jpeg_bytes, int(frame_meta.source_id), track_id, int(frame_meta.frame_num)), daemon=True).start()
                    mark_saved(track_id, now)

            l_obj = l_obj.next

        gc_tracks(now)
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK


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
    sink = make("fakesink", sync=False)

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


# =============================
# File: worker.py (RabbitMQ consumer + minibatch + MinIO I/O, GPU-heavy preprocess/alignment)
# =============================
#!/usr/bin/env python3
import os, time, cv2, json, queue, threading
import numpy as np
from io import BytesIO
from typing import List, Tuple

import pika
from minio import Minio
import onnxruntime
import face_alignment

# ---------------------- Konfig ----------------------------
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "videos-face-detection-v2")
MINIO_HOST = os.getenv("MINIO_HOST","127.0.0.1")
MINIO_PORT = os.getenv("MINIO_PORT","9000")
MINIO_USER = os.getenv("MINIO_USER","minioadmin")
MINIO_PASS = os.getenv("MINIO_PASS","minioadmin")
RAW_PREFIX    = os.getenv("RAW_PREFIX", "raw-crops/")
PROCESSED_PREFIX = os.getenv("PROCESSED_PREFIX", "processed/")

RMQ_URL   = os.getenv("RMQ_URL", "amqp://guest:guest@127.0.0.1:5672/%2F")
RMQ_QUEUE = os.getenv("RMQ_QUEUE", "faces")

BATCH_SIZE = int(os.getenv("BATCH", "16"))
MAX_WAIT_MS = int(os.getenv("MAX_WAIT_MS", "8"))
GPU_ID = int(os.getenv("GPU_ID","0"))
ARCFACE_MODEL = os.getenv("ARCFACE_MODEL", "/opt/models/recognition/w600k_r50.onnx")

# ---------------------- MinIO -----------------------------
minio_client = Minio(f"{MINIO_HOST}:{MINIO_PORT}", access_key=MINIO_USER, secret_key=MINIO_PASS, secure=False)
if not minio_client.bucket_exists(MINIO_BUCKET):
    minio_client.make_bucket(MINIO_BUCKET)

# ---------------------- RabbitMQ --------------------------
params = pika.URLParameters(RMQ_URL)
rmq_conn = pika.BlockingConnection(params)
rmq_chan = rmq_conn.channel()
rmq_chan.queue_declare(queue=RMQ_QUEUE, durable=False)
rmq_chan.basic_qos(prefetch_count=BATCH_SIZE)

job_q: "queue.Queue[Tuple[str,np.ndarray, int]]" = queue.Queue(maxsize=1024)

# ---------------------- Models ----------------------------
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda")

class ArcFaceONNX:
    def __init__(self, model_path: str, gpu_id: int = 0):
        providers = [
            ("TensorrtExecutionProvider", {"device_id": gpu_id}),
            ("CUDAExecutionProvider", {"device_id": gpu_id}),
            "CPUExecutionProvider",
        ]
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(model_path, sess_options=so, providers=providers)
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name
        ishape = self.session.get_inputs()[0].shape
        self.H = int(ishape[2] if ishape[2] not in (None, 'None', -1) else 112)
        self.W = int(ishape[3] if ishape[3] not in (None, 'None', -1) else 112)

    def preprocess_batch_gpuassist(self, crops: List[np.ndarray]) -> np.ndarray:
        """Resize + BGR2RGB on GPU via cv2.cuda; final CHW/normalize on CPU (lightweight)."""
        out = []
        for im in crops:
            gpu = cv2.cuda_GpuMat()
            gpu.upload(im)  # HxWx3 BGR
            gpu_res = cv2.cuda.resize(gpu, (self.W, self.H), interpolation=cv2.INTER_AREA)
            gpu_rgb = cv2.cuda.cvtColor(gpu_res, cv2.COLOR_BGR2RGB)
            rgb = gpu_rgb.download().astype(np.float32)
            rgb = (rgb - 127.5) / 127.5
            chw = np.transpose(rgb, (2,0,1))
            out.append(chw)
        return np.stack(out, axis=0)

    def infer(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0,128), dtype=np.float32)
        inp = self.preprocess_batch_gpuassist(crops)
        out = self.session.run([self.out_name], {self.in_name: inp})[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return (out / norms).astype(np.float32)

arc = ArcFaceONNX(ARCFACE_MODEL, gpu_id=GPU_ID)

class GPUFaceAligner:
    def __init__(self, crop_size: int = 112):
        self.crop_size = crop_size
        self.std = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        self.idx5 = [36,45,33,48,54]
    def align_face(self, img_bgr: np.ndarray, lm68: np.ndarray) -> np.ndarray | None:
        try:
            lm5 = lm68[self.idx5].astype(np.float32)
            M, _ = cv2.estimateAffinePartial2D(lm5, self.std, method=cv2.LMEDS)
            if M is None:
                return None
            # GPU warp
            gpu = cv2.cuda_GpuMat(); gpu.upload(img_bgr)
            gpu_warp = cv2.cuda.warpAffine(gpu, M, (self.crop_size, self.crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            return gpu_warp.download()
        except Exception:
            return None

aligner = GPUFaceAligner(112)

# ---------------------- Consumer --------------------------

def on_msg(ch, method, properties, body):
    try:
        m = json.loads(body)
        name = m["object_name"]
        resp = minio_client.get_object(MINIO_BUCKET, name)
        data = resp.read(); resp.close(); resp.release_conn()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            ch.basic_ack(method.delivery_tag)
            safe_remove(name)
            return
        job_q.put((name, img, method.delivery_tag), block=False)
    except Exception as e:
        print(f"[worker] on_msg error: {e}")
        ch.basic_nack(method.delivery_tag, requeue=True)

rmq_chan.basic_consume(queue=RMQ_QUEUE, on_message_callback=on_msg, auto_ack=False)

# ---------------------- Batch Döngüsü ----------------------

def process_pending(items: List[Tuple[str,np.ndarray,int]]):
    names = [n for (n,_,_) in items]
    imgs  = [im for (_,im,_) in items]
    tags  = [t for (_,_,t) in items]

    aligned = []
    for im in imgs:
        lms = fa.get_landmarks(im)
        a = aligner.align_face(im, lms[0]) if lms else None
        aligned.append(a)

    to_embed = [a for a in aligned if a is not None]
    embs = arc.infer(to_embed) if to_embed else np.empty((0,128), dtype=np.float32)
    it = iter(embs)

    for name, a, dtag in zip(names, aligned, tags):
        if a is None:
            rmq_chan.basic_ack(dtag)
            safe_remove(name)
            continue
        e = next(it, np.zeros((128,), np.float32))  # TODO: DB'ye yazılacaksa burada
        ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            minio_client.put_object(
                MINIO_BUCKET,
                f"{PROCESSED_PREFIX}aligned_{os.path.basename(name)}",
                BytesIO(buf), len(buf), content_type="image/jpeg"
            )
        rmq_chan.basic_ack(dtag)
        safe_remove(name)


def batch_worker_loop():
    pending = []
    last = time.time()
    while True:
        timeout = max(0.0, MAX_WAIT_MS/1000.0 - (time.time()-last))
        try:
            item = job_q.get(timeout=timeout)
            pending.append(item)
            if len(pending) >= BATCH_SIZE:
                process_pending(pending)
                pending = []
                last = time.time()
        except queue.Empty:
            if pending:
                process_pending(pending)
                pending = []
                last = time.time()


def safe_remove(object_name: str):
    try:
        minio_client.remove_object(MINIO_BUCKET, object_name)
        print(f"  -> silindi: {object_name}")
    except Exception as e:
        print(f"[WARN] remove {object_name}: {e}")


def main():
    print(f"[worker] BATCH={BATCH_SIZE} MAX_WAIT_MS={MAX_WAIT_MS} GPU_ID={GPU_ID}")
    t = threading.Thread(target=batch_worker_loop, daemon=True)
    t.start()
    try:
        rmq_chan.start_consuming()
    except KeyboardInterrupt:
        rmq_chan.stop_consuming()

if __name__ == "__main__":
    main()


# =============================
# Çalıştırma talimatları
# =============================
# =============================
# 1) Altyapıyı ayağa kaldırın:
#    docker compose up -d   # RabbitMQ (http://localhost:15672 guest/guest) ve MinIO (http://localhost:9001)
# 2) Ortam değişkenlerini ayarlayın (gerekirse .env dosyasından kopyalayın)
# 3) DeepStream makinesinde video_process.py'yi çalıştırın:
#    python3 video_process.py /path/to/video.mp4
# 4) Worker makinesinde (GPU'lu) worker.py'yi çalıştırın:
#    python3 worker.py
# 5) Çıktılar MinIO'da:
#    raw-crops/: DS tarafından atılan kırpmalar
#    processed/: worker'ın hizalayıp isim başına 'aligned_' ile yükledikleri
