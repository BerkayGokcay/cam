#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU crop + JPEG encode path for DeepStream (PGIE face det + SGIE arcface)
- Crops are produced on GPU with nvds_obj_enc_process (NV12 surface)
- JPEG bytes are attached as NVDS_CROP_IMAGE_META; we read them in the same probe
  after calling nvds_obj_enc_finish(ctx)
- We then do a tiny CPU step only to letterbox-pad to 112x112 (black pad) and save
- Embeddings are read from SGIE tensor meta and matched with cosine similarity

Notes / requirements:
* nvds_obj_enc_process expects NV12 input. Therefore the probe is attached to the
  SGIE **src** pad (before any RGBA conversion). Do **not** move it after nvdsosd.
* JPEG hardware encoder is required. Running under WSL is unsupported and can
  segfault. Prefer native Linux (Ubuntu) or Jetson. 
* DeepStream >= 7.1 Python bindings expose nvds_obj_enc_* in pyds.
"""

import sys, os, json, ctypes, argparse
from urllib.parse import urlparse
import numpy as np
import cv2
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import pyds

PGIE = "/opt/models/yolov10_face/pgie_yolov10_face.txt"
SGIE = "/opt/models/recognition/sgie_arcface.txt"  # must set output-tensor-meta=1

# ---------------------- helpers ----------------------

def make(factory, name=None, **props):
    el = Gst.ElementFactory.make(factory, name)
    if not el:
        raise RuntimeError(f"create fail: {factory}")
    for k, v in props.items():
        el.set_property(k, v)
    return el

def link_many(*els):
    for a, b in zip(els, els[1:]):
        if not a.link(b):
            raise RuntimeError(f"link fail: {a.get_name()} -> {b.get_name()}")

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def np_from_layer(layer: pyds.NvDsInferLayerInfo) -> np.ndarray:
    dt = layer.dataType
    ctype = {
        pyds.NvDsInferDataType.FLOAT: ctypes.c_float,
        pyds.NvDsInferDataType.HALF: ctypes.c_uint16,
        pyds.NvDsInferDataType.INT32: ctypes.c_int32,
        pyds.NvDsInferDataType.INT8: ctypes.c_int8,
    }.get(dt, ctypes.c_float)
    n = int(layer.inferDims.numElements)
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctype))
    arr = np.ctypeslib.as_array(ptr, shape=(n,))
    if dt == pyds.NvDsInferDataType.HALF:
        arr = arr.view(np.float16).astype(np.float32)
    return np.array(arr, copy=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    an = a / (np.linalg.norm(a) + 1e-10)
    bn = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(an, bn))

def letterbox_pad(img_bgr: np.ndarray, out_size: int = 112) -> np.ndarray:
    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = min(out_size / w, out_size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    xoff = (out_size - nw) // 2
    yoff = (out_size - nh) // 2
    canvas[yoff:yoff+nh, xoff:xoff+nw] = resized
    return canvas

# ---------------------- FaceDB ----------------------

class FaceDB:
    def __init__(self, out_dir, json_path, sim_thresh=0.5, save_interval_frames=60):
        self.out_dir = out_dir
        self.json_path = json_path
        self.sim_thresh = float(sim_thresh)
        self.save_interval_frames = int(save_interval_frames)
        os.makedirs(out_dir, exist_ok=True)
        self.data = {"identities": []}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass
        self.means = []
        self.last_frame = []
        for ident in self.data.get("identities", []):
            emb = np.array(ident.get("mean_embedding", []), dtype=np.float32)
            self.means.append(emb if emb.size else None)
            self.last_frame.append(int(ident.get("last_frame", -1)))

    def _persist(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def match(self, vec: np.ndarray):
        if not self.means:
            return -1, 0.0
        sims = [cosine_sim(vec, m) if m is not None else -1.0 for m in self.means]
        idx = int(np.argmax(sims))
        return (idx, sims[idx]) if sims[idx] >= self.sim_thresh else (-1, sims[idx])

    def _update_mean(self, idx, new_vec):
        ident = self.data["identities"][idx]
        n = ident.get("count", 0)
        old = self.means[idx]
        if old is None or n == 0:
            mean = new_vec
        else:
            mean = (old * n + new_vec) / (n + 1)
            mean = mean / (np.linalg.norm(mean) + 1e-10)
        self.means[idx] = mean.astype(np.float32)
        ident["mean_embedding"] = self.means[idx].tolist()
        ident["count"] = n + 1

    def add_or_update(self, vec: np.ndarray, crop_bgr: np.ndarray, meta: dict):
        idx, sim = self.match(vec)
        frame_num = int(meta.get("frame_num", -1))
        ts = int(meta.get("ntp_ts", 0))
        if idx >= 0:
            can_save = (frame_num - self.last_frame[idx]) >= self.save_interval_frames
            ident = self.data["identities"][idx]
            ident_id = ident["id"]
            if can_save and crop_bgr is not None:
                fname = f"{ident_id}_f{frame_num}.jpg"
                cv2.imwrite(os.path.join(self.out_dir, fname), crop_bgr)
                ident.setdefault("samples", []).append({
                    "frame_num": frame_num,
                    "ntp_ts": ts,
                    "similarity": sim,
                    "bbox": meta.get("bbox"),
                    "file": fname,
                })
                self.last_frame[idx] = frame_num
            self._update_mean(idx, vec)
            ident["last_frame"] = self.last_frame[idx]
            self._persist()
            return ident_id, sim, idx, False
        else:
            ident_id = f"id_{len(self.data['identities']):06d}"
            fname = f"{ident_id}_f{frame_num}.jpg" if crop_bgr is not None else None
            if crop_bgr is not None:
                cv2.imwrite(os.path.join(self.out_dir, fname), crop_bgr)
            entry = {
                "id": ident_id,
                "count": 1,
                "mean_embedding": (vec / (np.linalg.norm(vec) + 1e-10)).tolist(),
                "last_frame": frame_num,
                "samples": [{
                    "frame_num": frame_num,
                    "ntp_ts": ts,
                    "similarity": 1.0,
                    "bbox": meta.get("bbox"),
                    "file": fname,
                }],
            }
            self.data["identities"].append(entry)
            self.means.append(np.array(entry["mean_embedding"], dtype=np.float32))
            self.last_frame.append(frame_num)
            self._persist()
            return ident_id, 1.0, len(self.data["identities"]) - 1, True

# ---------------------- probes ----------------------

def sgie_src_probe(pad, info, user_data):
    """SGIE src (NV12) üzerinde çalışır: GPU JPEG crop + tensörden embedding okuma.
    Her obje için nvds_obj_enc_process çağrılır; nvds_obj_enc_finish(ctx) sonrası
    obje metasındaki NVDS_CROP_IMAGE_META'dan JPEG baytları okunur.
    """
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK

    db: FaceDB = user_data["db"]
    out_size = int(user_data.get("out_size", 112))
    verbose = bool(user_data.get("verbose", False))
    ctx = user_data.get("obj_ctx")
    if ctx is None:
        # ObjEnc context dışarıda create/destroy edilmeli.
        return Gst.PadProbeReturn.OK

    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch:
        return Gst.PadProbeReturn.OK

    # (ometa_ptr -> (vec, fmeta)) haritası
    vec_map = {}

    l_frame = batch.frame_meta_list
    while l_frame:
        fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = fmeta.obj_meta_list
        while l_obj:
            ometa = pyds.NvDsObjectMeta.cast(l_obj.data)

            # ---- SGIE tensöründen embedding oku ----
            vec = None
            l_user = ometa.obj_user_meta_list
            while l_user:
                um = pyds.NvDsUserMeta.cast(l_user.data)
                if um and um.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tmeta = pyds.NvDsInferTensorMeta.cast(um.user_meta_data)
                    if tmeta and tmeta.num_output_layers > 0:
                        layer = pyds.get_nvds_LayerInfo(tmeta, 0)
                        v = np_from_layer(layer)
                        # Güvenli L2 norm
                        n = np.linalg.norm(v)
                        if n > 0:
                            vec = v / n
                        else:
                            vec = v
                        break
                l_user = l_user.next

            if vec is None:
                l_obj = l_obj.next
                continue

            # C pointer anahtar olarak
            ometa_ptr = int(pyds.get_ptr(ometa))
            vec_map[ometa_ptr] = (vec, fmeta)

            # ---- Bu obje için GPU JPEG encode planla ----
            rp = ometa.rect_params
            bw = max(1, int(rp.width))
            bh = max(1, int(rp.height))
            scale = min(out_size / bw, out_size / bh)
            sw = max(1, int(round(bw * scale)))
            sh = max(1, int(round(bh * scale)))

            args = pyds.NvDsObjEncUsrArgs()
            # pybind, sıfırlanmış bir struct döndürür; alanları açıkça ayarlıyoruz.
            args.saveImg = False
            args.attachUsrMeta = True
            args.scaleImg = True
            args.quality = 95
            args.scaledWidth = sw
            args.scaledHeight = sh
            # saveImg=True olursa: args.fileNameImg = b"path"

            # Python bağlayıcısında NvBufSurface yerine gst_buffer adresi kullanılır.
            pyds.nvds_obj_enc_process(ctx, args, hash(buf), ometa, fmeta)

            l_obj = l_obj.next
        l_frame = l_frame.next

    # Encode'ları flush et; JPEG metaları iliştirilsin
    pyds.nvds_obj_enc_finish(ctx)

    # JPEG metasını oku ve DB'yi güncelle
    l_frame = batch.frame_meta_list
    while l_frame:
        fmeta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = fmeta.obj_meta_list
        while l_obj:
            ometa = pyds.NvDsObjectMeta.cast(l_obj.data)
            ometa_ptr = int(pyds.get_ptr(ometa))
            pair = vec_map.get(ometa_ptr)
            if pair is None:
                l_obj = l_obj.next
                continue

            vec, fmeta_ref = pair

            l_user = ometa.obj_user_meta_list
            while l_user:
                um = pyds.NvDsUserMeta.cast(l_user.data)
                if um and um.base_meta.meta_type == pyds.NvDsMetaType.NVDS_CROP_IMAGE_META:
                    outp = pyds.NvDsObjEncOutParams.cast(um.user_meta_data)
                    jpeg_arr = outp.outBuffer()  # np.ndarray (uint8)
                    if jpeg_arr is not None and getattr(jpeg_arr, "size", 0) > 0:
                        # Ek kopya yapmadan imdecode
                        img = cv2.imdecode(jpeg_arr, cv2.IMREAD_COLOR)
                        if img is None:
                            # Güvenlik: decode başarısızsa devam etme
                            break
                        crop112 = letterbox_pad(img, out_size)

                        rp = ometa.rect_params
                        meta = {
                            "frame_num": int(fmeta.frame_num),
                            "ntp_ts": int(getattr(fmeta, "ntp_timestamp", 0)),
                            "bbox": [int(rp.left), int(rp.top), int(rp.width), int(rp.height)],
                            "source_id": int(getattr(fmeta, "source_id", 0)),
                            "object_id": int(getattr(ometa, "object_id", -1)),
                        }
                        ident_id, sim, idx, is_new = db.add_or_update(vec, crop112, meta)
                        if verbose:
                            status = "NEW" if is_new else "UPD"
                            print(f"[{status}] frame={meta['frame_num']} ident={ident_id} "
                                  f"sim={sim:.3f} bbox={meta['bbox']} len={vec.size}", flush=True)
                l_user = l_user.next

            l_obj = l_obj.next
        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def set_gpu_ids(gpu_id, *elements):
    for el in elements:
        try:
            el.set_property("gpu-id", int(gpu_id))
        except Exception:
            pass
def set_unified_memory(*elements):
    """Uygun elemanlarda nvbuf-memory-type'ı CUDA Unified yapmayı dener.
    Jetson'da bazı elemanlarda desteklenmeyebilir; sessizce geçilir.
    """
    try:
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        for el in elements:
            try:
                el.set_property("nvbuf-memory-type", mem_type)
            except Exception:
                pass
    except Exception:
        pass


def build(uri, width, height, timeout, probe_userdata):
    p = Gst.Pipeline.new("p")
    u = urlparse(uri)

    if u.scheme and u.scheme.lower().startswith("rtsp"):
        rtspsrc = make("rtspsrc", "rtspsrc", location=uri, latency=200, drop_on_latency=True)
        depay_h264 = make("rtph264depay", "depay_h264")
        depay_h265 = make("rtph265depay", "depay_h265")
        h264parse = make("h264parse", "h264parse")
        h265parse = make("h265parse", "h265parse")
        dec = make("nvv4l2decoder", "dec")
        conv1 = make("nvvideoconvert", "conv1")
        capsf = make("capsfilter", "capsf")
        capsf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
        q0 = make("queue", "q0")
        mux = make("nvstreammux", "mux",
                   batch_size=1, width=width, height=height,
                   live_source=True, batched_push_timeout=timeout)

        for el in [rtspsrc, depay_h264, depay_h265, h264parse, h265parse, dec, conv1, capsf, q0, mux]:
            p.add(el)

        linked = {"video": False}

        def on_pad_added_rtspsrc(src, pad):
            if linked["video"]:
                return
            caps = pad.get_current_caps() or pad.query_caps()
            name = caps.to_string() if caps else ""
            if "application/x-rtp" not in name:
                return

            if "H265" in name or "encoding-name=H265" in name:
                sink = depay_h265.get_static_pad("sink")
                if not sink.is_linked():
                    pad.link(sink)
                if not depay_h265.is_linked():
                    depay_h265.link(h265parse)
                    link_many(h265parse, dec, conv1, capsf, q0)
            else:
                sink = depay_h264.get_static_pad("sink")
                if not sink.is_linked():
                    pad.link(sink)
                if not depay_h264.is_linked():
                    depay_h264.link(h264parse)
                    link_many(h264parse, dec, conv1, capsf, q0)

            sinkpad = mux.request_pad_simple("sink_0")
            srcpad = q0.get_static_pad("src")
            assert srcpad.link(sinkpad) == Gst.PadLinkReturn.OK
            linked["video"] = True

        rtspsrc.connect("pad-added", on_pad_added_rtspsrc)

    else:
        src = make("filesrc", "src", location=uri)
        demux = make("qtdemux", "demux")
        parse = make("h264parse", "parse")
        dec = make("nvv4l2decoder", "dec")
        conv1 = make("nvvideoconvert", "conv1")
        capsf = make("capsfilter", "capsf")
        capsf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12"))
        q0 = make("queue", "q0")
        mux = make("nvstreammux", "mux",
                   batch_size=1, width=width, height=height,
                   live_source=False, batched_push_timeout=timeout)

        for el in [src, demux, parse, dec, conv1, capsf, q0, mux]:
            p.add(el)

        def on_pad_added_demux(demux, pad, target):
            sinkpad = target.get_static_pad("sink")
            if not sinkpad.is_linked():
                pad.link(sinkpad)

        demux.connect("pad-added", on_pad_added_demux, parse)
        link_many(src, demux)
        link_many(parse, dec, conv1, capsf, q0)
        sinkpad = mux.request_pad_simple("sink_0")
        srcpad = q0.get_static_pad("src")
        assert srcpad.link(sinkpad) == Gst.PadLinkReturn.OK

    pgie = make("nvinfer", "pgie", config_file_path=PGIE)
    sgie = make("nvinfer", "sgie", config_file_path=SGIE, process_mode=2)

    # Görüntüleme dalı (opsiyonel)
    conv_rgba = make("nvvideoconvert", "conv_rgba")
    caps_rgba = make("capsfilter", "caps_rgba")
    caps_rgba.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
    osd = make("nvdsosd", "osd", process_mode=1, display_text=False)
    conv2 = make("nvvideoconvert", "conv2")
    vconv = make("videoconvert", "vconv")
    sink = make("fakesink", "sink", sync=False) if not os.environ.get("DISPLAY") \
        else make("nveglglessink", "sink", sync=False)

    for el in [pgie, sgie, conv_rgba, caps_rgba, osd, conv2, vconv, sink]:
        p.add(el)

    set_unified_memory(conv1, conv_rgba, conv2)
    set_gpu_ids(0, mux, pgie, sgie, conv1, conv_rgba, conv2, osd)

    # mux → pgie → sgie → (display branch)
    link_many(mux, pgie, sgie, conv_rgba, caps_rgba, osd, conv2, vconv, sink)

    # SGIE src (NV12) üzerine probe
    sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, sgie_src_probe, probe_userdata)

    return p

# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Dosya veya RTSP/HTTP URI")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--timeout", type=int, default=40000)
    parser.add_argument("--out_dir", default="faces_out")
    parser.add_argument("--json", default=None, help="JSON çıktı yolu (varsayılan: out_dir/faces.json)")
    parser.add_argument("--sim", type=float, default=0.5, help="Cosine similarity eşiği")
    parser.add_argument("--save_every", type=int, default=60, help="Aynı kimlik için en az kaç frame sonra kaydedilsin")
    parser.add_argument("--size", type=int, default=112, help="Crop çıktı boyutu (square)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    json_path = args.json or os.path.join(args.out_dir, "faces.json")
    db = FaceDB(args.out_dir, json_path, sim_thresh=args.sim, save_interval_frames=args.save_every)

    Gst.init(None)

    # create object encoder context on GPU 0
    obj_ctx = pyds.nvds_obj_enc_create_context(0)

    pipe = build(
        args.video,
        width=args.width,
        height=args.height,
        timeout=args.timeout,
        probe_userdata={"db": db, "out_size": args.size, "verbose": args.verbose, "obj_ctx": obj_ctx},
    )

    bus = pipe.get_bus(); bus.add_signal_watch(); loop = GLib.MainLoop()

    def on_msg(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print("ERROR:", err, "\n", dbg)
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()
        return True

    bus.connect("message", on_msg)
    pipe.set_state(Gst.State.PLAYING)
    print("Running… Ctrl+C ile çıkabilirsiniz.")
    try:
        loop.run()
    finally:
        try:
            # ensure encoder is flushed and destroyed
            pyds.nvds_obj_enc_finish(obj_ctx)
        except Exception:
            pass
        try:
            pyds.nvds_obj_enc_destroy_context(obj_ctx)
        except Exception:
            pass
        pipe.set_state(Gst.State.NULL)
    return 0

if __name__ == "__main__":
    sys.exit(main())
