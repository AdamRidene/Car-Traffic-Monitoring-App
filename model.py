from ultralytics import YOLO,solutions
import numpy as np
import os,time,ultralytics,tempfile
import cv2 as cv
from pathlib import Path


NCNN_MODEL_DIR = "yolo11n_ncnn_model" #if using NCNN
OPENVINO_INT8_DIR = "yolo11n_int8_openvino_model" #if using OPENVINO
OPENVINO_CALIB_DATA = "coco8.yaml"# Calibration dataset YAML (required for good INT8 PTQ)

def export_to_ncnn(): #if we want to use NCNN instead of openvino
    if not os.path.exists(NCNN_MODEL_DIR):
        print("Exporting YOLO11n to NCNN format...")
        model = YOLO("yolo11n.pt")
        model.export(
            format="ncnn",
            imgsz=640,
            half=False,
            dynamic=False,
            simplify=True,
        )
        print(f"Export complete: {NCNN_MODEL_DIR}/")
    else:
        print(f"NCNN model found at {NCNN_MODEL_DIR}/, skipping export.")
def _resolve_calib_yaml(yaml_path: str):
    #Return an existing calibration YAML path. Tries local path first, then Ultralytics built-in coco8.yaml.
    if not yaml_path:
        return None

    p = Path(yaml_path)
    if p.exists():
        return str(p)

    # If the user asked for coco8.yaml but it's not in the repo, we will use the packaged one
    if p.name.lower() == "coco8.yaml":
        ul_root = Path(ultralytics.__file__).resolve().parent
        packaged = ul_root / "cfg" / "datasets" / "coco8.yaml"
        if packaged.exists():
            return str(packaged)

    return None
def export_to_openvino_int8(): #If we want to use OPENVINO
    if not os.path.exists(OPENVINO_INT8_DIR):
        print("Exporting YOLO11n to OpenVINO INT8...")

        calib_yaml = _resolve_calib_yaml(OPENVINO_CALIB_DATA)
        if not calib_yaml:
            raise FileNotFoundError(
                f"OpenVINO INT8 export needs a calibration dataset YAML.\n"
                f"Tried: '{OPENVINO_CALIB_DATA}' (not found).\n"
                f"Fix: set OPENVINO_CALIB_DATA to your dataset data.yaml path."
            )

        model = YOLO("yolo11n.pt")
        model.export(
            format="openvino",
            imgsz=640,
            int8=True,
            data=calib_yaml,
            dynamic=False,
            simplify=True,
            project=".",
            name=OPENVINO_INT8_DIR,
        )
        print(f"Export complete: {OPENVINO_INT8_DIR}/")
    else:
        print(f"OpenVINO INT8 model found at {OPENVINO_INT8_DIR}/, skipping export.")
def _coerce_video_path(video_input):
    if isinstance(video_input, dict) and "path" in video_input:
        return video_input["path"]
    if isinstance(video_input, (list, tuple)) and len(video_input) > 0:
        return video_input[0]
    return video_input

def process_heatmap(selected_model,video_path, frame_skip=2):
    video_path = _coerce_video_path(video_path)
    def _make_writer(path,wid,hei,fps):
        fourcc = cv.VideoWriter_fourcc(*"avc1")
        w = cv.VideoWriter(path, fourcc, fps, (wid, hei))
        if not w.isOpened():
            print("Warning: H.264 (avc1) failed. Trying VP9 (vp09)...")
            fourcc = cv.VideoWriter_fourcc(*"vp09")
            w = cv.VideoWriter(path, fourcc, fps, (wid, hei))
        if not w.isOpened():
            print("Warning: VP9 failed. Falling back to mp4v (may not play in browser)...")
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            w = cv.VideoWriter(path, fourcc, fps, (wid, hei))
        return w
    if selected_model=="NCNN":
        export_to_ncnn()
        ncnn_model=YOLO(NCNN_MODEL_DIR)
        ncnn_model.predictor=None
        cap=cv.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Failed to open video."
        width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv.CAP_PROP_FPS) or 25.0
        heatmap = solutions.Heatmap(
        show=False,
        model=ncnn_model,
        colormap=cv.COLORMAP_PARULA,
        classes=[2, 5, 7],
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        out = _make_writer(output_path,width,height,fps)
        if not out.isOpened():
            cap.release()
            out.release()
            return None, "Failed to create video writer."

        frame_idx = 0
        last_result_frame = None
        n_infers = 0
        t_sum = 0.0
        total_detections = 0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            if frame_idx % int(frame_skip) == 0:
                t0 = time.perf_counter()
                results = heatmap(im0)
                t_sum += (time.perf_counter() - t0)

                last_result_frame = results.plot_im
                n_infers += 1
                try:
                    if getattr(heatmap, 'clss', None) is not None:
                        current_frame_count = sum(1 for cls in heatmap.clss if int(cls) in [2, 5, 7])
                        total_detections += current_frame_count
                except Exception:
                    pass

            if last_result_frame is not None:
                out.write(last_result_frame)

            frame_idx += 1

        cap.release()
        out.release()

        if n_infers > 0 and t_sum > 0:
            avg_ms = (t_sum / n_infers) * 1000.0
            fps_eff = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            bench_md = (
                "## Benchmark NCNN Model\n\n"
                "| Backend | Inferences | Avg time / inference (ms) | Approx. FPS |\n"
                "|---|---:|---:|---:|\n"
                f"| NCNN | {n_infers} | {avg_ms:.2f} | {fps_eff:.2f} |\n"
            )
        else:
            bench_md = "## Benchmark NCNN\n\nNo frames were processed."
        nb_detections=f"## {total_detections} vehicule"
        return output_path, bench_md,nb_detections
    else:
        export_to_openvino_int8()
        openvino_model=YOLO(OPENVINO_INT8_DIR)
        openvino_model.predictor=None
        cap=cv.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Failed to open video."
        width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv.CAP_PROP_FPS) or 25.0
        heatmap = solutions.Heatmap(
        show=False,
        model=openvino_model,
        colormap=cv.COLORMAP_PARULA,
        classes=[2, 5, 7],
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name

        out = _make_writer(output_path,width,height,fps)
        if not out.isOpened():
            cap.release()
            out.release()
            return None, "Failed to create video writer."

        frame_idx = 0
        last_result_frame = None
        n_infers = 0
        t_sum = 0.0
        total_detections=0

        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break

            if frame_idx % int(frame_skip) == 0:
                t0 = time.perf_counter()
                results = heatmap(im0)
                t_sum += (time.perf_counter() - t0)

                last_result_frame = results.plot_im
                n_infers += 1
                try:
                    if getattr(heatmap, 'clss', None) is not None:
                        current_frame_count = sum(1 for cls in heatmap.clss if int(cls) in [2, 5, 7])
                        total_detections += current_frame_count
                except Exception:
                    pass

            if last_result_frame is not None:
                out.write(last_result_frame)

            frame_idx += 1

        cap.release()
        out.release()

        if n_infers > 0 and t_sum > 0:
            avg_ms = (t_sum / n_infers) * 1000.0
            fps_eff = 1000.0 / avg_ms if avg_ms > 0 else 0.0
            bench_md = (
                "## Benchmark (OPENVINO INT8)\n\n"
                "| Backend | Inferences | Avg time / inference (ms) | Approx. FPS |\n"
                "|---|---:|---:|---:|\n"
                f"| OPENVINO INT8 | {n_infers} | {avg_ms:.2f} | {fps_eff:.2f} |\n"
            )
        else:
            bench_md = "## Benchmark (OPENVINO INT8)\n\nNo frames were processed."
        nb_detections=f"## {total_detections} vehicule "

        return output_path, bench_md,nb_detections
