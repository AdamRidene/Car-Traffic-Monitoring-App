import gradio as gr
import cv2
from ultralytics import solutions, YOLO
import os
import tempfile

NCNN_MODEL_DIR = "yolo11n_ncnn_model"

def export_to_ncnn():
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
        print(f"Export complete :{NCNN_MODEL_DIR}/")
    else:
        print(f"NCNN model found at {NCNN_MODEL_DIR}/, skipping export.")

export_to_ncnn()
ncnn_model = YOLO(NCNN_MODEL_DIR)


def process_heatmap(video_path, frame_skip=2):
    ncnn_model.predictor = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0

    heatmap = solutions.Heatmap(
        show=False,
        model=ncnn_model,
        colormap=cv2.COLORMAP_PARULA,
        classes=[2, 5, 7],
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_result_frame = None 

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Processing complete.")
            break

        if frame_idx % frame_skip == 0:
            # Process the frame through YOLO + heatmap
            results = heatmap(im0)
            last_result_frame = results.plot_im
        
        if last_result_frame is not None:
            out.write(last_result_frame)

        frame_idx += 1

    cap.release()
    out.release()
    return output_path


with gr.Blocks() as demo:
    gr.Markdown("# 🚦 AI Traffic Heatmap Generator")
    gr.Markdown("Upload a traffic video to generate a density heatmap using YOLO11n + NCNN.")

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Traffic Video")

            frame_skip_slider = gr.Slider(
                minimum=1,
                maximum=8,
                value=2,
                step=1,
                label="Frame Skip",
                info="1 = process every frame (slowest), 8 = process every 8th frame (fastest)"
            )

            process_btn = gr.Button("Generate Heatmap", variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Heatmap Output")

    process_btn.click(
        fn=process_heatmap,
        inputs=[input_video, frame_skip_slider],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch()
