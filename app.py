import gradio as gr
from model import process_heatmap

with gr.Blocks() as demo:
    gr.Markdown("# AI Traffic Heatmap Generator")
    gr.Markdown("Upload a traffic video to generate a density heatmap using YOLO11n")

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
            with gr.Column():
                process_btn_NCNN = gr.Button("Generate Heatmap + Benchmark NCNN", variant="primary")
                process_btn_openvino=gr.Button("Generate Heatmap + Benchmark (OPENVINO INT8)",variant="primary")

        with gr.Column():
            output_video = gr.Video(label="Heatmap Output")
            benchmark_md = gr.Markdown()
            nb_detections=gr.Markdown()

    process_btn_NCNN.click(
        fn=lambda video, fs:process_heatmap("NCNN",video,fs), #this is done because the inputs doesn't accept strings
        inputs=[input_video, frame_skip_slider],
        outputs=[output_video, benchmark_md,nb_detections],
    )
    process_btn_openvino.click(
        fn=lambda video, fs:process_heatmap("openvino",video,fs),
        inputs=[input_video, frame_skip_slider],
        outputs=[output_video, benchmark_md,nb_detections],
    )

if __name__ == "__main__":
    demo.launch()