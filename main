import gradio as gr
import src.p3g  # if under src/, ensure `pip install -e .` or PYTHONPATH includes src

def main():
    demo = src.p3g.build_ui()
    demo.queue(max_size=32).launch(server_port=7860, server_name="0.0.0.0")

if __name__ == "__main__":
    main()
