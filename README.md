# LlamaCppGradio
Run Llama Cpp with Python Bindings and Gradio UI

This code is meant to run Llama.cpp with python bindings and a Gradio user interface for simplicity locally

Clone this repo into a virtual environment

In terminal, install requirements
"pip3 install -r requirements.txt"

In line 5+6 of the code, replace your model path with whatever model you are currently using

MODEL_PATH  = "/Users/x/Downloads/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "/Users/x/Downloads/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"

https://huggingface.co/meta-llama
Use whatever model and mmproj of your choosing

Run the program and follow available open port for a localhost server
