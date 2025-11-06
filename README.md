🦙 Run Llama.cpp with Python Bindings and Gradio UI

A lightweight local interface for running Llama.cpp models (including vision-enabled GGUFs) using Python bindings and a Gradio web UI.
This lets you chat or test multimodal (text + image) models directly from your browser — no cloud APIs required.

⸻

🧠 What This Project Does

This script runs a Llama.cpp￼ model locally through the llama-cpp-python￼ bindings.
It automatically launches a Gradio interface in your web browser for chatting and, if your model supports it, processing images (via .mmproj projector files).

⸻

⚙️ Prerequisites

1. Hardware
	•	macOS (Apple Silicon strongly recommended — tested on M1/M2/M3/M4)
	•	At least 8 GB RAM for smaller models
(16 GB + recommended for 7B + models)

2. Software
	•	Python 3.10+
	•	git command-line tools installed

⸻

🚀 Setup Instructions (Plug & Play)

Step 1 — Clone this repository

git clone https://github.com/yourusername/llama-gradio-ui.git
cd llama-gradio-ui


Step 2 — Create and activate a virtual environment

python3 -m venv .venv
source .venv/bin/activate


Step 3 — Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

Note for Apple Silicon users:
To enable GPU acceleration with Metal, install llama-cpp-python like this:

CMAKE_ARGS="-DLLAMA_METAL=on" pip install --force-reinstall llama-cpp-python


🧩 Step 4 — Download a Model and Projector

You can use any model compatible with Llama.cpp (in .gguf format).

Example: Hugging Face – Meta Llama Models￼

Example model pair

Model
Qwen2-VL-7B-Instruct-Q4_K_M.gguf
Projector
mmproj-Qwen2-VL-7B-Instruct-f16.gguf

Download both to your local system, e.g. ~/Downloads/.


🧠 Step 5 — Configure the Script

Open the main Python file (e.g. app.py, main.py, or gradio_llama.py) and update these lines:

MODEL_PATH  = "/Users/yourname/Downloads/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"

MMPROJ_PATH = "/Users/yourname/Downloads/mmproj-Qwen2-VL-7B-Instruct-f16.gguf"

If you don’t have an mmproj file, set:

MMPROJ_PATH = None
The app will run in text-only mode.

🖥️ Step 6 — Run the Program


From your project folder:
python3 main.py

You should see output similar to:
Running on local URL: http://127.0.0.1:7860

💬 Using the Interface
	1.	Enter your text prompt in the box.
	2.	(Optional) Upload an image if your model supports vision.
	3.	Press Generate.
	4.	Watch responses stream live.

If no model is loaded, the app will guide you to provide valid file paths before chatting.

⸻

🧱 Features

✅ Load any GGUF model locally
✅ Optional vision with .mmproj
✅ Streaming responses for natural feel
✅ Works offline — no API keys required
✅ GPU acceleration via Metal (macOS) or CUDA (Windows/Linux)
✅ Interactive Gradio UI with model loading inside the browser

⸻

Switch to a different model anytime:
	1.	Open the Gradio UI.
	2.	Enter a new model or projector path.
	3.	Click Load Model again.


🪶 License

MIT License — free for personal and educational use.

⸻

🌐 Resources
	•	[Llama.cpp GitHub￼](https://github.com/ggml-org/llama.cpp)
	•	[Gradio Docs￼](https://www.gradio.app)
	•	[Hugging Face Models](https://huggingface.co/models)￼

