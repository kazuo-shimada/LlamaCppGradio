Local Multimodal LLM (GGUF + mmproj) — Gradio App

This repository contains a simple, offline Gradio application that lets you run local GGUF language models with optional vision capabilities (image understanding) using llama-cpp-python. You can:

•	Load any text-only GGUF model for chat.

•	Load a vision-capable model (e.g., Qwen2-VL / Qwen2.5-VL, LLaVA-family) together with a matching mmproj to ask questions about images.

•	Stream responses in the UI.

•	Change models at runtime without restarting the app.

The app is designed for beginners: drag-and-drop the model files, pick a handler, click “Load,” and start chatting (and optionally add images).

Contents

1.	Features

2.	Requirements

3.	Quick Start (most users)

4.	Step-by-step Installation (beginner friendly)

5.	Running the App

6.	Using the UI (what each control does)

7.	Choosing the Correct Handler (VL vs text-only)

8.	Typical Workflows

9.	Troubleshooting and Common Errors

10.	Performance Tips

11.	Security, Privacy, and Offline Notes

12.	FAQ

1) Features

•	Works completely offline after install.

•	Drag-and-drop model selection at runtime (no code edits required).

•	Supports both text-only and vision-language (VL) models.

•	Streams tokens for a responsive chat experience.

•	Simple, single-file app for easy modification.

2) Requirements

•	Operating system: macOS (Apple Silicon recommended), Linux, or Windows.

•	Python: 3.10–3.13.

•	RAM/VRAM: depends on the model you choose. Smaller quantized models (Q4_K_M, etc.) require less memory.

•	Disk space: enough to store the GGUF model and (if needed) its mmproj file.


3) Quick Start (most users)

1.	Install Python 3.10–3.13.

2.	Create and activate a virtual environment.

3.	Install dependencies with: pip install --upgrade "llama-cpp-python[server]" gradio

4.	Save the app script as app.py in a new folder.
	
5.	Run with: python app.py

6.	In your browser, open the printed local URL (usually http://127.0.0.1:7860).

7.	Drag in a model GGUF.
	•	For text-only: that’s all you need; choose “Raw text (no images)”.
	•	For vision: also drag in a matching mmproj GGUF and choose the right vision handler (see section 7).

8.	Click “Load / Reload model”, then chat. To analyze an image, upload an image and ask about it.

4) Step-by-step Installation (beginner friendly)

A) Install Python

•	macOS: Use the official Python installer or brew install python@3.11.

•	Windows: Use the official Python installer and check “Add Python to PATH”.

•	Linux: Use your package manager or python.org.

B) Create a project folder

•	Example: create local-vl-app and put app.py inside it.

C) Create a virtual environment

•	macOS/Linux:

•	python3 -m venv .venv

•	source .venv/bin/activate
	
•	Windows (PowerShell):

•	python -m venv .venv

•	.\.venv\Scripts\Activate.ps1

D) Upgrade pip and install dependencies

•	python -m pip install --upgrade pip

•	pip install --upgrade "llama-cpp-python[server]" gradio

On Apple Silicon, the prebuilt llama-cpp-python wheels use Metal acceleration automatically when available.

5) Running the App

•	Place app.py in your project folder.

•	Activate your virtual environment (see 4C).

•	Run: python app.py

•	Open the printed local URL in your browser (default port is 7860).

•	The UI will appear with controls to load models and chat.

6) Using the UI (what each control does)

•	Model GGUF: Drop your .gguf language model here (e.g., Qwen2-VL-7B-Instruct-Q4_K_M.gguf, Llama-3-8B.Q4_K_M.gguf, etc.).

•	mmproj GGUF: Drop the matching mmproj .gguf file here if you want vision (image) support. Text-only models do not need this.

•	Vision handler (dropdown):

•	Auto from filename: Tries a best guess from file names (e.g., detects “qwen2.5-vl”, “qwen2-vl”, “llava”).

•	Qwen2.5-VL (chat_format): For Qwen2.5-VL models. Requires an mmproj.

•	LLaVA15 (chat_handler): For Qwen2-VL (non-2.5) and other LLaVA-like models. Requires an mmproj.

•	Raw text (no images): For text-only models (no image ingestion).

•	n_ctx: Context window (token capacity). Larger helps, but uses more RAM/VRAM. 8192 is a good starting point for VL.

•	n_gpu_layers: Set -1 to offload as many layers to GPU as possible (recommended on Apple Silicon).
	
•	verbose: Prints extra logs to your terminal. Useful for debugging.

•	Load / Reload model: Applies your selected files and settings. Must be clicked after choosing a model and handler.

•	Prompt: Your user message.

•	Image (optional): Upload an image to discuss (only used if a vision handler is active and a matching mmproj is loaded).

•	Generate: Starts streaming the model’s response.

7) Choosing the Correct Handler (VL vs text-only)

•	Text-only model (e.g., “EdgeRunner-Light”, many LLaMA-based chat models):

•	Choose “Raw text (no images)”.

•	No mmproj required. If you upload an image, it will be ignored by design.

•	Vision model (Qwen2.5-VL + mmproj):

•	Choose “Qwen2.5-VL (chat_format)”.

•	Provide the matching mmproj file.
	
•	You can then upload an image and ask questions about it.
	
•	Vision model (Qwen2-VL non-2.5, LLaVA-family, InternVL-style):
	
•	Choose “LLaVA15 (chat_handler)”.

•	Provide the matching mmproj file.

•	You can then upload an image and ask questions about it.

If you are unsure, try “Auto from filename” first. If the app still reports “text-only”, explicitly pick the correct VL handler and ensure you provided an mmproj.

8) Typical Workflows

A) Pure text chat

•	Drag in a text-only .gguf.

•	Select “Raw text (no images)”.

•	Click “Load / Reload model”.

•	Type a prompt and click “Generate”.

B) Multimodal (image + text) with Qwen2.5-VL

•	Drag in Qwen2.5-VL-... .gguf as Model GGUF.

•	Drag in the matching mmproj-... .gguf as mmproj.

•	Select “Qwen2.5-VL (chat_format)”.

•	Click “Load / Reload model”.

•	Add an image then a prompt; click “Generate”.

C) Multimodal with Qwen2-VL (non-2.5) or LLaVA-family

•	Drag in a VL model .gguf.

•	Drag in its matching mmproj .gguf.

•	Select “LLaVA15 (chat_handler)”.

•	Load and then ask about an uploaded image.

9) Troubleshooting and Common Errors

“Model is not loaded. Load a model first.”

•	You need to click “Load / Reload model” after selecting files and options.

“I uploaded an image but the model says: please provide an image!”

•	You likely loaded a text-only model or chose “Raw text (no images)”.

•	Load a VL model and mmproj, and choose the appropriate VL handler.

“Current model/handler is text-only. Load a VL model and mmproj.”

•	The app is blocking image requests intentionally because your current config can’t process images. Load a VL model + mmproj.

“Load error: …” during model load

•	The GGUF might be corrupted or incompatible.

•	Ensure paths are correct and you’re using a recent llama-cpp-python.

•	Try a smaller model or reduce n_ctx.

The app runs, but answers are generic or it seems to ignore the image

•	Increase n_ctx (8192 or higher). Large image embeddings require more context.

•	Make sure the mmproj file truly matches the model family.

•	Use the correct handler (“Qwen2.5-VL (chat_format)” for Qwen2.5-VL, “LLaVA15 (chat_handler)” for Qwen2-VL or LLaVA-like).

“externally-managed-environment” when installing packages (macOS)

•	Create and use a virtual environment. Example:

•	python3 -m venv .venv

•	source .venv/bin/activate
	
•	pip install --upgrade "llama-cpp-python[server]" gradio

The “images kwarg” appears missing in Python checks

•	That’s expected with the chat-completions API. Vision is handled via the chat formatting/handler and the mmproj, not a direct “images” argument.

GPU/Metal memory errors or slow performance

•	Try a smaller model (lower parameter count or heavier quantization).

•	Keep n_ctx moderate (e.g., 8192).

•	Ensure n_gpu_layers is -1 on Apple Silicon to offload layers automatically.

10) Performance Tips

•	Prefer quantized models (e.g., Q4_K_M) for lower RAM/VRAM usage.

•	Apple Silicon users benefit from the default Metal acceleration in the prebuilt wheels.

•	Increase n_ctx only as needed; larger context uses more memory.

•	Keep the Gradio queue size modest (the default in the script is already reasonable).

•	Close other heavy applications while running large models.

11) Security, Privacy, and Offline Notes

•	This app runs locally and does not require an internet connection after you install dependencies.

•	All prompts, images, and model files remain on your machine.

•	Do not expose the Gradio server to the public internet unless you know what you’re doing. The default server_name="0.0.0.0" is convenient for LAN access; if you don’t need LAN access, you can set it to 127.0.0.1 to restrict to your machine.

12) FAQ

Q: Do I always need an mmproj?

A: Only for vision (image) models. Text-only models do not use an mmproj.

Q: How do I know if my model is vision-capable?

A: Model names often include “VL” (e.g., Qwen2-VL, Qwen2.5-VL) or belong to the LLaVA family. If unsure, try “Auto from filename,” then explicitly pick the handler if auto-detection is wrong.

Q: The app still says text-only even though I selected a VL handler.

A: Verify that you provided a valid mmproj path and that the mmproj matches the model family. Then click “Load / Reload model” again.

Q: Can I change models without restarting?

A: Yes. Load a different GGUF (and mmproj if needed), pick the handler, and click “Load / Reload model”.

Q: Why does the app stream partial text?

A: Streaming improves responsiveness so you can see tokens as they’re generated.

Q: Can I run larger models?
A: Yes, but they may require more memory and could be slower. Start with smaller models and scale up.
