from sympy import python
import gradio as gr
from llama_cpp import Llama

# 1. Load the Llama Model
model_path = "Qwen2-VL-7B-Instruct-Q4_K_M.gguf"   # Replace with your model's path
llm = Llama(model_path=model_path, n_ctx=2048)  # Adjust n_ctx as needed

# 2. Define the Gradio Interface
def generate_response(prompt):
    """
    This function takes the user's prompt and generates a response
    using the Llama model.
    """
    output = llm(prompt)
    return output["choices"][0]["text"]  # Extract the generated text

iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=5, placeholder="Enter your prompt here..."),
    outputs=gr.Textbox(lines=5),
    title="Llama CPP Server with Gradio",
    description="A simple interface to interact with a Llama model using llama-cpp-python."
)

# 3. Launch the Gradio Server
iface.launch(server_port=7860)  # Or any available port
