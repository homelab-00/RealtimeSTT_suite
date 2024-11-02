import os
import torch
from llama_cpp import Llama

# Configuration
# Path to the GGUF model file
model_path = r"C:\Users\Bill\.cache\huggingface\hub\Custom-models--bartowski--Llama\Llama-3.2-3B-Instruct-Q6_K.gguf"
cuda_enabled = torch.cuda.is_available()
device = "cuda" if cuda_enabled else "cpu"

# Load the model
print("Loading Llama model using llama-cpp-python...")
model = Llama(model_path=model_path, device=device, verbose=False)
print("Model loaded successfully!")

# Prompt configuration
prompt = "Tell me something interesting about space exploration."
print(f"Running prompt: {prompt}")

# Run inference
response = model(prompt, max_tokens=150, temperature=0.7)
print("Response:")
print(response)

# Save response to file (for easy review)
output_path = "response_output.txt"
with open(output_path, "w") as file:
    file.write(response)
print(f"Response saved to {output_path}")
