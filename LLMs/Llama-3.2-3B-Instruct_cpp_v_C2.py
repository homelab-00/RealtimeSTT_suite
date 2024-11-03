import os
import torch
from llama_cpp import Llama
import time

# Configuration
# Path to the GGUF model file
model_path = r"C:\Users\Bill\.cache\huggingface\hub\Custom-models--bartowski--Llama\Llama-3.2-3B-Instruct-Q6_K.gguf"
cuda_enabled = torch.cuda.is_available()

# Determine device
if cuda_enabled:
    device = "cuda"
    print("CUDA is available. Loading the model with GPU acceleration...")
else:
    device = "cpu"
    print("CUDA is not available. Loading the model on CPU...")

# Load the model
start_time = time.time()
model = Llama(model_path=model_path, device=device, use_cublas=True)
print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds!")

# Prompt configuration
prompt = "Tell me something interesting about space exploration."
print(f"Running prompt: {prompt}")

# Run inference
start_time = time.time()
response = model(prompt, max_tokens=150, temperature=0.7)
print(f"Inference completed in {time.time() - start_time:.2f} seconds")

# Print and save response
print("Response:")
print(response)

# Save response to file (for easy review)
output_path = "response_output.txt"
with open(output_path, "w") as file:
    file.write(response)
print(f"Response saved to {output_path}")
