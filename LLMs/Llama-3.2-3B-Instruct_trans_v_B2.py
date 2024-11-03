import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
model_name = "meta-llama/Llama-3.2-3B-Instruct"
cuda_enabled = torch.cuda.is_available()
device = "cuda" if cuda_enabled else "cpu"

# Load the tokenizer and model
print("Loading Llama model from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
print("Model loaded successfully!")

# Prompt configuration

with open('text_to_summarize.txt', 'r') as file:
    text = file.read()

prompt = "Summarize the following text:\n\n" + text
print(f"Running prompt: {prompt}")

# Tokenize input and generate response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output_tokens = model.generate(**inputs, max_length=150, temperature=0.7)
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Print and save response
print("Response:")
print(response)

# Save response to file (for easy review)
output_path = "response_output.txt"
with open(output_path, "w") as file:
    file.write(response)
print(f"Response saved to {output_path}")
