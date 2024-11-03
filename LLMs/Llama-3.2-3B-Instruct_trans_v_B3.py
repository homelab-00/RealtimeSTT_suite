import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
model_name = "meta-llama/Llama-3.2-3B-Instruct"
local_model_dir = r"C:\Users\Bill\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95"
cuda_enabled = torch.cuda.is_available()
device = "cuda" if cuda_enabled else "cpu"

# Load the tokenizer and model
print("Loading Llama model from Hugging Face...")

if not os.path.exists(local_model_dir):
    os.makedirs(local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer.save_pretrained(local_model_dir)
    model.save_pretrained(local_model_dir)
else:
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir)

model = model.to(device)
print("Model loaded successfully!")

# Prompt configuration
with open('text_to_summarize.txt', 'r') as file:
    text = file.read()

prompt = "Summarize the following text:\n\n" + text
print(f"Running prompt: {prompt}")

# Tokenize input and generate response
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output_tokens = model.generate(**inputs, max_new_tokens=1000, temperature=0.7)
response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Print and save response
print("Response:")
print(response)