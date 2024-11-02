import os
from llama_cpp import Llama

# Path to the GGUF model file
model_path = r"C:\Users\Bill\.cache\huggingface\hub\Custom-models--bartowski--Llama\Llama-3.2-3B-Instruct-Q6_K_L.gguf"

# Load the model using llama-cpp-python
llm = Llama(model_path=model_path, verbose=False)

# Generate text with a prompt
prompt = "Explain the theory of relativity."
response = llm(prompt)
print(response['choices'][0]['text'])