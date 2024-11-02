import torch
from transformers import pipeline

# Define model and load the pipeline
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,  # Optimized for performance
    device_map="auto"            # Uses GPU if available
)

# Function to correct text
def correct_text(stt_output):
    prompt = f"Correct the following text for formatting, grammatical, contextual and other and minor errors: {stt_output}"
    result = pipe(prompt, max_new_tokens=150, temperature=0.5)
    return result[0]["generated_text"]

# Example usage
stt_output = "this is the stt output that needs corrections"
corrected_output = correct_text(stt_output)
print("Corrected Output:", corrected_output)
