import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "meta-llama/Llama-3.2-3B-Instruct-QLORA_INT4_EO8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(file_path):
    # Read the input text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # Prepend instruction for summarization
    prompt_text = f"Summarize the following text:\n\n{input_text}"

    # Tokenize the prompt text
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Calculate max_new_tokens as one quarter of input tokens, with a minimum of 100 tokens
    input_token_length = inputs['input_ids'].shape[1]
    max_new_tokens = max(100, input_token_length // 4)

    # Generate the summary with the calculated max_new_tokens
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        length_penalty=2.0,
        early_stopping=True
    )

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    file_name = "text_to_summarize.txt"
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_name}' does not exist in the script's directory.")
    else:
        summary = summarize_text(file_path)
        print("Summary:")
        print(summary)
