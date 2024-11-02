import os
from llama_cpp import Llama

# Path to the GGUF model file
model_path = r"C:\Users\Bill\.cache\huggingface\hub\Custom-models--bartowski--Llama\Llama-3.2-3B-Instruct-Q6_K_L.gguf"

# Load the model
llm = Llama(model_path=model_path)

def summarize_text(file_path):
    # Read the input text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read()

    # Add instruction to summarize the text
    prompt_text = f"Summarize the following text:\n\n{input_text}"

    # Generate the summary with llama.cpp
    response = llm(prompt_text, max_tokens=100, temperature=0.7, top_p=0.9)

    # Extract the summary from the response
    summary = response["choices"][0]["text"].strip()
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
