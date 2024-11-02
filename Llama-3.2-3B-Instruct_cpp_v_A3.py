import os
from llama_cpp import Llama

# Path to the GGUF model file
model_path = r"C:\Users\Bill\.cache\huggingface\hub\Custom-models--bartowski--Llama\Llama-3.2-3B-Instruct-Q6_K_L.gguf"

# Load the model with a context window of 512 tokens
llm = Llama(model_path=model_path, n_ctx=512)

def summarize_text_in_chunks(file_path, max_input_tokens=400, max_summary_tokens=100):
    # Read the input text from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        input_text = file.read().encode('utf-8').decode('utf-8')  # Ensure UTF-8 encoding

    # Split the text into chunks that fit the model's context window
    tokens = llm.tokenize(input_text, add_bos=True, special=False)
    summaries = []

    # Process each chunk
    for i in range(0, len(tokens), max_input_tokens):
        chunk_tokens = tokens[i:i + max_input_tokens]
        chunk_text = llm.detokenize(chunk_tokens)

        # Add summarization prompt to each chunk
        prompt_text = f"Summarize the following text:\n\n{chunk_text}"

        # Generate the summary for the chunk
        response = llm(prompt_text, max_tokens=max_summary_tokens, temperature=0.7, top_p=0.9)
        summary = response["choices"][0]["text"].strip()
        summaries.append(summary)
        print(f"Chunk {i // max_input_tokens + 1} Summary:\n{summary}\n{'-'*40}")

    # Combine all chunk summaries into one text
    combined_summary_text = " ".join(summaries)

    # Optionally, summarize the combined summaries to create a final summary
    final_prompt = f"Summarize the following text:\n\n{combined_summary_text}"
    final_response = llm(final_prompt, max_tokens=max_summary_tokens, temperature=0.7, top_p=0.9)
    final_summary = final_response["choices"][0]["text"].strip()

    return final_summary

if __name__ == "__main__":
    file_name = "text_to_summarize.txt"
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    if not os.path.exists(file_path):
        print(f"Error: The file '{file_name}' does not exist in the script's directory.")
    else:
        final_summary = summarize_text_in_chunks(file_path)
        print("Final Summary:")
        print(final_summary)
