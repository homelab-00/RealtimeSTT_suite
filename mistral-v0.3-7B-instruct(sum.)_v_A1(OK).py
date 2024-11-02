from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# Path to the Mistral model and tokenizer
mistral_models_path = r"C:\Users\Bill\.cache\huggingface\hub\models--mistralai--Mistral-7B-Instruct-v0.3"

# Initialize tokenizer and model
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

# Read the input text from file
with open("text_to_summarize.txt", "r", encoding="utf-8") as file:
    text_to_summarize = file.read()

# Formulate the instruction prompt for summarization
completion_request = ChatCompletionRequest(messages=[UserMessage(content=f"Summarize the following text: {text_to_summarize}")])

# Encode tokens and generate summary
tokens = tokenizer.encode_chat_completion(completion_request).tokens
out_tokens, _ = generate([tokens], model, max_tokens=750, temperature=0.2, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

# Decode and print/save result
summary = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

# Output summary to a file
with open("summarized_text.txt", "w", encoding="utf-8") as output_file:
    output_file.write(summary)

print("Summary:", summary)
