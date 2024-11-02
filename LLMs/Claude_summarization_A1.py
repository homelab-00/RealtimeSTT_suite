from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def setup_model():
    model_name = "bigscience/bloomz-7b1"
    
    # Load model with 8-bit quantization and automatic device mapping
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def generate_summary(text, model, tokenizer, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=40,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary