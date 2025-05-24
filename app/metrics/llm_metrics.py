import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_flan_t5_model(model_name="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

def get_llm_similarity_score_flan_t5(tokenizer, model, device, sentence1, sentence2):
    prompt = f"""
    Sentence 1: "{sentence1}"
    Sentence 2: "{sentence2}"
    Question: How semantically similar are Sentence 1 and Sentence 2?
    Provide a similarity score from 0.0 (not similar) to 1.0 (identical in meaning).
    Answer (Score only, e.g., 0.75):
    """

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)

    outputs = model.generate(**inputs, max_new_tokens=10)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"(\d\.\d+)", response_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            raise ValueError(f"Could not parse float from: {response_text}")
    raise ValueError(f"Could not find score in: {response_text}")
