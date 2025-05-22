import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

llm_model_name = "google/flan-t5-small" 

tokenizer_llm = AutoTokenizer.from_pretrained(llm_model_name)
model_llm = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_llm.to(device)
print(f"Loaded LLM: {llm_model_name}")


# qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
# text_generator = pipeline("text-generation", model="gpt2") # Or a more capable model

def get_llm_similarity_score_prompt(llm_pipeline, sentence1, sentence2):
    
    prompt = f"""
    Rate the semantic similarity between the following two sentences on a continuous scale from 0 to 1, 
    where 0 means no similarity and 1 means perfect semantic equivalence.
    Sentence 1: "{sentence1}"
    Sentence 2: "{sentence2}"
    Similarity Score (0-1):
    """
    
    inputs = tokenizer_llm(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)

    outputs = model_llm.generate(**inputs, max_new_tokens=10) # Generate a short response
    response_text = tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
    
    match = re.search(r"(\d\.\d+)", response_text)

    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return f"Could not parse float from: {response_text}"
    else:
        return f"Could not find score in: {response_text}"



def calculate_sentence_log_prob(tokenizer, model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        log_likelihood = outputs.loss.item() * input_ids.shape[1] # Unnormalized
    return -log_likelihood # Higher is better (less negative)