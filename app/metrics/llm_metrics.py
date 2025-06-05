import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

def load_flan_t5_model(model_name="google/flan-t5-small"):
    logger.info2(f"Loading Model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info2(">>>")
    return tokenizer, model, device

def get_llm_similarity_score_flan_t5(tokenizer, model, device, sentence1, sentence2):
    logger.info2("Calculating LLM Score ...")
    prompt = f"""
    Strictly give score between 1 and 0, it must be in float

    Example:
    Sentence 1: "Apples are fruits."
    Sentence 2: "Apples are food."
    Similarity: 0.90

    Sentence 1: "Hello."
    Sentence 2: "Good morning."
    Similarity: 0.00

    Sentence 1: "{sentence1}"
    Sentence 2: "{sentence2}"
    Similarity:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=4,     
        num_beams=5,
        early_stopping=True,
        temperature=0.0
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    try:
        score = float(response)
        logger.info2(f"FlanT5-Small: {score:.4f} >>>")
        return max(0.0, min(1.0, score)) * 100
    except:
        return 0.0

