import re

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    
    return text