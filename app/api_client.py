import requests
from .config import API_URL

class HuggingFaceClient:
    def __init__(self, api_token):
        self.api_token = api_token
        self.headers = {"Authorization": f"Bearer {self.api_token}"}

    def get_similarity_scores(self, source_sentence, target_sentences):
        """Get similarity scores from Hugging Face API"""
        payload = {
            "inputs": {
                "source_sentence": source_sentence,
                "sentences": target_sentences
            }
        }
        
        response = requests.post(API_URL, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()