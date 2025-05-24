import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import csv
import nltk
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


sbert_model_name = 'all-MiniLM-L6-v2'


nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    logger.info("nltk Downloaded")


    

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

all_results = []
csv_path = os.path.join(project_root, "all_metrics_results.csv")

try:
    from data.samples import (
        source_sentence,
        target_sentences
    )
    from metrics.lexical_metrics import (
        calculate_bleu,
        calculate_rouge_l_f1, 
        calculate_meteor,
        calculate_chrf
    )
    from metrics.embedding_metrics import (
        calculate_sentence_bert_similarity,
        calculate_bertscore_f1
    )
    from metrics.llm_metrics import (
        load_flan_t5_model,
        get_llm_similarity_score_flan_t5
    )
except ImportError as e:
    print(f"ImportError: {e}. Make sure metric modules and functions are defined.")
    def calculate_bleu(*args, **kwargs): return 0.5
    def calculate_rouge_l_f1(*args, **kwargs): return 0.5
    def calculate_meteor(*args, **kwargs): return 0.5
    def calculate_chrf(*args, **kwargs): return 50.0
    def calculate_sentence_bert_similarity(*args, **kwargs): return 0.75
    def calculate_bertscore_f1(*args, **kwargs): return 0.85
    def get_llm_similarity_score_prompt(*args, **kwargs): return 0.65


class TestLexicalMetrics(unittest.TestCase):

    def setUp(self):
        self.source_sentence = source_sentence
        self.target_sentences = target_sentences
        self.results = []

    def test_lexical_metrics(self):
        for target in self.target_sentences:
            result_row = {
                "target_sentence": target,
                "BLEU": calculate_bleu(self.source_sentence, target),
                "ROUGE-L": calculate_rouge_l_f1(self.source_sentence, target),
                "METEOR": calculate_meteor(self.source_sentence, target),
                "chrF": calculate_chrf(self.source_sentence, target)
            }
            self.results.append(result_row)

            # Assert ranges
            self.assertTrue(0.0 <= result_row["BLEU"] <= 1.0)
            self.assertTrue(0.0 <= result_row["ROUGE-L"] <= 1.0)
            self.assertTrue(0.0 <= result_row["METEOR"] <= 1.0)
            self.assertTrue(0.0 <= result_row["chrF"] <= 100.0)

        # Save results to CSV
        csv_path = os.path.join(project_root, "test_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

        # return self.results


class TestEmbeddingMetrics(unittest.TestCase):

    def setUp(self):
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.sentence1 = source_sentence
        self.sentence2_similar = target_sentences[1]
        self.sentence3_different = target_sentences[2]
        self.results = []

    def test_calculate_sentence_bert_similarity(self):
        score_similar = calculate_sentence_bert_similarity(
            self.sbert_model, self.sentence1, self.sentence2_similar
        )
        print("SBERT similarity (similar):", score_similar)
        self.assertTrue(0.0 <= score_similar <= 1.0)
        self.assertGreater(score_similar, 0.5)

        score_different = calculate_sentence_bert_similarity(
            self.sbert_model, self.sentence1, self.sentence3_different
        )
        print("SBERT similarity (different):", score_different)
        self.assertTrue(0.0 <= score_different <= 1.0)
        self.assertLess(score_different, score_similar)

    def test_calculate_bertscore_f1(self):
        score_similar = calculate_bertscore_f1(self.sentence1, self.sentence2_similar)
        print("BERTScore F1 (similar):", score_similar)
        self.assertTrue(0.0 <= score_similar <= 1.0)
        self.assertGreater(score_similar, 0.5)

        score_different = calculate_bertscore_f1(self.sentence1, self.sentence3_different)
        print("BERTScore F1 (different):", score_different)
        self.assertTrue(0.0 <= score_different <= 1.0)
        self.assertLess(score_different, score_similar)
        
        csv_path = os.path.join(project_root, "test_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

class TestLLMFlanT5Similarity(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.tokenizer, self.model, self.device = load_flan_t5_model()
        self.sentence1 = source_sentence
        self.sentence2_similar = target_sentences[1]
        self.sentence3_different = target_sentences[2]
        self.results = []

    def test_similarity_score_similar_sentences(self):
        score = get_llm_similarity_score_flan_t5(
            self.tokenizer, self.model, self.device,
            self.sentence1, self.sentence2_similar
        )
        print(f"Similarity (similar sentences): {score}")
        self.assertTrue(0.0 <= score <= 1.0)

    def test_similarity_score_different_sentences(self):
        score = get_llm_similarity_score_flan_t5(
            self.tokenizer, self.model, self.device,
            self.sentence1, self.sentence3_different
        )
        print(f"Similarity (different sentences): {score}")
        self.assertTrue(0.0 <= score <= 1.0)
        
        csv_path = os.path.join(project_root, "test_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

    # def test_invalid_response_handling(self):
    #     # Simulate invalid decoding by forcing an invalid prompt
    #     with self.assertRaises(ValueError):
    #         get_llm_similarity_score_flan_t5(
    #             self.tokenizer, self.model, self.device,
    #             "asdfghjkl", "zxcvbnm"
    #         )

if __name__ == '__main__':
    download_nltk_data()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
