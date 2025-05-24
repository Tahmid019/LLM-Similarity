import unittest
import os
import sys
import csv
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


try:
    logger.info(" === Importing Materials ===")
    from data.samples import (
        source_sentence,
        target_sentences
    )
    logger.info("Data Samples Imported")
    from config.nltkSetup import (
        download_nltk_data
    )
    logger.info("NLTK config Imported")
    from metrics.lexical_metrics import (
        calculate_bleu,
        calculate_rouge_l_f1, 
        calculate_meteor,
        calculate_chrf
    )
    logger.info("Lexical Metrics Functions Imported")
    from metrics.embedding_metrics import (
        load_sbert_model,
        calculate_sentence_bert_similarity,
        calculate_bertscore_f1
    )
    logger.info("Embedding Metrics Functions Imported")
    from metrics.llm_metrics import (
        load_flan_t5_model,
        get_llm_similarity_score_flan_t5
    )
    logger.info("LLM Metrics Functions Imported")
except ImportError as e:
    logger.error(f"ImportError: {e}. Make sure metric modules and functions are defined.")
    print(f"ImportError: {e}. Make sure metric modules and functions are defined.")
    def calculate_bleu(*args, **kwargs): return 0.5
    def calculate_rouge_l_f1(*args, **kwargs): return 0.5
    def calculate_meteor(*args, **kwargs): return 0.5
    def calculate_chrf(*args, **kwargs): return 50.0
    def calculate_sentence_bert_similarity(*args, **kwargs): return 0.75
    def calculate_bertscore_f1(*args, **kwargs): return 0.85
    def get_llm_similarity_score_prompt(*args, **kwargs): return 0.65

class TestLexicalMetrics(unittest.TestCase):

    logger.info("Lexical Test")
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

            # self.assertTrue(0.0 <= result_row["BLEU"] <= 1.0)
            # self.assertTrue(0.0 <= result_row["ROUGE-L"] <= 1.0)
            # self.assertTrue(0.0 <= result_row["METEOR"] <= 1.0)
            # self.assertTrue(0.0 <= result_row["chrF"] <= 100.0)

        csv_path = os.path.join(project_root, "test_results.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)

class TestEmbeddingMetrics(unittest.TestCase):

    logger.info("Embedding Test")
    def setUp(self):
        self.sbert_model = load_sbert_model()
        self.source_sentence = source_sentence
        self.target_sentences = target_sentences
        self.results = []

    def test_embedding_metrics(self):
        
        for target in self.target_sentences:
            result_row = {
                "target_sentence" : target,
                "SBERT" : calculate_sentence_bert_similarity(self.sbert_model, self.source_sentence, target),
                "BERTSCORE_F1" : calculate_bertscore_f1(target, self.source_sentence)
            }
            self.results.append(result_row)
            
            # self.assertTrue(0.0 <= result_row['SBERT'] <= 1.0)
            # self.assertTrue(0.0 <= result_row['BERTSCORE_F1'] <= 1.0)
            
        csv_path = os.path.join(project_root, "test_results2.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
            
class TestLLMFlanT5Similarity(unittest.TestCase):

    logger.info("LLM Test")
    def setUp(self):
        self.tokenizer, self.model, self.device = load_flan_t5_model()
        self.source_sentence = source_sentence
        self.target_sentences = target_sentences
        self.results = []
        
    def test_llm_metrics(self):
        
        for target in self.target_sentences:
            result_row = {
                "target_sentence" : target,
                "LLM/Flan_t5" : get_llm_similarity_score_flan_t5(self.tokenizer, self.model, self.device, self.source_sentence, target),
            }
            self.results.append(result_row)
            
            # self.assertTrue(0.0 <= result_row['LLM/Flan_t5'] <= 1.0)
            
        csv_path = os.path.join(project_root, "test_results3.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)


if __name__ == '__main__':
    logger.info("Downloading NLTK data")
    download_nltk_data()
    
    logger.info("Starting Unittest")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
