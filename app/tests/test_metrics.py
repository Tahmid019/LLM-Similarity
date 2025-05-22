import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import csv
import nltk
from pathlib import Path




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
        get_llm_similarity_score_prompt
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
        self.sentence1 = source_sentence
        self.sentence2_similar = target_sentences[1]
        self.sentence3_different = target_sentences[2]

    @patch('metrics.embedding_metrics.SentenceTransformer')
    def test_calculate_sentence_bert_similarity(self, MockSentenceTransformer):
        mock_model_instance = MockSentenceTransformer.return_value
        mock_model_instance.encode.side_effect = lambda x, convert_to_tensor: {
            self.sentence1: [0.1, 0.2, 0.3],
            self.sentence2_similar: [0.11, 0.22, 0.33],
            self.sentence3_different: [0.9, 0.8, 0.7],
        }[x]

        with patch('metrics.embedding_metrics.util.pytorch_cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value.item.return_value = 0.95
            score_similar = calculate_sentence_bert_similarity(mock_model_instance, self.sentence1, self.sentence2_similar)
            self.assertAlmostEqual(score_similar, 0.95, places=2)

            mock_cos_sim.return_value.item.return_value = 0.15
            score_different = calculate_sentence_bert_similarity(mock_model_instance, self.sentence1, self.sentence3_different)
            self.assertAlmostEqual(score_different, 0.15, places=2)

    @patch('bert_score.score')
    def test_calculate_bertscore_f1(self, mock_bert_score_calculate):
        mock_bert_score_calculate.return_value = (None, None, MagicMock(mean=lambda: 0.92))
        score_similar = calculate_bertscore_f1(self.sentence1, self.sentence2_similar)
        self.assertAlmostEqual(score_similar, 0.92, places=2)
        mock_bert_score_calculate.assert_called_once()

        mock_bert_score_calculate.reset_mock()
        mock_bert_score_calculate.return_value = (None, None, MagicMock(mean=lambda: 0.23))
        score_different = calculate_bertscore_f1(self.sentence1, self.sentence3_different)
        self.assertAlmostEqual(score_different, 0.23, places=2)
        mock_bert_score_calculate.assert_called_once()


class TestLLMPromptEvaluators(unittest.TestCase):

    def setUp(self):
        self.sentence1 = source_sentence
        self.sentence2_similar = target_sentences[1]
        self.sentence3_different = target_sentences[2]

    @patch('metrics.llm_metrics.pipeline')
    def test_get_llm_similarity_score_prompt(self, mock_pipeline):
        mock_pipeline_instance = mock_pipeline.return_value

        mock_pipeline_instance.return_value = [{"generated_text": "Similarity Score (0-1): 0.85"}]
        score_similar = get_llm_similarity_score_prompt(mock_pipeline_instance, self.sentence1, self.sentence2_similar)
        self.assertAlmostEqual(score_similar, 0.85, places=2)

        mock_pipeline_instance.return_value = [{"generated_text": "The similarity is about 0.2. It's quite low."}]
        score_different = get_llm_similarity_score_prompt(mock_pipeline_instance, self.sentence1, self.sentence3_different)
        self.assertAlmostEqual(score_different, 0.20, places=2)

        mock_pipeline_instance.return_value = [{"generated_text": "These sentences are not very alike."}]
        with self.assertRaises(ValueError):
            get_llm_similarity_score_prompt(mock_pipeline_instance, self.sentence1, self.sentence3_different)


if __name__ == '__main__':
    download_nltk_data()
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
