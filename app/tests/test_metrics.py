import unittest
from unittest.mock import patch, MagicMock
import os
import sys


# Add the project root to the Python path to allow direct imports of modules
# This assumes tests are run from the 'SentenceSimilaritySuite' directory or its parent
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import functions from your metric modules
# These imports assume that the metric functions are defined in their respective files.
# If the files don't exist yet, these lines will cause an error until they are created.
try:
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
    # Define dummy functions if modules/functions are not yet created,
    # so the test structure can still be run and understood.
    def calculate_bleu(*args, **kwargs): return 0.5
    def calculate_rouge_l_f1(*args, **kwargs): return 0.5
    def calculate_meteor(*args, **kwargs): return 0.5
    def calculate_chrf(*args, **kwargs): return 50.0
    def calculate_sentence_bert_similarity(*args, **kwargs): return 0.75
    def calculate_bertscore_f1(*args, **kwargs): return 0.85
    def get_llm_similarity_score_prompt(*args, **kwargs): return 0.65


class TestLexicalMetrics(unittest.TestCase):
    
    @patch('metrics.embedding_metrics.calculate_sentence_bert_similarity')


    def setUp(self):
        self.source_sentence_perfect = "the cat sat on the mat"
        self.target_sentence_perfect = "the cat sat on the mat"
        self.target_sentence_similar = "a cat was on the mat"
        self.target_sentence_different = "a dog chased a ball"
        self.reference_list_perfect = [self.source_sentence_perfect.split()] # For NLTK BLEU

    def test_calculate_bleu(self):
        # Perfect match
        score_perfect = calculate_bleu(self.source_sentence_perfect, self.target_sentence_perfect)
        self.assertAlmostEqual(score_perfect, 1.0, places=2, msg="BLEU score for perfect match should be ~1.0")

        # Similar match
        score_similar = calculate_bleu(self.source_sentence_perfect, self.target_sentence_similar)
        self.assertTrue(0.0 < score_similar < 1.0, "BLEU score for similar sentence should be between 0 and 1")

        # Different match
        score_different = calculate_bleu(self.source_sentence_perfect, self.target_sentence_different)
        self.assertAlmostEqual(score_different, 0.0, places=2, msg="BLEU score for different sentences should be ~0.0")

    def test_calculate_rouge_l_f1(self):
        # Perfect match
        score_perfect = calculate_rouge_l_f1(self.source_sentence_perfect, self.target_sentence_perfect)
        self.assertAlmostEqual(score_perfect, 1.0, places=2)

        # Similar match
        score_similar = calculate_rouge_l_f1(self.source_sentence_perfect, self.target_sentence_similar)
        self.assertTrue(0.0 < score_similar < 1.0)

        # Different match
        score_different = calculate_rouge_l_f1(self.source_sentence_perfect, self.target_sentence_different)
        self.assertTrue(0.0 <= score_different < 0.3) # ROUGE might give some small overlap

    def test_calculate_meteor(self):
        # Perfect match
        score_perfect = calculate_meteor(self.source_sentence_perfect, self.target_sentence_perfect)
        self.assertAlmostEqual(score_perfect, 1.0, places=1) # METEOR can be slightly off 1.0

        # Similar match
        score_similar = calculate_meteor(self.source_sentence_perfect, self.target_sentence_similar)
        self.assertTrue(0.0 < score_similar < 1.0)

        # Different match - METEOR requires WordNet and can be > 0 if any word matches (e.g. 'a')
        score_different = calculate_meteor(self.source_sentence_perfect, self.target_sentence_different)
        self.assertTrue(0.0 <= score_different < 0.5)

    def test_calculate_chrf(self):
        # Perfect match (chrF scores are typically 0-100)
        score_perfect = calculate_chrf(self.source_sentence_perfect, self.target_sentence_perfect)
        self.assertAlmostEqual(score_perfect, 100.0, places=1)

        # Similar match
        score_similar = calculate_chrf(self.source_sentence_perfect, self.target_sentence_similar)
        self.assertTrue(0.0 < score_similar < 100.0)

        # Different match
        score_different = calculate_chrf(self.source_sentence_perfect, self.target_sentence_different)
        self.assertTrue(0.0 <= score_different < 30.0) # chrF might find some char n-gram overlap

class TestEmbeddingMetrics(unittest.TestCase):

    def setUp(self):
        self.sentence1 = "This is a test sentence."
        self.sentence2_similar = "This is a test example."
        self.sentence3_different = "An apple a day keeps the doctor away."

    @patch('metrics.embedding_metrics.SentenceTransformer') # Mock the SentenceTransformer class
    def test_calculate_sentence_bert_similarity(self, MockSentenceTransformer):
        # Configure the mock model and its encode method
        mock_model_instance = MockSentenceTransformer.return_value
        mock_model_instance.encode.side_effect = lambda x, convert_to_tensor: {
            self.sentence1: [0.1, 0.2, 0.3],
            self.sentence2_similar: [0.11, 0.22, 0.33], # Similar embedding
            self.sentence3_different: [0.9, 0.8, 0.7], # Different embedding
        }[x]

        # Mock util.pytorch_cos_sim
        with patch('metrics.embedding_metrics.util.pytorch_cos_sim') as mock_cos_sim:
            # Test similar sentences
            mock_cos_sim.return_value.item.return_value = 0.95 # High similarity
            score_similar = calculate_sentence_bert_similarity(mock_model_instance, self.sentence1, self.sentence2_similar)
            self.assertAlmostEqual(score_similar, 0.95, places=2)

            # Test different sentences
            mock_cos_sim.return_value.item.return_value = 0.15 # Low similarity
            score_different = calculate_sentence_bert_similarity(mock_model_instance, self.sentence1, self.sentence3_different)
            self.assertAlmostEqual(score_different, 0.15, places=2)

    @patch('metrics.embedding_metrics.bert_score_calculate') # Mock the bert_score.score function
    def test_calculate_bertscore_f1(self, mock_bert_score_calculate):
        # Configure the mock to return (P, R, F1) tensors
        # F1 tensor with a single value
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
        self.sentence1 = "The weather is sunny today."
        self.sentence2_similar = "It's a bright and sunny day."
        self.sentence3_different = "I like to eat pizza."

    @patch('metrics.llm_prompt_evaluators.pipeline') # If using Hugging Face pipeline
    # Or @patch('your_llm_client_library.ChatCompletion.create') if using OpenAI API, etc.
    def test_get_llm_similarity_score_prompt(self, mock_llm_pipeline_or_client):
        # --- Scenario 1: LLM returns a valid score string ---
        # Mock the LLM's response generation
        # If using HF pipeline:
        mock_pipeline_instance = mock_llm_pipeline_or_client.return_value
        # This mock simulates the LLM outputting text that contains a score.
        # The actual llm_prompt_evaluators.py needs to parse this text.
        mock_pipeline_instance.return_value = [{"generated_text": "Similarity Score (0-1): 0.85"}]

        # Assuming get_llm_similarity_score_prompt internally calls the pipeline and parses the score
        score_similar = get_llm_similarity_score_prompt(mock_pipeline_instance, self.sentence1, self.sentence2_similar)
        self.assertAlmostEqual(score_similar, 0.85, places=2)

        # --- Scenario 2: LLM returns a different score string ---
        mock_pipeline_instance.return_value = [{"generated_text": "The similarity is about 0.2. It's quite low."}]
        score_different = get_llm_similarity_score_prompt(mock_pipeline_instance, self.sentence1, self.sentence3_different)
        self.assertAlmostEqual(score_different, 0.20, places=2) # Assuming your parsing logic finds "0.2"

        # --- Scenario 3: LLM returns text without a clear score (test parsing robustness) ---
        mock_pipeline_instance.return_value = [{"generated_text": "These sentences are not very alike."}]
        # Depending on how your get_llm_similarity_score_prompt handles unparsable responses:
        with self.assertRaises(ValueError): # Or check for a default low score, or None
             get_llm_similarity_score_prompt(mock_pipeline_instance, self.sentence1, self.sentence3_different)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)