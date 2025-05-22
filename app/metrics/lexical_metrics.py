from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import sacrebleu


def calculate_bleu(reference_sentences, candidate_sentence):
    # NLTK expects tokenized input
    reference_tokens = [ref.split() for ref in reference_sentences]
    candidate_tokens = candidate_sentence.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie) / 100

def calculate_meteor(reference_sentence, candidate_sentence):
    # NLTK expects string input for single_meteor_score
    
    return single_meteor_score(reference_sentence, candidate_sentence)

def calculate_rouge_l_f1(reference_sentence, candidate_sentence):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_sentence, candidate_sentence)
    return scores['rougeL'].fmeasure


def calculate_chrf(source: str, target: str) -> float:
    """
    Calculate chrF (Character n-gram F-score) between two sentences.

    Args:
        source (str): Reference sentence.
        target (str): Hypothesis sentence.

    Returns:
        float: chrF score (0-100)
    """
    score = sacrebleu.sentence_chrf(target, [source]).score
    return score