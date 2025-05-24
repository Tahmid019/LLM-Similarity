from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import sacrebleu
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

def calculate_bleu(reference_sentences, candidate_sentence):
    logger.info2("Calculating Bleu ...")
    reference_tokens = [ref.split() for ref in reference_sentences]
    candidate_tokens = candidate_sentence.split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie) / 100 
    logger.info2(">>>")
    return score


def calculate_meteor(reference, hypothesis):
    logger.info2("Calculating Meteor ...")
    try:
        score = meteor_score([word_tokenize(reference)], word_tokenize(hypothesis))
        logger.info2(">>>")
        return score
    except Exception as e:
        logger.error(f"Note: METEOR encountered an issue with '{hypothesis}', score set to 0. Error: {e}")
        return 0.0
    
    
def calculate_rouge_l_f1(reference_sentence, candidate_sentence):
    logger.info2("Calculating Rouge_l_F1 ...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_sentence, candidate_sentence)
    score = scores['rougeL'].fmeasure 
    logger.info2(">>>")
    return score


def calculate_chrf(source: str, target: str) -> float:
    """
    Calculate chrF (Character n-gram F-score) between two sentences.

    Args:
        source (str): Reference sentence.
        target (str): Hypothesis sentence.

    Returns:
        float: chrF score (0-100)
    """
    logger.info2("Calculating Chrf ...")
    score = sacrebleu.sentence_chrf(target, [source]).score
    logger.info2(">>>")
    return score