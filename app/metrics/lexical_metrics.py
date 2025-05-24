from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import sacrebleu
import sys
from pathlib import Path
from utils import normalize_text

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

def calculate_bleu(source, target):
    logger.info2("Calculating Bleu ...")
    tokenized_src = normalize_text(source).split()
    tokenized_tgt = normalize_text(target).split()
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([tokenized_src], tokenized_tgt, smoothing_function=smoothie) / 100 
    logger.info2(">>>")
    return score


def calculate_meteor(source, target):
    logger.info2("Calculating Meteor ...")
    try:
        score = single_meteor_score(source, target)
        logger.info2(">>>")
        return score
    except Exception as e:
        logger.error(f"Note: METEOR encountered an issue with '{target}', score set to 0. Error: {e}")
        return 0.0
    
    
def calculate_rouge_l_f1(source, target):
    logger.info2("Calculating Rouge_l_F1 ...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(source, target)
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