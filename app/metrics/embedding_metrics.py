import torch
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_calc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from app.__init__ import logger

def load_sbert_model():
    logger.info2("Loading SBert Model ...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    logger.info2(">>>")
    return sbert_model
    
    logger.info2(">>>")
    return sbert_model

def calculate_sentence_bert_similarity(model, sentence1, sentence2):
    logger.info2("Calculating SBERT similarity ...")
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding1, embedding2).item() 
    logger.info2(">>>")
    return score

def calculate_bertscore_f1(reference_sentence, candidate_sentence):
    logger.info2("Calculating BERTSCORE F1 ...")
    P, R, F1 = bert_score_calc([candidate_sentence], [reference_sentence], lang="en", verbose=False, idf=False)
    logger.info2(">>>")
    return F1.item()
