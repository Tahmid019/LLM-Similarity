from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_calc
# from comet import download_model, load_from_checkpoint # If using COMET

# Initialize models (consider loading them once in main_app.py or a config module)
# sentence_model = SentenceTransformer('all-mpnet-base-v2')
# comet_model = load_from_checkpoint(download_model("wmt20-comet-da")) # Example

def calculate_sentence_bert_similarity(model, sentence1, sentence2):
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding1, embedding2).item()

def calculate_bertscore_f1(reference_sentence, candidate_sentence):
    P, R, F1 = bert_score_calc([candidate_sentence], [reference_sentence], lang="en", verbose=False, idf=False)
    return F1.item()

# def calculate_comet_score(source_sentence, mt_sentence, reference_sentence):
#     data = [{"src": source_sentence, "mt": mt_sentence, "ref": reference_sentence}]
#     model_output = comet_model.predict(data, batch_size=8, gpus=1) # Adjust gpus
#     return model_output[0][0] # Or specific score from output