import streamlit as st
import pandas as pd
from pathlib import Path
import openpyxl

from metrics.lexical_metrics import calculate_bleu, calculate_rouge_l_f1, calculate_meteor, calculate_chrf
from metrics.embedding_metrics import load_sbert_model, calculate_sentence_bert_similarity, calculate_bertscore_f1
from metrics.llm_metrics import load_flan_t5_model, get_llm_similarity_score_flan_t5

@st.cache_resource
def load_models():
    sbert_model = load_sbert_model()
    tokenizer, flan_model, device = load_flan_t5_model()
    return sbert_model, tokenizer, flan_model, device

sbert_model, tokenizer, flan_model, device = load_models()

st.title("Sentence Similarity Explorer")

mode = st.sidebar.selectbox("Mode", ["Single Pair", "Batch CSV"])

if mode == "Single Pair":
    st.header("Single Sentence Pair Similarity")
    src = st.text_area("Source sentence", "")
    tgt = st.text_area("Target sentence", "")
    if st.button("Compute Similarities"):
        if src and tgt:
            st.subheader("Lexical Metrics")
            st.write({
                "BLEU": calculate_bleu(src, tgt),
                "ROUGE-L (F1)": calculate_rouge_l_f1(src, tgt),
                "METEOR": calculate_meteor(src, tgt),
                "chrF": calculate_chrf(src, tgt),
            })
            st.subheader("Embedding Metrics")
            st.write({
                "SBERT": calculate_sentence_bert_similarity(sbert_model, src, tgt),
                "BERTScore F1": calculate_bertscore_f1(tgt, src),
            })
            st.subheader("LLM-based Metric")
            st.write({
                "Flan-T5": get_llm_similarity_score_flan_t5(tokenizer, flan_model, device, src, tgt)
            })
        else:
            st.error("Please provide both source and target sentences.")

else:
    st.header("Batch Similarity via File Upload")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xls", "xlsx"])
    if uploaded:
        ext = Path(uploaded.name).suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

        required = ["source", "target"]
        if all(col in df.columns for col in required):
            results = []
            for _, row in df.iterrows():
                src = row["source"]
                tgt = row["target"]
                results.append({
                    "source": src,
                    "target": tgt,
                    "BLEU": calculate_bleu(src, tgt),
                    "ROUGE-L": calculate_rouge_l_f1(src, tgt),
                    "METEOR": calculate_meteor(src, tgt),
                    "chrF": calculate_chrf(src, tgt),
                    "SBERT": calculate_sentence_bert_similarity(sbert_model, src, tgt),
                    "BERTScore": calculate_bertscore_f1(tgt, src),
                    "Flan-T5": get_llm_similarity_score_flan_t5(tokenizer, flan_model, device, src, tgt)
                })
            res_df = pd.DataFrame(results)
            st.download_button("Download Results", data=res_df.to_csv(index=False), file_name="similarity_results.csv")
            st.dataframe(res_df)
        else:
            st.error(f"File must contain columns: {required}")

