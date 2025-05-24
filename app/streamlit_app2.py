import streamlit as st
import pandas as pd
from pathlib import Path

# import your metric functions
from metrics.lexical_metrics import calculate_bleu, calculate_rouge_l_f1, calculate_meteor, calculate_chrf
from metrics.embedding_metrics import load_sbert_model, calculate_sentence_bert_similarity, calculate_bertscore_f1
from metrics.llm_metrics import load_flan_t5_model, get_llm_similarity_score_flan_t5

# load models once
def load_models():
    sbert_model = load_sbert_model()
    tokenizer, flan_model, device = load_flan_t5_model()
    return sbert_model, tokenizer, flan_model, device

@st.cache_resource
def _load():
    return load_models()

sbert_model, tokenizer, flan_model, device = _load()

st.title("Sentence Similarity Explorer")
mode = st.sidebar.selectbox("Mode", ["Single Pair", "Batch File"])

if mode == "Single Pair":
    st.header("Single Sentence Pair Similarity")
    src = st.text_area("Source sentence", "")
    tgt = st.text_area("Target sentence", "")
    if st.button("Compute Similarities"):
        if not (src and tgt):
            st.error("Please provide both source and target sentences.")
        else:
            # Initialize progress bar
            total_steps = 7
            progress = st.progress(0)
            step = 0

            # Lexical metrics
            bleu = calculate_bleu(src, tgt)
            step += 1; progress.progress(step/total_steps)
            rouge = calculate_rouge_l_f1(src, tgt)
            step += 1; progress.progress(step/total_steps)
            meteor = calculate_meteor(src, tgt)
            step += 1; progress.progress(step/total_steps)
            chrf = calculate_chrf(src, tgt)
            step += 1; progress.progress(step/total_steps)

            # Embedding metrics
            sbert = calculate_sentence_bert_similarity(sbert_model, src, tgt)
            step += 1; progress.progress(step/total_steps)
            bertscore = calculate_bertscore_f1(tgt, src)
            step += 1; progress.progress(step/total_steps)

            # LLM metric
            flan = get_llm_similarity_score_flan_t5(tokenizer, flan_model, device, src, tgt)
            step += 1; progress.progress(step/total_steps)

            # Build results table
            result = {
                "source": src,
                "target": tgt,
                "BLEU": bleu,
                "ROUGE-L": rouge,
                "METEOR": meteor,
                "chrF": chrf,
                "SBERT": sbert,
                "BERTScore": bertscore,
                "Flan-T5": flan
            }
            df_result = pd.DataFrame([result])
            st.subheader("Similarity Scores")
            st.dataframe(df_result)
            # Download button
            st.download_button(
                label="Download Scores as CSV", 
                data=df_result.to_csv(index=False), 
                file_name="single_similarity.csv"
            )

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
        if not all(col in df.columns for col in required):
            st.error(f"File must contain columns: {required}")
        else:
            total = len(df)
            progress = st.progress(0)
            results = []
            for i, row in df.iterrows():
                src = row["source"]
                tgt = row["target"]
                # compute metrics
                res = {
                    "source": src,
                    "target": tgt,
                    "BLEU": calculate_bleu(src, tgt),
                    "ROUGE-L": calculate_rouge_l_f1(src, tgt),
                    "METEOR": calculate_meteor(src, tgt),
                    "chrF": calculate_chrf(src, tgt),
                    "SBERT": calculate_sentence_bert_similarity(sbert_model, src, tgt),
                    "BERTScore": calculate_bertscore_f1(tgt, src),
                    "Flan-T5": get_llm_similarity_score_flan_t5(tokenizer, flan_model, device, src, tgt)
                }
                results.append(res)
                # update progress
                progress.progress((i + 1) / total)

            res_df = pd.DataFrame(results)
            st.subheader("Batch Similarity Results")
            st.dataframe(res_df)
            st.download_button(
                label="Download Batch Results as CSV", 
                data=res_df.to_csv(index=False), 
                file_name="batch_similarity.csv"
            )

