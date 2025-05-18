import streamlit as st
from .api_client import HuggingFaceClient
from .utils import process_target_sentences, validate_inputs
from .config import DEFAULT_SOURCE_SENTENCE, DEFAULT_TARGET_SENTENCES

def setup_sidebar():
    """Configure the app sidebar"""
    with st.sidebar:
        st.header("Instructions")
        st.write("""
        1. Enter your Hugging Face API token
        2. Enter a source sentence to compare against
        3. Enter target sentences (one per line)
        4. Click "Calculate Similarity" to see results
        """)
        st.markdown("[Get API token](https://huggingface.co/settings/tokens)")
        
        st.header("About")
        st.write("""
        Uses the `msmarco-distilbert-base-tas-b` model for semantic similarity.
        Scores range from 0 (no similarity) to 1 (very similar).
        """)

def display_results(source_sentence, target_sentences, similarities):
    """Display the similarity results"""
    st.header("Similarity Results")
    st.write(f"Source sentence: **{source_sentence}**")
    
    # Create results table
    results = []
    for sentence, similarity in zip(target_sentences, similarities):
        results.append({
            "Target Sentence": sentence,
            "Similarity Score": f"{similarity:.3f}",
            "Match Percentage": f"{similarity*100:.0f}%"
        })
    
    st.table(results)
    
    # Add visualization
    st.subheader("Similarity Visualization")
    st.bar_chart(
        data={sentence: similarity for sentence, similarity in zip(target_sentences, similarities)},
        height=400
    )

def main():
    """Main app function"""
    st.set_page_config(page_title="Sentence Similarity", layout="wide")
    st.title("Sentence Similarity Comparison")
    
    setup_sidebar()
    
    # Input section
    api_token = st.text_input("Hugging Face API Token:", type="password")
    source_sentence = st.text_area(
        "Source sentence:", 
        DEFAULT_SOURCE_SENTENCE
    )
    target_sentences_text = st.text_area(
        "Target sentences (one per line):", 
        DEFAULT_TARGET_SENTENCES,
        height=150
    )
    target_sentences = process_target_sentences(target_sentences_text)
    
    # Calculate button
    if st.button("Calculate Similarity"):
        error = validate_inputs(api_token, source_sentence, target_sentences)
        if error:
            st.error(error)
            return
            
        with st.spinner("Calculating similarities..."):
            try:
                client = HuggingFaceClient(api_token)
                similarities = client.get_similarity_scores(source_sentence, target_sentences)
                display_results(source_sentence, target_sentences, similarities)
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()