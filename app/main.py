import streamlit as st
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

# Set up the app
st.set_page_config(page_title="Sentence Similarity", layout="wide")
st.title("Sentence Similarity Comparison")

# Model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ],
    index=0
)

# Initialize model (cached to avoid reloading)
@st.cache_resource
def load_model(model_name):
    try:
        # Try loading with sentence-transformers first (simpler)
        return SentenceTransformer(model_name), None
    except:
        # Fall back to transformers + manual pooling
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, tokenizer

model, tokenizer = load_model(model_name)

def calculate_similarity(source_sentence, target_sentences):
    """Calculate similarity scores using the selected model"""
    if tokenizer is None:
        # Using sentence-transformers
        source_embedding = model.encode(source_sentence, convert_to_tensor=True)
        target_embeddings = model.encode(target_sentences, convert_to_tensor=True)
        similarities = util.cos_sim(source_embedding, target_embeddings)[0]
        return similarities.tolist()
    else:
        # Using transformers with manual pooling
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Encode source
        encoded_source = tokenizer([source_sentence], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_source)
        source_embedding = mean_pooling(model_output, encoded_source['attention_mask'])
        source_embedding = F.normalize(source_embedding, p=2, dim=1)

        # Encode targets
        encoded_targets = tokenizer(target_sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_targets)
        target_embeddings = mean_pooling(model_output, encoded_targets['attention_mask'])
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

        # Calculate similarities
        similarities = F.cosine_similarity(source_embedding, target_embeddings)
        return similarities.tolist()

# Input section
source_sentence = st.text_area(
    "Source sentence to compare against:",
    "Machine learning is so easy."
)

target_sentences = st.text_area(
    "Target sentences to compare (one per line):",
    "Deep learning is so straightforward.\nThis is so difficult, like rocket science.\nI can't believe how much I struggled with this.",
    height=150
)

# Process input
target_list = [s.strip() for s in target_sentences.split('\n') if s.strip()]

# Calculate button
if st.button("Calculate Similarity"):
    if not source_sentence.strip():
        st.error("Please enter a source sentence.")
    elif not target_list:
        st.error("Please enter at least one target sentence.")
    else:
        with st.spinner("Calculating similarities..."):
            try:
                similarities = calculate_similarity(source_sentence, target_list)
                
                # Display results
                st.subheader("Results")
                st.write(f"Using model: `{model_name}`")
                
                # Create results table
                results = []
                for sentence, similarity in zip(target_list, similarities):
                    results.append({
                        "Target Sentence": sentence,
                        "Similarity Score": f"{similarity:.3f}",
                        "Match": f"{similarity*100:.1f}%"
                    })
                
                st.table(results)
                
                # Visualization
                st.subheader("Similarity Visualization")
                st.bar_chart(
                    data={sentence: sim for sentence, sim in zip(target_list, similarities)},
                    height=400
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Sidebar info
st.sidebar.header("About")
st.sidebar.write("""
This app calculates semantic similarity between sentences using state-of-the-art transformer models.

**Key Features:**
- Multiple model options
- No API token required
- Runs completely locally after initial download
""")

st.sidebar.header("Model Information")
st.sidebar.write(f"""
Selected Model: `{model_name}`

**Available Models:**
1. `all-mpnet-base-v2` - Best quality (768-dimensional embeddings)
2. `all-MiniLM-L6-v2` - Good balance (384-dimensional)
3. `paraphrase-multilingual` - For multilingual text
""")