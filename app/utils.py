def process_target_sentences(target_sentences_text):
    """Convert multiline text input to list of sentences"""
    return [s.strip() for s in target_sentences_text.split('\n') if s.strip()]

def validate_inputs(api_token, source_sentence, target_sentences):
    """Validate user inputs"""
    if not api_token:
        return "Please enter your Hugging Face API token."
    if not source_sentence.strip():
        return "Please enter a source sentence."
    if not target_sentences:
        return "Please enter at least one target sentence."
    return None