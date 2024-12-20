from transformers import pipeline

def map_text_to_category(user_input):
    """Map text input to sound categories using zero-shot classification."""
    nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    categories = ["pad", "bass", "flute", "ambient"]
    result = nlp(user_input, candidate_labels=categories)
    return result["labels"][0]  # Return the top category

if __name__ == "__main__":
    user_input = "Generate a peaceful pad sound."
    category = map_text_to_category(user_input)
    print(f"Mapped input '{user_input}' to category: {category}")
