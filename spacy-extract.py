"""
    SpaCy is primarily known as a powerful library for natural language processing (NLP), including tasks such as part-of-speech tagging, named entity recognition, and dependency parsing.
    However, it doesn't directly provide a keyword extraction algorithm like some other libraries do.
    We can, however, use SpaCy in combination with other techniques to achieve keyword extraction.
"""
import spacy
import string
from collections import Counter
# python -m spacy download en_core_web_sm

# Load English tokenizer, tagger, parser, and NER
nlp = spacy.load("en_core_web_sm")

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    text = text.lower()
    return text

# Function to extract keywords using SpaCy
def extract_top_keywords_spacy(text, num_keywords=200):
    # Tokenize the text and perform part-of-speech tagging
    doc = nlp(text)
    # Filter out nouns, proper nouns, and adjectives
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    # Count the frequency of each keyword
    keyword_counts = Counter(keywords)
    # Get the top 200 keywords based on their frequency
    top_keywords = keyword_counts.most_common(num_keywords)
    # Normalize the scores
    total_count = sum(keyword_counts.values())
    top_keywords_with_scores = [(keyword, count / total_count) for keyword, count in top_keywords]
    
    return top_keywords_with_scores

# Main function
def main():
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/spacy-keywords.txt"
    text = read_text_file(input_file_path)
    preprocessed_text = preprocess_text(text)
    top_keywords_with_scores = extract_top_keywords_spacy(preprocessed_text)
    formatted_keywords = [f"{keyword} - {score:.4f}" for keyword, score in top_keywords_with_scores]

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(formatted_keywords))

    print("Keywords extracted and saved to ", output_file_path)

if __name__ == "__main__":
    main()
