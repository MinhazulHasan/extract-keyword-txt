from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from nltk.corpus import stopwords
from collections import Counter
from nltk import pos_tag

# Download NLTK resources
# import nltk
# nltk.download('averaged_perceptron_tagger')

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_keywords(text):
    # Define a set of English stopwords
    stop_words = set(stopwords.words('english'))

    # Load pre-trained model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

    # Create a pipeline
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)

    # Extract keywords
    keywords = nlp(text)

    # Filter out the 'O' (non-keyword) labels, stopwords, and words with special characters
    keywords = [word['word'] for word in keywords if word['entity'] != 'O' and word['word'] not in stop_words and word['word'].isalpha()]

    # Tag each keyword with its part of speech
    pos_keywords = pos_tag(keywords)

    # Filter out words that are not nouns, verbs, adverbs, or adjectives
    allowed_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
    keywords = [word for word, pos in pos_keywords if pos in allowed_pos]

    # Count the frequency of each keyword
    keyword_freq = Counter(keywords)

    # Get the top-k most common keywords
    keywords = keyword_freq.most_common(200)

    return keywords

def main():
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/h1-transformer3-keywords.txt"
    text = read_text_file(input_file_path)

    # Extract keywords
    keywords = extract_keywords(text)
    
    # Write keywords to a text file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for keyword, freq in keywords:
            output_file.write(keyword + "\n")

if __name__ == "__main__":
    main()