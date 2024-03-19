"""
    NLTK (Natural Language Toolkit) provides various algorithms and tools for natural language processing, including keyword extraction.
    One popular method for keyword extraction in NLTK is TF-IDF (Term Frequency-Inverse Document Frequency).
"""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# Function to extract keywords using TF-IDF
def extract_top_keywords_tf_idf(text, num_keywords=200):
    words = preprocess_text(text)
    # Calculate term frequency
    freq_dist = FreqDist(words)
    # Calculate inverse document frequency
    num_documents = 1  # We only have one document
    idf = {word: num_documents / (1 + freq_dist.get(word, 0)) for word in set(words)}
    # Calculate TF-IDF
    tf_idf = {word: freq_dist[word] * idf[word] for word in set(words)}
    # Sort keywords by TF-IDF score
    sorted_keywords = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)
    # Select top 200 keywords
    top_keywords = sorted_keywords[:num_keywords]
    return top_keywords

# Main function
def main():
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/nltk-keywords.txt"
    text = read_text_file(input_file_path)
    top_keywords_with_scores = extract_top_keywords_tf_idf(text)
    formatted_keywords = [f"{keyword} - {score:.4f}" for keyword, score in top_keywords_with_scores]

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(formatted_keywords))

    print("Keywords extracted and saved to ", output_file_path)

if __name__ == "__main__":
    main()
