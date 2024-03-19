from rake_nltk import Rake
import string

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase all words
    text = text.lower()
    return text

# Function to extract keywords using RAKE
def extract_top_keywords_rake(words, num_keywords=200):
    r = Rake()
    r.extract_keywords_from_text(' '.join(words))
    ranked_keywords = r.get_ranked_phrases_with_scores()[:num_keywords]
    formatted_keywords = [f"Keyword: {keyword}  ==>>  Score: {score:.4f}" for score, keyword in ranked_keywords]
    return formatted_keywords

# Main function
def main():
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/rake-keywords.txt"
    text = read_text_file(input_file_path)
    words = preprocess_text(text).split()
    top_keywords_with_scores = extract_top_keywords_rake(words)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(top_keywords_with_scores))

    print("Keywords extracted and saved to ", output_file_path)

if __name__ == "__main__":
    main()
