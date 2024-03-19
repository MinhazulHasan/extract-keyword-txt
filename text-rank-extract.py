from summa import keywords
import string

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# Function to extract keywords using TextRank
def extract_top_keywords_textrank(text, num_keywords):
    keywords_with_scores = keywords.keywords(text, words=num_keywords, scores=True)
    return keywords_with_scores

# Main function
def main():
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/text-rank-keywords.txt"
    text = read_text_file(input_file_path)
    preprocessed_text = preprocess_text(text)
    top_keywords_with_scores = extract_top_keywords_textrank(preprocessed_text, 200)
    formatted_keywords = [f"{keyword}  ==>>  {score:.4f}" for keyword, score in top_keywords_with_scores]

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(formatted_keywords))

    print("Keywords extracted and saved to ", output_file_path)

if __name__ == "__main__":
    main()
