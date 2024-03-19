import os
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to extract keywords using TF-IDF
def extract_keywords_tfidf(text):
    vectorizer = TfidfVectorizer(max_features=1000) 
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    keyword_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
    sorted_keywords = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:200]
    # return keywords
    formatted_keywords = [f"{keyword} - {keyword_scores[keyword]:.4f}" for keyword in sorted_keywords]
    return formatted_keywords

# Main function
def main():
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/tf-idf-keywords.txt"
    text = read_text_file(input_file_path)
    top_keywords_with_scores  = extract_keywords_tfidf(text)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(top_keywords_with_scores ))

    print("Keywords extracted and saved to ", output_file_path)


if __name__ == "__main__":
    main()
