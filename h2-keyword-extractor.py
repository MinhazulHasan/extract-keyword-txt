# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from transformers import pipeline
# from nltk.corpus import stopwords
# from collections import Counter
# from nltk import pos_tag

# def read_text_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         text = file.read()
#     return text

# def extract_keywords(text):
#     # Define a set of English stopwords
#     stop_words = set(stopwords.words('english'))

#     # Load pre-trained model and tokenizer
#     model = AutoModelForTokenClassification.from_pretrained("distilbert-base-cased")
#     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

#     # Create a pipeline
#     nlp = pipeline("ner", model=model, tokenizer=tokenizer)

#     # Extract keywords
#     keywords = nlp(text)

#     # Filter out the 'O' (non-keyword) labels, stopwords, and words with special characters
#     keywords = [word['word'] for word in keywords if word['entity'] != 'O' and word['word'] not in stop_words and word['word'].isalpha()]

#     # Tag each keyword with its part of speech
#     pos_keywords = pos_tag(keywords)

#     # Filter out words that are not nouns, verbs, adverbs, or adjectives
#     allowed_pos = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
#     keywords = [word for word, pos in pos_keywords if pos in allowed_pos]

#     # Count the frequency of each keyword
#     keyword_freq = Counter(keywords)

#     # Get the top-k most common keywords
#     keywords = keyword_freq.most_common(200)

#     return keywords

# def main():
#     input_file_path = "EnvironmentalReport.txt"
#     output_file_path = "./keywords/h2-transformer3-keywords.txt"
#     text = read_text_file(input_file_path)

#     # Extract keywords
#     keywords = extract_keywords(text)
    
#     # Write keywords to a text file
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         for keyword, freq in keywords:
#             output_file.write(keyword + "\n")

# if __name__ == "__main__":
#     main()








from transformers import pipeline

# Define the keyword extraction pipeline using the h2-text-extract model
keyword_extractor = pipeline("keyword-extraction", model="h2-text-extract/conceptual-keyphrase-extractor")

def extract_keywords(text_file_path, num_keywords=5):
    try:
        # Read text from the file
        with open(text_file_path, 'r', encoding="utf-8") as file:
            text = file.read()

        print("\n\n ++++++++++++++++++++++++ Text Extracted ++++++++++++++++++++++++ n\n")

        # Extract keywords using the pipeline
        keywords = keyword_extractor(text, k=num_keywords)  # Change k to adjust the number of keywords

        # Extract the actual keyword text from the results
        extracted_keywords = [keyword["keyword"] for keyword in keywords]

        return extracted_keywords

    except FileNotFoundError:
        print(f"Error: File '{text_file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    text_file_path = "EnvironmentalReport.txt"  # Replace with your actual file path
    num_keywords = 10  # Adjust the number of keywords as needed

    keywords = extract_keywords(text_file_path, num_keywords)

    if keywords:
        print("Extracted Keywords:")
        for keyword in keywords:
            print(keyword)
    else:
        print("No keywords extracted.")
