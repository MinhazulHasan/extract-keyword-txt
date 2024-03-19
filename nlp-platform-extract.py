from transformers import BertTokenizer, BertForMaskedLM
import torch
from transformers import logging
logging.set_verbosity_error()

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_model():
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    return tokenizer, model

def tokenize_text(text, tokenizer):
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True, padding=True)
    return tokens

def extract_keywords(tokens, model, tokenizer, num_keywords=200):
    # Generate predictions (hidden states) for each token
    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get the hidden states of the last layer

    # Calculate the importance score for each token
    token_importance = torch.mean(hidden_states, dim=1).squeeze()

    # Get the top-k most important tokens as keywords
    _, indices = torch.topk(token_importance, k=num_keywords)
    keywords = [tokenizer.decode([token.item()]) for token in indices]

    return keywords

def main():
    # Load pre-trained BERT model
    tokenizer, model = load_model()

    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/nlp-patform-keywords.txt"
    text = read_text_file(input_file_path)

    # Tokenize text
    tokens = tokenize_text(text, tokenizer)

    # Extract keywords
    keywords = extract_keywords(tokens, model, tokenizer)
    print("\n\n\nKEYWORDS : ", keywords)

    # with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #     output_file.write("\n".join(keywords))

if __name__ == "__main__":
    main()
