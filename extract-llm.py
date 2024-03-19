import openai
from dotenv import load_dotenv
import os
load_dotenv()

client = openai.OpenAI()

key = os.environ.get('OPENAI_API_KEY')
openai.api_key = key

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# Define the function to generate keywords using GPT-3.5
def generate_keywords(text):
    prompt = "Extract the top 20 keywords from the text file based on keyword score. Keyword MUST BE A SINGLE WORD. The Text File is: \n\n" + text
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "YOU ARE A KEYWORD EXTRACTION EXPERT FROM LARGE TEXT DATA."},
            {"role": "user", "content": prompt},
        ],
        temperature = 0.0
    )
    response_message = response.choices[0].message.content
    print(response_message )


if __name__ == "__main__":
    input_file_path = "EnvironmentalReport.txt"
    output_file_path = "./keywords/nltk-keywords.txt"
    pdf_text = read_text_file(input_file_path)
    generate_keywords(pdf_text)