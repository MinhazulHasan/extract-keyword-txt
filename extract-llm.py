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
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a summarizer. Your can summarize long texts into short ones. The file what you need to summarize is: \n\n" + text
    })
    messages.append({
        "role": "user",
        "content": "Summarize the text file."
    })
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature = 0.0
    )
    response_message = response.choices[0].message.content
    return response_message


if __name__ == "__main__":
    file_name = "Alphabet Inc. (Google) - Alphabet-Sustainability-Bond-Framework-Second-Party-Opinion"

    input_file_path = f"./text-from-html/{file_name}.txt"
    output_file_path = f"./summery-from-html/GPT_4_{file_name}.txt"

    pdf_text = read_text_file(input_file_path)
    summary = generate_keywords(pdf_text)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)