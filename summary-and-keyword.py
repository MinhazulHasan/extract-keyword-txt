import PyPDF2
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
import os
load_dotenv()

HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text


def extract_text_from_html(html_path):
    with open(html_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
    return text


def save_text_to_file(text, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)


def summarize_by_mistral(payload):
    try:
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print("Error: ", e)


def summarize_by_falcon(payload):
    API_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def summarize_by_llama2(payload):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response.json())



def main():
    # file_name = "Alphabet Inc. (Google) - Alphabet ESG Index"
    file_name = "Alphabet Inc. (Google) - Alphabet ESG Index"

    pdf_path = f"./pdf-files/{file_name}.pdf"
    html_path = f"./html-files/{file_name}.html"
    
    text_output_path_pdf = f"./text-from-pdf/{file_name}.txt"
    text_output_path_html = f"./text-from-html/{file_name}.txt"
    
    summery_path_pdf = f"./summery-from-pdf/Falcon_{file_name}.txt"
    summery_path_html = f"./summery-from-html/Falcon_{file_name}.txt"

    text_from_pdf = extract_text_from_pdf(pdf_path)
    save_text_to_file(text_from_pdf, text_output_path_pdf)
    
    text_from_html = extract_text_from_html(html_path)
    save_text_to_file(text_from_html, text_output_path_html)

    # payload_pdf = { "inputs": f"Summarize the large content: \n {text_from_pdf}" }
    # summary = summarize_by_mistral(payload_pdf)
    # save_text_to_file(summary[0].get("generated_text"), summery_path_pdf)


    # payload_pdf = {"inputs": text_from_pdf}
    # summary_pdf = summarize_by_falcon(payload_pdf)
    # save_text_to_file(summary_pdf[0].get("summary_text"), summery_path_pdf)
    
    
    # payload_html = {"inputs": text_from_html}
    # summary_html = summarize_by_falcon(payload_html)
    # save_text_to_file(summary_html[0].get("summary_text"), summery_path_html)


    payload_pdf = {"inputs": text_from_pdf}
    summarize_by_llama2(payload_pdf)
    # save_text_to_file(summary_pdf[0].get("summary_text"), summery_path_pdf)


if __name__ == "__main__":
    main()
