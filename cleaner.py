import os
import re
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """ Извлекает текст из PDF. """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.strip()

def normalize_text(text):
    """ 
    Нормализует текст:
    - Приведение к нижнему регистру
    - Удаление спецсимволов, пунктуации
    - Замена множественных пробелов на один
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    text = re.sub(r'\s+', ' ', text).strip()  # Удаление множественных пробелов
    return text

def process_pdfs(input_folder, output_file):
    """ Обрабатывает PDF-файлы и сохраняет нормализованный текст в файл. """
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(input_folder, file_name)
                text = extract_text_from_pdf(pdf_path)
                normalized_text = normalize_text(text)
                f_out.write(normalized_text + "\n")

if __name__ == "__main__":
    process_pdfs("data/medical_articles_dermatomycosis", "data/medical_dataset_dermatomycosis.txt")
    process_pdfs("data/medical_articles_migraine", "data/medical_dataset_migraine.txt")
    process_pdfs("data/vkr_articles", "data/vkr_dataset.txt")
