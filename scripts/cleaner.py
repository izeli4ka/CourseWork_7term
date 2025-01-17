import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """ Извлекает текст из PDF. """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    """ Очищает текст (приведение к нижнему регистру, удаление лишних пробелов). """
    return ' '.join(text.lower().split())

def process_pdfs(input_folder, output_file):
    """ Обрабатывает PDF-файлы и сохраняет текст в файл. """
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(input_folder, file_name)
                text = extract_text_from_pdf(pdf_path)
                clean_text_content = clean_text(text)
                f_out.write(clean_text_content + "\n")

if __name__ == "__main__":
    process_pdfs("data/medical_articles_dermatomycosis", "data/medical_dataset_dermatomycosis.txt")
    process_pdfs("data/medical_articles_migraine", "data/medical_dataset_migraine.txt")
    process_pdfs("data/vkr_articles", "data/vkr_dataset.txt")
