import json
import os
import numpy as np
import random
from gensim.models import Word2Vec
from PyPDF2 import PdfReader


def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF файла."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text.strip()
    except Exception as e:
        print(f"Ошибка при обработке {pdf_path}: {e}")
        return ""


def get_random_text_from_folder(folder_path, num_samples=2):
    """Выбирает случайные тексты из PDF файлов в указанной папке."""
    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка {folder_path} не найдена!")
        return []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"Ошибка: Нет PDF файлов в {folder_path}!")
        return []

    sampled_files = random.sample(pdf_files, min(len(pdf_files), num_samples))
    texts = [extract_text_from_pdf(os.path.join(folder_path, file)) for file in sampled_files]
    return [text for text in texts if text]  # Убираем пустые тексты


def categorize_text(text, model):
    """Преобразует текст в векторное представление."""
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]

    if not vectors:
        print(f"Предупреждение: все слова в '{text[:50]}...' отсутствуют в модели!")
        return None
    return np.mean(vectors, axis=0)


def create_reference_system(data_models):
    """Создаёт справочную систему на основе моделей."""
    reference_data = {}
    for category, (texts, model_path) in data_models.items():
        try:
            print(f"Загрузка модели для категории '{category}' из {model_path}")
            model = Word2Vec.load(model_path)

            vectors = [categorize_text(text, model) for text in texts]
            vectors = [v for v in vectors if v is not None]

            if not vectors:
                print(f"Предупреждение: нет данных для категории '{category}'.")
                continue

            reference_data[category] = np.mean(vectors, axis=0).tolist()
        except FileNotFoundError:
            print(f"Ошибка: модель {model_path} не найдена.")

    with open("reference_system.json", "w", encoding="utf-8") as f:
        json.dump(reference_data, f, ensure_ascii=False, indent=4)
    print("Справочная система сохранена в reference_system.json")


if __name__ == "__main__":
    # Пути к датасетам (по вашим данным)
    dermatomycosis_articles = get_random_text_from_folder("data/medical_articles_dermatomycosis", num_samples=2)
    migraine_articles = get_random_text_from_folder("data/medical_articles_migraine", num_samples=2)
    vkr_articles = get_random_text_from_folder("data/vkr_articles", num_samples=2)

    data_models = {
        "dermatomycosis": (dermatomycosis_articles, "models/medical_word2vec_dermatomycosis.model"),
        "migraine": (migraine_articles, "models/medical_word2vec_migraine.model"),
        "vkr": (vkr_articles, "models/vkr_word2vec.model"),
    }

    create_reference_system(data_models)
