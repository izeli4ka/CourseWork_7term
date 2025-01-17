import os
import numpy as np
from gensim.models import Word2Vec
import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    """
    Извлекает текст из PDF файла.
    """
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            return text
    except Exception as e:
        print(f"Ошибка при обработке файла {pdf_path}: {e}")
        return ""

def categorize_text(text, model):
    """
    Преобразует текст в векторное представление.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]

    if not vectors:
        return None, words  # Вернуть пустой вектор, если нет известных слов

    avg_vector = np.mean(vectors, axis=0)
    missing_words = [word for word in words if word not in model.wv]
    return avg_vector, missing_words

def process_dataset(dataset_path, model_path):
    """
    Обрабатывает датасет и возвращает векторы для каждого текста.
    """
    model = Word2Vec.load(model_path)
    file_vectors = {}

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                text = extract_text_from_pdf(file_path)
                avg_vector, missing_words = categorize_text(text, model)

                if avg_vector is not None:
                    file_vectors[file] = avg_vector
                else:
                    print(f"В файле {file} отсутствуют слова, представленные в модели.")

    return file_vectors

def categorize_and_sort(vectors):
    """
    Сортирует тексты на две категории по значениям векторов.
    """
    sorted_vectors = sorted(vectors.items(), key=lambda x: np.linalg.norm(x[1]))
    midpoint = len(sorted_vectors) // 2

    category_1 = sorted_vectors[:midpoint]
    category_2 = sorted_vectors[midpoint:]

    avg_category_1 = np.mean([vec for _, vec in category_1], axis=0)
    avg_category_2 = np.mean([vec for _, vec in category_2], axis=0)

    return category_1, category_2, avg_category_1, avg_category_2

def get_closest_words(model, vector, topn=5):
    """
    Возвращает ближайшие слова для заданного вектора.
    """
    return model.wv.similar_by_vector(vector, topn=topn)

if __name__ == "__main__":
    # Пути к датасетам
    medical_dataset_dermatomycosis_path = "data/medical_articles_dermatomycosis"
    medical_dataset_migraine_path = "data/medical_articles_migraine"
    vkr_dataset_path = "data/vkr_articles"

    # Пути к обученным моделям
    medical_model_dermatomycosis_path = "models/medical_word2vec_dermatomycosis.model"
    medical_model_migraine_path = "models/medical_word2vec_migraine.model"
    vkr_model_path = "models/vkr_word2vec.model"

    # Обработка медицинского датасета (Dermatomycosis)
    print("Обработка медицинских статей (Dermatomycosis)...")
    medical_vectors_dermatomycosis = process_dataset(medical_dataset_dermatomycosis_path, medical_model_dermatomycosis_path)

    # Обработка медицинского датасета (Migraine)
    print("Обработка медицинских статей (Migraine)...")
    medical_vectors_migraine = process_dataset(medical_dataset_migraine_path, medical_model_migraine_path)

    # Обработка ВКР датасета
    print("Обработка статей ВКР...")
    vkr_vectors = process_dataset(vkr_dataset_path, vkr_model_path)

    # Рубрицирование медицинских статей (Dermatomycosis)
    print("Рубрицирование медицинских статей (Dermatomycosis)...")
    med_derm_cat1, med_derm_cat2, med_derm_avg1, med_derm_avg2 = categorize_and_sort(medical_vectors_dermatomycosis)

    # Рубрицирование медицинских статей (Migraine)
    print("Рубрицирование медицинских статей (Migraine)...")
    med_mig_cat1, med_mig_cat2, med_mig_avg1, med_mig_avg2 = categorize_and_sort(medical_vectors_migraine)

    # Рубрицирование статей ВКР
    print("Рубрицирование статей ВКР...")
    vkr_cat1, vkr_cat2, vkr_avg1, vkr_avg2 = categorize_and_sort(vkr_vectors)

    # Вывод результатов
    print("Медицинская категория 1 (Dermatomycosis, средний вектор):", med_derm_avg1)
    print("Медицинская категория 2 (Dermatomycosis, средний вектор):", med_derm_avg2)

    print("Медицинская категория 1 (Migraine, средний вектор):", med_mig_avg1)
    print("Медицинская категория 2 (Migraine, средний вектор):", med_mig_avg2)

    print("ВКР категория 1 (средний вектор):", vkr_avg1)
    print("ВКР категория 2 (средний вектор):", vkr_avg2)

    # Получение ключевых слов
    medical_model_dermatomycosis = Word2Vec.load(medical_model_dermatomycosis_path)
    medical_model_migraine = Word2Vec.load(medical_model_migraine_path)
    vkr_model = Word2Vec.load(vkr_model_path)

    print("Ключевые слова для медицинской категории 1 (Dermatomycosis):", get_closest_words(medical_model_dermatomycosis, med_derm_avg1))
    print("Ключевые слова для медицинской категории 2 (Dermatomycosis):", get_closest_words(medical_model_dermatomycosis, med_derm_avg2))

    print("Ключевые слова для медицинской категории 1 (Migraine):", get_closest_words(medical_model_migraine, med_mig_avg1))
    print("Ключевые слова для медицинской категории 2 (Migraine):", get_closest_words(medical_model_migraine, med_mig_avg2))

    print("Ключевые слова для категории ВКР 1:", get_closest_words(vkr_model, vkr_avg1))
    print("Ключевые слова для категории ВКР 2:", get_closest_words(vkr_model, vkr_avg2))
