import json
import numpy as np
import re
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

# Путь к JSON-файлу
REFERENCE_FILE = "reference_system.json"

# Простая база знаний (HashMap) на случай отсутствия моделей
HASHMAP_DB = {
    "dermatomycosis": "Грибковая инфекция кожи, вызываемая различными патогенными грибками.",
    "migraine": "Неврологическое заболевание, сопровождающееся сильными головными болями.",
    "вкр": "Выпускная квалификационная работа — это итоговое исследование студента.",
}

def load_reference_data(file_path=REFERENCE_FILE):
    """Загрузка категорий и векторов из reference_system.json."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Ошибка: reference_system.json не найден.")
        return None
    except json.JSONDecodeError:
        print("Ошибка: Ошибка в формате JSON.")
        return None

def preprocess_text(text):
    """Очистка и токенизация текста."""
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return words

def query_to_vector(query, model_path):
    """Преобразование запроса в средний вектор с использованием модели Word2Vec."""
    try:
        model = Word2Vec.load(model_path)
    except FileNotFoundError:
        print(f"Ошибка: модель {model_path} не найдена.")
        return None

    words = preprocess_text(query)
    vectors = [model.wv[word] for word in words if word in model.wv]

    if not vectors:
        print(f"Все слова в запросе '{query}' отсутствуют в модели!")
        return None

    return np.mean(vectors, axis=0)

def find_closest_category(query, reference_data):
    """Находит ближайшую категорию по косинусному сходству."""
    best_match = None
    highest_similarity = -1
    query_vector = None

    # Поиск первой подходящей модели
    for category in reference_data.keys():
        model_path = f"models/medical_word2vec_{category}.model"
        query_vector = query_to_vector(query, model_path)
        if query_vector is not None:
            break

    if query_vector is None:
        return None, None, None  # Исправлено: теперь функция всегда возвращает 3 значения

    # Сравнение запроса с категориями
    for category, data in reference_data.items():
        category_vector = np.array(data["vector"])
        similarity = 1 - cosine(query_vector, category_vector)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = category

    return best_match, highest_similarity if best_match else None, None  # Исправлено: возвращаем 3 значения

if __name__ == "__main__":
    reference_data = load_reference_data()

    if reference_data:
        while True:
            user_query = input("\nВведите ваш запрос (или 'выход' для завершения): ")
            if user_query.lower() in ["выход", "exit"]:
                print("Выход из системы.")
                break

            closest_category, similarity, _ = find_closest_category(user_query, reference_data)  # Исправлено

            # Если не удалось найти ближайшую категорию, используем HashMap
            if closest_category is None:
                print(f"Не удалось найти категорию по модели. Используем HashMap...")
                for key in HASHMAP_DB.keys():
                    if key in user_query.lower():
                        closest_category = key
                        similarity = "N/A"
                        break

            if closest_category is None:
                print("Нет информации по данному запросу.")
                continue

            response = reference_data.get(closest_category, {}).get("description", HASHMAP_DB.get(closest_category, "Нет информации."))

            similarity_text = f" (Сходство: {similarity:.2f})" if isinstance(similarity, float) else ""
            print(f"\n Ближайшая категория: {closest_category}{similarity_text}")
            print(f"Ответ: {response}")
