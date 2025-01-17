import json
import random
from fuzzywuzzy import fuzz

def load_intents(file_path):
    """ Загрузка intents.json. """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def find_best_match(user_input, patterns):
    """ Находит лучшее совпадение пользовательского ввода с шаблонами. """
    best_match = None
    highest_score = 0

    for pattern in patterns:
        score = fuzz.partial_ratio(user_input.lower(), pattern.lower())
        if score > highest_score:
            highest_score = score
            best_match = pattern

    return best_match, highest_score

def chatbot_response(user_input, intents, threshold=70):
    """ Генерация ответа на запрос пользователя с учётом совпадений. """
    for intent in intents['intents']:
        best_match, score = find_best_match(user_input, intent['patterns'])
        if score >= threshold:
            return random.choice(intent['responses'])

    return "Извините, я вас не понял. Попробуйте переформулировать запрос."

if __name__ == "__main__":
    intents = load_intents("intents.json")

    if intents:
        print("Добро пожаловать в чат-бот! Введите ваш запрос или напишите 'выход' для завершения.")

        while True:
            user_input = input("Вы: ")
            if user_input.lower() in ["выход", "exit"]:
                print("Чат-бот: До свидания!")
                break

            response = chatbot_response(user_input, intents)
            print(f"Чат-бот: {response}")
