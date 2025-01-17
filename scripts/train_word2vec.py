from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def train_word2vec(data_file, model_path):
    """ Обучает модель Word2Vec. """
    with open(data_file, 'r', encoding='utf-8') as f:
        sentences = [simple_preprocess(line) for line in f]
    
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

if __name__ == "__main__":
    train_word2vec("data/medical_dataset_dermatomycosis.txt", "models/medical_word2vec_dermatomycosis.model")
    train_word2vec("data/medical_dataset_migraine.txt", "models/medical_word2vec_migraine.model")
    train_word2vec("data/vkr_dataset.txt", "models/vkr_word2vec.model")
