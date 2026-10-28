import os
import sys

# Отключаем лишний вывод от библиотек Google
os.environ["GRPC_VERBOSITY"] = "NONE"      # Полностью выключаем логи gRPC
os.environ["GLOG_minloglevel"] = "3"       # Выключаем C++ логи Google (3 = только фатальные ошибки)

import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Не найден GEMINI_API_KEY в файле .env")

genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 768

# Список расширен на 40 предложений на английском языке в исходном стиле (кошки, собаки, ананасы, кодинг)
texts_to_index = [
    # Оригинальный текст
    "Кошка сидит на окне",
    "Собака играет в парке",
    "Ананас растет в тропиках",
    "Кот спит на диване",
    "Пес лает на почтальона",
    "Фрукт ананас очень вкусный",
    "Домашняя кошка любит ласку",
    "Верный пес охраняет дом",
    "Спелый ананас полон витаминов",
    "Кошки любят рыбу и мясо",
    "Основной рацион кошек - это белок",
    "Чем кормить котенка?",
    "Лучший корм для кошек - сбалансированный",
    "Коты едят сухой и влажный корм",
    "Нельзя кормить кошку шоколадом",
    "Молоко не всегда полезно для кошек",
    "Я обожаю программировать.",
    "Программирование – это то, что мне очень нравится.",
    "Меня увлекает разработка программного обеспечения.",
    "Я испытываю страсть к написанию кода.",
    "Программирование приносит мне огромное удовольствие.",
    "Мне интересно заниматься программированием.",
    "Моё хобби – это кодинг.",
    "В свободное время я занимаюсь программированием.",
    "Кодинг – это моё любимое увлечение.",
    "Я увлекаюсь кодингом на досуге.",
    "Программирование – это моё хобби и страсть.",
    "Когда есть свободное время, я кодирую.",
    "Кодинг - это моё хобби, которым я наслаждаюсь."

    # Cats & Felines
    "The cat is sitting by the window looking outside.",
    "A fluffy kitten is sleeping soundly on the living room sofa.",
    "Domestic cats generally prefer a diet high in protein.",
    "Felines are known to enjoy fresh fish and meat dishes.",
    "What is the best way to feed a newborn kitten?",
    "A balanced diet is essential for a cat's long-term health.",
    "Cats consume both dry kibble and wet canned food.",
    "Never feed your cat chocolate as it is highly toxic.",
    "Cow's milk is not always beneficial for adult felines.",
    "The house cat purrs softly when it receives affection.",

    # Dogs & Canines
    "A cheerful dog is playing with a ball in the public park.",
    "The loyal guard dog barks loudly at the approaching postman.",
    "A faithful puppy protects the house and the backyard.",
    "Dogs require regular walks and outdoor exercise every day.",
    "Training a young puppy requires patience and consistency.",
    "The golden retriever enjoys running across the green lawn.",
    "Canines use their sense of smell to explore the environment.",
    "A healthy dog needs a well-balanced nutritional plan.",
    "The energetic hound chased after the wooden stick.",
    "Many dogs show incredible loyalty toward their owners.",

    # Pineapples & Fruits
    "Pineapples grow abundantly in tropical climate zones.",
    "The tropical pineapple fruit is exceptionally delicious and sweet.",
    "A ripe pineapple is packed with essential vitamins and minerals.",
    "Exotic fruits thrive in warm and humid plantations.",
    "How do you properly select a fresh pineapple at the market?",
    "Pineapple juice contains enzymes that aid organic digestion.",
    "The sweet aroma of sliced tropical fruit filled the kitchen.",
    "Farmers harvest ripe pineapples during the peak summer season.",
    "A healthy diet includes a variety of fresh citrus fruits.",
    "The spiky skin protects the juicy yellow fruit inside.",

    # Programming & Coding
    "I absolutely love programming and developing software applications.",
    "Writing clean code is a passion that brings me great joy.",
    "Software engineering involves solving complex logical problems.",
    "I spend my spare time coding personal side projects.",
    "Computer programming is both my professional job and favorite hobby.",
    "Developing applications provides immense intellectual satisfaction.",
    "I enjoy learning new programming languages and algorithms.",
    "Coding in Python allows for rapid application prototyping.",
    "In my free time, I focus on building modern software solutions.",
    "Writing efficient algorithms is a highly rewarding experience."
]


def get_embedding(text, task_type):
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIMENSION
    )
    return np.asarray(response["embedding"], dtype=np.float32)


def build_index(texts):
    embeddings = []
    for i, text in enumerate(texts, start=1):
        print(f"Индексация {i}/{len(texts)}: {text}")
        vector = get_embedding(text, task_type="RETRIEVAL_DOCUMENT")
        embeddings.append(vector)

    embeddings_array = np.ascontiguousarray(np.stack(embeddings), dtype=np.float32)
    faiss.normalize_L2(embeddings_array)

    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)
    return index


def semantic_search(query, index, texts, k=4):
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")
    query_array = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
    faiss.normalize_L2(query_array)

    k = min(k, index.ntotal)
    scores, indices = index.search(query_array, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": texts[idx],
            "similarity": float(score)
        })
    return results

#    search_query = "Питание кошек"
def get_search_query_from_user():
    """Запрашивает поисковый запрос у пользователя через консоль."""
    print("\n" + "=" * 40)
    try:
        query = input("Введите поисковый запрос (или наберите 'exit' для выхода): ").strip()
        if query.lower() == 'exit' or not query:
            print("Выход из программы.")
            sys.exit(0)
        return query
    except (KeyboardInterrupt, EOFError):
        print("\nПрограмма принудительно завершена.")
        sys.exit(0)


def generate_combined_context(search_results):
    """Объединяет тексты из топ-4 результатов поиска в одну строку контекста."""
    extracted_texts = [result["text"] for result in search_results]
    # Объединение через пробел. При необходимости можно использовать переход на новую строку "\n"
    combined_context = " ".join(extracted_texts)
    return combined_context


if __name__ == "__main__":
    # Построение индекса Faiss
    print("--- Инициализация и построение индекса ---")
    index = build_index(texts_to_index)

    print("\nРазмерность индекса:", index.d)
    print("Количество векторов:", index.ntotal)

    # Интерактивное получение запроса от пользователя
    search_query = get_search_query_from_user()

    # Выполнение семантического поиска (топ-4)
    results = semantic_search(
        search_query,
        index,
        texts_to_index,
        k=4
    )

    print("\n--- Результаты семантического поиска ---")
    print(f"Запрос: {search_query}")
    for result in results:
        print(f"- {result['text']} (similarity: {result['similarity']:.4f})")

    # Формирование объединенного контекста для генеративной модели
    final_context = generate_combined_context(results)

    print("\n--- Сформированный контекст для генеративной модели ---")
    print(final_context)
