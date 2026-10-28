import os
# Отключаем лишний вывод от библиотек Google, чтобы они не забивали консоль
os.environ["GRPC_VERBOSITY"] = "NONE"      # Полностью выключаем логи gRPC
os.environ["GLOG_minloglevel"] = "3"       # Выключаем C++ логи Google (3 = только фатальные ошибки)

import google.generativeai as genai
from google.api_core import exceptions  # Импортируем модуль исключений Google API
from dotenv import load_dotenv
from datetime import datetime

from loaders import loading_animation

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
# print(api_key[:10])
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.8-flash")

current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
prompt = f"Контекст: Текущая дата и время на устройстве пользователя: {current_time_str}.\nВопрос: Какой сегодня день недели, месяц и год?"

# Переменная для хранения ответа
response = None

try:
    # Запускаем анимацию
    with loading_animation(message="Запрос к Gemini API", style="spin"):
        response = model.generate_content(prompt)
except exceptions.ResourceExhausted as e:
    # Перехватываем ошибку исчерпания лимитов (код 429)
    print(f"\n[Ошибка квоты] Вы исчерпали лимит бесплатных запросов Gemini API.")
    print("Пожалуйста, повторите попытку позже или проверьте лимиты в Google AI Studio.")
except exceptions.GoogleAPICallError as e:
    # Перехватываем любые другие ошибки самого API (например, неверный ключ)
    print(f"\n[Ошибка API] Произошел сбой при обращении к Google Gemini: {e}")
except Exception as e:
    # На случай непредвиденных ошибок (например, пропал интернет)
    print(f"\n[Ошибка сети/системы] Не удалось выполнить запрос: {e}")

# Выводим ответ модели только в том случае, если он был успешно получен
if response:
    print("\nОтвет модели:")
    print(response.text)
else:
    print("\n[Инфо] Программа завершена без получения ответа от модели.")
