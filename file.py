import os
import google.generativeai as genai
from dotenv import load_dotenv
import time
from datetime import datetime  # Для получения текущей даты

# Импортируем наш контекстный менеджер
from loaders import loading_animation

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Создаем модель
model = genai.GenerativeModel("gemini-3.8-flash")

# 1. Получаем текущую дату и время на компьютере
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. Формируем промпт, передавая актуальное время контекстом
prompt = f"Контекст: Текущая дата и время на устройстве пользователя: {current_time_str}.\nВопрос: Какой сегодня день недели, месяц и год?"

# 3. Оборачиваем долгий запрос к API в анимацию спиннера
with loading_animation(message="Запрос к Gemini API", style="spin"):
    response = model.generate_content(prompt)

# Выводим ответ модели
print("\nОтвет модели:")
print(response.text)
