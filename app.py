import os
import io
import json
import re
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import cadquery as cq

app = FastAPI(title="Precision AI CAD Engine")

EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Ошибка: файл index.html не найден.</h1>"

def parse_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables!")

    # Очистка ключа от случайных символов разметки
    api_key = re.sub(r'\[.*?\]|\(|\)', '', api_key).strip()

    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
    Ты — ведущий инженер-конструктор ЧПУ. Твоя задача — идеально прочитать чертеж токарной детали и извлечь все до единого размеры.

    ВНИМАТЕЛЬНО ИЗУЧИ ЧЕРТЕЖ СЛЕВА НАПРАВО:
    1. Найди общую длину детали (например, 61 мм).
    2. Разбей внешнюю геометрию на элементы (конусы и цилиндры):
       - Указывай 'start_diameter' и 'end_diameter' для КАЖДОГО участка. 
       - Если участок цилиндрический, start_diameter == end_diameter.
       - Если участок конический (например, с Ø4.5 до Ø18), укажи начальный и конечный диаметры и его длину (например, 23.5 мм).
    3. Найди внутреннее отверстие:
       - Извлеки диаметр (например, Ø8).
       - Извлеки глубину глухого отверстия (например, 26.5 мм от правого торца).

    Верни ТОЛЬКО валидный JSON без текста и markdown-тегов:
    {
      "part_name": "Токарная деталь",
      "total_length": 61.0,
      "outer_profile": [
        {
          "type": "cone",
          "start_diameter": 4.5,
          "end_diameter": 18.0,
          "length": 23.5
        },
        {
          "type": "cylinder",
          "start_diameter": 18.0,
          "end_diameter": 18.0,
          "length": 16.0
        },
        {
          "type": "cone",
          "start_diameter": 18.0,
          "end_diameter": 16.0,
          "length": 21.5
        }
      ],
      "bores": [
        {
          "diameter": 8.0,
          "depth": 26.5,
          "from_side": "right"
        }
      ]
    }
    Соблюдай абсолютную точность всех чисел в миллиметрах (мм)!
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }]
    }

    # Список доступных моделей (запросы будут строиться безопасно)
    models_to_try = [
        ("v1beta", "gemini-2.5-flash"),
        ("v1", "gemini-1.5-flash"),
        ("v1beta", "gemini-2.0-flash"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-1.5-pro")
    ]

    res_data = None
    last_error = None

    for api_ver, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"
        params = {"key": api_key}
        
        try:
            res = requests.post(url, params=params, json=payload, timeout=25)
            if res.status_code == 200:
                res_data = res.json()
                break
            else:
                last_error = f"HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_error = str(e)

    if not res_data:
        raise ValueError(f"Ошибка ИИ: {last_error}")

    try:
        text_content = res_data['candidates'][0]['content']['parts'][0]['text']
        clean_json = text_content.replace("```json", "").replace("
