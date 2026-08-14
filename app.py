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

# Создаем папку для экспорта файлов STEP и STL
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables на сервере!")

    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
    Ты — главный инженер-технолог по ЧПУ и эксперт по чтению машиностроительных чертежей ISO/ГОСТ.
    Проанализируй чертеж и извлеки ПОЛНУЮ точную геометрию детали.

    Верни ТОЛЬКО чистый валидный JSON без markdown-тегов в следующем формате:
    {
      "part_name": "Деталь",
      "type": "revolve",
      "axis_steps": [
        {"diameter": 50.0, "length": 30.0},
        {"diameter": 35.0, "length": 40.0},
        {"diameter": 20.0, "length": 20.0}
      ],
      "inner_bore": {
        "diameter": 16.0,
        "through": true
      },
      "chamfers": [
        {"size": 1.5, "location": "front_outer"}
      ]
    }

    Требования:
    1. Все размеры указывай строго в миллиметрах (мм).
    2. 'axis_steps' описывает внешние цилиндрические/конические ступени детали слева направо.
    3. 'inner_bore' описывает центральное внутреннее отверстие (если есть).
    4. Игнорируй рамки чертежа, штамп, текстовые примечания и выносные стрелки — учитывай только точную геометрию тела детали.
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }]
    }

    # Поочередный перебор эндпоинтов Google Gemini
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    ]

    res_data = None
    last_error = None

    for url in endpoints:
        try:
            res = requests.post(url, json=payload, timeout=25)
            if res.status_code == 200:
                res_data = res.json()
                break
            else:
                last_error = f"HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_error = str(e)

    if not res_data:
        raise ValueError(f"Ошибка ИИ при запросе к Gemini API: {last_error}")

    try:
        text_content = res_data['candidates'][0]['content']['parts'][0]['text']
        clean_json = re.sub(r'```json\s*|\s*```', '', text_content).strip()
        return json.loads(clean_json)
    except Exception as parse_err:
        raise ValueError(f"Ошибка парсинга JSON от ИИ: {parse_err}")

def build_cad_model(spec: dict, filename_base: str):
    """Построение точной CAD-модели с помощью ядра CadQuery (OpenCASCADE)"""
    steps = spec.get("axis_steps", [])
    if not steps:
        raise ValueError("В ответе ИИ не найдены геометрические ступени детали (axis_steps).")

    current_z = 0.0
    result = None

    # 1. Построение внешнего контура (ступеней детали)
    for step in steps:
        r = float(step.get("diameter", 10.0)) / 2.0
        l = float(step.get("length", 10.0))
        cyl = cq.Workplane("XY").workplane(offset=current_z).circle(r).extrude(l)
        if result is None:
            result = cyl
        else:
            result = result.union(cyl)
        current_z += l

    # 2. Выполнение сквозного или глухого центрального отверстия
    bore = spec.get("inner_bore")
    if bore and float(bore.get("diameter", 0)) > 0:
        bore_r = float(bore["diameter"]) / 2.0
        result = result.faces("<Z").workplane().circle(bore_r).cutThruAll()

    # 3. Экспорт в промышленные форматы STEP и STL
    step_path = os.path.join(EXPORTS_DIR, f"{filename_base}.step")
    stl_path = os.path.join(EXPORTS_DIR, f"{filename_base}.stl")

    cq.exporters.export(result, step_path)
    cq.exporters.export(result, stl_path)

    return step_path, stl_path

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Поддерживаются только изображения чертежей (PNG, JPG, JPEG).")

    contents = await file.read()
    filename_base = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(file.filename)[0])

    try:
        # Распознавание геометрических параметров ИИ
        spec = parse_drawing_with_gemini(contents)

        # Генерация 3D STEP и STL моделей
        step_file, stl_file = build_cad_model(spec, filename_base)

        return {
            "status": "success",
            "spec": spec,
            "stl_url": f"/download/{filename_base}.stl",
            "step_url": f"/download/{filename_base}.step"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(EXPORTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Запрошенный файл не найден.")
