import os
import io
import json
import re
import base64
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cadquery as cq

app = FastAPI(title="AI Precision CAD Engine")

# Создаем папку для экспорта файлов
os.makedirs("exports", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

def parse_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables!")

    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
    Ты — главный инженер-технолог по ЧПУ и эксперт по чтению чертежей ISO/ГОСТ.
    Проанализируй чертеж и извлеки ПОЛНУЮ точную геометрию детали.
    
    Верни ТОЛЬКО валидный JSON со строгой структурой:
    {
      "part_name": "Деталь",
      "type": "revolve",  // "revolve" для токарных деталей или "extrude" для корпусных
      "axis_steps": [     // Для деталей вращения: слева направо ступенчато
        {"diameter": 50.0, "length": 30.0},
        {"diameter": 35.0, "length": 40.0},
        {"diameter": 20.0, "length": 20.0}
      ],
      "inner_bore": {     // Центральное отверстие (если есть)
        "diameter": 16.0,
        "through": true
      },
      "chamfers": [      // Фаски
        {"size": 1.5, "location": "front_outer"},
        {"size": 1.0, "location": "back_outer"}
      ]
    }
    Строго соблюдай все числовые размеры с чертежа в миллиметрах (мм). Не выдумывай размеры.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }]
    }

    res = requests.post(url, json=payload, timeout=25)
    if res.status_code != 200:
        raise ValueError(f"Ошибка ИИ: {res.status_code} - {res.text}")

    res_data = res.json()
    text_content = res_data['candidates'][0]['content']['parts'][0]['text']
    clean_json = re.sub(r'```json\s*|\s*```', '', text_content).strip()

    return json.loads(clean_json)

def build_cad_model(spec: dict, filename_base: str):
    """Генерация точной CAD-модели с помощью CadQuery (OpenCASCADE)"""
    
    # 1. Построение детали вращения
    steps = spec.get("axis_steps", [])
    if not steps:
        raise ValueError("Не найдены ступенчатые элементы детали.")

    # Строим профиль
    current_z = 0.0
    result = None

    for step in steps:
        r = step["diameter"] / 2.0
        l = step["length"]
        cyl = cq.Workplane("XY").workplane(offset=current_z).circle(r).extrude(l)
        if result is None:
            result = cyl
        else:
            result = result.union(cyl)
        current_z += l

    # 2. Выполнение центрального отверстия
    bore = spec.get("inner_bore")
    if bore and bore.get("diameter", 0) > 0:
        bore_r = bore["diameter"] / 2.0
        result = result.faces("<Z").workplane().circle(bore_r).cutThruAll()

    # 3. Сохранение в STEP и STL
    step_path = f"exports/{filename_base}.step"
    stl_path = f"exports/{filename_base}.stl"

    cq.exporters.export(result, step_path)
    cq.exporters.export(result, stl_path)

    return step_path, stl_path

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename_base = os.path.splitext(file.filename)[0]

    try:
        # Шаг 1: Распознавание ИИ
        spec = parse_drawing_with_gemini(contents)
        
        # Шаг 2: Параметрическая генерация 3D STEP / STL
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
    path = f"exports/{filename}"
    if os.path.exists(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="Файл не найден.")
