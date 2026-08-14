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

    api_key = re.sub(r'\[.*?\]|\(|\)', '', api_key).strip()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
    Ты — главный инженер-технолог по ЧПУ. Твоя задача — детально разобрать машиностроительный чертеж детали и сформировать JSON-описание внешней и внутренней геометрии.

    ВНИМАТЕЛЬНО ИЗУЧИ ЧЕРТЕЖ:
    1. ВНЕШНИЙ ПРОФИЛЬ (outer_profile):
       - Разбей геометрию слева направо на последовательные цилиндрические и конические участки/канавки/фаски.
       - Указывай номинальные размеры без допусков (например, из '18+0.2' бери 18.0, из '22-0.05' бери 22.0).
       - Обязательно укажи start_diameter, end_diameter и length для каждого участка.

    2. ВНУТРЕННИЕ ОТВЕРСТИЯ И РАСТОЧКИ (bores):
       - Найди ступенчатые отверстия и расточки (разрез A-A).
       - Укажи диаметр (diameter), глубину (depth) и с какого торца идет расточка (from_side: 'right' или 'left').

    Формат ответа (ТОЛЬКО ЧИСТЫЙ JSON):
    {
      "part_name": "Ступенчатая деталь",
      "total_length": 45.0,
      "outer_profile": [
        {"type": "cylinder", "start_diameter": 18.0, "end_diameter": 18.0, "length": 5.0},
        {"type": "cylinder", "start_diameter": 24.0, "end_diameter": 24.0, "length": 5.0},
        {"type": "cylinder", "start_diameter": 22.0, "end_diameter": 22.0, "length": 12.5},
        {"type": "cylinder", "start_diameter": 35.0, "end_diameter": 35.0, "length": 10.0},
        {"type": "cylinder", "start_diameter": 36.0, "end_diameter": 36.0, "length": 12.5}
      ],
      "bores": [
        {"diameter": 30.0, "depth": 5.0, "from_side": "right"},
        {"diameter": 22.0, "depth": 10.0, "from_side": "right"},
        {"diameter": 20.0, "depth": 11.2, "from_side": "right"}
      ]
    }
    Строго соблюдай все номинальные размеры в миллиметрах!
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }]
    }

    # Поочередная попытка быстрых и стабильных моделей
    models_to_try = [
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-2.0-flash"),
        ("v1beta", "gemini-2.5-flash")
    ]

    res_data = None
    last_error = None

    for api_ver, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"
        params = {"key": api_key}
        
        try:
            # Таймаут увеличен до 90 секунд
            res = requests.post(url, params=params, json=payload, timeout=90)
            if res.status_code == 200:
                res_data = res.json()
                break
            else:
                last_error = f"HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_error = str(e)

    if not res_data:
        raise ValueError(f"Ошибка обращения к ИИ: {last_error}")

    try:
        text_content = res_data['candidates'][0]['content']['parts'][0]['text']
        match = re.search(r'\{.*\}', text_content, re.DOTALL)
        if match:
            clean_json = match.group(0)
        else:
            clean_json = text_content.strip()
        return json.loads(clean_json)
    except Exception as parse_err:
        raise ValueError(f"Ошибка парсинга ответа ИИ: {parse_err}")

def build_cad_model(spec: dict, filename_base: str):
    """Генерация CAD-модели в CadQuery с поддержкой ступенчатых валов и расточек"""
    profile = spec.get("outer_profile", [])
    if not profile:
        raise ValueError("В ответе ИИ не найден внешний профиль детали.")

    current_z = 0.0
    result = None

    # 1. Построение внешнего контура
    for seg in profile:
        d1 = float(seg.get("start_diameter", 10.0))
        d2 = float(seg.get("end_diameter", 10.0))
        l = float(seg.get("length", 10.0))

        if l <= 0:
            continue

        r1 = d1 / 2.0
        r2 = d2 / 2.0

        if abs(r1 - r2) < 0.001:
            solid = cq.Workplane("XY").workplane(offset=current_z).circle(r1).extrude(l)
        else:
            solid = (
                cq.Workplane("XY")
                .workplane(offset=current_z)
                .circle(r1)
                .workplane(offset=l)
                .circle(r2)
                .loft(combine=True)
            )

        if result is None:
            result = solid
        else:
            result = result.union(solid)

        current_z += l

    # 2. Выполнение ступенчатых отверстий и расточек
    bores = spec.get("bores", [])
    # Сортируем отверстия от наибольшего диаметра к наименьшему для правильного формирования расточек
    bores_sorted = sorted(bores, key=lambda x: float(x.get("diameter", 0)), reverse=True)

    for bore in bores_sorted:
        bd = float(bore.get("diameter", 0))
        depth = float(bore.get("depth", 0))
        from_side = bore.get("from_side", "right")

        if bd > 0 and depth > 0:
            if from_side == "right":
                result = result.faces(">Z").workplane().hole(bd, depth)
            else:
                result = result.faces("<Z").workplane().hole(bd, depth)

    step_path = os.path.join(EXPORTS_DIR, f"{filename_base}.step")
    stl_path = os.path.join(EXPORTS_DIR, f"{filename_base}.stl")

    cq.exporters.export(result, step_path)
    cq.exporters.export(result, stl_path)

    return step_path, stl_path

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    contents = await file.read()
    filename_base = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(file.filename)[0])

    try:
        spec = parse_drawing_with_gemini(contents)
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
    raise HTTPException(status_code=404, detail="Файл не найден.")
