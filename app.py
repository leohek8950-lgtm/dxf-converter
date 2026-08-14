import os
import io
import json
import re
import base64
import requests
from PIL import Image
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

def compress_image_bytes(img_bytes: bytes, max_dim=1200) -> bytes:
    """Оптимизация чертежа для мгновенной передачи в ИИ"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buf, format='JPEG', quality=90)
        return buf.getvalue()
    except Exception:
        return img_bytes

def parse_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен!")

    api_key = re.sub(r'\[.*?\]|\(|\)', '', api_key).strip()
    small_img_bytes = compress_image_bytes(img_bytes)
    base64_image = base64.b64encode(small_img_bytes).decode("utf-8")

    prompt = """
    Ты — эксперт-технолог по ЧПУ и CAD-проектированию.
    Проведи визуальный и размерный анализ машиностроительного чертежа.

    ИНСТРУКЦИЯ ПО ВЫЧИСЛЕНИЮ И СРАВНЕНИЮ РАЗМЕРОВ:
    1. ВИЗУАЛЬНЫЙ АУДИТ: 
       - Сопоставь внешний вид и разрез (A-A). Найди ВСЕ канавки, проточки, фаски и уступы.
       - Не пропусти канавку (например, канавку под выгрузку/уплотнение на диаметрах) и внутренние ступенчатые выточки.

    2. ВЫЧИСЛЕНИЕ НЕДОСТАЮЩИХ РАЗМЕРОВ (Цепочка размеров):
       - Если длина какого-то элемента (например, паза или уступа) не проставлена напрямую, ВЫЧИСЛИ ЕЁ через размерную цепь:
         Длина_элемента = Габаритная_длина - Сумма(Остальных_известных_длин).
       - Учитывай номинальные значения размеров (без допусков ±).

    3. ФОРМИРОВАНИЕ ГЕОМЕТРИИ (слева направо):
       - 'outer_profile': массив последовательных участков (cylinder, cone, chamfer, groove).
       - 'bores': массив всех внутренних расточек и отверстий с указанием 'diameter', 'depth' и стороны 'from_side' ('right' или 'left').

    Верни ТОЛЬКО валидный JSON без разметки markdown:
    {
      "part_name": "Деталь",
      "total_length": 45.0,
      "outer_profile": [
        {"type": "cylinder", "start_diameter": 18.0, "end_diameter": 18.0, "length": 5.0},
        {"type": "groove", "start_diameter": 15.0, "end_diameter": 15.0, "length": 1.5},
        {"type": "cylinder", "start_diameter": 24.0, "end_diameter": 24.0, "length": 5.0},
        {"type": "cylinder", "start_diameter": 22.0, "end_diameter": 22.0, "length": 12.5},
        {"type": "cylinder", "start_diameter": 35.0, "end_diameter": 35.0, "length": 10.0},
        {"type": "cylinder", "start_diameter": 36.0, "end_diameter": 36.0, "length": 11.0}
      ],
      "bores": [
        {"diameter": 30.0, "depth": 5.0, "from_side": "right"},
        {"diameter": 22.0, "depth": 10.0, "from_side": "right"},
        {"diameter": 20.0, "depth": 11.2, "from_side": "right"}
      ]
    }
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    models_to_try = [
        ("v1beta", "gemini-2.0-flash"),
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-2.5-flash")
    ]

    last_err = None
    for api_ver, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"
        params = {"key": api_key}
        try:
            res = requests.post(url, params=params, json=payload, timeout=50)
            if res.status_code == 200:
                res_data = res.json()
                text_content = res_data['candidates'][0]['content']['parts'][0]['text']
                return json.loads(text_content.strip())
            else:
                last_err = f"HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_err = str(e)

    raise ValueError(f"Ошибка обращения к ИИ: {last_err}")

def build_cad_model(spec: dict, filename_base: str):
    profile = spec.get("outer_profile", [])
    if not profile:
        raise ValueError("Не найден внешний профиль детали.")

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

    # 2. Построение внутренних расточек
    bores = spec.get("bores", [])
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
    
    # Корректная очистка имени файла
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(file.filename)[0])
    if not clean_name:
        clean_name = "cad_model"
    filename_base = f"part_{clean_name}"

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
        return FileResponse(
            file_path, 
            media_type="application/octet-stream", 
            filename=filename
        )
    raise HTTPException(status_code=404, detail="Файл не найден на сервере.")
