import os
import io
import json
import re
import base64
import asyncio
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cadquery as cq

app = FastAPI(title="Precision AI CAD Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Ошибка: файл index.html не найден в корне проекта.</h1>"

def compress_image_bytes(img_bytes: bytes, max_dim=800) -> bytes:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()
    except Exception:
        return img_bytes

def parse_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables!")

    api_key = re.sub(r'\[.*?\]|\(|\)', '', api_key).strip()
    small_bytes = compress_image_bytes(img_bytes)
    base64_image = base64.b64encode(small_bytes).decode("utf-8")

    prompt = """
    Ты — главный инженер-конструктор ЧПУ. Оцифруй чертеж детали в JSON.

    ИНСТРУКЦИЯ ПО ВЫЧИСЛЕНИЮ И СРАВНЕНИЮ РАЗМЕРОВ:
    1. ВИЗУАЛЬНЫЙ АУДИТ: Найди ВСЕ канавки, фаски, пазы и ступенчатые выточки на главном виде и разрезе A-A.
    2. ВЫЧИСЛЕНИЕ РАЗМЕРНЫХ ЦЕПЕЙ: Если длина элемента/паза не указана прямым числом, ВЫЧИСЛИ её: 
       L_паза = L_общее - Sum(L_остальных_известных_сегментов).
    3. Используй ТОЛЬКО номинальные размеры (без допусков ±).

    Верни ТОЛЬКО чистый JSON без markdown:
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
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    models_to_try = [
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-2.0-flash")
    ]

    last_err = None
    for api_ver, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"
        params = {"key": api_key}
        try:
            res = requests.post(url, params=params, json=payload, timeout=20)
            if res.status_code == 200:
                res_data = res.json()
                text_content = res_data['candidates'][0]['content']['parts'][0]['text']
                cleaned_text = re.sub(r'```json\s*|\s*```', '', text_content.strip())
                return json.loads(cleaned_text)
            else:
                last_err = f"[{model_name}] HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_err = f"[{model_name}] {str(e)}"

    raise ValueError(f"Ошибка обращения к ИИ: {last_err}")

def build_cad_model(spec: dict, filename_base: str):
    profile = spec.get("outer_profile", [])
    if not profile:
        raise ValueError("В ответе ИИ отсутствует внешний профиль детали.")

    current_z = 0.0
    result = None

    for seg in profile:
        d1 = float(seg.get("start_diameter", 10.0))
        d2 = float(seg.get("end_diameter", d1))
        l = float(seg.get("length", 0.0))

        if l <= 0.001:
            continue

        r1 = max(d1 / 2.0, 0.1)
        r2 = max(d2 / 2.0, 0.1)

        try:
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
        except Exception as e:
            print(f"Пропущен элемент профиля ({seg}): {e}")
            continue

        current_z += l

    if result is None:
        raise ValueError("Не удалось построить геометрию из переданного JSON.")

    bores = spec.get("bores", [])
    bores_sorted = sorted(bores, key=lambda x: float(x.get("diameter", 0)), reverse=True)

    for bore in bores_sorted:
        bd = float(bore.get("diameter", 0))
        depth = float(bore.get("depth", 0))
        from_side = bore.get("from_side", "right")

        if bd > 0 and depth > 0:
            try:
                if from_side == "right":
                    result = result.faces(">Z").workplane().hole(bd, depth)
                else:
                    result = result.faces("<Z").workplane().hole(bd, depth)
            except Exception as e:
                print(f"Ошибка при расточке ({bore}): {e}")

    step_path = os.path.join(EXPORTS_DIR, f"{filename_base}.step")
    stl_path = os.path.join(EXPORTS_DIR, f"{filename_base}.stl")

    cq.exporters.export(result, step_path)
    cq.exporters.export(result, stl_path)

    return step_path, stl_path

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(file.filename)[0])
        if not clean_name:
            clean_name = "cad_model"
        filename_base = f"part_{clean_name}"

        spec = await asyncio.to_thread(parse_drawing_with_gemini, contents)
        step_file, stl_file = await asyncio.to_thread(build_cad_model, spec, filename_base)

        return JSONResponse(content={
            "status": "success",
            "spec": spec,
            "stl_url": f"/download/{filename_base}.stl",
            "step_url": f"/download/{filename_base}.step"
        })
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "error": str(e),
                "spec": {"error": str(e)}
            }
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(EXPORTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="application/octet-stream", 
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            filename=filename
        )
    return JSONResponse(status_code=404, content={"error": "Файл не найден на сервере."})
