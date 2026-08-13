import os
import io
import json
import re
import math
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import ezdxf
import ezdxf.recover
import google.generativeai as genai

app = FastAPI(title="AI CAD Converter - Gemini Vision Engine")

# Безопасная инициализация API Ключа
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"Ошибка конфигурации Gemini API: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Ошибка: файл index.html не найден.</h1>"

def analyze_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY не найден в Environment Variables!")

    genai.configure(api_key=api_key)

    try:
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise ValueError("Не удалось прочитать загруженное изображение.")

    prompt = """
    Ты — инженер-конструктор и эксперт по распознаванию чертежей токарных деталей.
    Проанализируй чертеж детали вращения:
    1. ПОЛНОСТЬЮ ИГНОРИРУЙ все выносные линии, размерные стрелки, допуски, тексты, рамку и штамп чертежа.
    2. Определи центральную ось вращения детали.
    3. Найди верхний контур металлической детали (радиусный профиль) от левого до правого края.
    4. Сформируй координаты ступеней/переходов детали слева направо.
    
    Верни ТОЛЬКО чистый JSON-массив без markdown-тегов и текста в формате:
    [
      {"x": 0, "y": 15},
      {"x": 20, "y": 15},
      {"x": 20, "y": 25},
      {"x": 60, "y": 25}
    ]
    где 'x' — координата по длине (растет от 0), а 'y' — радиус детали относительно оси.
    Передай от 20 до 60 ключевых точек ступеней контура.
    """

    # Подбор доступной модели в порядке приоритета
    candidate_models = [
        'gemini-1.5-flash',
        'gemini-1.5-flash-latest',
        'gemini-1.5-pro'
    ]

    response = None
    last_error = None

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([image, prompt])
            if response and response.text:
                break
        except Exception as err:
            last_error = err
            continue

    if not response or not response.text:
        raise ValueError(f"Ошибка ИИ: {str(last_error)}")

    text_content = response.text.strip()
    clean_json = re.sub(r'```json\s*|\s*```', '', text_content).strip()

    try:
        profile = json.loads(clean_json)
        return profile
    except Exception:
        raise ValueError("ИИ вернул ответ в некорректном формате JSON.")

def process_dxf_file(contents: bytes):
    try:
        doc, auditor = ezdxf.recover.read(io.BytesIO(contents))
    except Exception:
        text_data = contents.decode('utf-8', errors='ignore')
        doc, auditor = ezdxf.recover.read(io.StringIO(text_data))

    msp = doc.modelspace()
    contours = []

    for entity in msp:
        e_type = entity.dxftype()
        if e_type == 'LINE':
            s, e = entity.dxf.start, entity.dxf.end
            line_style = getattr(entity.dxf, 'linetype', '').upper()
            layer_name = getattr(entity.dxf, 'layer', '').upper()
            if 'CENTER' in line_style or 'DASHDOT' in line_style or 'DIM' in layer_name:
                continue
            contours.append([
                {"x": round(s.x, 2), "y": round(s.y, 2)},
                {"x": round(e.x, 2), "y": round(e.y, 2)}
            ])
        elif e_type in ('CIRCLE', 'ARC'):
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r = entity.dxf.radius
            start_angle = 0.0 if e_type == 'CIRCLE' else math.radians(entity.dxf.start_angle)
            end_angle = (2 * math.pi) if e_type == 'CIRCLE' else math.radians(entity.dxf.end_angle)
            if end_angle <= start_angle:
                end_angle += 2 * math.pi

            steps = max(16, int(math.degrees(end_angle - start_angle) / 5))
            pts = []
            for i in range(steps + 1):
                a = start_angle + i * ((end_angle - start_angle) / steps)
                pts.append({"x": round(cx + r * math.cos(a), 2), "y": round(cy + r * math.sin(a), 2)})
            contours.append(pts)

    return contours, {"DXF_ENTITIES": len(contours)}

@app.post("/process-dxf")
async def process_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(('.dxf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Формат файла не поддерживается.")

    contents = await file.read()

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            profile = analyze_drawing_with_gemini(contents)
            return {
                "filename": file.filename,
                "status": "success",
                "contours_count": 1,
                "entity_breakdown": {"GEMINI_AI": True, "POINTS_COUNT": len(profile)},
                "contours": [profile]
            }
        else:
            contours, breakdown = process_dxf_file(contents)
            return {
                "filename": file.filename,
                "status": "success",
                "contours_count": len(contours),
                "entity_breakdown": breakdown,
                "contours": contours
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
