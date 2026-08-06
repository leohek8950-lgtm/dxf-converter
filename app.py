from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import ezdxf
import ezdxf.recover
import io
import math
import cv2
import numpy as np

app = FastAPI(title="AI CAD Converter & 3D Visualizer")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Ошибка: файл index.html не найден.</h1>"

def process_dxf_file(contents: bytes):
    """Обработка классических векторов DXF"""
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
            # Игнорируем осевые линии при векторизации DXF
            if 'CENTER' in line_style or 'DASHDOT' in line_style:
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

def process_image_file(contents: bytes):
    """Алгоритм токарного сканирования для растровых чертежей (PNG / JPG)"""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Бинаризация (выделение темных линий чертежа)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Определение оси симметрии детали (центр изображения по Y)
    center_y = h // 2
    profile_points = []

    # 3. Сканирование верхней половины чертежа (слева направо)
    # Игнорирует нижние размерные стрелки и рамку
    step = max(1, w // 250)  # Шаг выборки пикселей для точности
    for x in range(0, w, step):
        col = thresh[0:center_y, x]
        black_pixels = np.where(col > 0)[0]
        
        if len(black_pixels) > 0:
            top_y = black_pixels[0]  # Самый верхний пиксель детали
            radius = center_y - top_y  # Расстояние от оси до края
            profile_points.append({
                "x": float(x),
                "y": float(max(0, radius))
            })

    # Запасной алгоритм поиска контура, если сканирование не дало результат
    if len(profile_points) < 3:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            approx = cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True)
            profile_points = [{"x": float(pt[0][0]), "y": float(h - pt[0][1])} for pt in approx]

    return [profile_points], {"PROFILE_POINTS": len(profile_points)}

@app.post("/process-dxf")
async def process_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(('.dxf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Формат не поддерживается. Загрузите DXF, PNG или JPG.")

    contents = await file.read()

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            contours, breakdown = process_image_file(contents)
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
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
