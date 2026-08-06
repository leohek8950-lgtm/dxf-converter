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

def filter_dimension_spikes(y_coords, window_size=17, spike_threshold=10.0):
    """
    Удаляет узкие пики (выносные размерные линии), сравнивая 
    точки профиля с локальной медианой окрестности.
    """
    if len(y_coords) < window_size:
        return y_coords

    arr = np.array(y_coords, dtype=np.float32)
    half = window_size // 2
    padded = np.pad(arr, (half, half), mode='edge')

    # Вычисление скользящей медианы
    medians = np.array([np.median(padded[i : i + window_size]) for i in range(len(arr))])

    # Если точка резко выскакивает вверх относительно медианы — это размерная линия
    cleaned = np.where(arr > medians + spike_threshold, medians, arr)

    # Итоговый проход скользящей медианы для сглаживания граней
    padded_cleaned = np.pad(cleaned, (half, half), mode='edge')
    final_y = [float(np.median(padded_cleaned[i : i + window_size])) for i in range(len(cleaned))]

    return final_y

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
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Бинаризация изображения
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 2. Удаление мелкого одиночного шума и стрелок (морфологическая очистка)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    center_y = h // 2
    raw_x = []
    raw_y = []

    # 3. Сканирование профиля детали от оси вращения Y вверх
    step = max(1, w // 300)
    for x in range(0, w, step):
        col = thresh_cleaned[0:center_y, x]
        black_pixels = np.where(col > 0)[0]
        
        if len(black_pixels) > 0:
            top_y = black_pixels[0]
            radius = center_y - top_y
            raw_x.append(float(x))
            raw_y.append(float(max(0, radius)))

    if len(raw_y) < 5:
        raise ValueError("Контур детали не найден.")

    # 4. Фильтрация выносных размерных линий (убирает узкие вертикальные диски)
    filtered_y = filter_dimension_spikes(raw_y, window_size=19, spike_threshold=8.0)

    # 5. Формирование итогового профиля
    profile_points = [{"x": x, "y": y} for x, y in zip(raw_x, filtered_y)]

    return [profile_points], {"PROFILE_POINTS": len(profile_points)}

@app.post("/process-dxf")
async def process_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(('.dxf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Формат не поддерживается. Используйте DXF, PNG или JPG.")

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
