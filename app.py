from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import io
import math
import ezdxf
import ezdxf.recover

app = FastAPI(title="AI CAD Converter - Fast Engine")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Ошибка: файл index.html не найден.</h1>"

def clean_engineering_drawing(img_bytes: bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать файл изображения.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Автоматический бинарный порог Оцу (адаптируется под любой фон и освещение)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Авто-проверка фонового цвета (если край картинки оказался белым после инверсии)
    border_pixels = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    if np.mean(border_pixels) > 127:
        binary = cv2.bitwise_not(binary)

    # 2. Мягкая очистка шума (ядро 2x2 не разрушает тонкие контуры)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    clean_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 3. Выделение компонентов
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_binary)

    if num_labels <= 1:
        clean_binary = binary
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_binary)

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        raise ValueError("Чертеж слишком бледный или не содержит линий.")

    # 4. Поиск детали (мягкий фильтр + автоматический выбор крупного объекта)
    valid_indices = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        if area > (w * h * 0.0003) or width > (w * 0.02):
            valid_indices.append(i)

    # Если мягкий фильтр отсек всё — берем наибольший объект на изображении
    if not valid_indices:
        valid_indices = [np.argmax(areas) + 1]

    clean_mask = np.zeros_like(clean_binary)
    for idx in valid_indices:
        clean_mask[labels == idx] = 255

    # 5. Вычисление оси вращения
    M = cv2.moments(clean_mask)
    center_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else h // 2

    x_indices = np.where(clean_mask > 0)[1]
    if len(x_indices) == 0:
        raise ValueError("Не удалось определить границы контура.")

    min_x, max_x = np.min(x_indices), np.max(x_indices)

    raw_x, raw_r = [], []
    step = max(1, (max_x - min_x) // 200)

    # 6. Сканирование радиусов
    for x in range(min_x, max_x, step):
        col = clean_mask[0:center_y, x]
        pixels = np.where(col > 0)[0]
        if len(pixels) > 0:
            top_y = pixels[0]
            raw_x.append(float(x - min_x))
            raw_r.append(float(center_y - top_y))

    # Страховочный проход, если верхняя часть детали была перекрыта
    if not raw_x:
        for x in range(min_x, max_x, step):
            col = clean_mask[:, x]
            pixels = np.where(col > 0)[0]
            if len(pixels) > 0:
                radius = (pixels[-1] - pixels[0]) / 2.0
                raw_x.append(float(x - min_x))
                raw_r.append(float(radius))

    # Медианное сглаживание ступеней
    window = 7
    if len(raw_r) >= window:
        padded = np.pad(raw_r, (window // 2, window // 2), mode='edge')
        smoothed_r = [float(np.median(padded[i : i + window])) for i in range(len(raw_r))]
    else:
        smoothed_r = raw_r

    profile = [{"x": x, "y": r} for x, r in zip(raw_x, smoothed_r)]
    return profile, center_y

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
        raise HTTPException(status_code=400, detail="Формат не поддерживается.")

    contents = await file.read()

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            profile, axis_y = clean_engineering_drawing(contents)
            return {
                "filename": file.filename,
                "status": "success",
                "contours_count": 1,
                "entity_breakdown": {"POINTS": len(profile), "AXIS_Y": axis_y},
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
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
