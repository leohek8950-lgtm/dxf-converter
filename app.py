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

def extract_engineering_profile(img_bytes: bytes):
    """
    Инженерный анализатор чертежей:
    - Подавление тонких выносных/размерных линий (ГОСТ 2.303-68)
    - Поиск основного видимого контура детали
    - Построение радиусного профиля относительно оси симметрии
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Бинаризация (инвертируем: чертеж -> белый, фон -> черный)
    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # 2. Фильтрация по толщине линий (ГОСТ: Основная линия толще размерной)
    # Морфологическое "открытие" стирает тонкие выносные линии и стрелки (1-2px)
    kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    main_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_thick)

    # 3. Фильтрация мелких текстовых компонентов и размеров
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(main_lines)
    clean_mask = np.zeros_like(main_lines)
    
    # Оставляем только крупные связные области (саму деталь)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Если элемент имеет физический размер детали (не мелкая цифра или стрелка)
        if area > (w * h * 0.002) and width > (w * 0.08):
            clean_mask[labels == i] = 255

    # 4. Поиск горизонтальной оси симметрии (центр масс детали)
    M = cv2.moments(clean_mask)
    if M["m00"] != 0:
        center_y = int(M["m01"] / M["m00"])
    else:
        center_y = h // 2

    # 5. Сканирование профиля детали от оси вращения (Center Y) ВВЕРХ
    raw_x = []
    raw_r = []

    # Определяем границы детали по X
    x_indices = np.where(clean_mask > 0)[1]
    if len(x_indices) == 0:
        raise ValueError("Не удалось выделить главный контур детали.")

    min_x, max_x = np.min(x_indices), np.max(x_indices)
    
    # Сканируем только тело детали
    step = max(1, (max_x - min_x) // 200)
    for x in range(min_x, max_x, step):
        col = clean_mask[0:center_y, x]
        black_pixels = np.where(col > 0)[0]
        
        if len(black_pixels) > 0:
            top_y = black_pixels[0]
            radius = center_y - top_y
            raw_x.append(float(x - min_x)) # Смещаем начало к 0
            raw_r.append(float(radius))

    # 6. Аппроксимация ступеней (Токарная обработка состоит из цилиндров и конусов)
    # Медианное сглаживание для удаления остаточного пиксельного «мусора»
    window = 11
    smoothed_r = []
    half = window // 2
    padded = np.pad(raw_r, (half, half), mode='edge')
    
    for i in range(len(raw_r)):
        smoothed_r.append(float(np.median(padded[i : i + window])))

    # Формируем контур для Three.js
    profile_points = [{"x": x, "y": r} for x, r in zip(raw_x, smoothed_r)]

    return [profile_points], {"CONTOUR_POINTS": len(profile_points), "AXIS_Y": center_y}

def process_dxf_file(contents: bytes):
    """Анализатор векторов DXF"""
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
            if 'CENTER' in line_style or 'DASHDOT' in line_style or 'DIM' in entity.dxf.layer.upper():
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
        raise HTTPException(status_code=400, detail="Формат не поддерживается. Используйте DXF, PNG или JPG.")

    contents = await file.read()

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            contours, breakdown = extract_engineering_profile(contents)
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
        raise HTTPException(status_code=500, detail=f"Ошибка анализа чертежа: {str(e)}")
