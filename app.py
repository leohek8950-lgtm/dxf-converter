from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import io
import math
import ezdxf
import ezdxf.recover
import os

# Проверяем наличие нейросети ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

app = FastAPI(title="AI CAD Converter - Neural Vision")

# Загрузка нейросети YOLOv8 Segmentation
# При первом запуске автоматически скачается лёгкая модель (14 МБ)
yolo_model = None
if YOLO_AVAILABLE:
    try:
        yolo_model = YOLO("yolov8n-seg.pt")
        print(">>> Нейросеть YOLOv8-seg успешно загружена! <<<")
    except Exception as e:
        print(f"Предупреждение: Не удалось загрузить YOLO: {e}")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Ошибка: файл index.html не найден.</h1>"

def process_image_with_yolo(img):
    """
    Сегментация детали с помощью нейросети.
    Возвращает маску чистого физического тела детали без размерных стрелок и надписей.
    """
    h, w, _ = img.shape
    
    # Прогон через нейросеть
    results = yolo_model(img, conf=0.15, verbose=False)
    
    detail_mask = None
    
    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            
            # Находим самую крупную маску (тело детали)
            max_area = 0
            best_mask = None
            for mask in masks:
                mask_resized = cv2.resize(mask, (w, h))
                area = np.sum(mask_resized > 0.5)
                if area > max_area:
                    max_area = area
                    best_mask = mask_resized
            
            if best_mask is not None:
                detail_mask = (best_mask > 0.5).astype(np.uint8) * 255

    return detail_mask

def process_image_fallback(img):
    """
    Запасной фильтр по толщине линий ГОСТ (если нейросеть не смогла выделить силуэт)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)

    # Морфологическое удаление тонких выносных линий и стрелок
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thick_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Выделение главного элемента детали
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thick_lines)
    clean_mask = np.zeros_like(thick_lines)
    
    h, w = gray.shape
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        if area > (w * h * 0.003) and width > (w * 0.05):
            clean_mask[labels == i] = 255

    return clean_mask

def extract_lathe_profile(clean_mask):
    """Извлечение токарного радиусного профиля из бинарной маски"""
    h, w = clean_mask.shape

    # Вычисление оси симметрии деталей (центр масс)
    M = cv2.moments(clean_mask)
    if M["m00"] != 0:
        center_y = int(M["m01"] / M["m00"])
    else:
        center_y = h // 2

    x_indices = np.where(clean_mask > 0)[1]
    if len(x_indices) == 0:
        raise ValueError("Не удалось распознать силуэт детали на чертеже.")

    min_x, max_x = np.min(x_indices), np.max(x_indices)

    raw_x = []
    raw_r = []

    # Лучевое сканирование профиля детали
    step = max(1, (max_x - min_x) // 250)
    for x in range(min_x, max_x, step):
        col = clean_mask[0:center_y, x]
        pixels = np.where(col > 0)[0]
        
        if len(pixels) > 0:
            top_y = pixels[0]
            radius = center_y - top_y
            raw_x.append(float(x - min_x))
            raw_r.append(float(radius))

    # Скользящая медианная аппроксимация для получения идеальных цилиндров и конусов
    window = 9
    half = window // 2
    padded = np.pad(raw_r, (half, half), mode='edge')
    smoothed_r = [float(np.median(padded[i : i + window])) for i in range(len(raw_r))]

    profile_points = [{"x": x, "y": r} for x, r in zip(raw_x, smoothed_r)]

    return profile_points, center_y

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
            layer_name = entity.dxf.layer.upper()
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
        raise HTTPException(status_code=400, detail="Формат не поддерживается. Загрузите DXF, PNG или JPG.")

    contents = await file.read()

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Ошибка чтения изображения.")

            mask = None
            engine_used = "OpenCV Engineering Filter"

            # 1. Сначала запускаем нейросеть YOLOv8
            if yolo_model is not None:
                try:
                    mask = process_image_with_yolo(img)
                    if mask is not None:
                        engine_used = "AI YOLOv8 Neural Network"
                except Exception as ex:
                    print(f"Ошибка работы YOLO: {ex}")

            # 2. Если нейросеть не вернула маску, применяем классический фильтр ГОСТ
            if mask is None:
                mask = process_image_fallback(img)

            # 3. Извлекаем токарный профиль
            profile, axis_y = extract_lathe_profile(mask)

            return {
                "filename": file.filename,
                "status": "success",
                "engine": engine_used,
                "contours_count": 1,
                "entity_breakdown": {"POINTS_COUNT": len(profile), "AXIS_Y": axis_y},
                "contours": [profile]
            }

        else:
            contours, breakdown = process_dxf_file(contents)
            return {
                "filename": file.filename,
                "status": "success",
                "engine": "DXF Vector Parser",
                "contours_count": len(contours),
                "entity_breakdown": breakdown,
                "contours": contours
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
