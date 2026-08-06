from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import ezdxf
import ezdxf.recover
import ezdxf.path
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
        return "<h1>Error: index.html not found.</h1>"

def extract_primitive_entities(entities):
    for entity in entities:
        if entity.dxftype() == 'INSERT':
            try:
                yield from extract_primitive_entities(entity.virtual_entities())
            except Exception:
                pass
        else:
            yield entity

def process_dxf_entity(entity, contours_list):
    e_type = entity.dxftype()

    if e_type == 'LINE':
        s, e = entity.dxf.start, entity.dxf.end
        line_style = getattr(entity.dxf, 'linetype', '').upper()
        if 'CENTER' in line_style or 'DASHDOT' in line_style:
            return
        contours_list.append([
            {"x": round(s.x, 2), "y": round(s.y, 2)},
            {"x": round(e.x, 2), "y": round(e.y, 2)}
        ])

    elif e_type in ('CIRCLE', 'ARC'):
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        if e_type == 'CIRCLE':
            start_angle, end_angle = 0.0, 2 * math.pi
        else:
            start_angle = math.radians(entity.dxf.start_angle)
            end_angle = math.radians(entity.dxf.end_angle)
            if end_angle <= start_angle:
                end_angle += 2 * math.pi

        steps = max(16, int(math.degrees(end_angle - start_angle) / 5))
        pts = []
        for i in range(steps + 1):
            a = start_angle + i * ((end_angle - start_angle) / steps)
            pts.append({"x": round(cx + r * math.cos(a), 2), "y": round(cy + r * math.sin(a), 2)})
        contours_list.append(pts)

def process_image_file(contents: bytes):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    h, w = img.shape[:2]
    min_area = (w * h) * 0.001  # Фильтр: игнорируем элементы меньше 0.1% от площади чертежа (текст, стрелки)
    
    formatted_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue  # Пропускаем размерные цифры и мелкие стрелки

        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        if len(approx) < 3:
            continue

        poly_pts = []
        for pt in approx:
            poly_pts.append({
                "x": float(pt[0][0]),
                "y": float(h - pt[0][1])  # Инвертируем Y для координатной сетки
            })
        formatted_contours.append(poly_pts)

    return formatted_contours

@app.post("/process-dxf")
async def process_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not filename.endswith(('.dxf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Поддерживаются файлы .dxf, .png, .jpg, .jpeg")

    contents = await file.read()
    contours_list = []

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            contours_list = process_image_file(contents)
            entity_counts = {"CLEAN_CONTOURS": len(contours_list)}
        else:
            doc, auditor = ezdxf.recover.read(io.BytesIO(contents))
            msp = doc.modelspace()
            entity_counts = {}
            for entity in extract_primitive_entities(msp):
                e_type = entity.dxftype()
                entity_counts[e_type] = entity_counts.get(e_type, 0) + 1
                process_dxf_entity(entity, contours_list)

        return {
            "filename": file.filename,
            "status": "success",
            "contours_count": len(contours_list),
            "entity_breakdown": entity_counts,
            "contours": contours_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
