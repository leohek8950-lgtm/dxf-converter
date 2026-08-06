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

def process_dxf_entity(entity, geometry_segments):
    e_type = entity.dxftype()

    if e_type == 'LINE':
        s, e = entity.dxf.start, entity.dxf.end
        line_style = getattr(entity.dxf, 'linetype', '').upper()
        if 'CENTER' in line_style or 'DASHDOT' in line_style:
            return
        geometry_segments.append({
            "type": "line",
            "x1": round(s.x, 4), "y1": round(s.y, 4),
            "x2": round(e.x, 4), "y2": round(e.y, 4)
        })

    elif e_type == 'CIRCLE':
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        steps = 64
        for i in range(steps):
            a1 = 2 * math.pi * i / steps
            a2 = 2 * math.pi * (i + 1) / steps
            geometry_segments.append({
                "type": "line",
                "x1": round(cx + r * math.cos(a1), 4),
                "y1": round(cy + r * math.sin(a1), 4),
                "x2": round(cx + r * math.cos(a2), 4),
                "y2": round(cy + r * math.sin(a2), 4)
            })

    elif e_type == 'ARC':
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        
        if end_angle <= start_angle:
            end_angle += 2 * math.pi
            
        steps = max(12, int(math.degrees(end_angle - start_angle) / 5))
        angle_step = (end_angle - start_angle) / steps
        
        for i in range(steps):
            a1 = start_angle + i * angle_step
            a2 = start_angle + (i + 1) * angle_step
            geometry_segments.append({
                "type": "line",
                "x1": round(cx + r * math.cos(a1), 4),
                "y1": round(cy + r * math.sin(a1), 4),
                "x2": round(cx + r * math.cos(a2), 4),
                "y2": round(cy + r * math.sin(a2), 4)
            })

    else:
        try:
            path = ezdxf.path.make_path(entity)
            vertices = list(ezdxf.path.path_to_vertices(path, distance=0.1))
            for i in range(len(vertices) - 1):
                geometry_segments.append({
                    "type": "line",
                    "x1": round(vertices[i].x, 4), "y1": round(vertices[i].y, 4),
                    "x2": round(vertices[i+1].x, 4), "y2": round(vertices[i+1].y, 4)
                })
        except Exception:
            pass

def process_image_file(contents: bytes):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось прочитать изображение.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    geometry_segments = []
    h, w = img.shape[:2]

    for cnt in contours:
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        n = len(approx)
        if n < 2:
            continue

        for i in range(n):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % n][0]
            geometry_segments.append({
                "type": "line",
                "x1": float(pt1[0]),
                "y1": float(h - pt1[1]),
                "x2": float(pt2[0]),
                "y2": float(h - pt2[1])
            })

    return geometry_segments

@app.post("/process-dxf")
async def process_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    valid_exts = ('.dxf', '.png', '.jpg', '.jpeg')
    if not filename.endswith(valid_exts):
        raise HTTPException(status_code=400, detail="Поддерживаются файлы .dxf, .png, .jpg, .jpeg")

    contents = await file.read()
    geometry_segments = []

    try:
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            geometry_segments = process_image_file(contents)
            entity_counts = {"IMAGE_CONTOURS": len(geometry_segments)}
        else:
            try:
                doc, auditor = ezdxf.recover.read(io.BytesIO(contents))
            except Exception:
                text_data = contents.decode('utf-8', errors='ignore')
                doc, auditor = ezdxf.recover.read(io.StringIO(text_data))

            msp = doc.modelspace()
            entity_counts = {}

            for entity in extract_primitive_entities(msp):
                e_type = entity.dxftype()
                entity_counts[e_type] = entity_counts.get(e_type, 0) + 1
                process_dxf_entity(entity, geometry_segments)

        return {
            "filename": file.filename,
            "status": "success",
            "entities_processed": len(geometry_segments),
            "entity_breakdown": entity_counts,
            "segments": geometry_segments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")
