from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import ezdxf
import ezdxf.recover
import ezdxf.path
import io
import math

app = FastAPI(title="AI CAD Converter & 3D Visualizer")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found.</h1>"

def extract_primitive_entities(entities):
    """Рекурсивный разбор блоков INSERT в базовые сущности"""
    for entity in entities:
        if entity.dxftype() == 'INSERT':
            try:
                yield from extract_primitive_entities(entity.virtual_entities())
            except Exception:
                pass
        else:
            yield entity

def process_entity(entity, geometry_segments):
    e_type = entity.dxftype()

    # 1. Прямые линии (LINE)
    if e_type == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        geometry_segments.append({
            "type": "line",
            "x1": round(start.x, 4), "y1": round(start.y, 4),
            "x2": round(end.x, 4), "y2": round(end.y, 4)
        })

    # 2. Окружности (CIRCLE) — явное построение 64 сегментами
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

    # 3. Дуги (ARC) — расчет по начальному и конечному углам
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

    # 4. Полилинии, сплайны и эллипсы (LWPOLYLINE, POLYLINE, SPLINE, ELLIPSE)
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
            if e_type in ['LWPOLYLINE', 'POLYLINE']:
                try:
                    pts = list(entity.get_points(format='xy'))
                    for i in range(len(pts) - 1):
                        geometry_segments.append({
                            "type": "line",
                            "x1": round(pts[i][0], 4), "y1": round(pts[i][1], 4),
                            "x2": round(pts[i+1][0], 4), "y2": round(pts[i+1][1], 4)
                        })
                    if getattr(entity, 'closed', False) and len(pts) > 2:
                        geometry_segments.append({
                            "type": "line",
                            "x1": round(pts[-1][0], 4), "y1": round(pts[-1][1], 4),
                            "x2": round(pts[0][0], 4), "y2": round(pts[0][1], 4)
                        })
                except Exception:
                    pass

@app.post("/process-dxf")
async def process_dxf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.dxf'):
        raise HTTPException(status_code=400, detail="Поддерживаются только .dxf файлы.")
    
    try:
        contents = await file.read()
        
        try:
            doc, auditor = ezdxf.recover.read(io.BytesIO(contents))
        except Exception:
            text_data = None
            for enc in ['utf-8', 'cp1251', 'latin-1']:
                try:
                    text_data = contents.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text_data is None:
                text_data = contents.decode('utf-8', errors='ignore')
            doc, auditor = ezdxf.recover.read(io.StringIO(text_data))

        msp = doc.modelspace()
        geometry_segments = []
        entity_counts = {}

        for entity in extract_primitive_entities(msp):
            e_type = entity.dxftype()
            entity_counts[e_type] = entity_counts.get(e_type, 0) + 1
            process_entity(entity, geometry_segments)

        return {
            "filename": file.filename,
            "status": "success",
            "entities_processed": len(geometry_segments),
            "entity_breakdown": entity_counts,
            "segments": geometry_segments,
            "message": "Чертеж успешно проанализирован!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки DXF: {str(e)}")
