from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import ezdxf
import ezdxf.recover
from ezdxf.path import path_to_vertices, make_path
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

def process_entity(entity, geometry_segments):
    """Рекурсивная и точная обработка любых типов сущностей DXF"""
    e_type = entity.dxftype()

    # 1. Линии (LINE)
    if e_type == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        geometry_segments.append({
            "type": "line",
            "x1": start.x, "y1": start.y,
            "x2": end.x, "y2": end.y
        })

    # 2. Полилинии (LWPOLYLINE / POLYLINE)
    elif e_type in ['LWPOLYLINE', 'POLYLINE']:
        try:
            path = make_path(entity)
            vertices = list(path_to_vertices(path, distance=0.1))
            for i in range(len(vertices) - 1):
                geometry_segments.append({
                    "type": "line",
                    "x1": vertices[i].x, "y1": vertices[i].y,
                    "x2": vertices[i+1].x, "y2": vertices[i+1].y
                })
        except Exception:
            # Резервный метод для полилиний
            points = list(entity.get_points(format='xy'))
            for i in range(len(points) - 1):
                geometry_segments.append({
                    "type": "line",
                    "x1": points[i][0], "y1": points[i][1],
                    "x2": points[i+1][0], "y2": points[i+1][1]
                })
            if getattr(entity, 'closed', False) and len(points) > 2:
                geometry_segments.append({
                    "type": "line",
                    "x1": points[-1][0], "y1": points[-1][1],
                    "x2": points[0][0], "y2": points[0][1]
                })

    # 3. Окружности (CIRCLE)
    elif e_type == 'CIRCLE':
        center = entity.dxf.center
        radius = entity.dxf.radius
        geometry_segments.append({
            "type": "circle",
            "cx": center.x, "cy": center.y,
            "r": radius
        })

    # 4. Дуги скругления (ARC) - Точный расчет по углам
    elif e_type == 'ARC':
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        r = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        
        if end_angle < start_angle:
            end_angle += 2 * math.pi
            
        steps = max(8, int(math.degrees(end_angle - start_angle) / 5))
        angle_step = (end_angle - start_angle) / steps
        
        for i in range(steps):
            a1 = start_angle + i * angle_step
            a2 = start_angle + (i + 1) * angle_step
            geometry_segments.append({
                "type": "line",
                "x1": cx + r * math.cos(a1), "y1": cy + r * math.sin(a1),
                "x2": cx + r * math.cos(a2), "y2": cy + r * math.sin(a2)
            })

    # 5. Сплайны и эллипсы (SPLINE, ELLIPSE)
    elif e_type in ['SPLINE', 'ELLIPSE']:
        try:
            path = make_path(entity)
            vertices = list(path_to_vertices(path, distance=0.1))
            for i in range(len(vertices) - 1):
                geometry_segments.append({
                    "type": "line",
                    "x1": vertices[i].x, "y1": vertices[i].y,
                    "x2": vertices[i+1].x, "y2": vertices[i+1].y
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

        # Сначала раскрываем все вложенные блоки (INSERT)
        for insert in list(msp.query('INSERT')):
            try:
                insert.explode()
            except Exception:
                pass

        # Перебираем всю полученную виртуальную геометрию
        for entity in msp:
            e_type = entity.dxftype()
            entity_counts[e_type] = entity_counts.get(e_type, 0) + 1
            try:
                process_entity(entity, geometry_segments)
            except Exception:
                continue

        return {
            "filename": file.filename,
            "status": "success",
            "entities_processed": len(geometry_segments),
            "entity_breakdown": entity_counts,
            "segments": geometry_segments,
            "message": "Чертеж успешно проанализирован и подготовлен для 3D!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки DXF: {str(e)}")
