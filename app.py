from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import ezdxf
import ezdxf.recover
import ezdxf.path
import io

app = FastAPI(title="AI CAD Converter & 3D Visualizer")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found.</h1>"

def extract_primitive_entities(entities):
    """Рекурсивно разворачивает блоки (INSERT) с сохранением их координат и масштаба"""
    for entity in entities:
        if entity.dxftype() == 'INSERT':
            try:
                yield from extract_primitive_entities(entity.virtual_entities())
            except Exception:
                pass
        else:
            yield entity

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

        # Рекурсивная обработка всех элементов и блоков
        for entity in extract_primitive_entities(msp):
            e_type = entity.dxftype()
            entity_counts[e_type] = entity_counts.get(e_type, 0) + 1

            try:
                # ezdxf.path автоматически рассчитывает bulge полилиний, дуги, окружности и сплайны
                path = ezdxf.path.make_path(entity)
                vertices = list(ezdxf.path.path_to_vertices(path, distance=0.05))
                
                for i in range(len(vertices) - 1):
                    geometry_segments.append({
                        "type": "line",
                        "x1": round(vertices[i].x, 4),
                        "y1": round(vertices[i].y, 4),
                        "x2": round(vertices[i+1].x, 4),
                        "y2": round(vertices[i+1].y, 4)
                    })
            except Exception:
                # Запасной вариант для простых линий LINE
                if e_type == 'LINE':
                    s, e = entity.dxf.start, entity.dxf.end
                    geometry_segments.append({
                        "type": "line",
                        "x1": round(s.x, 4), "y1": round(s.y, 4),
                        "x2": round(e.x, 4), "y2": round(e.y, 4)
                    })

        return {
            "filename": file.filename,
            "status": "success",
            "entities_processed": len(geometry_segments),
            "entity_breakdown": entity_counts,
            "segments": geometry_segments,
            "message": "Чертеж проанализирован с точным построением радиусов и скруглений!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки DXF: {str(e)}")
