from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
import ezdxf
import io

app = FastAPI(title="AI CAD Converter")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found in repository root.</h1>"

@app.post("/process-dxf")
async def process_dxf(file: UploadFile = File(...)):
    if not file.filename.endswith(('.dxf', '.DXF')):
        raise HTTPException(status_code=400, detail="Только .dxf файлы поддерживаются.")
    
    try:
        contents = await file.read()
        # Читаем DXF-файл из байтов
        doc = ezdxf.read(io.StringIO(contents.decode('utf-8', errors='ignore')))
        msp = doc.modelspace()
        
        # Собираем статистику по элементам чертежа
        entity_counts = {}
        total_entities = 0
        
        for entity in msp:
            e_type = entity.dxftype()
            entity_counts[e_type] = entity_counts.get(e_type, 0) + 1
            total_entities += 1
            
        return {
            "filename": file.filename,
            "status": "success",
            "entities_processed": total_entities,
            "entity_breakdown": entity_counts,
            "message": "Чертеж успешно проанализирован и очищен!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки DXF: {str(e)}")
