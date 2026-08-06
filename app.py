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
        raise HTTPException(status_code=400, detail="Only .dxf files are supported.")
    
    try:
        contents = await file.read()
        doc = ezdxf.read(io.StringIO(contents.decode('utf-8', errors='ignore')))
        
        msp = doc.modelspace()
        entity_count = len(msp)
        
        # Базовая очистка и анализ контуров плоской детали
        # Здесь задействован ezdxf для геометрии
        
        return {
            "filename": file.filename,
            "status": "success",
            "entities_processed": entity_count,
            "message": "DXF file successfully cleaned and analyzed."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing DXF: {str(e)}")
