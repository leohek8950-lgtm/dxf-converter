import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import ezdxf
from ezdxf.recover import readfile
ezdxf.path

app = FastAPI(title="AI CAD Converter API")

# Настройка CORS, чтобы фронтенд мог свободно общаться с бэкендом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/process-part/")
async def process_part(file: UploadFile = File(...), part_type: str = "flat_laser"):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        if part_type == "flat_laser":
            doc, auditor = readfile(file_path)
            msp = doc.modelspace()
            
            # Интеллектуальный анализ геометрии контуров
            paths = list(make_paths(msp))
            closed_contours_count = sum(1 for p in paths if p.is_closed)
            open_contours_count = sum(1 for p in paths if not p.is_closed)
            
            # Создаем очищенный файл для лазерной резки
            out_doc = ezdxf.new(doc.dxfversion)
            out_msp = out_doc.modelspace()
            
            allowed_types = {'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'POLYLINE'}
            copied_count = 0
            
            for entity in msp:
                if entity.dxftype() in allowed_types:
                    out_msp.add_entity(entity.copy())
                    copied_count += 1
                    
            output_filename = f"laser_{file.filename}"
            output_path = os.path.join(PROCESSED_DIR, output_filename)
            out_doc.saveas(output_path)
            
            return {
                "status": "success",
                "message": "Чертеж успешно проанализирован и очищен!",
                "analysis": {
                    "total_primitives": copied_count,
                    "closed_contours": closed_contours_count,
                    "open_contours": open_contours_count,
                    "errors_found": len(auditor.errors)
                },
                "download_url": f"/download/{output_filename}"
            }
        else:
            return JSONResponse(status_code=400, content={"error": "Этот тип обработки пока не реализован."})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения чертежа: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/dxf", filename=filename)
    raise HTTPException(status_code=404, detail="Файл не найден")
