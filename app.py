import os
import io
import json
import re
import base64
import asyncio
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cadquery as cq

app = FastAPI(title="Precision AI CAD Engine")

# Включаем CORS для корректной загрузки 3D файлов браузером
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Встроенный HTML-интерфейс с зафиксированной трехмерной визуализацией Three.js
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Precision AI CAD</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        body { background-color: #0f172a; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }
        #sidebar { width: 360px; background-color: #1e293b; padding: 20px; display: flex; flex-direction: column; gap: 15px; border-right: 1px solid #334155; z-index: 10; }
        h2 { color: #38bdf8; font-size: 1.2rem; display: flex; align-items: center; gap: 8px; }
        .upload-box { border: 2px dashed #475569; border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; transition: 0.3s; background: #0f172a; }
        .upload-box:hover { border-color: #38bdf8; background: #1e293b; }
        input[type="file"] { display: none; }
        button { background-color: #0284c7; color: white; border: none; padding: 12px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; }
        button:hover { background-color: #0369a1; }
        button:disabled { background-color: #475569; cursor: not-allowed; }
        .download-btn { background-color: #16a34a; }
        .download-btn:hover { background-color: #15803d; }
        #json-output { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; font-family: monospace; font-size: 0.75rem; color: #38bdf8; overflow-y: auto; flex-grow: 1; white-space: pre-wrap; }
        #viewport { flex-grow: 1; position: relative; background: #020617; }
        #status-msg { position: absolute; top: 20px; left: 20px; padding: 10px 18px; border-radius: 6px; font-weight: 500; font-size: 0.9rem; z-index: 20; display: none; }
        .status-success { background: #166534; color: #4ade80; border: 1px solid #22c55e; }
        .status-error { background: #991b1b; color: #fca5a5; border: 1px solid #ef4444; }
        .status-loading { background: #1e3a8a; color: #93c5fd; border: 1px solid #3b82f6; }
    </style>
</head>
<body>
    <div id="sidebar">
        <h2>⚙️ Precision AI CAD</h2>
        <div class="upload-box" onclick="document.getElementById('fileInput').click()">
            <span id="file-label">📄 Выбрать чертеж (.jpg, .png)</span>
            <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
        </div>
        <button id="processBtn" onclick="uploadDrawing()" disabled>⚡ Сгенерировать STEP модель</button>
        <button id="downloadBtn" class="download-btn" style="display:none;" onclick="downloadStep()">💾 Скачать файл .STEP</button>
        <div style="font-size:0.8rem; color:#94a3b8; font-weight:bold;">РАСПОЗНАННАЯ ГЕОМЕТРИЯ (JSON):</div>
        <div id="json-output">// Ожидание файла...</div>
    </div>
    <div id="viewport">
        <div id="status-msg"></div>
    </div>

    <script>
        let selectedFile = null;
        let stepDownloadUrl = "";
        let scene, camera, renderer, controls, currentMesh;

        function init3D() {
            const container = document.getElementById('viewport');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x020617);

            camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(100, 100, 100);

            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight1.position.set(1, 1, 1).normalize();
            scene.add(dirLight1);

            const dirLight2 = new THREE.DirectionalLight(0x38bdf8, 0.5);
            dirLight2.position.set(-1, -1, -1).normalize();
            scene.add(dirLight2);

            const gridHelper = new THREE.GridHelper(200, 20, 0x334155, 0x1e293b);
            scene.add(gridHelper);

            window.addEventListener('resize', onWindowResize);
            animate();
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        function onWindowResize() {
            const container = document.getElementById('viewport');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }

        function showStatus(text, type) {
            const msg = document.getElementById('status-msg');
            msg.className = 'status-' + type;
            msg.innerText = text;
            msg.style.display = 'block';
        }

        function handleFileSelect(e) {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                document.getElementById('file-label').innerText = 'Выбран файл:\n' + selectedFile.name;
                document.getElementById('processBtn').disabled = false;
            }
        }

        async function uploadDrawing() {
            if (!selectedFile) return;

            showStatus('⏳ Выполняется визуальный и размерный анализ чертежа...', 'loading');
            document.getElementById('processBtn').disabled = true;
            document.getElementById('downloadBtn').style.display = 'none';
            document.getElementById('json-output').innerText = '// Обработка детали...';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const res = await fetch('/api/analyze', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.status === 'success') {
                    showStatus('✅ 3D Модель успешно построена!', 'success');
                    document.getElementById('json-output').innerText = JSON.stringify(data.spec, null, 2);
                    stepDownloadUrl = data.step_url;
                    document.getElementById('downloadBtn').style.display = 'block';
                    loadSTL(data.stl_url);
                } else {
                    showStatus('❌ Ошибка: ' + (data.error || 'Неизвестный сбой'), 'error');
                    document.getElementById('json-output').innerText = '// Ошибка:\n' + JSON.stringify(data, null, 2);
                }
            } catch (err) {
                showStatus('❌ Ошибка соединения с сервером', 'error');
                document.getElementById('json-output').innerText = '// Ошибка сети:\n' + err.message;
            } finally {
                document.getElementById('processBtn').disabled = false;
            }
        }

        function loadSTL(url) {
            const loader = new THREE.STLLoader();
            loader.load(url, function (geometry) {
                if (currentMesh) scene.remove(currentMesh);

                geometry.computeVertexNormals();
                const material = new THREE.MeshStandardMaterial({
                    color: 0x94a3b8,
                    roughness: 0.3,
                    metalness: 0.6
                });
                currentMesh = new THREE.Mesh(geometry, material);

                geometry.center();
                geometry.computeBoundingBox();
                const box = geometry.boundingBox;
                const maxDim = Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z);
                
                camera.position.set(maxDim * 1.5, maxDim * 1.5, maxDim * 1.5);
                controls.target.set(0, 0, 0);

                scene.add(currentMesh);
            }, undefined, function (error) {
                showStatus('❌ Ошибка визуализации STL модели', 'error');
                console.error(error);
            });
        }

        function downloadStep() {
            if (stepDownloadUrl) {
                window.location.href = stepDownloadUrl;
            }
        }

        window.onload = init3D;
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    return HTML_TEMPLATE

def compress_image_bytes(img_bytes: bytes, max_dim=800) -> bytes:
    """Уменьшение картинки для отклика Gemini API в пределах 3-5 секунд"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buf, format='JPEG', quality=80)
        return buf.getvalue()
    except Exception:
        return img_bytes

def parse_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables!")

    api_key = re.sub(r'\[.*?\]|\(|\)', '', api_key).strip()
    small_bytes = compress_image_bytes(img_bytes)
    base64_image = base64.b64encode(small_bytes).decode("utf-8")

    prompt = """
    Ты — главный инженер-конструктор ЧПУ. Оцифруй чертеж детали в JSON.

    ИНСТРУКЦИЯ ПО ВЫЧИСЛЕНИЮ И СРАВНЕНИЮ РАЗМЕРОВ:
    1. ВИЗУАЛЬНЫЙ АУДИТ: Найди ВСЕ канавки, фаски, пазы и ступенчатые выточки на главном виде и разрезе A-A.
    2. ВЫЧИСЛЕНИЕ РАЗМЕРНЫХ ЦЕПЕЙ: Если длина элемента/паза не указана прямым числом, ВЫЧИСЛИ её: 
       L_паза = L_общее - Sum(L_остальных_известных_сегментов).
    3. Используй ТОЛЬКО номинальные размеры (без допусков ±).

    Верни ТОЛЬКО чистый JSON без markdown:
    {
      "part_name": "Деталь",
      "total_length": 45.0,
      "outer_profile": [
        {"type": "cylinder", "start_diameter": 18.0, "end_diameter": 18.0, "length": 5.0},
        {"type": "groove", "start_diameter": 15.0, "end_diameter": 15.0, "length": 1.5},
        {"type": "cylinder", "start_diameter": 24.0, "end_diameter": 24.0, "length": 5.0},
        {"type": "cylinder", "start_diameter": 22.0, "end_diameter": 22.0, "length": 12.5},
        {"type": "cylinder", "start_diameter": 35.0, "end_diameter": 35.0, "length": 10.0},
        {"type": "cylinder", "start_diameter": 36.0, "end_diameter": 36.0, "length": 11.0}
      ],
      "bores": [
        {"diameter": 30.0, "depth": 5.0, "from_side": "right"},
        {"diameter": 22.0, "depth": 10.0, "from_side": "right"},
        {"diameter": 20.0, "depth": 11.2, "from_side": "right"}
      ]
    }
    """

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    models_to_try = [
        ("v1beta", "gemini-1.5-flash"),
        ("v1beta", "gemini-2.0-flash")
    ]

    last_err = None
    for api_ver, model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/{api_ver}/models/{model_name}:generateContent"
        params = {"key": api_key}
        try:
            res = requests.post(url, params=params, json=payload, timeout=20)
            if res.status_code == 200:
                res_data = res.json()
                text_content = res_data['candidates'][0]['content']['parts'][0]['text']
                cleaned_text = re.sub(r'```json\s*|\s*```', '', text_content.strip())
                return json.loads(cleaned_text)
            else:
                last_err = f"[{model_name}] HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_err = f"[{model_name}] {str(e)}"

    raise ValueError(f"Ошибка обращения к ИИ: {last_err}")

def build_cad_model(spec: dict, filename_base: str):
    profile = spec.get("outer_profile", [])
    if not profile:
        raise ValueError("В ответе ИИ отсутствует внешний профиль детали.")

    current_z = 0.0
    result = None

    for seg in profile:
        d1 = float(seg.get("start_diameter", 10.0))
        d2 = float(seg.get("end_diameter", d1))
        l = float(seg.get("length", 0.0))

        if l <= 0.001:
            continue

        r1 = max(d1 / 2.0, 0.1)
        r2 = max(d2 / 2.0, 0.1)

        try:
            if abs(r1 - r2) < 0.001:
                solid = cq.Workplane("XY").workplane(offset=current_z).circle(r1).extrude(l)
            else:
                solid = (
                    cq.Workplane("XY")
                    .workplane(offset=current_z)
                    .circle(r1)
                    .workplane(offset=l)
                    .circle(r2)
                    .loft(combine=True)
                )

            if result is None:
                result = solid
            else:
                result = result.union(solid)
        except Exception as e:
            print(f"Пропущен элемент профиля ({seg}): {e}")
            continue

        current_z += l

    if result is None:
        raise ValueError("Не удалось построить геометрию из переданного JSON.")

    bores = spec.get("bores", [])
    bores_sorted = sorted(bores, key=lambda x: float(x.get("diameter", 0)), reverse=True)

    for bore in bores_sorted:
        bd = float(bore.get("diameter", 0))
        depth = float(bore.get("depth", 0))
        from_side = bore.get("from_side", "right")

        if bd > 0 and depth > 0:
            try:
                if from_side == "right":
                    result = result.faces(">Z").workplane().hole(bd, depth)
                else:
                    result = result.faces("<Z").workplane().hole(bd, depth)
            except Exception as e:
                print(f"Ошибка при расточке ({bore}): {e}")

    step_path = os.path.join(EXPORTS_DIR, f"{filename_base}.step")
    stl_path = os.path.join(EXPORTS_DIR, f"{filename_base}.stl")

    cq.exporters.export(result, step_path)
    cq.exporters.export(result, stl_path)

    return step_path, stl_path

@app.post("/api/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', os.path.splitext(file.filename)[0])
        if not clean_name:
            clean_name = "cad_model"
        filename_base = f"part_{clean_name}"

        spec = await asyncio.to_thread(parse_drawing_with_gemini, contents)
        step_file, stl_file = await asyncio.to_thread(build_cad_model, spec, filename_base)

        return JSONResponse(content={
            "status": "success",
            "spec": spec,
            "stl_url": f"/download/{filename_base}.stl",
            "step_url": f"/download/{filename_base}.step"
        })
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "error": str(e),
                "spec": {"error": str(e)}
            }
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(EXPORTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="application/octet-stream", 
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            filename=filename
        )
    return JSONResponse(status_code=404, content={"error": "Файл не найден на сервере."})
