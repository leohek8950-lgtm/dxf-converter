def analyze_drawing_with_gemini(img_bytes: bytes):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables!")

    # Конвертируем изображение в Base64
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
    Ты — инженер-конструктор и эксперт по распознаванию чертежей токарных деталей.
    Проанализируй чертеж детали вращения:
    1. ПОЛНОСТЬЮ ИГНОРИРУЙ все выносные линии, размерные стрелки, допуски, тексты, рамку и штамп чертежа.
    2. Определи центральную ось вращения детали.
    3. Найди верхний контур металлической детали (радиусный профиль) от левого до правого края.
    4. Сформируй координаты ступеней/переходов детали слева направо.
    
    Верни ТОЛЬКО чистый JSON-массив без markdown-тегов и текста в формате:
    [
      {"x": 0, "y": 15},
      {"x": 20, "y": 15},
      {"x": 20, "y": 25},
      {"x": 60, "y": 25}
    ]
    где 'x' — координата по длине (растет от 0), а 'y' — радиус детали относительно оси.
    Передай от 20 до 60 ключевых точек ступеней контура.
    """

    # Актуальные эндпоинты v1 и v1beta для современной линейки Gemini
    endpoints = [
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    ]

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
        }]
    }

    response_text = None
    last_error = None

    for url in endpoints:
        try:
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code == 200:
                res_data = res.json()
                # Извлекаем текст ответа
                response_text = res_data['candidates'][0]['content']['parts'][0]['text']
                break
            else:
                last_error = f"HTTP {res.status_code}: {res.text}"
        except Exception as e:
            last_error = str(e)

    if not response_text:
        raise ValueError(f"Ошибка вызова Gemini REST API: {last_error}")

    # Очистка ответа от ```json ... ```
    clean_json = re.sub(r'```json\s*|\s*```', '', response_text).strip()

    try:
        profile = json.loads(clean_json)
        return profile
    except Exception:
        raise ValueError("ИИ вернул ответ в неверном формате JSON.")
