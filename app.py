def analyze_drawing_with_gemini(img_bytes: bytes):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY не установлен в Environment Variables на Render!")

    try:
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        raise ValueError("Ошибка чтения файла изображения.")

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

    # Список имен моделей по приоритету
    models_to_try = [
        'gemini-1.5-flash-latest',
        'gemini-2.5-flash',
        'gemini-1.5-pro',
        'models/gemini-1.5-flash'
    ]

    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([image, prompt])
            if response and response.text:
                break
        except Exception as e:
            last_error = e
            continue

    if not response or not response.text:
        raise ValueError(f"Ошибка запроса к Gemini API: {str(last_error)}")

    text_content = response.text.strip()

    # Очистка JSON от возможных знаков разметки ```json ... ```
    clean_json = re.sub(r'```json\s*|\s*```', '', text_content).strip()

    try:
        profile = json.loads(clean_json)
        return profile
    except Exception:
        raise ValueError("ИИ вернул ответ в неверном формате JSON. Попробуйте еще раз.")
