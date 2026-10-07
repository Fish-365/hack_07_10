import hashlib
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Импортируем классы из ваших файлов
from inference import Classifiter
from llmanalyze import VacancyLLMAnalyzer

app = FastAPI()

# Хранилище: IP -> { "quick": hash, "detailed": hash }
user_last_text = {}

class VacancyReq(BaseModel):
    vacancy_text: str

# Инициализация моделей при запуске
try:
    classifier = Classifiter(
        path_catboost="models/catboost_scam_detector.cbm", 
        path_fasttext_compressed="models/fasttext_compressed.bin", 
        path_pipeline_meta="models/pipeline_meta.pkl"
    )
    analyzer = VacancyLLMAnalyzer(prompt_path="SYSTEM_PROMPT.txt")
    print("Модели успешно загружены.")
except Exception as e:
    print(f"ВНИМАНИЕ: Ошибка загрузки моделей! {e}")
    classifier = None
    analyzer = None

def check_cooldown(ip: str, mode: str, text: str):
    """Функция проверки кулдауна с раздельным учетом режимов"""
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if ip not in user_last_text:
        user_last_text[ip] = {}
        
    if user_last_text[ip].get(mode) == text_hash:
        raise HTTPException(
            status_code=429, 
            detail="Вы уже сделали этот анализ для данного текста. Измените его для нового запроса."
        )
    user_last_text[ip][mode] = text_hash

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/analyze")
async def analyze_vacancy_quick(req: VacancyReq, request: Request):
    """БЫСТРЫЙ АНАЛИЗ (Только CatBoost) - работает мгновенно"""
    if not classifier:
        raise HTTPException(status_code=500, detail="Модель CatBoost не загружена.")
    
    # Проверка кулдауна только для быстрого анализа
    check_cooldown(request.client.host, "quick", req.vacancy_text)
    
    try:
        # 1. CatBoost (отработает за миллисекунды)
        vectors = classifier.vectorize(req.vacancy_text)
        cb_prob = classifier.predict(vectors=vectors)
        
        # 2. Быстрая эвристика без LLM, чтобы не было задержек
        level = "low"
        if cb_prob >= 0.7: level = "high"
        elif cb_prob >= 0.35: level = "medium"

        lower_text = req.vacancy_text.lower()
        red_flags = []
        if any(w in lower_text for w in ["депозит", "взнос", "оплатить", "вложение", "перевести"]):
            red_flags.append("Возможна просьба заплатить деньги до начала работы.")
        if any(w in lower_text for w in ["cvv", "смс", "карты", "карт"]):
            red_flags.append("Упоминаются банковские карты или SMS-коды.")

        return {
            "risk_level": level,
            "scam_probability": cb_prob,
            "summary": "Это результат быстрого машинного анализа. Для глубокого поиска скрытых уловок нейросетью нажмите «Подробный разбор».",
            "red_flags": red_flags,
            "missing_information": ["Быстрый анализ не выявляет скрытые смыслы. Используйте подробный разбор."],
            "questions_to_employer": ["Уточните ИНН организации и формат договора."],
            "safe_actions": ["Не переводите деньги.", "Проверьте отзывы о работодателе."],
            "salary_comment": "Если зарплата кажется неадекватно высокой за простые задачи — будьте осторожны.",
            "ml_signal_comment": f"CatBoost оценил вероятность мошенничества в {cb_prob*100:.1f}%."
        }
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detailed-analysis")
async def analyze_vacancy_detailed(req: VacancyReq, request: Request):
    """ПОДРОБНЫЙ АНАЛИЗ (CatBoost + LLM) - занимает несколько секунд"""
    if not classifier or not analyzer:
        raise HTTPException(status_code=500, detail="Модели не загружены.")
        
    # Проверка кулдауна только для подробного анализа
    check_cooldown(request.client.host, "detailed", req.vacancy_text)
    
    try:
        vectors = classifier.vectorize(req.vacancy_text)
        cb_prob = classifier.predict(vectors=vectors)
        
        result = analyzer.analyze(vacancy_text=req.vacancy_text, scam_probability=cb_prob)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)