from __future__ import annotations

import json
import os
import re
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()


class VacancyLLMAnalyzer:
    # Класс для анализа вакансии 

    def __init__(
        self,
        model: str | None = None,
        prompt_path: str = "SYSTEM_PROMPT.txt",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        

        key = api_key or os.getenv("LLM_API_KEY")

        if not key:
            raise ValueError("Не найден API-ключ")

        # Получаем адрес API.
        api_base = base_url or os.getenv("LLM_BASE_URL")

        if not api_base:
            raise ValueError("Не найден адрес API")

        self.model = model or os.getenv("LLM_MODEL")
        if not self.model:
            raise ValueError("Не указана модель")

        self.client = OpenAI(api_key=key, base_url=api_base)

        # Загружаем системный промпт
        with open(prompt_path, "r", encoding="utf-8") as file:
            self.system_prompt = file.read().strip()

        if not self.system_prompt:
            raise ValueError( f"Файл с промптом '{prompt_path}' пуст.")

    def analyze(
        self,
        vacancy_text: str,
        scam_probability: float | None = None,
    ) -> dict[str, Any]:
        # Анализирует вакансию

        vacancy_text = vacancy_text.strip()

        if not vacancy_text:
            raise ValueError("Текст вакансии пуст")

        # Формируем блок с результатом CatBoost
        if scam_probability is not None:
            if not 0 <= scam_probability <= 1:
                raise ValueError("scam_probability должна быть числом от 0 до 1")

            ml_block = (f"Вероятность скама: {scam_probability * 100:.1f}%")
        else:
            ml_block = (
                "Результат модели не передан"
                "Оцени вакансию только по её тексту"
                )

        # Формируем сообщение с конкретной вакансией.
        user_prompt = f"""
Проанализируй следующую вакансию.

РЕЗУЛЬТАТ МОДЕЛИ:
{ml_block}

ТЕКСТ ВАКАНСИИ:
{vacancy_text}
""".strip()

        # Отправляем запрос в LLM.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        # Получаем текст ответа.
        answer = response.choices[0].message.content

        if not answer:
            raise ValueError("LLM вернула пустой ответ.")

        return self._parse_json(answer)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        # Превращает ответ LLM из JSON-строки в Python-словарь

        cleaned = text.strip()

        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM вернула ответ не в формате JSON.\n\n"
                f"Ответ модели:\n{text}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError("Ожидался JSON-объект.")

        return result


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent
    PROMPT_PATH = BASE_DIR / "SYSTEM_PROMT.txt"

    analyzer = VacancyLLMAnalyzer(prompt_path=PROMPT_PATH)

    # Тестовая вакансия
    vacancy = """
    Младший специалист по сопровождению клиентских сервисов.
    В связи с масштабированием финтех-направления открыта позиция в распределенной команде. 
    Обязанности: первичная валидация обращений пользователей корпоративного портала, маршрутизация инцидентов в системе Service Desk, подготовка еженедельных отчетов по SLA. 
    Требования: уверенный пользователь ПК, базовое понимание регламентов информационной безопасности, грамотная устная и письменная речь, готовность пройти регламентированный внутренний комплаенс-аудит. 
    Мы предлагаем: полностью удаленный формат работы, гибкий график 5/2 (с плавающим началом дня), предоставление корпоративной техники после завершения адаптационного периода, оплачиваемый отпуск.
    """

    result = analyzer.analyze(
        vacancy_text=vacancy,
        scam_probability=None,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))