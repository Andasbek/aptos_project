import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


BACKEND_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_ROOT / ".env")


SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "ru", "kk")
DEFAULT_LANGUAGE = "en"


_LANGUAGE_INSTRUCTIONS = {
    "en": "Always respond in English.",
    "ru": "Всегда отвечай на русском языке.",
    "kk": "Барлық жауаптарыңды қазақ тілінде беріңіз.",
}

_BASE_SYSTEM_PROMPT = (
    "You are a medical assistant helping a patient understand the result of a "
    "diabetic retinopathy screening produced by a deep learning classifier "
    "trained on the APTOS 2019 dataset. The model classifies retinal fundus "
    "images into five stages: No DR, Mild, Moderate, Severe, Proliferative DR. "
    "Always answer clearly and empathetically. Explain what the predicted stage "
    "means, what risk factors are typical, and what next steps are reasonable. "
    "Always remind the user that this is a research prototype and not a medical "
    "diagnosis, and recommend consulting an ophthalmologist for confirmation. "
    "Keep answers concise and structured."
)


_EXPLAIN_INSTRUCTIONS = {
    "en": (
        "Generate a structured patient-facing explanation for the prediction "
        "below.\n\n{prediction}\n\nCover: 1) what this stage means, 2) how "
        "confident the model is and how to interpret confidence, 3) typical "
        "recommended next steps, 4) general lifestyle and follow-up advice. "
        "End with a short disclaimer."
    ),
    "ru": (
        "Сформируй структурированное объяснение для пациента по результату "
        "ниже.\n\n{prediction}\n\nРаскрой: 1) что означает эта стадия, "
        "2) насколько модель уверена и как интерпретировать уверенность, "
        "3) типичные рекомендованные следующие шаги, 4) общие советы по "
        "образу жизни и наблюдению. Заверши коротким дисклеймером."
    ),
    "kk": (
        "Төмендегі болжам бойынша науқасқа арналған құрылымдық түсіндірме "
        "құрастыр.\n\n{prediction}\n\nҚамту керек: 1) бұл саты нені "
        "білдіреді, 2) модельдің сенімділігі қандай және оны қалай "
        "түсіндіру керек, 3) ұсынылатын келесі қадамдар, 4) өмір салты мен "
        "бақылау бойынша жалпы кеңестер. Қысқа ескертумен аяқта."
    ),
}


class LLMConfigurationError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set. Create backend/.env from .env.example "
            "and provide a valid key."
        )
    return OpenAI(api_key=api_key)


def _get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.5")


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def supported_languages() -> list[str]:
    return list(SUPPORTED_LANGUAGES)


def _normalize_language(language: str | None) -> str:
    if language and language in SUPPORTED_LANGUAGES:
        return language
    return DEFAULT_LANGUAGE


def _system_prompt(language: str) -> str:
    return f"{_BASE_SYSTEM_PROMPT}\n\n{_LANGUAGE_INSTRUCTIONS[language]}"


def _format_prediction(prediction: dict) -> str:
    probabilities = prediction.get("probabilities", {})
    probability_lines = "\n".join(
        f"- {class_name}: {value * 100:.1f}%"
        for class_name, value in probabilities.items()
    )
    confidence = prediction.get("confidence", 0.0)
    classifier = prediction.get("model_name", "unknown")
    return (
        f"Classifier: {classifier}\n"
        f"Predicted stage: {prediction.get('class_name')} "
        f"(class index {prediction.get('predicted_class')})\n"
        f"Model confidence: {confidence * 100:.1f}%\n"
        f"Per-class probabilities:\n{probability_lines}"
    )


def _get_temperature() -> float | None:
    raw = os.getenv("OPENAI_TEMPERATURE")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _chat_completion(messages: Iterable[dict]) -> str:
    client = _get_client()
    kwargs: dict = {
        "model": _get_model(),
        "messages": list(messages),
    }
    temperature = _get_temperature()
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        response = client.chat.completions.create(**kwargs)
    except OpenAIError as exc:
        message = str(exc)
        if "temperature" in message and "temperature" in kwargs:
            kwargs.pop("temperature", None)
            try:
                response = client.chat.completions.create(**kwargs)
            except OpenAIError as retry_exc:
                raise LLMRequestError(
                    f"OpenAI request failed: {retry_exc}"
                ) from retry_exc
        else:
            raise LLMRequestError(f"OpenAI request failed: {exc}") from exc

    choice = response.choices[0]
    content = choice.message.content or ""
    return content.strip()


def generate_explanation(prediction: dict, language: str = DEFAULT_LANGUAGE) -> str:
    lang = _normalize_language(language)
    user_prompt = _EXPLAIN_INSTRUCTIONS[lang].format(
        prediction=_format_prediction(prediction)
    )
    return _chat_completion(
        [
            {"role": "system", "content": _system_prompt(lang)},
            {"role": "user", "content": user_prompt},
        ]
    )


def chat_about_prediction(
    user_messages: list[dict],
    prediction: dict | None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    lang = _normalize_language(language)
    messages: list[dict] = [{"role": "system", "content": _system_prompt(lang)}]
    if prediction is not None:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Current model prediction context:\n"
                    f"{_format_prediction(prediction)}"
                ),
            }
        )
    messages.extend(user_messages)
    return _chat_completion(messages)
