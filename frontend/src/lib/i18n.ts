export type Language = "en" | "ru" | "kk";

export const SUPPORTED_LANGUAGES: Language[] = ["en", "ru", "kk"];

export const LANGUAGE_LABELS: Record<Language, string> = {
  en: "English",
  ru: "Русский",
  kk: "Қазақша",
};

export type TranslationKey = keyof typeof translations.en;

export const translations = {
  en: {
    "header.eyebrow": "APTOS 2019 Blindness Detection",
    "header.title": "Diabetic Retinopathy Classifier",
    "header.subtitle":
      "Upload a retinal fundus image and analyze it with a trained PyTorch model. The system returns the predicted severity class, confidence, and class probabilities for all five APTOS labels.",
    "header.languagePicker": "Language",
    "disclaimer":
      "This system is a research prototype and does not replace consultation with an ophthalmologist.",

    "uploader.eyebrow": "Upload",
    "uploader.title": "Fundus Image",
    "uploader.reset": "Reset",
    "uploader.resetTitle": "Reset selected image",
    "uploader.model": "Model",
    "uploader.placeholder.title": "Choose an image",
    "uploader.placeholder.subtitle":
      "PNG, JPG, or JPEG fundus image for APTOS class prediction.",
    "uploader.errors.noFile": "Select a retinal fundus image first.",
    "uploader.errors.predictionFailed": "Prediction failed.",
    "uploader.modelMissingSuffix": "(missing)",
    "uploader.analyze": "Analyze Image",

    "modelInfo.eyebrow": "Backend",
    "modelInfo.title": "Model Info",
    "modelInfo.api": "API",
    "modelInfo.device": "Device",
    "modelInfo.llm": "LLM",
    "modelInfo.llm.ready": "Ready",
    "modelInfo.llm.noKey": "No API key",
    "modelInfo.checkpoint.ready": "Ready",
    "modelInfo.checkpoint.missing": "Missing",
    "modelInfo.availableModels": "Available models",
    "modelInfo.default": "default",

    "result.eyebrow": "Prediction",
    "result.classLabel": "Class",
    "result.confidence": "Confidence",

    "explanation.eyebrow": "AI Assistant",
    "explanation.title": "Explanation",
    "explanation.loading": "Generating explanation...",

    "chat.eyebrow": "AI Assistant",
    "chat.title": "Ask a question",
    "chat.subtitle":
      "Ask about the result, recommendations, or what the stage means.",
    "chat.empty": 'No messages yet. Try: "What should I do next?"',
    "chat.thinking": "Thinking...",
    "chat.placeholder": "Type your question...",
    "chat.send": "Send",
    "chat.errors.failed": "Chat request failed.",
  },
  ru: {
    "header.eyebrow": "APTOS 2019 Blindness Detection",
    "header.title": "Классификатор диабетической ретинопатии",
    "header.subtitle":
      "Загрузите снимок глазного дна и проанализируйте его обученной моделью PyTorch. Система вернёт предсказанную стадию, уверенность и вероятности всех пяти классов APTOS.",
    "header.languagePicker": "Язык",
    "disclaimer":
      "Это исследовательский прототип, он не заменяет консультацию офтальмолога.",

    "uploader.eyebrow": "Загрузка",
    "uploader.title": "Снимок глазного дна",
    "uploader.reset": "Сбросить",
    "uploader.resetTitle": "Сбросить выбранное изображение",
    "uploader.model": "Модель",
    "uploader.placeholder.title": "Выберите изображение",
    "uploader.placeholder.subtitle":
      "PNG, JPG или JPEG снимок глазного дна для классификации по APTOS.",
    "uploader.errors.noFile": "Сначала выберите снимок глазного дна.",
    "uploader.errors.predictionFailed": "Предсказание не выполнено.",
    "uploader.modelMissingSuffix": "(нет файла)",
    "uploader.analyze": "Проанализировать",

    "modelInfo.eyebrow": "Backend",
    "modelInfo.title": "О модели",
    "modelInfo.api": "API",
    "modelInfo.device": "Устройство",
    "modelInfo.llm": "LLM",
    "modelInfo.llm.ready": "Готов",
    "modelInfo.llm.noKey": "Нет API-ключа",
    "modelInfo.checkpoint.ready": "Готов",
    "modelInfo.checkpoint.missing": "Отсутствует",
    "modelInfo.availableModels": "Доступные модели",
    "modelInfo.default": "по умолчанию",

    "result.eyebrow": "Предсказание",
    "result.classLabel": "Класс",
    "result.confidence": "Уверенность",

    "explanation.eyebrow": "AI-ассистент",
    "explanation.title": "Объяснение",
    "explanation.loading": "Формирую объяснение...",

    "chat.eyebrow": "AI-ассистент",
    "chat.title": "Задать вопрос",
    "chat.subtitle":
      "Спросите о результате, рекомендациях или что означает эта стадия.",
    "chat.empty": "Сообщений пока нет. Попробуйте: «Что мне делать дальше?»",
    "chat.thinking": "Думаю...",
    "chat.placeholder": "Введите вопрос...",
    "chat.send": "Отправить",
    "chat.errors.failed": "Не удалось получить ответ.",
  },
  kk: {
    "header.eyebrow": "APTOS 2019 Blindness Detection",
    "header.title": "Диабеттік ретинопатия классификаторы",
    "header.subtitle":
      "Көз түбінің суретін жүктеп, оқытылған PyTorch моделімен талдаңыз. Жүйе болжанған сатыны, сенімділікті және APTOS-тің бес класының ықтималдықтарын қайтарады.",
    "header.languagePicker": "Тіл",
    "disclaimer":
      "Бұл — зерттеу прототипі, офтальмолог консультациясын алмастырмайды.",

    "uploader.eyebrow": "Жүктеу",
    "uploader.title": "Көз түбі суреті",
    "uploader.reset": "Тазалау",
    "uploader.resetTitle": "Таңдалған суретті тазалау",
    "uploader.model": "Модель",
    "uploader.placeholder.title": "Сурет таңдаңыз",
    "uploader.placeholder.subtitle":
      "APTOS классификациясы үшін PNG, JPG немесе JPEG көз түбі суреті.",
    "uploader.errors.noFile": "Алдымен көз түбі суретін таңдаңыз.",
    "uploader.errors.predictionFailed": "Болжам жасалмады.",
    "uploader.modelMissingSuffix": "(файл жоқ)",
    "uploader.analyze": "Талдау",

    "modelInfo.eyebrow": "Backend",
    "modelInfo.title": "Модель туралы",
    "modelInfo.api": "API",
    "modelInfo.device": "Құрылғы",
    "modelInfo.llm": "LLM",
    "modelInfo.llm.ready": "Дайын",
    "modelInfo.llm.noKey": "API-кілт жоқ",
    "modelInfo.checkpoint.ready": "Дайын",
    "modelInfo.checkpoint.missing": "Жоқ",
    "modelInfo.availableModels": "Қолжетімді модельдер",
    "modelInfo.default": "әдепкі",

    "result.eyebrow": "Болжам",
    "result.classLabel": "Класс",
    "result.confidence": "Сенімділік",

    "explanation.eyebrow": "AI-көмекші",
    "explanation.title": "Түсіндірме",
    "explanation.loading": "Түсіндірме дайындалуда...",

    "chat.eyebrow": "AI-көмекші",
    "chat.title": "Сұрақ қою",
    "chat.subtitle":
      "Нәтиже, ұсыныстар немесе саты нені білдіретіні туралы сұраңыз.",
    "chat.empty": "Әзірге хабарлама жоқ. Көріңіз: «Маған не істеу керек?»",
    "chat.thinking": "Ойланудамын...",
    "chat.placeholder": "Сұрағыңызды жазыңыз...",
    "chat.send": "Жіберу",
    "chat.errors.failed": "Жауап алу мүмкін болмады.",
  },
} as const;

export function isLanguage(value: unknown): value is Language {
  return (
    typeof value === "string" &&
    (SUPPORTED_LANGUAGES as string[]).includes(value)
  );
}
