# Web-прототип

Web-прототип состоит из FastAPI backend и Next.js frontend. Помимо инференса
обученной модели, прототип использует OpenAI LLM для объяснения результата и
чата с пользователем, а интерфейс полностью локализован на трёх языках.

## Возможности

- Загрузка снимка глазного дна и инференс одной из четырёх обученных моделей.
- Автоматическое объяснение результата от AI-ассистента после предсказания.
- Чат с ассистентом в контексте текущего предсказания.
- Глобальный переключатель языка интерфейса и ответов LLM:
  English / Русский / Қазақша.
- Отображение списка доступных checkpoints и их статуса (Ready / Missing).
- Markdown-рендер ответов ассистента (заголовки, списки, жирный, ссылки,
  таблицы) через `react-markdown` + `remark-gfm`.
- Логотип AT University в шапке ([frontend/public/logo.png](../frontend/public/logo.png)).

## Выбор модели

`/model-info` возвращает список доступных моделей и наличие чекпоинтов:

```json
{
  "default_model": "resnet50",
  "available_models": [
    { "name": "resnet50",        "checkpoint_exists": true,  "is_default": true,  "checkpoint_path": "..." },
    { "name": "efficientnet_b0", "checkpoint_exists": true,  "is_default": false, "checkpoint_path": "..." },
    { "name": "mobilenet_v2",    "checkpoint_exists": true,  "is_default": false, "checkpoint_path": "..." },
    { "name": "cnn",             "checkpoint_exists": true,  "is_default": false, "checkpoint_path": "..." }
  ],
  "device": "cpu",
  "classes": { "0": "No DR", "1": "Mild", "2": "Moderate", "3": "Severe", "4": "Proliferative DR" },
  "llm_enabled": true,
  "supported_languages": ["en", "ru", "kk"]
}
```

Frontend в `ImageUploader` показывает `<select>` с моделями. Если checkpoint
отсутствует — пункт помечается как `(missing)` и недоступен для выбора.
Пользовательский выбор передаётся в `/predict` как form-поле `model`.

Backend кеширует загруженные модели через `lru_cache(maxsize=4)`, поэтому
переключение между моделями не пересоздаёт уже загруженные веса.

## LLM-ассистент

### Endpoints

| Метод | Endpoint | Тело запроса | Ответ |
|---|---|---|---|
| POST | `/explain` | `{ prediction, language }` | `{ explanation, language }` |
| POST | `/chat` | `{ messages[], prediction?, language }` | `{ reply, language }` |

`prediction` — объект, который вернул `/predict` (включая `model_name`,
`probabilities`, `confidence`).

`messages[]` — история чата:

```json
[
  { "role": "user", "content": "What should I do next?" },
  { "role": "assistant", "content": "..." },
  { "role": "user", "content": "Is it dangerous?" }
]
```

### Промпты

Базовый системный промпт описывает роль ассистента и объясняет 5 классов
APTOS. К нему добавляется языковая инструкция (`Always respond in English` /
`Всегда отвечай на русском` / `Барлық жауаптарыңды қазақ тілінде беріңіз`).

Контекст текущего предсказания передаётся отдельным system-сообщением:

```text
Classifier: resnet50
Predicted stage: Moderate (class index 2)
Model confidence: 87.3%
Per-class probabilities:
- No DR: 2.1%
- Mild: 6.4%
- Moderate: 87.3%
- Severe: 3.0%
- Proliferative DR: 1.2%
```

Шаблоны промптов для `/explain` различаются по языку и просят раскрыть:

1. что означает эта стадия;
2. насколько модель уверена и как интерпретировать confidence;
3. рекомендованные следующие шаги;
4. общие советы по образу жизни и наблюдению;
5. короткий дисклеймер.

### Ошибки

| Код | Когда |
|---|---|
| `503` | `OPENAI_API_KEY` не задан |
| `502` | OpenAI API вернул ошибку (rate limit, network, и т.д.) |

`/predict` продолжает работать без ключа — без LLM просто не будет
объяснения и чата на UI.

## Многоязычный UI

### Архитектура

- [frontend/src/lib/i18n.ts](../frontend/src/lib/i18n.ts) — словари переводов
  для трёх языков (`en`, `ru`, `kk`). Ключи плоские с dot-notation
  (`uploader.title`, `chat.send`, `modelInfo.availableModels` и т.д.).
- [frontend/src/lib/LanguageContext.tsx](../frontend/src/lib/LanguageContext.tsx)
  — React Context с провайдером, хук `useLanguage()` и функция `t(key)`.
- [frontend/src/components/LanguageSelector.tsx](../frontend/src/components/LanguageSelector.tsx)
  — компактный селектор в правом верхнем углу шапки.

Состояние хранится в `localStorage` (ключ `aptos-ui-language`). При первой
загрузке язык определяется по `navigator.language` (`ru-*` → ru, `kk-*` →
kk, иначе en). Атрибут `<html lang>` обновляется при смене языка.

### Что переключается

Глобальный язык управляет одновременно:

- всеми UI-строками (шапка, формы, кнопки, плейсхолдеры, ошибки);
- языком ответов от LLM в `/explain` и `/chat`.

### Что НЕ переключается

- Имена классов APTOS (`No DR`, `Mild`, `Moderate`, `Severe`,
  `Proliferative DR`) — оставлены на английском как стандартные медицинские
  термины из датасета.
- Имена архитектур (`ResNet50`, `EfficientNet-B0`, `MobileNetV2`,
  `Custom CNN`) — собственные имена.

### Добавление нового языка

1. Расширить тип `Language` в `i18n.ts` (`"en" | "ru" | "kk" | "..."`).
2. Добавить запись в `SUPPORTED_LANGUAGES`, `LANGUAGE_LABELS` и
   `translations` со всеми ключами.
3. В backend дополнить:
   - `SUPPORTED_LANGUAGES` в `backend/app/llm.py`;
   - `_LANGUAGE_INSTRUCTIONS` (короткая инструкция стиля «отвечай на …»);
   - `_EXPLAIN_INSTRUCTIONS` (полный шаблон промпта для `/explain`);
   - тип `Language` в `backend/app/schemas.py`.

## Markdown-рендер ответов ассистента

Ответы LLM содержат markdown (списки, заголовки, жирный). Они проходят
через [MarkdownContent.tsx](../frontend/src/components/MarkdownContent.tsx),
который использует `react-markdown` + `remark-gfm` и стилизует элементы
через Tailwind:

- два варианта: `default` (для блока explanation) и `chat` (компактнее, для
  пузырьков чата);
- кастомные компоненты для `h1-h4`, `p`, `ul/ol/li`, `strong`, `em`, `a`,
  `code`, `pre`, `blockquote`, `hr`, `table/th/td`.

Сообщения пользователя рендерятся как обычный текст с
`whitespace-pre-wrap` — пользователь обычно не пишет markdown.

## Конфигурация LLM

В `backend/.env`:

```text
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL` опционален, по умолчанию `gpt-4o-mini`. Файл `.env`
исключён из git. Шаблон — `backend/.env.example`.

Загрузка переменных выполняется через `python-dotenv` в
[backend/app/llm.py](../backend/app/llm.py) при импорте модуля.

## Запуск

См. [setup-and-run.md](setup-and-run.md), раздел «Запуск backend» и «Запуск
frontend».
