# Архитектура кода

Проект состоит из трех основных частей:

- `src/` - обучение, оценка и сравнение моделей.
- `backend/` - FastAPI inference + LLM API.
- `frontend/` - Next.js интерфейс с многоязычностью и AI-ассистентом.

## ML-пайплайн

Общий поток обучения:

```text
train.py
+-- set_seed()
+-- create_dirs()
+-- get_dataloaders()
|   +-- read train.csv
|   +-- stratified train/validation split
|   +-- APTOSDataset
|   +-- DataLoader
+-- get_model()
+-- get_class_weights()
+-- training loop
+-- save best checkpoint
+-- save history CSV
+-- plot training history
```

Оценка выполняется похожим образом:

```text
evaluate.py
+-- get_dataloaders()
+-- get_model()
+-- load results/saved_models/best_<model>.pth
+-- predict validation split
+-- calculate metrics
+-- save classification report
+-- plot confusion matrix
```

## `src/config.py`

Центральный файл конфигурации.

Основные параметры:

| Параметр | Значение | Назначение |
|---|---:|---|
| `IMAGE_SIZE` | 224 | Размер входного изображения. |
| `BATCH_SIZE` | 16 | Размер batch. |
| `NUM_EPOCHS` | 15 | Количество эпох обучения. |
| `LEARNING_RATE` | `1e-4` | Начальный learning rate. |
| `NUM_CLASSES` | 5 | Количество классов. |
| `RANDOM_SEED` | 42 | Seed для воспроизводимости. |

Также здесь задаются пути:

- `DATA_DIR`
- `TRAIN_CSV`
- `IMAGE_DIR`
- `RESULTS_DIR`
- `FIGURES_DIR`
- `METRICS_DIR`
- `SAVED_MODELS_DIR`

## `src/dataset.py`

Отвечает за чтение данных и подготовку `DataLoader`.

Ключевые элементы:

- `APTOSDataset` - PyTorch Dataset для чтения `.png` изображений по `id_code`.
- `get_train_transforms()` - аугментации для обучения.
- `get_val_transforms()` - трансформации для валидации.
- `get_class_weights()` - расчет весов классов для функции потерь.
- `get_dataloaders()` - создание train и validation dataloaders.

## `src/models.py`

Содержит модели и фабрику `get_model()`.

Поддерживаемые значения `model_name`:

- `cnn`
- `resnet50`
- `efficientnet_b0`
- `mobilenet_v2`

Для transfer learning моделей используется актуальный API `weights` из
`torchvision.models`.

## `src/train.py`

Скрипт обучения.

Что делает:

- выбирает модель;
- создает loss с весами классов;
- использует Adam optimizer;
- подключает `ReduceLROnPlateau`;
- при CUDA включает mixed precision через `torch.amp`;
- сохраняет лучший checkpoint по `val_accuracy`;
- сохраняет CSV с историей обучения;
- строит графики loss, accuracy и learning rate.

CLI:

```powershell
python src/train.py --model resnet50
python src/train.py --model resnet50 --freeze_backbone
```

## `src/evaluate.py`

Скрипт оценки модели.

Метрики:

- accuracy;
- weighted precision;
- weighted recall;
- weighted F1-score;
- classification report по классам;
- confusion matrix.

CLI:

```powershell
python src/evaluate.py --model resnet50
```

## `src/compare_models.py`

Собирает summary-файлы после оценки моделей и формирует:

- `results/metrics/model_comparison.csv`;
- `results/figures/model_comparison.png`.

## `src/plots.py`

Функции визуализации:

- `plot_training_history()`;
- `plot_confusion_matrix()`;
- `plot_model_comparison()`.

## `src/utils.py`

Вспомогательные функции:

- `set_seed()`;
- `create_dirs()`;
- `save_history()`;
- `count_parameters()`;
- `get_gpu_info()`;
- `save_json()`.

## Backend

Backend находится в `backend/app/` и предоставляет FastAPI API для
inference и взаимодействия с LLM.

Основной поток `/predict` с выбираемой моделью:

```text
main.py POST /predict
+-- validate UploadFile
+-- validate model name (cnn / resnet50 / efficientnet_b0 / mobilenet_v2)
+-- read image bytes
+-- utils.read_image()
+-- inference.predict_image(image, model_name)
|   +-- model_loader.load_model(model_name)   lru_cache(maxsize=4)
|   +-- resize to 224x224
|   +-- ToTensor
|   +-- ImageNet normalization
|   +-- model inference with torch.no_grad()
|   +-- softmax probabilities
+-- return PredictionResponse(model_name=...)
```

Поток LLM:

```text
main.py POST /explain
+-- ExplainRequest(prediction, language)
+-- llm.generate_explanation(prediction, language)
|   +-- load .env (один раз при импорте)
|   +-- _system_prompt(language)
|   +-- _EXPLAIN_INSTRUCTIONS[language].format(prediction=...)
|   +-- OpenAI client.chat.completions.create()
+-- return ExplainResponse(explanation, language)

main.py POST /chat
+-- ChatRequest(messages, prediction?, language)
+-- llm.chat_about_prediction(messages, prediction, language)
|   +-- system: base prompt + language instruction
|   +-- system: prediction context (если задан)
|   +-- user/assistant messages
|   +-- OpenAI client.chat.completions.create()
+-- return ChatResponse(reply, language)
```

### `backend/app/main.py`

Создает FastAPI-приложение, настраивает CORS для локального frontend и
объявляет endpoints:

| Метод | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | Проверка состояния backend. |
| GET | `/model-info` | `default_model`, список `available_models`, device, классы, поддерживаемые языки, флаг `llm_enabled`. |
| POST | `/predict` | Инференс выбранной моделью (form-поле `model`). |
| POST | `/explain` | Объяснение предсказания через LLM на выбранном языке. |
| POST | `/chat` | Чат с LLM в контексте предсказания. |

### `backend/app/model_loader.py`

Реестр доступных моделей и кеш загруженных весов.

Ключевые элементы:

- `AVAILABLE_MODELS` — словарь `{ "resnet50": "best_resnet50.pth", ... }`.
- `DEFAULT_MODEL_NAME = "resnet50"`.
- `DEVICE = cuda`, если CUDA доступна, иначе `cpu`.
- `available_models()` — возвращает список `{ name, checkpoint_path,
  checkpoint_exists, is_default }` для UI.
- `load_model(model_name)` — `lru_cache(maxsize=4)`, чтобы каждая модель
  загружалась только один раз. Поддерживает три формата чекпоинта: чистый
  `state_dict`, `{"state_dict": ...}` и `{"model_state_dict": ...}`.

### `backend/app/inference.py`

Содержит preprocessing и `predict_image(image, model_name)`.

Предобработка:

1. Resize до `224x224`.
2. `ToTensor()`.
3. Normalize с ImageNet mean/std.

Результат включает:

- `predicted_class`;
- `class_name`;
- `confidence`;
- `probabilities`;
- `model_name` — какая модель использовалась.

### `backend/app/llm.py`

Обёртка над OpenAI Python SDK.

Ключевые элементы:

- При импорте загружает `backend/.env` через `python-dotenv`.
- `_get_client()` — `lru_cache(maxsize=1)` для OpenAI-клиента, читает
  `OPENAI_API_KEY`. Если ключа нет — `LLMConfigurationError`.
- `_get_model()` — читает `OPENAI_MODEL` (по умолчанию `gpt-4o-mini`).
- `SUPPORTED_LANGUAGES = ("en", "ru", "kk")`.
- `_LANGUAGE_INSTRUCTIONS` — короткая инструкция стиля «отвечай на
  английском/русском/казахском», добавляется к системному промпту.
- `_EXPLAIN_INSTRUCTIONS` — полные многоязычные шаблоны для `/explain`.
- `_format_prediction()` — преобразует словарь предсказания в человекочитаемый
  блок текста для LLM.
- `generate_explanation(prediction, language)` — структурированное объяснение.
- `chat_about_prediction(messages, prediction, language)` — обработка чата
  с контекстом предсказания.

Ошибки:

- `LLMConfigurationError` → endpoint возвращает `503`.
- `LLMRequestError` (OpenAI API упал) → endpoint возвращает `502`.

### `backend/app/schemas.py`

Pydantic-схемы:

- `HealthResponse`;
- `ModelOption` — описание одного варианта модели для `/model-info`;
- `ModelInfoResponse` — `default_model`, `available_models`, `device`,
  `classes`, `llm_enabled`, `supported_languages`;
- `PredictionResponse` — поля включают `model_name`;
- `ExplainRequest` — `{ prediction, language }`;
- `ExplainResponse` — `{ explanation, language }`;
- `ChatMessage` — `{ role: "user" | "assistant", content }`;
- `ChatRequest` — `{ messages[], prediction?, language }`;
- `ChatResponse` — `{ reply, language }`.

`Language = Literal["en", "ru", "kk"]`.

### `backend/app/utils.py`

Содержит список классов, mapping классов и безопасное чтение изображения из
байтов.

## Frontend

Frontend находится в `frontend/` и использует Next.js App Router.

### `frontend/src/app/layout.tsx`

Корневой layout. Оборачивает приложение в `<LanguageProvider>`, чтобы
любой клиентский компонент мог обратиться к `useLanguage()`.

### `frontend/src/app/page.tsx`

Главный экран приложения (server component):

- логотип AT University слева;
- заголовок и подзаголовок (через `LocalizedHeader`);
- глобальный `LanguageSelector` справа в шапке;
- `ImageUploader` — основная форма;
- `ModelInfo` — статус backend и моделей;
- `LocalizedDisclaimer` — предупреждение, что прототип не заменяет врача.

### `frontend/src/components/ImageUploader.tsx`

Клиентский компонент для основного workflow:

- выбор файла через file input и preview;
- селектор модели (`<select>`), варианты приходят из `/model-info`;
  отсутствующие чекпоинты помечаются как `(missing)` и недоступны;
- сброс выбранного файла;
- отправка файла + имени модели в `/predict`;
- после успешного ответа рендерит `PredictionResult`,
  `PredictionExplanation` и `PredictionChat`.

### `frontend/src/components/PredictionResult.tsx`

Отображает:

- предсказанный класс;
- confidence;
- вероятности всех пяти классов в виде шкал.

### `frontend/src/components/PredictionExplanation.tsx`

После получения предсказания автоматически вызывает `/explain` с текущим
языком. Ответ рендерится через `MarkdownContent`.

### `frontend/src/components/PredictionChat.tsx`

Чат с ассистентом по результату предсказания:

- хранит локальную историю сообщений;
- отправляет всю историю + `prediction` + `language` в `/chat`;
- сообщения пользователя — обычный текст;
- сообщения ассистента — markdown через `MarkdownContent` с компактным
  вариантом стилей.

### `frontend/src/components/MarkdownContent.tsx`

Обёртка над `react-markdown` + `remark-gfm`. Стилизует все markdown-элементы
через Tailwind, поддерживает два варианта: `default` (для `PredictionExplanation`)
и `chat` (для `PredictionChat`). Ссылки открываются в новом табе с
`rel="noreferrer noopener"`.

### `frontend/src/components/ModelInfo.tsx`

Запрашивает `/model-info` и показывает:

- текущий API base URL;
- device;
- статус LLM (`Ready` / `No API key`);
- список всех доступных моделей с чекпоинт-статусом и пометкой `default`.

### `frontend/src/components/LanguageSelector.tsx`

Глобальный селектор языка с двумя вариантами: компактный (для шапки) и
обычный (для inline-форм). Использует `useLanguage()` и переключает язык
на лету. Сохраняет выбор в `localStorage` (через provider).

### `frontend/src/components/LocalizedHeader.tsx`, `LocalizedDisclaimer.tsx`

Маленькие клиентские обёртки над текстами шапки и дисклеймера. Нужны,
чтобы `page.tsx` остался серверным компонентом.

### `frontend/src/lib/i18n.ts`

Содержит:

- тип `Language = "en" | "ru" | "kk"`;
- `SUPPORTED_LANGUAGES`, `LANGUAGE_LABELS`;
- словарь `translations` с тремя вариантами для каждого ключа;
- тип `TranslationKey`;
- `isLanguage()` — type guard.

Ключи плоские: `header.title`, `uploader.analyze`, `chat.empty` и т.д.

### `frontend/src/lib/LanguageContext.tsx`

React Context Provider:

- начальный язык определяется по `localStorage` (`aptos-ui-language`),
  иначе по `navigator.language` (`ru-*` → ru, `kk-*` → kk, иначе en);
- сохраняет язык в `localStorage` и обновляет `<html lang>` при смене;
- предоставляет `{ language, setLanguage, t }` через `useLanguage()`.

### `frontend/src/lib/api.ts`

Клиентский wrapper для backend:

- `API_BASE_URL`, по умолчанию `http://localhost:8000`;
- типы `ModelOption`, `ModelInfo`, `PredictionResponse`, `ChatMessage`;
- `fetchModelInfo()`;
- `predictImage(file, modelName)`;
- `explainPrediction(prediction, language)`;
- `chatWithLLM(messages, prediction, language)`;
- общий разбор ошибок API.

URL backend можно переопределить переменной окружения:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```
