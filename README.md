# APTOS 2019 Blindness Detection

Проект для классификации стадий диабетической ретинопатии по изображениям
глазного дна из датасета **APTOS 2019 Blindness Detection**.

В репозитории есть две части:

- ML-пайплайн на PyTorch для обучения, оценки и сравнения четырёх моделей.
- Web-прототип: FastAPI backend для inference + LLM-ассистента и Next.js
  frontend с многоязычным интерфейсом (English / Русский / Қазақша).

## Поддерживаемые классы

| Метка | Класс |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Модели

В `src/models.py` реализованы четыре архитектуры, и web-прототип позволяет
выбирать любую из них во время инференса (если соответствующий checkpoint
лежит в `results/saved_models/`):

- Custom CNN — `best_cnn.pth`
- ResNet50 — `best_resnet50.pth` (по умолчанию)
- EfficientNet-B0 — `best_efficientnet_b0.pth`
- MobileNetV2 — `best_mobilenet_v2.pth`

## Возможности web-прототипа

- Загрузка снимка глазного дна и инференс выбранной моделью.
- Переключение классификатора на лету через UI.
- AI-ассистент на базе OpenAI: автоматическое объяснение результата и чат
  по диагнозу.
- Глобальный переключатель языка (English / Русский / Қазақша) — управляет
  одновременно UI и языком ответов LLM.
- Markdown-рендер ответов ассистента (заголовки, списки, жирный текст).

## Структура проекта

```text
aptos_project/
+-- backend/                  FastAPI inference + LLM API
|   +-- app/
|   |   +-- main.py            Endpoints: /health, /model-info, /predict, /explain, /chat
|   |   +-- inference.py       Preprocessing and prediction
|   |   +-- model_loader.py    Loading of all 4 trained models
|   |   +-- llm.py             OpenAI client, multilingual prompts
|   |   +-- schemas.py         Pydantic request/response schemas
|   |   +-- utils.py           Image and class helpers
|   +-- .env.example           Template for OPENAI_API_KEY
|   +-- requirements.txt
+-- frontend/                 Next.js App Router UI
|   +-- public/
|   |   +-- logo.png           AT University logo (header)
|   +-- src/
|   |   +-- app/               Layout + page (server)
|   |   +-- components/        UI (uploader, chat, model info, language switcher)
|   |   +-- lib/api.ts         Backend API wrapper
|   |   +-- lib/i18n.ts        Translation dictionaries (en/ru/kk)
|   |   +-- lib/LanguageContext.tsx  Global language Provider
|   +-- package.json
+-- data/                     APTOS dataset, ignored by Git
+-- docs/                     Project documentation
+-- notebooks/                Research notebooks
+-- results/                  Metrics, figures, checkpoints, ignored by Git
+-- src/                      Training and evaluation scripts
+-- images.png                Source logo (copied to frontend/public/logo.png)
+-- requirements.txt          ML dependencies
+-- README.md
```

## Быстрый старт: web-прототип

Перед запуском убедитесь, что в `results/saved_models/` лежит хотя бы один
checkpoint (по умолчанию `best_resnet50.pth`).

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# открыть .env и вписать OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend будет доступен на `http://localhost:8000`. Без OpenAI ключа
`/predict` продолжит работать, а `/explain` и `/chat` вернут `503` с
понятным сообщением.

### 2. Frontend

Во втором терминале:

```powershell
cd frontend
npm install
npm run dev
```

Frontend будет доступен на `http://localhost:3000`. По умолчанию обращается
к `http://localhost:8000`. URL backend можно переопределить:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```

## API

| Метод | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | Проверка состояния backend |
| GET | `/model-info` | Список доступных моделей, статус чекпоинтов, устройство, поддерживаемые языки, флаг доступности LLM |
| POST | `/predict` | Инференс выбранной моделью (form-поле `model`) |
| POST | `/explain` | Объяснение предсказания через LLM (требует `OPENAI_API_KEY`) |
| POST | `/chat` | Чат с ассистентом по результату предсказания |

Пример ответа `/predict`:

```json
{
  "predicted_class": 2,
  "class_name": "Moderate",
  "confidence": 0.87,
  "probabilities": {
    "No DR": 0.02,
    "Mild": 0.06,
    "Moderate": 0.87,
    "Severe": 0.03,
    "Proliferative DR": 0.02
  },
  "model_name": "resnet50"
}
```

`/explain` и `/chat` принимают поле `language: "en" | "ru" | "kk"`. Ответ
ассистента возвращается на выбранном языке.

## Конфигурация LLM

В `backend/.env`:

```text
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini
```

Файл `backend/.env` не коммитится (исключён в `.gitignore`). Шаблон —
`backend/.env.example`.

## Обучение и оценка

Установка зависимостей для ML-части:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Ожидаемая структура данных:

```text
data/
+-- train.csv
+-- train_images/
```

Запуск обучения:

```powershell
python src/train.py --model cnn
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
python src/train.py --model mobilenet_v2
```

Оценка моделей:

```powershell
python src/evaluate.py --model cnn
python src/evaluate.py --model resnet50
python src/evaluate.py --model efficientnet_b0
python src/evaluate.py --model mobilenet_v2
```

Сравнение сохранённых метрик:

```powershell
python src/compare_models.py
```

## Документация

Подробные разделы находятся в [`docs/`](docs/README.md):

- обзор проекта;
- установка и запуск;
- web-прототип (LLM, мультиязычность, выбор модели);
- данные;
- архитектура кода;
- модели;
- обучение и оценка;
- результаты;
- ноутбуки.

## Важно

Проект является исследовательским прототипом и не заменяет консультацию
офтальмолога. Ответы LLM-ассистента — образовательного характера, не
медицинская рекомендация.
