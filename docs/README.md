# Документация проекта

Этот каталог содержит документацию для проекта **APTOS 2019 Blindness
Detection**: обучение и сравнение моделей глубокого обучения, FastAPI backend
для inference и LLM-ассистента, многоязычный Next.js frontend для
демонстрации классификации изображений глазного дна.

## Навигация

| Документ | Описание |
|---|---|
| [Обзор проекта](./project-overview.md) | Цель, задача классификации, структура репозитория, web-прототип и стек технологий. |
| [Установка и запуск](./setup-and-run.md) | Подготовка окружения, ML-скрипты, backend (включая `.env` для OpenAI) и frontend. |
| [Web-прототип](./web-prototype.md) | LLM-ассистент, выбор модели, многоязычный UI, markdown-рендер. |
| [Данные](./data.md) | Формат датасета APTOS, структура каталогов, классы и дисбаланс выборки. |
| [Архитектура кода](./architecture.md) | Назначение модулей `src`, backend и frontend, основные функции и поток выполнения. |
| [Модели](./models.md) | Реализованные архитектуры: Custom CNN, ResNet50, EfficientNet-B0 и MobileNetV2. |
| [Обучение и оценка](./training-and-evaluation.md) | Команды обучения, метрики, сохранение checkpoint и сравнение моделей. |
| [Результаты](./results.md) | Артефакты в `results/`, таблица результатов и интерпретация метрик. |
| [Ноутбуки](./notebooks.md) | Назначение исследовательских Jupyter Notebook-файлов. |

## Быстрый старт ML

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py --model resnet50
python src/evaluate.py --model resnet50
python src/compare_models.py
```

Перед запуском убедитесь, что данные находятся в `data/train.csv` и
`data/train_images/`.

## Быстрый старт web

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# впишите OPENAI_API_KEY
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Откройте `http://localhost:3000`. Backend должен быть доступен на
`http://localhost:8000`. В `results/saved_models/` должен лежать хотя
бы один checkpoint (по умолчанию `best_resnet50.pth`).

В правом верхнем углу UI — переключатель языка (English / Русский /
Қазақша). В блоке загрузки — селектор модели (ResNet50 / EfficientNet-B0
/ MobileNetV2 / Custom CNN). Без `OPENAI_API_KEY` доступен только
inference; объяснение результата и чат с ассистентом потребуют ключ.
