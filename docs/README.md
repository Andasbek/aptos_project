# Документация проекта

Этот каталог содержит документацию для проекта **APTOS 2019 Blindness
Detection**: обучение и сравнение моделей глубокого обучения, FastAPI backend
для inference и Next.js frontend для демонстрации классификации изображений
глазного дна.

## Навигация

| Документ | Описание |
|---|---|
| [Обзор проекта](./project-overview.md) | Цель, задача классификации, структура репозитория, web-прототип и стек технологий. |
| [Установка и запуск](./setup-and-run.md) | Подготовка окружения, запуск ML-скриптов, backend и frontend. |
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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd frontend
npm install
npm run dev
```

Откройте `http://localhost:3000`. Backend должен быть доступен на
`http://localhost:8000`, а checkpoint ResNet50 должен лежать в
`results/saved_models/best_resnet50.pth`.
