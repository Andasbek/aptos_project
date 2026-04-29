# APTOS 2019 Blindness Detection

Проект для классификации стадий диабетической ретинопатии по изображениям
глазного дна из датасета **APTOS 2019 Blindness Detection**.

В репозитории есть две части:

- ML-пайплайн на PyTorch для обучения, оценки и сравнения моделей.
- Web-прототип: FastAPI backend для inference и Next.js frontend для загрузки
  изображения и просмотра результата.

## Поддерживаемые классы

| Метка | Класс |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Модели

В `src/models.py` реализованы:

- Custom CNN
- ResNet50
- EfficientNet-B0
- MobileNetV2

Web-прототип по умолчанию использует лучший checkpoint ResNet50:

```text
results/saved_models/best_resnet50.pth
```

## Структура проекта

```text
aptos_project/
+-- backend/                  FastAPI inference API
|   +-- app/
|   |   +-- main.py            Endpoints: /health, /model-info, /predict
|   |   +-- inference.py       Preprocessing and prediction
|   |   +-- model_loader.py    Loading ResNet50 checkpoint
|   |   +-- schemas.py         Pydantic response schemas
|   |   +-- utils.py           Image and class helpers
|   +-- requirements.txt
+-- frontend/                 Next.js App Router UI
|   +-- src/
|   |   +-- app/
|   |   +-- components/
|   |   +-- lib/api.ts         Client API wrapper
|   +-- package.json
+-- data/                     APTOS dataset, ignored by Git
+-- docs/                     Project documentation
+-- notebooks/                Research notebooks
+-- results/                  Metrics, figures, checkpoints, ignored by Git
+-- src/                      Training and evaluation scripts
+-- requirements.txt          ML dependencies
+-- README.md
```

## Быстрый старт: web-прототип

Сначала убедитесь, что checkpoint лежит здесь:

```text
results/saved_models/best_resnet50.pth
```

Запуск backend из корня проекта:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend будет доступен на `http://localhost:8000`.

Запуск frontend во втором терминале:

```powershell
cd frontend
npm install
npm run dev
```

Frontend будет доступен на `http://localhost:3000`.

По умолчанию frontend обращается к `http://localhost:8000`. При необходимости
URL можно изменить через переменную:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```

## API

| Метод | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | Проверка состояния backend |
| GET | `/model-info` | Информация о модели, checkpoint и устройстве |
| POST | `/predict` | Загрузка изображения и получение предсказания |

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
  }
}
```

Если checkpoint отсутствует, `/predict` возвращает `503` с понятным сообщением.

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

Сравнение сохраненных метрик:

```powershell
python src/compare_models.py
```

## Документация

Подробные разделы находятся в [`docs/`](docs/README.md):

- обзор проекта;
- установка и запуск;
- данные;
- архитектура кода;
- модели;
- обучение и оценка;
- результаты;
- ноутбуки.

## Важно

Проект является исследовательским прототипом и не заменяет консультацию
офтальмолога.
