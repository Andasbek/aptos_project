# Обзор проекта

## Назначение

Проект предназначен для исследования, обучения и сравнения моделей глубокого
обучения на датасете **APTOS 2019 Blindness Detection**. Основная задача -
классифицировать изображение глазного дна по одной из пяти стадий
диабетической ретинопатии.

Помимо ML-скриптов, в проекте есть web-прототип. FastAPI backend загружает
любую из четырёх обученных моделей и выполняет inference. Next.js frontend
позволяет выбрать классификатор и язык интерфейса, отправить изображение в
API, получить предсказанный класс с вероятностями и автоматическое
объяснение от AI-ассистента (OpenAI). Поддерживается чат с ассистентом
по результату.

## Классы

| Метка | Класс | Описание |
|---:|---|---|
| 0 | No DR | Признаков диабетической ретинопатии нет. |
| 1 | Mild | Легкая стадия. |
| 2 | Moderate | Умеренная стадия. |
| 3 | Severe | Тяжелая стадия. |
| 4 | Proliferative DR | Пролиферативная диабетическая ретинопатия. |

## Что делает проект

ML-часть поддерживает полный цикл эксперимента:

1. Загрузка метаданных из `data/train.csv`.
2. Чтение изображений из `data/train_images/`.
3. Стратифицированное разделение train/validation в пропорции 80/20.
4. Аугментация обучающих изображений.
5. Обучение одной из четырех моделей.
6. Сохранение лучшего checkpoint по `val_accuracy`.
7. Построение графиков обучения.
8. Оценка модели на validation split.
9. Сравнение моделей по accuracy, precision, recall и F1-score.

Web-часть добавляет пользовательский сценарий inference и объяснения:

1. Frontend получает изображение через file input и показывает preview.
2. Пользователь выбирает классификатор (`resnet50` / `efficientnet_b0` /
   `mobilenet_v2` / `cnn`) и язык (en / ru / kk).
3. Файл и имя модели отправляются в `/predict` как `multipart/form-data`.
4. Backend проверяет тип файла, открывает изображение через Pillow и
   применяет ту же базовую предобработку, что validation pipeline.
5. Выбранная модель возвращает logits, которые преобразуются в вероятности
   через `softmax`.
6. Frontend отображает предсказанный класс и распределение вероятностей.
7. После предсказания фронтенд автоматически вызывает `/explain` и
   получает подробное объяснение от LLM на выбранном языке.
8. Пользователь может задать дополнительные вопросы в чате (`/chat`),
   ответ возвращается на том же языке.

## Структура репозитория

```text
aptos_project/
+-- backend/
|   +-- app/
|   |   +-- main.py            FastAPI: /health, /model-info, /predict, /explain, /chat
|   |   +-- inference.py       Preprocessing + predict_image(model_name)
|   |   +-- model_loader.py    Reestr 4 моделей, lru_cache(maxsize=4)
|   |   +-- llm.py             OpenAI client + многоязычные промпты
|   |   +-- schemas.py         Pydantic-схемы запросов/ответов
|   |   +-- utils.py
|   +-- .env.example           Шаблон для OPENAI_API_KEY
|   +-- requirements.txt
+-- frontend/
|   +-- public/
|   |   +-- logo.png           Логотип AT University в шапке
|   +-- src/
|   |   +-- app/               layout.tsx + page.tsx
|   |   +-- components/        ImageUploader, ModelInfo, PredictionResult,
|   |                          PredictionExplanation, PredictionChat,
|   |                          LanguageSelector, MarkdownContent, Localized*
|   |   +-- lib/api.ts         API-клиент
|   |   +-- lib/i18n.ts        Словари переводов en/ru/kk
|   |   +-- lib/LanguageContext.tsx  Глобальный Provider языка
|   +-- package.json
+-- data/
|   +-- train.csv
|   +-- test.csv
|   +-- sample_submission.csv
|   +-- train_images/
|   +-- test_images/
+-- docs/
+-- models/
+-- notebooks/
|   +-- 01_eda.ipynb
|   +-- 02_cnn.ipynb
|   +-- 03_mobilenetv2.ipynb
|   +-- 04_efficientnet_b0.ipynb
|   +-- 05_resnet50.ipynb
+-- results/
|   +-- figures/
|   +-- metrics/
|   +-- saved_models/         best_cnn.pth, best_resnet50.pth,
|                             best_efficientnet_b0.pth, best_mobilenet_v2.pth
+-- src/
|   +-- compare_models.py
|   +-- config.py
|   +-- dataset.py
|   +-- evaluate.py
|   +-- models.py
|   +-- plots.py
|   +-- train.py
|   +-- utils.py
+-- images.png                 Исходный логотип
+-- requirements.txt
+-- README.md
```

## Технологический стек

ML:

- Python
- PyTorch
- Torchvision
- CUDA, если доступна GPU-версия PyTorch
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- tqdm
- OpenCV Python

Backend:

- FastAPI
- Uvicorn
- python-multipart
- Pillow
- PyTorch и Torchvision
- OpenAI Python SDK
- python-dotenv

Frontend:

- Next.js App Router
- React 18
- TypeScript
- Tailwind CSS
- lucide-react
- react-markdown + remark-gfm

## Основные ограничения

- Скрипты рассчитаны на локальную структуру данных Kaggle APTOS.
- Валидация строится из `train.csv`, отдельный inference pipeline для
  `test_images/` в `src` пока не реализован.
- Размер входного изображения зафиксирован в `src/config.py` и
  `backend/app/inference.py`: `224x224`.
- Для Windows в `DataLoader` используется `num_workers=0`.
- Web-прототип использует чекпоинты из `results/saved_models/`. Если файл
  отсутствует — соответствующий вариант в селекторе модели становится
  недоступным.
- LLM-функциональность требует `OPENAI_API_KEY` в `backend/.env`. Без
  ключа `/explain` и `/chat` возвращают `503`, инференс продолжает работать.
- Проект является исследовательским прототипом и не заменяет медицинскую
  диагностику. Ответы LLM-ассистента носят образовательный характер.
