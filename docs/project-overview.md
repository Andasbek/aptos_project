# Обзор проекта

## Назначение

Проект предназначен для исследования, обучения и сравнения моделей глубокого
обучения на датасете **APTOS 2019 Blindness Detection**. Основная задача -
классифицировать изображение глазного дна по одной из пяти стадий
диабетической ретинопатии.

Помимо ML-скриптов, в проекте есть web-прототип: FastAPI backend загружает
обученный checkpoint ResNet50 и выполняет inference, а Next.js frontend
позволяет выбрать изображение, отправить его в API и увидеть предсказанный
класс, confidence и вероятности по всем классам.

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

Web-часть добавляет пользовательский сценарий inference:

1. Frontend получает изображение через file input и показывает preview.
2. Файл отправляется в backend как `multipart/form-data`.
3. Backend проверяет тип файла, открывает изображение через Pillow и применяет
   ту же базовую предобработку, что validation pipeline.
4. ResNet50 возвращает logits, которые преобразуются в вероятности через
   `softmax`.
5. Frontend отображает предсказанный класс и распределение вероятностей.

## Структура репозитория

```text
aptos_project/
+-- backend/
|   +-- app/
|   |   +-- main.py
|   |   +-- inference.py
|   |   +-- model_loader.py
|   |   +-- schemas.py
|   |   +-- utils.py
|   +-- requirements.txt
+-- frontend/
|   +-- src/
|   |   +-- app/
|   |   +-- components/
|   |   +-- lib/api.ts
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
|   +-- saved_models/
+-- src/
|   +-- compare_models.py
|   +-- config.py
|   +-- dataset.py
|   +-- evaluate.py
|   +-- models.py
|   +-- plots.py
|   +-- train.py
|   +-- utils.py
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

Frontend:

- Next.js App Router
- React
- TypeScript
- Tailwind CSS
- lucide-react

## Основные ограничения

- Скрипты рассчитаны на локальную структуру данных Kaggle APTOS.
- Валидация строится из `train.csv`, отдельный inference pipeline для
  `test_images/` в `src` пока не реализован.
- Размер входного изображения зафиксирован в `src/config.py` и
  `backend/app/inference.py`: `224x224`.
- Для Windows в `DataLoader` используется `num_workers=0`.
- Web-прототип использует только `results/saved_models/best_resnet50.pth`.
- Проект является исследовательским прототипом и не заменяет медицинскую
  диагностику.
