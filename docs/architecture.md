# Архитектура кода

Проект состоит из трех основных частей:

- `src/` - обучение, оценка и сравнение моделей.
- `backend/` - FastAPI inference API.
- `frontend/` - Next.js интерфейс для загрузки изображения и просмотра
  результата.

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

Backend находится в `backend/app/` и предоставляет FastAPI API для inference.

Основной поток `/predict`:

```text
main.py
+-- validate UploadFile
+-- read image bytes
+-- utils.read_image()
+-- inference.predict_image()
|   +-- model_loader.load_model()
|   +-- resize to 224x224
|   +-- ToTensor
|   +-- ImageNet normalization
|   +-- model inference with torch.no_grad()
|   +-- softmax probabilities
+-- return PredictionResponse
```

### `backend/app/main.py`

Создает FastAPI-приложение, настраивает CORS для локального frontend и
объявляет endpoints:

| Метод | Endpoint | Назначение |
|---|---|---|
| GET | `/health` | Проверка состояния backend. |
| GET | `/model-info` | Имя модели, путь к checkpoint, наличие checkpoint, device и классы. |
| POST | `/predict` | Загрузка изображения и получение результата классификации. |

### `backend/app/model_loader.py`

Загружает модель ResNet50 через `get_model()` из `src/models.py`.

Важные константы:

- `MODEL_NAME = "resnet50"`
- `CHECKPOINT_PATH = results/saved_models/best_resnet50.pth`
- `DEVICE = cuda`, если CUDA доступна, иначе `cpu`

Модель кэшируется через `lru_cache(maxsize=1)`, поэтому checkpoint загружается
один раз при первом inference-запросе.

### `backend/app/inference.py`

Содержит preprocessing и `predict_image()`.

Предобработка:

1. Resize до `224x224`.
2. `ToTensor()`.
3. Normalize с ImageNet mean/std.

Результат включает:

- `predicted_class`;
- `class_name`;
- `confidence`;
- `probabilities`.

### `backend/app/schemas.py`

Описывает Pydantic-схемы ответов API:

- `HealthResponse`;
- `ModelInfoResponse`;
- `PredictionResponse`.

### `backend/app/utils.py`

Содержит список классов, mapping классов и безопасное чтение изображения из
байтов.

## Frontend

Frontend находится в `frontend/` и использует Next.js App Router.

### `frontend/src/app/page.tsx`

Главный экран приложения:

- заголовок проекта;
- компонент загрузки изображения;
- панель информации о backend и checkpoint;
- предупреждение, что прототип не заменяет врача.

### `frontend/src/components/ImageUploader.tsx`

Клиентский компонент для основного workflow:

- выбор файла через file input;
- локальный preview изображения;
- сброс выбранного файла;
- отправка файла в `/predict`;
- отображение ошибки или результата.

### `frontend/src/components/PredictionResult.tsx`

Отображает:

- предсказанный класс;
- confidence;
- вероятности всех пяти классов в виде шкал.

### `frontend/src/components/ModelInfo.tsx`

Запрашивает `/model-info` и показывает:

- текущий API base URL;
- имя модели;
- device;
- статус checkpoint: `Ready` или `Missing`.

### `frontend/src/lib/api.ts`

Клиентский wrapper для backend:

- `API_BASE_URL`, по умолчанию `http://localhost:8000`;
- `fetchModelInfo()`;
- `predictImage()`;
- общий разбор ошибок API.

URL backend можно переопределить переменной окружения:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```
