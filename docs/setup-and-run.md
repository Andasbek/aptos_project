# Установка и запуск

## Требования

Рекомендуемая среда:

- Python 3.10 или новее.
- Windows PowerShell или совместимая командная оболочка.
- Node.js 64-bit для frontend.
- GPU NVIDIA с CUDA - опционально, но желательно для ускорения обучения.
- Датасет APTOS 2019, размещенный в папке `data/`.

## Установка ML-окружения

Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Файл `requirements.txt` содержит:

```text
torch
torchvision
torchaudio
pandas
numpy
matplotlib
scikit-learn
pillow
tqdm
opencv-python
```

Если нужна CUDA-версия PyTorch, установите `torch`, `torchvision` и
`torchaudio` командой с официального сайта PyTorch под вашу версию CUDA, затем
установите остальные зависимости.

## Проверка структуры данных

Минимальная структура для обучения:

```text
data/
+-- train.csv
+-- train_images/
```

В текущем проекте также могут присутствовать:

```text
data/
+-- test.csv
+-- sample_submission.csv
+-- test_images/
```

## Обучение модели

```powershell
python src/train.py --model cnn
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
python src/train.py --model mobilenet_v2
```

Для pretrained-моделей можно заморозить backbone и обучать только
классификатор:

```powershell
python src/train.py --model resnet50 --freeze_backbone
python src/train.py --model efficientnet_b0 --freeze_backbone
python src/train.py --model mobilenet_v2 --freeze_backbone
```

Лучшие checkpoint сохраняются в:

```text
results/saved_models/best_<model>.pth
```

## Оценка модели

После обучения:

```powershell
python src/evaluate.py --model resnet50
```

Скрипт загрузит checkpoint из `results/saved_models/best_resnet50.pth`,
посчитает метрики на validation split и сохранит результаты в
`results/metrics/` и `results/figures/`.

## Сравнение моделей

После оценки нескольких моделей:

```powershell
python src/compare_models.py
```

Команда собирает файлы `*_summary.json` из `results/metrics/`, создает
`model_comparison.csv` и график `model_comparison.png`.

## Запуск backend

Backend нужен для web-прототипа и inference через API. Он использует отдельный
`backend/requirements.txt`.

Из корня проекта:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend будет доступен на:

```text
http://localhost:8000
```

Проверка:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/model-info
```

Для `/predict` нужен checkpoint:

```text
results/saved_models/best_resnet50.pth
```

Если файла нет, endpoint вернет `503` с сообщением, куда положить checkpoint.

## Запуск frontend

Во втором терминале:

```powershell
cd frontend
npm install
npm run dev
```

Frontend будет доступен на:

```text
http://localhost:3000
```

Next.js использует native SWC binary, поэтому на Windows нужен 64-bit Node.js.
32-bit Node.js может падать при `npm run dev` или `npm run build`.

## Настройка URL backend

Frontend читает API base URL из:

```text
frontend/src/lib/api.ts
```

Значение по умолчанию:

```text
http://localhost:8000
```

Переопределить URL можно через переменную окружения:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL="http://localhost:8000"
npm run dev
```

## Проверка GPU

Проект автоматически выбирает устройство:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

При доступной CUDA обучение использует mixed precision через `torch.amp` и
включает `pin_memory=True` в `DataLoader`.
