# Установка и запуск

## Требования

Рекомендуемая среда:

- Python 3.10 или новее.
- Windows PowerShell или совместимая командная оболочка.
- Node.js 64-bit для frontend (рекомендуется 20.19+ или 22 LTS).
- GPU NVIDIA с CUDA — опционально, но желательно для ускорения обучения.
- Датасет APTOS 2019, размещенный в папке `data/`.
- OpenAI API-ключ — опционально, нужен для функций объяснения и чата
  ассистента в web-прототипе. Без ключа `/predict` продолжает работать.

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

Backend нужен для web-прототипа. Он использует отдельный
`backend/requirements.txt`, в котором перечислены FastAPI, PyTorch для
inference, OpenAI SDK и `python-dotenv`.

Из корня проекта:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Настройка .env (для LLM)

Скопируйте шаблон и впишите ключ:

```powershell
copy .env.example .env
```

Откройте `backend/.env` и заполните:

```text
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_MODEL` опционален, по умолчанию `gpt-4o-mini`. Файл `.env` не
коммитится — он исключён в `.gitignore`. Без ключа `/predict` будет
работать, а `/explain` и `/chat` вернут `503` с понятным сообщением.

### Запуск сервера

```powershell
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

`/model-info` вернёт список всех четырёх моделей с признаком наличия
чекпоинта (`checkpoint_exists`), флаг `llm_enabled` и поддерживаемые языки.

Для `/predict` нужен хотя бы один checkpoint в `results/saved_models/`.
По умолчанию используется `best_resnet50.pth`. Если выбранной модели нет,
endpoint вернёт `503` с сообщением, куда положить файл.

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

Next.js использует native SWC binary, поэтому на Windows нужен 64-bit
Node.js. 32-bit Node.js может падать при `npm run dev` или `npm run build`.

`npm install` может выдать `EBADENGINE` для пакета `eslint-visitor-keys` —
это лишь предупреждение, на работу не влияет. При желании можно обновить
Node до 20.19+ или 22 LTS.

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

## Языки и модели в UI

После запуска frontend в правом верхнем углу появится переключатель
языка (English / Русский / Қазақша). Он сохраняет выбор в `localStorage`
и одновременно меняет UI и язык ответов LLM.

В блоке загрузки изображения есть селектор модели — он динамически
заполняется из `/model-info`. Если соответствующего файла
`best_<model>.pth` нет, пункт помечается как `(missing)` и недоступен.

## Проверка GPU

Проект автоматически выбирает устройство:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

При доступной CUDA обучение использует mixed precision через `torch.amp` и
включает `pin_memory=True` в `DataLoader`. Backend использует то же
устройство для inference.

## Типичные проблемы

### `Fatal error in launcher: Unable to create process using ...`

Случается, если `.venv` создан в одной папке, а потом папка проекта была
перенесена. Скрипты venv хранят абсолютный путь к python.exe, при переносе
ломаются. Решение — пересоздать venv:

```powershell
Remove-Item -Recurse -Force .\backend\.venv
py -m venv .\backend\.venv
.\backend\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### `OPENAI_API_KEY is not set`

Откройте `backend/.env` и убедитесь, что строка `OPENAI_API_KEY=sk-...`
заполнена. Перезапустите uvicorn — переменные читаются один раз при
импорте `backend/app/llm.py`.

### `Checkpoint for '<model>' was not found`

`/predict` вызвали с моделью, для которой нет файла в
`results/saved_models/`. Либо обучите её:

```powershell
python src/train.py --model <model>
```

либо выберите другую модель в UI.
