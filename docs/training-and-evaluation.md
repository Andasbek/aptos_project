# Обучение и оценка

## Конфигурация обучения

Параметры по умолчанию находятся в `src/config.py`:

| Параметр | Значение |
|---|---:|
| `IMAGE_SIZE` | 224 |
| `BATCH_SIZE` | 16 |
| `NUM_EPOCHS` | 15 |
| `LEARNING_RATE` | `1e-4` |
| `NUM_CLASSES` | 5 |
| `RANDOM_SEED` | 42 |

## Запуск обучения

```powershell
python src/train.py --model cnn
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
python src/train.py --model mobilenet_v2
```

Аргументы:

| Аргумент | Обязательный | Описание |
|---|---|---|
| `--model` | Да | Одна из моделей: `cnn`, `resnet50`, `efficientnet_b0`, `mobilenet_v2`. |
| `--freeze_backbone` | Нет | Замораживает backbone pretrained-модели и обучает только классификатор. |

## Что происходит во время обучения

1. Устанавливается seed для воспроизводимости.
2. Создаются папки `results/figures`, `results/metrics`, `results/saved_models`.
3. Данные делятся на train/validation.
4. Создается модель.
5. Рассчитываются веса классов.
6. Создается `CrossEntropyLoss(weight=class_weights)`.
7. Используется optimizer Adam.
8. Используется scheduler `ReduceLROnPlateau`.
9. При CUDA включается mixed precision.
10. Лучшая модель сохраняется по максимальной `val_accuracy`.

## Артефакты обучения

Для модели `<model>` создаются:

```text
results/
├── figures/
│   └── <model>_training_history.png
├── metrics/
│   ├── <model>_history.csv
│   └── <model>_training_summary.json
└── saved_models/
    └── best_<model>.pth
```

## Запуск оценки

```powershell
python src/evaluate.py --model cnn
python src/evaluate.py --model resnet50
python src/evaluate.py --model efficientnet_b0
python src/evaluate.py --model mobilenet_v2
```

Скрипт ожидает, что файл `results/saved_models/best_<model>.pth` уже существует.

## Метрики оценки

Оценка проводится на validation split и сохраняет:

- accuracy;
- weighted precision;
- weighted recall;
- weighted F1-score;
- classification report;
- confusion matrix.

Артефакты оценки:

```text
results/
├── figures/
│   └── <model>_confusion_matrix.png
└── metrics/
    ├── <model>_classification_report.txt
    └── <model>_summary.json
```

## Сравнение моделей

После оценки моделей:

```powershell
python src/compare_models.py
```

Скрипт читает `results/metrics/*_summary.json` и создает:

```text
results/
├── figures/
│   └── model_comparison.png
└── metrics/
    └── model_comparison.csv
```

Если summary-файлов нет, скрипт завершится с ошибкой `FileNotFoundError`.

## Рекомендованный порядок эксперимента

1. Проверить наличие данных.
2. Обучить легкую baseline-модель:

```powershell
python src/train.py --model cnn
python src/evaluate.py --model cnn
```

3. Обучить transfer learning модели:

```powershell
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
python src/train.py --model mobilenet_v2
```

4. Оценить каждую модель:

```powershell
python src/evaluate.py --model resnet50
python src/evaluate.py --model efficientnet_b0
python src/evaluate.py --model mobilenet_v2
```

5. Сравнить результаты:

```powershell
python src/compare_models.py
```
