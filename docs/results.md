# Результаты

## Где лежат результаты

Все артефакты сохраняются в `results/`:

```text
results/
├── figures/
│   ├── *_training_history.png
│   ├── *_confusion_matrix.png
│   └── model_comparison.png
├── metrics/
│   ├── *_history.csv
│   ├── *_training_summary.json
│   ├── *_classification_report.txt
│   ├── *_summary.json
│   └── model_comparison.csv
└── saved_models/
    └── best_*.pth
```

## Текущие результаты обучения

В текущем каталоге уже есть training summary для четырех моделей. Значения ниже взяты из `results/metrics/*_training_summary.json`.

| Модель | Best val accuracy | Final train accuracy | Final val accuracy | Final train loss | Final val loss | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Custom CNN | 0.6889 | 0.6767 | 0.6876 | 1.0770 | 1.1043 | 1 702 181 |
| ResNet50 | 0.8336 | 0.9594 | 0.8213 | 0.1370 | 1.1516 | 23 518 277 |
| EfficientNet-B0 | 0.8226 | 0.9256 | 0.7940 | 0.2484 | 1.2262 | 4 013 953 |
| MobileNetV2 | 0.7940 | 0.9041 | 0.7885 | 0.3477 | 1.0786 | 2 230 277 |

## Интерпретация

По текущим training summary лучший результат по `best_val_accuracy` показывает **ResNet50**: `0.8336`.

EfficientNet-B0 близок по качеству, но содержит значительно меньше параметров, чем ResNet50. MobileNetV2 еще легче, но в текущем эксперименте уступает по validation accuracy. Custom CNN заметно проще и полезна как baseline без transfer learning.

## Важное замечание

Файлы `*_training_summary.json` описывают результаты обучения. Для полноценного сравнения по `precision`, `recall` и `F1-score` нужно запустить:

```powershell
python src/evaluate.py --model cnn
python src/evaluate.py --model resnet50
python src/evaluate.py --model efficientnet_b0
python src/evaluate.py --model mobilenet_v2
python src/compare_models.py
```

После этого появятся `*_summary.json`, `*_classification_report.txt`, confusion matrix для каждой модели и итоговая таблица `model_comparison.csv`.

## Как читать графики

`*_training_history.png` содержит:

- train и validation loss;
- train и validation accuracy;
- learning rate по эпохам, если колонка `learning_rate` есть в history CSV.

Если train accuracy растет, а validation accuracy стагнирует или падает, модель может переобучаться. В таком случае стоит попробовать:

- больше аугментаций;
- заморозку backbone;
- меньше эпох;
- регуляризацию;
- другой learning rate;
- более аккуратную предобработку изображений глазного дна.
