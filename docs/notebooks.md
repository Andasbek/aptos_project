# Ноутбуки

В каталоге `notebooks/` находятся исследовательские Jupyter Notebook-файлы. Они дополняют production-like скрипты из `src/` и подходят для анализа, визуализации и отдельных экспериментов.

## Список ноутбуков

| Ноутбук | Назначение |
|---|---|
| `01_eda.ipynb` | Первичный анализ данных: структура датасета, распределение классов, просмотр изображений. |
| `02_cnn.ipynb` | Эксперименты с пользовательской CNN-моделью. |
| `03_mobilenetv2.ipynb` | Эксперименты с MobileNetV2. |
| `04_efficientnet_b0.ipynb` | Эксперименты с EfficientNet-B0. |
| `05_resnet50.ipynb` | Эксперименты с ResNet50. |

## Как использовать

Активируйте окружение проекта и запустите Jupyter:

```powershell
.\.venv\Scripts\activate
pip install notebook
jupyter notebook
```

Ноутбуки удобно использовать для исследования гипотез, а финальные воспроизводимые запуски лучше выполнять через CLI-скрипты:

```powershell
python src/train.py --model resnet50
python src/evaluate.py --model resnet50
```

Так результаты будут сохранены в стандартной структуре `results/`.
