# Данные

## Источник

Проект использует датасет **APTOS 2019 Blindness Detection**. В нем изображения глазного дна размечены по пяти стадиям диабетической ретинопатии.

## Ожидаемые файлы

Для обучения обязательны:

```text
data/
├── train.csv
└── train_images/
```

Для тестовой выборки и Kaggle submission могут использоваться:

```text
data/
├── test.csv
├── sample_submission.csv
└── test_images/
```

## Формат `train.csv`

| Колонка | Тип | Описание |
|---|---|---|
| `id_code` | string | Имя файла изображения без расширения `.png`. |
| `diagnosis` | int | Класс заболевания от 0 до 4. |

Пример соответствия:

```text
id_code=000c1434d8d7
image=data/train_images/000c1434d8d7.png
diagnosis=2
```

## Размеры выборок в текущем каталоге

| Выборка | Количество записей | Каталог изображений |
|---|---:|---|
| Train | 3662 | `data/train_images/` |
| Test | 1928 | `data/test_images/` |

## Распределение классов в `train.csv`

| Класс | Название | Количество |
|---:|---|---:|
| 0 | No DR | 1805 |
| 1 | Mild | 370 |
| 2 | Moderate | 999 |
| 3 | Severe | 193 |
| 4 | Proliferative DR | 295 |

Выборка несбалансирована: класс `0` встречается чаще всего, классы `3` и `4` представлены заметно слабее. Для компенсации дисбаланса в `src/train.py` используется `CrossEntropyLoss` с весами классов, рассчитанными функцией `get_class_weights()` из `src/dataset.py`.

## Разделение train/validation

Функция `get_dataloaders()` выполняет `train_test_split`:

- `test_size=0.2`
- `random_state=42`
- `stratify=df["diagnosis"]`

Это сохраняет пропорции классов в обучающей и валидационной частях.

## Предобработка и аугментации

Обучающие изображения:

1. Resize до `224x224`.
2. RandomHorizontalFlip с вероятностью `0.5`.
3. RandomRotation до `15` градусов.
4. ColorJitter по яркости, контрасту, насыщенности и оттенку.
5. ToTensor.
6. Normalize с ImageNet mean/std.

Валидационные изображения:

1. Resize до `224x224`.
2. ToTensor.
3. Normalize с ImageNet mean/std.
