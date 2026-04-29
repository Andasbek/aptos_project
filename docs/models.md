# Модели

Проект реализует четыре архитектуры. Все модели возвращают logits для пяти классов.

## Custom CNN

Внутренняя сверточная сеть из `src/models.py`.

Состав:

- 5 сверточных блоков;
- BatchNorm после каждой свертки;
- ReLU activation;
- MaxPool в первых четырех блоках;
- AdaptiveAvgPool2d в конце feature extractor;
- классификатор с Dropout и двумя Linear слоями.

Параметры по текущему обучению:

| Показатель | Значение |
|---|---:|
| Всего параметров | 1 702 181 |
| Обучаемых параметров | 1 702 181 |

Запуск:

```powershell
python src/train.py --model cnn
```

## ResNet50

Pretrained ResNet50 из `torchvision.models`.

Изменение в проекте:

- исходный `fc` заменяется на `Linear(in_features, 5)`.

Параметры по текущему обучению:

| Показатель | Значение |
|---|---:|
| Всего параметров | 23 518 277 |
| Обучаемых параметров | 23 518 277 |

Запуск:

```powershell
python src/train.py --model resnet50
```

С замороженным backbone:

```powershell
python src/train.py --model resnet50 --freeze_backbone
```

## EfficientNet-B0

Pretrained EfficientNet-B0 из `torchvision.models`.

Изменение в проекте:

- `classifier[1]` заменяется на `Linear(in_features, 5)`.

Параметры по текущему обучению:

| Показатель | Значение |
|---|---:|
| Всего параметров | 4 013 953 |
| Обучаемых параметров | 4 013 953 |

Запуск:

```powershell
python src/train.py --model efficientnet_b0
```

С замороженным backbone:

```powershell
python src/train.py --model efficientnet_b0 --freeze_backbone
```

## MobileNetV2

Pretrained MobileNetV2 из `torchvision.models`.

Изменение в проекте:

- `classifier[1]` заменяется на `Linear(in_features, 5)`.

Параметры по текущему обучению:

| Показатель | Значение |
|---|---:|
| Всего параметров | 2 230 277 |
| Обучаемых параметров | 2 230 277 |

Запуск:

```powershell
python src/train.py --model mobilenet_v2
```

С замороженным backbone:

```powershell
python src/train.py --model mobilenet_v2 --freeze_backbone
```

## Заморозка backbone

Флаг `--freeze_backbone` доступен для pretrained-моделей. Он отключает обучение всех параметров, кроме классификационной головы:

- ResNet50: `fc`;
- EfficientNet-B0: `classifier`;
- MobileNetV2: `classifier`.

Это полезно для быстрых baseline-экспериментов или при ограниченных вычислительных ресурсах. Полное дообучение обычно дает больше гибкости, но требует больше времени и GPU-памяти.
