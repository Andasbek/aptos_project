import torch
from PIL import Image
from torchvision import transforms

from .model_loader import DEVICE, load_model
from .utils import CLASS_NAMES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def predict_image(image: Image.Image) -> dict:
    model = load_model()
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_class = int(torch.argmax(probabilities_tensor).item())
    confidence = float(probabilities_tensor[predicted_class].item())
    probabilities = {
        class_name: float(probabilities_tensor[index].item())
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": confidence,
        "probabilities": probabilities,
    }

