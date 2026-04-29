from io import BytesIO

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def read_image(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
        return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def class_mapping() -> dict[int, str]:
    return {index: class_name for index, class_name in enumerate(CLASS_NAMES)}

