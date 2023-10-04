import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from kornia.enhance import denormalize


def torch2numpy(image_torch, mean, std):
    image_torch = denormalize(
        image_torch[None],
        torch.tensor(mean).to(image_torch.device),
        torch.tensor(std).to(image_torch.device),
    )[0]
    image_np = image_torch.permute(1, 2, 0).cpu().numpy()
    image_np = np.rint(np.clip(image_np, 0.0, 1.0) * 255).astype(np.uint8)
    return image_np


def add_text_to_image(image, text, position=(10, 10), font_color=(255, 255, 255)):
    """
    Add text to an image and return the modified image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array (HWC format, dtype=uint8).
        text (str): The text to add to the image.
        position (tuple): Position to place the text (x, y).
        font_color (tuple): RGB color for the text.

    Returns:
        numpy.ndarray: Image with text added (HWC format, dtype=uint8).
    """
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Load a font (you may need to specify the font file)
    font = ImageFont.load_default()

    # Calculate text size and position
    text_size = draw.textsize(text, font=font)
    text_x = position[0]
    text_y = position[1]

    # Draw the text on the image
    draw.text((text_x, text_y), text, fill=font_color, font=font)

    # Convert the modified PIL Image back to a NumPy array
    modified_image = np.array(pil_image)

    return modified_image
