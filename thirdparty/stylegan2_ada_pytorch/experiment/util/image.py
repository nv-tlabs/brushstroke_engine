import numpy as np
import PIL.Image
import torch

def convert_geom_data(image_path):
    """
    Read an image and convert it to the format for feeding the geometry encoder.
    @param image_path: The path to an image of shape [W, H, C]
    @return a torch.Tensor of shape [1, 1, W, H], and the value lies within the range [-1, 1]
    """
    geom_image = np.asarray(PIL.Image.open(image_path))
    assert len(geom_image.shape) == 3
    geom_image = np.transpose(geom_image, (2, 0, 1))
    return torch.from_numpy(geom_image[:1, :, :]).to(torch.float32).unsqueeze(0) / 255.0