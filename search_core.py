import os

os.environ["KERAS_BACKEND"] = "torch"
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import torchvision.transforms.v2 as v2
import torch.utils.data as data
import torch
import pathlib
from PIL import Image


def to_model_input(image):
    transformed = (
        transform_image(torch.tensor(image).unsqueeze(1)).permute((0, 2, 3, 1)).numpy()
    )

    return preprocess_input(transformed)


def transform_image(image):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Resize((96, 96)),
            v2.Grayscale(3),
        ]
    )

    return transform(image)


def transform_query_image(image):
    image = Image.fromarray(image)
    if image.mode == "RGBA":
        # Make a new RGBA image that is white
        background = Image.new("RGBA", image.size, "white")
        # Replace transparency with white
        image = image.alpha_composite(background, image)

    transform = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Grayscale(3)]
    )


def single_to_model_input(image):
    pass
