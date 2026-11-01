import numpy as np
import pathlib
import search_core
from keras.applications import MobileNetV2
import torch.utils.data as torch_data
import shutil
import os


def load_dataset(path: pathlib.Path):
    data = np.load(path, allow_pickle=False)
    return data


def find_paths(directory: pathlib.Path):
    return [file for file in directory.iterdir() if file.suffix == ".npz"]


def preprocess_data(data):
    return search_core.to_model_input(data)


def embed_data(data, model):
    embedded_data = []
    loader = torch_data.DataLoader(data, batch_size=64)
    for batch in loader:
        batch = batch.numpy()
        embedded = model.predict(batch, batch_size=64)
        norms = np.linalg.norm(embedded, axis=-1, keepdims=True)
        embedded_data.append(embedded / norms)
    return np.concatenate(embedded_data, axis=0)


def save_embeddings(embeddings, file_name="index/embeddings"):
    np.save(file_name + ".npy", embeddings)


def move_images():
    if not os.path.isdir("index"):
        os.mkdir("./index")

    shutil.move("subset_gallery.npz", "index/subset_gallery.npz")


def main(file):

    model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(96, 96, 3), pooling="avg"
    )
    model.trainable = False

    data = load_dataset(file)

    dat = []
    for key in data.keys():
        preprocessed_data = preprocess_data(data[key])
        embeddings = embed_data(preprocessed_data, model)
        dat.append(embeddings)
    save_embeddings(np.concatenate(dat, axis=0))
    move_images()


if __name__ == "__main__":
    path = pathlib.Path("subset_gallery.npz")
    main(path)
