import numpy as np
import pathlib
import argparse


def sample_category(category, gen: np.random.Generator, count: int = 500):
    idx = gen.integers(0, category.shape[0] - 1, size=count)
    return category[idx]


def split_category(category, split: list[int] = [450, 50]):
    cat1 = category[: split[0]]
    cat2 = category[split[0] :]
    return cat1, cat2


def make_sets(directory, gen: np.random.Generator):
    gallery = {}
    search = {}
    for file in directory:
        category_name = file.stem.rpartition("_")[-1]
        category = np.load(file, allow_pickle=False)

        partial_category = sample_category(category.reshape(-1, 28, 28), gen)
        for_gallery, for_search = split_category(partial_category)

        gallery[category_name] = for_gallery
        search[category_name] = for_search
    return gallery, search


def save_subset(gallery, file_name: str = "subset.npz"):
    np.savez_compressed(file_name, **gallery)


def save_datasets(data_directory, gen, gallery_name="gallery", search_name="search"):
    gallery, search = make_sets(data_directory, gen)
    save_subset(gallery, f"subset_{gallery_name}.npz")
    save_subset(search, f"subset_{search_name}.npz")


def main(args):
    image_directory = pathlib.Path(args.image_directory)
    image_files = [
        file
        for file in image_directory.iterdir()
        if (file.stem[0] != ".") and (file.stem.rpartition("_")[-1] != "triangle")
    ]
    triangle = [image_directory.joinpath("full_numpy_bitmap_triangle.npy")]

    gen = np.random.default_rng(seed=253532)
    save_datasets(image_files, gen)
    save_datasets(
        triangle, gen, gallery_name="triangle_gallery", search_name="triangle_search"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image-directory", type=str, help="The directory containing the image .npy"
    )

    args = parser.parse_args()

    main(args)
