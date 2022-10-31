import gzip
import math
import struct
import sys
from typing import Any, Iterable, Iterator, List, Optional, Sized, Union

import numpy as np

from .autograd import Tensor


def byteorder():
    if sys.byteorder == "little":
        return "<"
    return ">"


def parse_image(filename):
    image_bytes = gzip.open(filename, "rb").read()

    _, num_of_images, num_of_rows, num_of_columns = struct.unpack(
        f"{byteorder()}4l", image_bytes[:16]
    )
    pixels = []
    for (pixel,) in struct.iter_unpack(f"{byteorder()}B", image_bytes[16:]):
        pixels.append(pixel)

    return np.array(pixels, dtype=np.float32).reshape(-1, 784)


def parse_label(filename):
    image_bytes = gzip.open(filename, "rb").read()

    _, number_of_items = struct.unpack(f"{byteorder()}2l", image_bytes[:8])

    labels = []
    for (label,) in struct.iter_unpack(f"{byteorder()}B", image_bytes[8:]):
        labels.append(label)

    return np.array(labels, dtype=np.uint8)


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    x = parse_image(image_filename) / 255
    y = parse_label(label_filename)

    return x, y


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """Horizonally flip an image, specified as n H x W x C NDArray.

        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return np.flip(img, 1)

        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.

        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        h, w, _ = img.shape

        padded_img = np.pad(
            img,
            ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
            "constant",
        )

        return padded_img[
            self.padding + shift_x : self.padding + shift_x + h,
            self.padding + shift_y : self.padding + shift_y + w,
            :,
        ]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len(dataset)), range(batch_size, len(dataset), batch_size)
            )

    def __iter__(self):
        self.batch_index = 0
        if self.shuffle:
            ranges = np.arange(len(self.dataset))
            np.random.shuffle(ranges)
            self.ordering = np.array_split(
                ranges, range(self.batch_size, len(self.dataset), self.batch_size)
            )
        return self

    def __next__(self):
        if self.batch_index >= len(self.ordering):
            raise StopIteration
        batch = self.ordering[self.batch_index]
        self.batch_index += 1

        return tuple(Tensor(data) for data in self.dataset[batch])


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.X, self.y = parse_mnist(image_filename, label_filename)

        self.X = self.X.reshape(-1, 28, 28, 1)
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        data = self.X[index]
        label = self.y[index]

        if self.transforms is None:
            return data.reshape(-1, 784), label

        for transform in self.transforms:
            data = transform(data)

        return data.reshape(-1, 784), label

    def __len__(self) -> int:
        return self.X.shape[0]


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple(a[i] for a in self.arrays)
