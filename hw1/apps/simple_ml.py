import gzip
import math
import struct
import sys

import numpy as np

sys.path.append("python/")
import needle as ndl


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


def softmax_loss(Z, y):
    """Return softmax loss.  Note that for the purposes of this assignment, you don't need to worry
    about "nicely" scaling the numerical properties of the log-sum-exp computation, but can just
    compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    return (
        ndl.summation(ndl.log(ndl.summation(ndl.exp(Z), axes=1)) - ndl.summation(Z * y, axes=1))
        / Z.shape[0]
    )


def one_hot(y, k):
    return np.eye(k)[y]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the weights W1 and W2
    (with no bias terms):

        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    batch_num = math.ceil(X.shape[0] / batch)
    for ind in range(batch_num):
        data = ndl.Tensor(X[ind * batch : (ind + 1) * batch, :])
        label = y[ind * batch : (ind + 1) * batch]
        label_one_hot = ndl.Tensor(one_hot(label, W2.shape[1]))

        loss = softmax_loss(ndl.relu(data @ W1) @ W2, label_one_hot)
        loss.backward()

        W1 = ndl.Tensor((W1 - lr * W1.grad).numpy())
        W2 = ndl.Tensor((W2 - lr * W2.grad).numpy())

    return W1, W2


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error."""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
