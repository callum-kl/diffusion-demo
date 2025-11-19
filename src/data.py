from typing import Tuple, Optional
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

"""
src/data.py

Utilities to fetch MNIST dataset.

Provides:
- load_mnist(...): download (via sklearn.fetch_openml) and return train/test arrays.

Example:
    x_train, y_train, x_test, y_test = load_mnist(flatten=False, normalize=True)
"""


def load_mnist(
    *,
    data_home: Optional[str] = None,
    flatten: bool = True,
    normalize: bool = True,
    one_hot: bool = False,
    test_size: int = 10000,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch MNIST and return (x_train, y_train, x_test, y_test).

    Args:
        data_home: optional directory to store/download the dataset (passed to sklearn).
        flatten: if True return shape (N, 784); if False return (N, 1, 28, 28).
        normalize: if True scale pixel values to [0, 1] (float32).
        one_hot: if True convert labels to one-hot vectors of length 10.
        test_size: number of samples to reserve for test set (default 10000).
        random_state: RNG seed for splitting.

    Returns:
        x_train, y_train, x_test, y_test
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=data_home)
    X = mnist["data"].astype(np.float32)  # shape (70000, 784)
    y = mnist["target"].astype(np.int64)  # shape (70000,)

    if normalize:
        X /= 255.0

    # split into train/test (original MNIST has 60k train / 10k test; we mimic that)
    # fetch_openml returns 70000 samples (train+test). We stratify by label.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if not flatten:
        X_train = X_train.reshape((-1, 1, 28, 28))
        X_test = X_test.reshape((-1, 1, 28, 28))

    if one_hot:

        def to_one_hot(labels: np.ndarray) -> np.ndarray:
            out = np.zeros((labels.shape[0], 10), dtype=np.float32)
            out[np.arange(labels.shape[0]), labels] = 1.0
            return out

        y_train = to_one_hot(y_train)
        y_test = to_one_hot(y_test)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # quick sanity check and save to npz when run as script
    x_tr, y_tr, x_te, y_te = load_mnist(flatten=False, normalize=True)
    np.savez_compressed(
        "mnist_sample.npz", x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te
    )
    print(
        "Saved mnist_sample.npz with shapes:",
        x_tr.shape,
        y_tr.shape,
        x_te.shape,
        y_te.shape,
    )
