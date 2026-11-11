from pathlib import Path

import numpy as np


HAR_CHANNELS = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)


def make_adding(n: int, sequence_length: int, seed: int):
    """Create the assignment's deterministic two-marker adding dataset."""
    if n <= 0 or sequence_length < 2:
        raise ValueError("n must be positive and sequence_length must be >= 2")
    rng = np.random.default_rng(seed)
    x = np.zeros((n, sequence_length, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, sequence_length))
    first = rng.integers(0, sequence_length // 2, n)
    second = rng.integers(sequence_length // 2, sequence_length, n)
    rows = np.arange(n)
    x[rows, first, 1] = 1.0
    x[rows, second, 1] = 1.0
    y = x[rows, first, 0] + x[rows, second, 0]
    return x, y.astype(np.float32)


def assert_adding_data(x: np.ndarray, y: np.ndarray) -> None:
    """Check marker count, marker halves and target construction."""
    assert x.ndim == 3 and x.shape[-1] == 2
    assert y.shape == (x.shape[0],)
    markers = x[:, :, 1]
    length = x.shape[1]
    assert np.all(markers.sum(axis=1) == 2)
    assert np.all(markers[:, : length // 2].sum(axis=1) == 1)
    assert np.all(markers[:, length // 2 :].sum(axis=1) == 1)
    np.testing.assert_allclose(y, (x[:, :, 0] * markers).sum(axis=1))


def _find_har_root(root: Path) -> Path:
    candidates = (root, root / "UCI HAR Dataset")
    for candidate in candidates:
        if (candidate / "train" / "y_train.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find UCI HAR Dataset under {root}. "
        "Expected train/y_train.txt and train/Inertial Signals/."
    )


def _load_har_split(root: Path, split: str):
    signals_dir = root / split / "Inertial Signals"
    signals = [
        np.loadtxt(signals_dir / f"{channel}_{split}.txt", dtype=np.float32)
        for channel in HAR_CHANNELS
    ]
    x = np.stack(signals, axis=-1)
    y = np.loadtxt(root / split / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(
        root / split / f"subject_{split}.txt", dtype=np.int64
    )
    if x.shape[1:] != (128, 9) or len(x) != len(y) or len(y) != len(subjects):
        raise ValueError(f"Unexpected or inconsistent HAR {split} shapes")
    return x, y, subjects


def load_har(root: str | Path, seed: int):
    """Load HAR and make the required subject-wise train/validation split.

    Normalization is fitted only on the training subjects. The official test
    set remains untouched until the caller evaluates a selected model.
    """
    root = _find_har_root(Path(root))
    x_train_all, y_train_all, subjects = _load_har_split(root, "train")
    x_test, y_test, _ = _load_har_split(root, "test")

    train_subjects = np.unique(subjects)
    if len(train_subjects) != 21:
        raise ValueError(f"Expected 21 HAR training subjects; found {len(train_subjects)}")
    val_subjects = np.random.default_rng(seed).choice(
        train_subjects, 4, replace=False
    )
    is_val = np.isin(subjects, val_subjects)
    x_train, y_train = x_train_all[~is_val], y_train_all[~is_val]
    x_val, y_val = x_train_all[is_val], y_train_all[is_val]

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    return {
        "x_train": (x_train - mean) / std,
        "y_train": y_train,
        "x_val": (x_val - mean) / std,
        "y_val": y_val,
        "x_test": (x_test - mean) / std,
        "y_test": y_test,
        "train_subjects": np.setdiff1d(train_subjects, val_subjects),
        "val_subjects": np.sort(val_subjects),
        "mean": mean,
        "std": std,
    }
