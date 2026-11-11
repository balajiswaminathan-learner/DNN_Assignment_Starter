from pathlib import Path
import urllib.request

import numpy as np
import torch


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def load_tiny_shakespeare(path: str | Path = "data/input.txt"):
    """Load the assignment text, downloading it once if not already present."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    text = path.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    if len(vocabulary) != 65:
        raise ValueError(f"Expected 65 characters; found {len(vocabulary)}")
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded = np.fromiter((char_to_id[c] for c in text), dtype=np.int64)
    split = int(0.9 * len(encoded))
    return (
        text,
        vocabulary,
        char_to_id,
        torch.from_numpy(encoded[:split]),
        torch.from_numpy(encoded[split:]),
    )


def get_batch(data: torch.Tensor, batch_size: int, context: int, device):
    """Sample next-character windows without crossing the selected split."""
    if len(data) <= context:
        raise ValueError("The token split must be longer than the context")
    starts = torch.randint(len(data) - context, (batch_size,))
    x = torch.stack([data[i : i + context] for i in starts])
    y = torch.stack([data[i + 1 : i + context + 1] for i in starts])
    return x.to(device), y.to(device)
