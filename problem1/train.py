import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x, y, batch_size: int, shuffle: bool):
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_classifier(
    model,
    train_loader,
    val_loader,
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-3,
):
    """Train with Adam and return epoch times and the best validation weights."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    epoch_seconds = []
    best_val_loss = float("inf")
    best_state = None

    for _ in range(epochs):
        started = time.perf_counter()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        epoch_seconds.append(time.perf_counter() - started)

        model.eval()
        total_loss, total_items = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                batch_loss = criterion(model(x), y)
                total_loss += batch_loss.item() * len(y)
                total_items += len(y)
        val_loss = total_loss / total_items
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    return model, epoch_seconds, best_val_loss


def evaluate_classifier(model, loader, device: torch.device):
    model.eval()
    labels, predictions = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(labels, predictions),
    }


def train_regressor(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs: int,
    batch_size: int,
    device: torch.device,
    learning_rate: float = 1e-3,
    gradient_clip_norm: float | None = None,
):
    """Train one adding-problem run and return per-epoch train/validation MSE."""
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    model.to(device)
    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    x_val = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_mse": [], "val_mse": [], "sec_per_epoch": []}

    for _ in range(epochs):
        started = time.perf_counter()
        model.train()
        total_loss, total_items = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.nn.functional.mse_loss(prediction, y)
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip_norm
                )
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_items += len(y)
        history["train_mse"].append(total_loss / total_items)
        history["sec_per_epoch"].append(time.perf_counter() - started)
        model.eval()
        with torch.no_grad():
            val_mse = torch.nn.functional.mse_loss(model(x_val), y_val).item()
        history["val_mse"].append(val_mse)
    return model, history


def predict_mse(model, x, y, device: torch.device) -> float:
    model.eval()
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    with torch.no_grad():
        return torch.nn.functional.mse_loss(model(x), y).item()
