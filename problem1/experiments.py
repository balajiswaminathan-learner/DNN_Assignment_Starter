from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from problem1.data import assert_adding_data, load_har, make_adding
from problem1.models import (
    RecurrentClassifier,
    RecurrentRegressor,
    count_parameters,
    run_recurrent_module,
)
from problem1.train import (
    evaluate_classifier,
    make_loader,
    predict_mse,
    set_seed,
    train_classifier,
    train_regressor,
)
from results import append_result


CELLS = ("RNN", "LSTM", "GRU")


def parameter_count_reference(input_size: int, hidden_size: int) -> dict:
    """Return hand-counted one-bias and PyTorch two-bias cell totals."""
    gates = {"RNN": 1, "LSTM": 4, "GRU": 3}
    return {
        cell: {
            "one_bias": gates[cell]
            * hidden_size
            * (input_size + hidden_size + 1),
            "two_bias": gates[cell]
            * hidden_size
            * (input_size + hidden_size + 2),
        }
        for cell in CELLS
    }


def plot_har_class_balance(labels, figure_dir: str | Path) -> Path:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    classes, counts = np.unique(labels, return_counts=True)
    figure_path = figure_dir / "har_train_class_balance.png"
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(classes + 1, counts, color="#287c76", width=0.72)
    axis.set(xticks=np.arange(1, 7), xlabel="Activity class", ylabel="Training windows")
    axis.set_title("UCI HAR training class balance")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), std


def _append_result(row: dict, results_path: str | Path) -> None:
    append_result(row, path=results_path)


def _plot_confusion_matrix(matrix, figure_dir: str | Path, title: str) -> Path:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "har_best_confusion_matrix.png"
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Windows")
    axis.set(
        xticks=np.arange(6),
        yticks=np.arange(6),
        xticklabels=np.arange(1, 7),
        yticklabels=np.arange(1, 7),
        xlabel="Predicted activity",
        ylabel="True activity",
        title=title,
    )
    axis.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def run_har_comparison(
    har_root: str | Path,
    seeds,
    hidden_size: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
    results_path: str | Path,
    figure_dir: str | Path,
):
    records = []
    best = None
    for seed in seeds:
        har = load_har(har_root, seed=seed)
        train_loader = make_loader(
            har["x_train"], har["y_train"], batch_size, shuffle=True
        )
        val_loader = make_loader(
            har["x_val"], har["y_val"], batch_size, shuffle=False
        )
        test_loader = make_loader(
            har["x_test"], har["y_test"], batch_size, shuffle=False
        )
        for cell in CELLS:
            set_seed(seed)
            model = RecurrentClassifier(cell, 9, hidden_size, 6)
            model, epoch_seconds, val_loss = train_classifier(
                model, train_loader, val_loader, epochs, device, learning_rate=1e-3
            )
            metrics = evaluate_classifier(model, test_loader, device)
            record = {
                "cell": cell,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "params": count_parameters(model),
                "sec_per_epoch": float(np.mean(epoch_seconds)),
                "val_loss": val_loss,
            }
            records.append(record)
            _append_result(
                {
                    "problem": "problem1_har",
                    "model": cell,
                    "seed": seed,
                    "seq_len": 128,
                    "test_acc": record["accuracy"],
                    "macro_f1": record["macro_f1"],
                    "params": record["params"],
                    "sec_per_epoch": record["sec_per_epoch"],
                    "val_loss_nats": val_loss,
                    "notes": f"hidden_size={hidden_size}; batch_size={batch_size}; epochs={epochs}",
                },
                results_path,
            )
            if best is None or val_loss < best["val_loss"]:
                best = {
                    **record,
                    "confusion_matrix": metrics["confusion_matrix"],
                }

    for cell in CELLS:
        cell_records = [record for record in records if record["cell"] == cell]
        accuracy = _mean_std([record["accuracy"] for record in cell_records])
        macro_f1 = _mean_std([record["macro_f1"] for record in cell_records])
        seconds = _mean_std([record["sec_per_epoch"] for record in cell_records])
        print(
            f"HAR {cell}: accuracy={accuracy[0]:.4f} +/- {accuracy[1]:.4f}, "
            f"macro-F1={macro_f1[0]:.4f} +/- {macro_f1[1]:.4f}, "
            f"params={cell_records[0]['params']}, "
            f"sec/epoch={seconds[0]:.3f} +/- {seconds[1]:.3f}"
        )

    figure_path = _plot_confusion_matrix(
        best["confusion_matrix"],
        figure_dir,
        f"Best validation run: {best['cell']} (seed {best['seed']})",
    )
    print(f"Best validation run test confusion matrix saved to {figure_path}")
    return records, best


def run_hidden_size_sweep(
    har_root: str | Path,
    seeds,
    hidden_sizes,
    batch_size: int,
    epochs: int,
    device: torch.device,
    results_path: str | Path,
):
    records = []
    for seed in seeds:
        har = load_har(har_root, seed=seed)
        train_loader = make_loader(
            har["x_train"], har["y_train"], batch_size, shuffle=True
        )
        val_loader = make_loader(
            har["x_val"], har["y_val"], batch_size, shuffle=False
        )
        test_loader = make_loader(
            har["x_test"], har["y_test"], batch_size, shuffle=False
        )
        for cell in ("LSTM", "GRU"):
            for hidden_size in hidden_sizes:
                set_seed(seed)
                model = RecurrentClassifier(cell, 9, hidden_size, 6)
                model, epoch_seconds, val_loss = train_classifier(
                    model, train_loader, val_loader, epochs, device
                )
                metrics = evaluate_classifier(model, test_loader, device)
                record = {
                    "cell": cell,
                    "hidden_size": hidden_size,
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "params": count_parameters(model),
                    "sec_per_epoch": float(np.mean(epoch_seconds)),
                }
                records.append(record)
                _append_result(
                    {
                        "problem": "problem1_har_hidden_sweep",
                        "model": f"{cell}_h{hidden_size}",
                        "seed": seed,
                        "seq_len": 128,
                        "test_acc": record["accuracy"],
                        "macro_f1": record["macro_f1"],
                        "params": record["params"],
                        "sec_per_epoch": record["sec_per_epoch"],
                        "val_loss_nats": val_loss,
                        "notes": f"batch_size={batch_size}; epochs={epochs}",
                    },
                    results_path,
                )

    for cell in ("LSTM", "GRU"):
        for hidden_size in hidden_sizes:
            subset = [
                record
                for record in records
                if record["cell"] == cell and record["hidden_size"] == hidden_size
            ]
            accuracy = _mean_std([record["accuracy"] for record in subset])
            macro_f1 = _mean_std([record["macro_f1"] for record in subset])
            print(
                f"HAR sweep {cell} h={hidden_size}: "
                f"accuracy={accuracy[0]:.4f} +/- {accuracy[1]:.4f}, "
                f"macro-F1={macro_f1[0]:.4f} +/- {macro_f1[1]:.4f}, "
                f"params={subset[0]['params']}"
            )
    return records


def _hidden_gradient_norms(model, x, y, device: torch.device) -> np.ndarray:
    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)
    state = None
    hidden_states = []
    for time_index in range(x.shape[1]):
        _, state = run_recurrent_module(
            model.recurrent, x[:, time_index : time_index + 1], state
        )
        hidden = state[0] if model.cell == "LSTM" else state
        hidden.retain_grad()
        hidden_states.append(hidden)
    prediction = model.regressor(hidden_states[-1][-1]).squeeze(-1)
    F.mse_loss(prediction, y).backward()
    return (
        torch.stack([hidden.grad[-1].norm(dim=1).mean() for hidden in hidden_states])
        .detach()
        .cpu()
        .numpy()
    )


def _plot_adding_learning_curves(histories, figure_dir: str | Path) -> Path:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "adding_t200_learning_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cell in CELLS:
        train = np.asarray(histories[cell]["train_mse"])
        val = np.asarray(histories[cell]["val_mse"])
        epochs = np.arange(1, train.shape[-1] + 1)
        axes[0].plot(epochs, train.mean(axis=0), label=cell)
        axes[1].plot(epochs, val.mean(axis=0), label=cell)
    axes[0].set(title="Training", xlabel="Epoch", ylabel="MSE")
    axes[1].set(title="Validation", xlabel="Epoch", ylabel="MSE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Adding problem learning curves, T=200 (mean over seeds)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _plot_gradient_norms(gradient_norms, figure_dir: str | Path) -> Path:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / "adding_t100_hidden_gradient_norms.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for cell in CELLS:
        for axis, stage in zip(axes, ("before", "after")):
            values = np.maximum(gradient_norms[cell][stage], 1e-12)
            axis.semilogy(np.arange(len(values)), values, label=cell)
    axes[0].set(title="Before training", xlabel="Time step", ylabel="Mean $||\\partial L/\\partial h_t||_2$")
    axes[1].set(title="After training", xlabel="Time step")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    fig.suptitle("Adding problem hidden-state gradient norms, T=100")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def run_adding_trials(
    seeds,
    sequence_lengths,
    hidden_size: int,
    batch_size: int,
    epochs: int,
    train_size: int,
    val_size: int,
    test_size: int,
    device: torch.device,
    results_path: str | Path,
    figure_dir: str | Path,
):
    records = []
    t200_histories = {cell: {"train_mse": [], "val_mse": []} for cell in CELLS}
    gradient_norms = {cell: {} for cell in CELLS}
    test_baselines = {sequence_length: [] for sequence_length in sequence_lengths}

    for sequence_length in sequence_lengths:
        for seed in seeds:
            x_train, y_train = make_adding(train_size, sequence_length, seed * 100 + 1)
            x_val, y_val = make_adding(val_size, sequence_length, seed * 100 + 2)
            x_test, y_test = make_adding(test_size, sequence_length, seed * 100 + 3)
            assert_adding_data(x_train, y_train)
            assert_adding_data(x_val, y_val)
            assert_adding_data(x_test, y_test)
            test_baselines[sequence_length].append(
                float(np.mean((y_test - 1.0) ** 2))
            )
            for cell in CELLS:
                set_seed(seed)
                model = RecurrentRegressor(cell, hidden_size=hidden_size)
                track_gradients = sequence_length == 100 and seed == seeds[0]
                if track_gradients:
                    x_probe, y_probe = make_adding(256, sequence_length, seed * 100 + 4)
                    gradient_norms[cell]["before"] = _hidden_gradient_norms(
                        model, x_probe, y_probe, device
                    )
                model, history = train_regressor(
                    model,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    epochs,
                    batch_size,
                    device,
                )
                test_mse = predict_mse(model, x_test, y_test, device)
                record = {
                    "cell": cell,
                    "sequence_length": sequence_length,
                    "seed": seed,
                    "test_mse": test_mse,
                    "solved": test_mse < 0.05,
                    "params": count_parameters(model),
                    "sec_per_epoch": float(np.mean(history["sec_per_epoch"])),
                }
                records.append(record)
                _append_result(
                    {
                        "problem": "problem1_adding",
                        "model": cell,
                        "seed": seed,
                        "seq_len": sequence_length,
                        "test_mse": test_mse,
                        "params": record["params"],
                        "sec_per_epoch": record["sec_per_epoch"],
                        "notes": f"solved={record['solved']}; hidden_size={hidden_size}; epochs={epochs}",
                    },
                    results_path,
                )
                if sequence_length == 200:
                    t200_histories[cell]["train_mse"].append(history["train_mse"])
                    t200_histories[cell]["val_mse"].append(history["val_mse"])
                if track_gradients:
                    gradient_norms[cell]["after"] = _hidden_gradient_norms(
                        model, x_probe, y_probe, device
                    )

    for sequence_length in sequence_lengths:
        baseline = _mean_std(test_baselines[sequence_length])
        print(
            f"Adding T={sequence_length} constant baseline: "
            f"test MSE={baseline[0]:.5f} +/- {baseline[1]:.5f}"
        )
        for cell in CELLS:
            subset = [
                record
                for record in records
                if record["sequence_length"] == sequence_length
                and record["cell"] == cell
            ]
            mse = _mean_std([record["test_mse"] for record in subset])
            solved = sum(record["solved"] for record in subset)
            print(
                f"Adding T={sequence_length} {cell}: test MSE="
                f"{mse[0]:.5f} +/- {mse[1]:.5f}; solved {solved}/{len(subset)}"
            )

    learning_curve_path = _plot_adding_learning_curves(t200_histories, figure_dir)
    gradient_path = _plot_gradient_norms(gradient_norms, figure_dir)
    print(f"T=200 learning curves saved to {learning_curve_path}")
    print(f"T=100 gradient plots saved to {gradient_path}")
    return records, gradient_norms


def _set_forget_bias(model: RecurrentRegressor, value: float) -> None:
    hidden_size = model.recurrent.hidden_size
    with torch.no_grad():
        model.recurrent.bias_ih_l0[hidden_size : 2 * hidden_size].fill_(value)
        model.recurrent.bias_hh_l0[hidden_size : 2 * hidden_size].zero_()


def run_selected_ablation(
    student_id_last_digit: int,
    seeds,
    hidden_size: int,
    batch_size: int,
    epochs: int,
    train_size: int,
    val_size: int,
    test_size: int,
    device: torch.device,
    results_path: str | Path,
):
    if student_id_last_digit not in range(10):
        raise ValueError("student_id_last_digit must be an integer from 0 to 9")
    even_id = student_id_last_digit % 2 == 0
    cell = "RNN" if even_id else "LSTM"
    records = []

    for seed in seeds:
        x_train, y_train = make_adding(train_size, 200, seed * 100 + 1)
        x_val, y_val = make_adding(val_size, 200, seed * 100 + 2)
        x_test, y_test = make_adding(test_size, 200, seed * 100 + 3)
        variants = (("none", None), ("clip_1.0", 1.0)) if even_id else (("forget_bias_0", 0.0), ("forget_bias_1", 1.0))
        for variant, setting in variants:
            set_seed(seed)
            model = RecurrentRegressor(cell, hidden_size=hidden_size)
            if not even_id:
                _set_forget_bias(model, setting)
            model, history = train_regressor(
                model,
                x_train,
                y_train,
                x_val,
                y_val,
                epochs,
                batch_size,
                device,
                gradient_clip_norm=setting if even_id else None,
            )
            test_mse = predict_mse(model, x_test, y_test, device)
            record = {
                "cell": cell,
                "variant": variant,
                "seed": seed,
                "test_mse": test_mse,
                "solved": test_mse < 0.05,
                "sec_per_epoch": float(np.mean(history["sec_per_epoch"])),
                "params": count_parameters(model),
            }
            records.append(record)
            _append_result(
                {
                    "problem": "problem1_adding_ablation",
                    "model": f"{cell}_{variant}",
                    "seed": seed,
                    "seq_len": 200,
                    "test_mse": test_mse,
                    "params": record["params"],
                    "sec_per_epoch": record["sec_per_epoch"],
                    "notes": f"student_id_last_digit={student_id_last_digit}; solved={record['solved']}",
                },
                results_path,
            )

    for variant, _ in variants:
        subset = [record for record in records if record["variant"] == variant]
        mse = _mean_std([record["test_mse"] for record in subset])
        solved = sum(record["solved"] for record in subset)
        print(
            f"Ablation {cell} {variant}: test MSE={mse[0]:.5f} +/- "
            f"{mse[1]:.5f}; solved {solved}/{len(subset)}"
        )
    return records
