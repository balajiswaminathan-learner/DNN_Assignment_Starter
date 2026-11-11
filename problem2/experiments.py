from pathlib import Path
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from problem1.data import load_har
from problem1.models import count_parameters
from problem1.train import evaluate_classifier, make_loader, set_seed, train_classifier
from problem2.data import get_batch
from problem2.from_scratch import (
    EncoderBlock,
    SinusoidalPositionalEncoding,
    make_causal_mask,
)
from results import append_result


POSITION_VARIANTS = ("none", "sinusoidal", "learned")


def _save_path(figure_dir: str | Path, filename: str) -> Path:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir / filename


def _append(row: dict, results_path: str | Path) -> None:
    append_result(row, path=results_path)


def print_problem1_baseline(results_path: str | Path):
    results_path = Path(results_path)
    if not results_path.is_file():
        print("Problem 1 baseline unavailable: results CSV not found.")
        return None
    grouped = {}
    with results_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row.get("problem") != "problem1_har" or not row.get("test_acc"):
                continue
            grouped.setdefault(row["model"], []).append(row)
    if not grouped:
        print("Problem 1 baseline unavailable: no problem1_har rows in results CSV.")
        return None
    summaries = []
    for model_name, rows in grouped.items():
        accuracy = float(np.mean([float(row["test_acc"]) for row in rows]))
        params = float(np.mean([float(row["params"]) for row in rows if row["params"]]))
        seconds = float(
            np.mean([float(row["sec_per_epoch"]) for row in rows if row["sec_per_epoch"]])
        )
        summaries.append((accuracy, model_name, params, seconds))
    accuracy, model_name, params, seconds = max(summaries)
    summary = {
        "model": model_name,
        "accuracy": accuracy,
        "params": params,
        "sec_per_epoch": seconds,
    }
    print(
        f"Best Problem 1 baseline ({model_name}): accuracy={accuracy:.4f}, "
        f"params={params:.0f}, sec/epoch={seconds:.3f}"
    )
    return summary


class HARTransformerClassifier(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 6,
        channels: int = 9,
        patch_size: int = 8,
        positional: str = "sinusoidal",
        dropout: float = 0.1,
    ):
        super().__init__()
        if positional not in POSITION_VARIANTS:
            raise ValueError(f"positional must be one of {POSITION_VARIANTS}")
        self.patch_size = patch_size
        self.embedding = nn.Linear(channels * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        max_tokens = 128 // patch_size + 1
        self.positional = positional
        if positional == "sinusoidal":
            self.position_encoding = SinusoidalPositionalEncoding(d_model, max_tokens)
        elif positional == "learned":
            self.position_embedding = nn.Parameter(torch.empty(1, max_tokens, d_model))
            nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, num_heads, 4 * d_model, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch, steps, channels = x.shape
        if steps % self.patch_size:
            raise ValueError("HAR sequence length must be divisible by patch_size")
        patches = x.reshape(batch, steps // self.patch_size, self.patch_size * channels)
        tokens = self.embedding(patches)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if self.positional == "sinusoidal":
            tokens = self.position_encoding(tokens)
        elif self.positional == "learned":
            tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        tokens = self.embedding_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        return self.classifier(self.final_norm(tokens[:, 0]))


class CharacterTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        context: int = 128,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.context = context
        self.causal = causal
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model, context)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, num_heads, 4 * d_model, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.language_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, return_attentions: bool = False):
        _, sequence_length = token_ids.shape
        if sequence_length > self.context:
            raise ValueError("input sequence is longer than the configured context")
        hidden = self.position_encoding(self.token_embedding(token_ids))
        hidden = self.embedding_dropout(hidden)
        mask = None
        if self.causal:
            mask = make_causal_mask(sequence_length, device=token_ids.device)
        attentions = []
        for block in self.blocks:
            if return_attentions:
                hidden, weights = block(hidden, mask=mask, return_attention=True)
                attentions.append(weights)
            else:
                hidden = block(hidden, mask=mask)
        logits = self.language_head(self.final_norm(hidden))
        if return_attentions:
            return logits, attentions
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = 20,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
            raise ValueError("prompt must have shape (1, nonzero_sequence_length)")
        was_training = self.training
        self.eval()
        generated = prompt
        for _ in range(max_new_tokens):
            context_ids = generated[:, -self.context :]
            logits = self(context_ids)[:, -1, :] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive or None")
                k = min(top_k, logits.shape[-1])
                cutoff = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        self.train(was_training)
        return generated


def run_har_positional_comparison(
    har_root: str | Path,
    seeds,
    d_model: int,
    num_heads: int,
    num_layers: int,
    batch_size: int,
    epochs: int,
    device: torch.device,
    results_path: str | Path,
    dropout: float = 0.1,
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
        for positional in POSITION_VARIANTS:
            set_seed(seed)
            model = HARTransformerClassifier(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_layers,
                positional=positional,
                dropout=dropout,
            )
            model, epoch_seconds, val_loss = train_classifier(
                model, train_loader, val_loader, epochs, device
            )
            metrics = evaluate_classifier(model, test_loader, device)
            record = {
                "positional": positional,
                "seed": seed,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "params": count_parameters(model),
                "sec_per_epoch": float(np.mean(epoch_seconds)),
                "val_loss": val_loss,
            }
            records.append(record)
            _append(
                {
                    "problem": "problem2_har",
                    "model": f"encoder_{positional}",
                    "seed": seed,
                    "seq_len": 128,
                    "test_acc": record["accuracy"],
                    "macro_f1": record["macro_f1"],
                    "params": record["params"],
                    "sec_per_epoch": record["sec_per_epoch"],
                    "val_loss_nats": val_loss,
                    "notes": f"d={d_model}; heads={num_heads}; layers={num_layers}; patch={model.patch_size}",
                },
                results_path,
            )

    for positional in POSITION_VARIANTS:
        subset = [row for row in records if row["positional"] == positional]
        accuracy = np.asarray([row["accuracy"] for row in subset])
        std = accuracy.std(ddof=1) if len(accuracy) > 1 else 0.0
        seconds = np.asarray([row["sec_per_epoch"] for row in subset])
        seconds_std = seconds.std(ddof=1) if len(seconds) > 1 else 0.0
        print(
            f"HAR encoder {positional}: accuracy={accuracy.mean():.4f} +/- {std:.4f}, "
            f"params={subset[0]['params']}, "
            f"sec/epoch={seconds.mean():.3f} +/- {seconds_std:.3f}"
        )
    return records


def _evaluate_language_model(
    model,
    tokens: torch.Tensor,
    context: int,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(tokens, batch_size, context, device)
            logits = model(x)
            losses.append(
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item()
            )
    model.train()
    return float(np.mean(losses))


def _fit_language_model(
    model,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    context: int,
    batch_size: int,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    learning_rate: float,
    device: torch.device,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = {"step": [], "train_loss": [], "val_loss": []}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_tokens, batch_size, context, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step == 1 or step % eval_interval == 0 or step == steps:
            train_loss = _evaluate_language_model(
                model, train_tokens, context, batch_size, eval_batches, device
            )
            val_loss = _evaluate_language_model(
                model, val_tokens, context, batch_size, eval_batches, device
            )
            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            print(
                f"step {step}/{steps}: train={train_loss:.4f} nats, "
                f"validation={val_loss:.4f} nats"
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds_per_step = (time.perf_counter() - started) / steps
    return model, history, history["val_loss"][-1], seconds_per_step


def _plot_lm_history(history, figure_path: str | Path) -> Path:
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history["step"], history["train_loss"], label="Train")
    axis.plot(history["step"], history["val_loss"], label="Validation")
    axis.set(xlabel="Training step", ylabel="Cross-entropy (nats)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def train_character_model(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    vocab_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    context: int,
    batch_size: int,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    seed: int,
    device: torch.device,
    results_path: str | Path,
    figure_dir: str | Path,
    samples_path: str | Path,
    id_to_char: dict[int, str],
    top_k: int = 20,
    causal: bool = True,
):
    set_seed(seed)
    model = CharacterTransformerLM(
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        context,
        causal=causal,
    )
    model, history, val_loss, seconds_per_step = _fit_language_model(
        model,
        train_tokens,
        val_tokens,
        context,
        batch_size,
        steps,
        eval_interval,
        eval_batches,
        3e-4,
        device,
    )
    bits_per_character = val_loss / np.log(2.0)
    _append(
        {
            "problem": "problem2_char_lm",
            "model": "causal_transformer" if causal else "unmasked_transformer",
            "seed": seed,
            "seq_len": context,
            "params": count_parameters(model),
            "val_loss_nats": val_loss,
            "bits_per_char": bits_per_character,
            "notes": (
                f"d={d_model}; heads={num_heads}; layers={num_layers}; steps={steps}; "
                f"sec_per_step={seconds_per_step:.6f}"
            ),
        },
        results_path,
    )
    figure_path = _plot_lm_history(
        history, Path(figure_dir) / "character_lm_learning_curve.png"
    )
    prompt = val_tokens[:1].reshape(1, 1).to(device)
    saved_samples = save_temperature_samples(
        model, prompt, id_to_char, samples_path, top_k=top_k, new_characters=500
    )
    print(
        f"validation={val_loss:.4f} nats, {bits_per_character:.4f} bits/character; "
        f"learning curve saved to {figure_path}; samples saved to {saved_samples}"
    )
    return model, {
        "val_loss_nats": val_loss,
        "bits_per_character": bits_per_character,
        "sec_per_step": seconds_per_step,
        "history": history,
        "samples_path": saved_samples,
    }


def save_temperature_samples(
    model,
    prompt: torch.Tensor,
    id_to_char: dict[int, str],
    samples_path: str | Path,
    top_k: int = 20,
    new_characters: int = 500,
) -> Path:
    samples_path = Path(samples_path)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for temperature in (0.2, 0.8, 1.5):
        generated = model.generate(
            prompt.to(next(model.parameters()).device),
            new_characters,
            temperature=temperature,
            top_k=top_k,
        )[0, prompt.shape[1] :].tolist()
        text = "".join(id_to_char[index] for index in generated)
        sections.append(f"Temperature {temperature}\n{text}")
    samples_path.write_text("\n\n".join(sections), encoding="utf-8")
    return samples_path


def plot_attention_heads(
    model,
    token_ids: torch.Tensor,
    figure_path: str | Path,
    heads: tuple[int, int] = (0, 1),
) -> Path:
    model.eval()
    with torch.no_grad():
        _, attentions = model(
            token_ids.to(next(model.parameters()).device), return_attentions=True
        )
    weights = attentions[0][0]
    sequence_length = weights.shape[-1]
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 5))
    if len(heads) == 1:
        axes = [axes]
    for axis, head in zip(axes, heads):
        image = axis.imshow(weights[head, :sequence_length, :sequence_length].cpu(), cmap="magma")
        axis.set(title=f"Layer 1, head {head + 1}", xlabel="Key position", ylabel="Query position")
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def run_language_ablations(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    context: int,
    batch_size: int,
    steps: int,
    eval_interval: int,
    eval_batches: int,
    seeds,
    device: torch.device,
    results_path: str | Path,
    figure_dir: str | Path,
    samples_path: str | Path,
    id_to_char: dict[int, str],
):
    records = []
    causal_histories = {True: [], False: []}
    for seed in seeds:
        for num_heads in (1, 4, 8):
            if d_model % num_heads:
                raise ValueError("d_model must be divisible by each ablation head count")
            set_seed(seed)
            model = CharacterTransformerLM(
                vocab_size, d_model, num_heads, num_layers, context, causal=True
            )
            model, history, val_loss, seconds = _fit_language_model(
                model,
                train_tokens,
                val_tokens,
                context,
                batch_size,
                steps,
                eval_interval,
                eval_batches,
                3e-4,
                device,
            )
            records.append({"condition": f"heads_{num_heads}", "seed": seed, "val_loss_nats": val_loss})
            _append(
                {
                    "problem": "problem2_heads_ablation",
                    "model": f"heads_{num_heads}",
                    "seed": seed,
                    "seq_len": context,
                    "params": count_parameters(model),
                    "val_loss_nats": val_loss,
                    "bits_per_char": val_loss / np.log(2.0),
                    "notes": (
                        f"d={d_model}; layers={num_layers}; steps={steps}; "
                        f"sec_per_step={seconds:.6f}"
                    ),
                },
                results_path,
            )

        for causal in (True, False):
            set_seed(seed)
            model = CharacterTransformerLM(
                vocab_size, d_model, 4, num_layers, context, causal=causal
            )
            model, history, val_loss, seconds = _fit_language_model(
                model,
                train_tokens,
                val_tokens,
                context,
                batch_size,
                steps,
                eval_interval,
                eval_batches,
                3e-4,
                device,
            )
            causal_histories[causal].append(history)
            condition = "causal_mask" if causal else "no_causal_mask"
            records.append({"condition": condition, "seed": seed, "val_loss_nats": val_loss})
            _append(
                {
                    "problem": "problem2_causal_mask_ablation",
                    "model": condition,
                    "seed": seed,
                    "seq_len": context,
                    "params": count_parameters(model),
                    "val_loss_nats": val_loss,
                    "bits_per_char": val_loss / np.log(2.0),
                    "notes": (
                        f"d={d_model}; heads=4; layers={num_layers}; steps={steps}; "
                        f"sec_per_step={seconds:.6f}"
                    ),
                },
                results_path,
            )
            if not causal:
                prompt = val_tokens[:1].reshape(1, 1).to(device)
                generated = model.generate(prompt, 500, temperature=0.8, top_k=20)[0]
                generated = generated[prompt.shape[1] :].tolist()
                text = "".join(id_to_char[index] for index in generated)
                output_path = Path(samples_path).with_name(
                    f"no_mask_generation_seed{seed}.txt"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text, encoding="utf-8")
    _plot_ablation_losses(records, figure_dir)
    _plot_mask_learning_curves(causal_histories, figure_dir)
    return records


def _plot_mask_learning_curves(histories, figure_dir: str | Path) -> Path:
    figure_path = _save_path(figure_dir, "causal_mask_learning_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for causal, label in ((True, "Causal mask"), (False, "No causal mask")):
        runs = histories[causal]
        if not runs:
            continue
        steps = np.asarray(runs[0]["step"])
        for axis, key, title in zip(
            axes, ("train_loss", "val_loss"), ("Training", "Validation")
        ):
            losses = np.asarray([run[key] for run in runs])
            axis.plot(steps, losses.mean(axis=0), label=label)
            axis.set(xlabel="Training step", ylabel="Cross-entropy (nats)", title=title)
            axis.grid(alpha=0.25)
            axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def _plot_ablation_losses(records, figure_dir: str | Path) -> Path:
    figure_path = _save_path(figure_dir, "language_model_ablations.png")
    grouped = {}
    for row in records:
        grouped.setdefault(row["condition"], []).append(row["val_loss_nats"])
    labels = list(grouped)
    means = [np.mean(grouped[label]) for label in labels]
    errors = [np.std(grouped[label], ddof=1) if len(grouped[label]) > 1 else 0.0 for label in labels]
    fig, axis = plt.subplots(figsize=(9, 4))
    axis.bar(labels, means, yerr=errors, color="#287c76", capsize=3)
    axis.set(ylabel="Validation loss (nats)", title="Language-model ablations")
    axis.tick_params(axis="x", labelrotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return figure_path


def benchmark_context_lengths(
    vocab_size: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    contexts=(64, 128, 256, 512),
    device: torch.device = torch.device("cpu"),
    results_path: str | Path = "results.csv",
    figure_dir: str | Path = "problem2/figures",
    repeats: int = 3,
):
    records = []
    batch_size = 1
    for context in contexts:
        model = CharacterTransformerLM(
            vocab_size, d_model, num_heads, num_layers, context, dropout=0.0
        ).to(device)
        tokens = torch.randint(vocab_size, (batch_size, context), device=device)
        targets = torch.randint(vocab_size, (batch_size, context), device=device)
        elapsed = []
        for _ in range(repeats + 1):
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                if len(elapsed) == 0:
                    torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)
        seconds = float(np.mean(elapsed[1:]))
        attention_estimate = (
            num_layers * batch_size * num_heads * context * context * 4
        )
        memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else int(attention_estimate)
        )
        record = {
            "context": context,
            "sec_per_step": seconds,
            "memory_bytes": memory_bytes,
            "attention_estimate_bytes": int(attention_estimate),
        }
        records.append(record)
        _append(
            {
                "problem": "problem2_context_benchmark",
                "model": f"d{d_model}_h{num_heads}_l{num_layers}",
                "seq_len": context,
                "params": count_parameters(model),
                "notes": (
                    f"batch={batch_size}; sec_per_step={seconds:.6f}; "
                    f"measured_cuda_peak_bytes={memory_bytes}"
                    if device.type == "cuda"
                    else f"batch={batch_size}; sec_per_step={seconds:.6f}; "
                    f"CPU attention-score estimate bytes={memory_bytes}"
                ),
            },
            results_path,
        )
        print(
            f"context={context}: sec/step={seconds:.4f}, "
            f"memory_bytes={memory_bytes}"
        )
        del model, tokens, targets, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    figure_path = _save_path(figure_dir, "context_time_memory.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([row["context"] for row in records], [row["sec_per_step"] for row in records], marker="o")
    axes[0].set(xlabel="Context length", ylabel="Seconds per train step", title="Measured time")
    axes[1].plot([row["context"] for row in records], [row["memory_bytes"] / (1024**2) for row in records], marker="o")
    axes[1].set(xlabel="Context length", ylabel="Memory (MiB)", title="Peak CUDA / CPU attention estimate")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    return records
