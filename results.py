import csv
from pathlib import Path


RESULT_COLUMNS = (
    "problem",
    "model",
    "seed",
    "seq_len",
    "test_acc",
    "macro_f1",
    "test_mse",
    "params",
    "sec_per_epoch",
    "val_loss_nats",
    "bits_per_char",
    "notes",
)


def append_result(row: dict, path: str | Path = "results.csv") -> None:
    """Append one run while preserving the assignment's exact CSV schema."""
    extra = set(row) - set(RESULT_COLUMNS)
    if extra:
        raise ValueError(f"Unexpected result columns: {sorted(extra)}")
    path = Path(path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
