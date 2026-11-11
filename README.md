# DNN Programming Assignment Starter

## Environment

Python 3.10+ and the pinned dependencies in `requirements.txt` are expected.
Run `python -m pip install -r requirements.txt` in a fresh environment.
For long runs, use a Colab T4 GPU. HAR and the recurrent experiments also work
on CPU; the full character language model is intended for a GPU.

## Data

Download and extract UCI HAR so the directory `UCI HAR Dataset` contains
`train/`, `test/`, the label/subject files, and `Inertial Signals/`. Place it
at `data/UCI HAR Dataset` (or set `HAR_ROOT` in the notebook).
Tiny Shakespeare is downloaded by the Problem 2 notebook to `data/`.
No model checkpoints or datasets should be included in the submitted zip.

## Run order

1. Open `problem1/problem1.ipynb`; set your variant values in its first setup
   cell, then run all cells. The code creates a subject-wise validation split,
   computes training-only normalization statistics, generates the adding
   problem, and trains the standard-library recurrent baselines.
2. Implement and verify the scratch cells in `problem1/from_scratch.py`, then
   complete the notebook's cell-comparison section.
3. Open `problem2/problem2.ipynb`. Implement the required attention, MHA and
   encoder block in `problem2/from_scratch.py` before running dependent cells.
4. Save notebook outputs and call `results.append_result(...)` after every
   measured run to append a row to the supplied `results.csv`.

The notebook run time depends on epochs, model variant and hardware; time a
small pilot run and report the full observed T4 runtime. Do not report example
or placeholder results as measured results.

## Student-specific settings

The supplied DOCX refers to an individualisation table but does not contain
that table. Set the hidden size, Transformer width and three prescribed seeds
in the notebook after obtaining the mapping from the instructor. The starter
uses clearly labelled provisional values only.

## Incomplete by design

The scratch implementations, experimental results, figures, report PDF,
generated samples and AI-use declaration must be completed by the student.
`results.csv` starts with the required schema and no fabricated results.
