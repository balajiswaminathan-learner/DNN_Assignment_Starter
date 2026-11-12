# DNN Assignment

## Environment and execution

Python 3.10+; install the pinned packages with `python -m pip install -r requirements.txt`.
The recorded runs were executed on a personal Windows laptop with PyTorch
DirectML (`privateuseone:0`). The GPU model was not captured. The notebooks
save their execution outputs and generated figures under `problem1/` and
`problem2/`.

Place UCI HAR at `problem1/data/UCI HAR Dataset` (or set `HAR_ROOT` in the
Problem 1 notebook). Problem 2 downloads Tiny Shakespeare to its `data/`
folder when needed. Restart and run `problem1/problem1.ipynb`, then
`problem2/problem2.ipynb`, top to bottom. The Problem 2 notebook uses the
Problem 1 HAR results.

## Recorded variant and runtime

The executed settings were hidden size 64, Transformer width 128, seeds
11/21/31, and student-ID last digit 3 (forget-gate bias ablation). Measured
mean HAR training time per epoch was 8.56 s (RNN), 18.86 s (LSTM), and
21.04 s (GRU) in Problem 1; Problem 2 encoder variants took about 4.37–4.67
s/epoch. The 5,000-step character model averaged 0.244 s/step.

**T4 total runtime was not measured**: the recorded runs used the laptop's
DirectML backend, not a Colab T4. Timings are device- and backend-specific.
The installed DirectML backend warned that an Adam `lerp` operation falls
back to CPU, which can affect performance.

## Deliverables and limitations

See [REPORT.pdf](./REPORT.pdf) for the results and analysis. The notebook's
three 500-character samples are in [generated_samples.txt](./generated_samples.txt).
The language model reaches 2.3768 validation bits/character. The adding
problem is solved by the GRU at T=50, but not at T=100–400; the RNN and LSTM
do not meet the recorded solved threshold. The no-causal-mask language-model
result leaks future tokens and is shown only as an ablation, not a valid
language-model score.
