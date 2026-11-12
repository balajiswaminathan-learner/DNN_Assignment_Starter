# DNN Assignment 2: RNN, LSTM, GRU and Transformer

Scratch PyTorch recurrent cells and attention are checked against reference outputs.
Problem 1 evaluates recurrent models on UCI HAR and adding; Problem 2 evaluates
Transformer encoders on HAR and character-level language modelling.

| Student  | Student ID  | Executed values                                                                |
|----------|-------------|--------------------------------------------------------------------------------|
| BALAJI S | 2026AK05103 | RNN/LSTM/GRU hidden size 64; LM width 128, 4 heads, 4 layers; seeds 11, 21, 31 |

**5(c) ablation:** ID last digit 3; LSTM forget-gate bias 0 vs. 1 at T=200.
**Hardware:** Personal Windows laptop, PyTorch DirectML (`privateuseone:0`);
GPU model not recorded. **T4 runtime:** not measured; no Colab T4 run was
performed. Recorded timings below are laptop measurements.

## Results at a glance

HAR and ablation values are mean ± sample SD over 3 seeds.

**Problem 1 HAR, hidden size 64**

| Model | Test accuracy | Macro-F1       | Parameters |
|-------|--------------:|---------------:|-----------:|
| RNN   | 0.623 ± 0.056 | 0.577 ± 0.097  |      5,190 |
| LSTM  | 0.826 ± 0.067 | 0.822 ± 0.073  |     19,590 |
| GRU   | 0.868 ± 0.011 | 0.867 ± 0.011  |     14,790 |

**Adding task: runs solved (test MSE < 0.05) out of 3**

| Model | T = 50 | T = 100 | T = 200 | T = 400 |
|-------|-------:|--------:|--------:|--------:|
| RNN   |    0/3 |     0/3 |     0/3 |     0/3 |
| LSTM  |    0/3 |     0/3 |     0/3 |     0/3 |
| GRU   |    3/3 |     0/3 |     0/3 |     0/3 |

**Problem 2:** HAR encoder accuracy—none 0.914 ± 0.007, learned 0.907 ± 0.007,
sinusoidal 0.900 ± 0.011. Character LM (5,000 steps): 1.6475 validation nats
(2.3768 bits/character).

## Quick start

1. Install: `python -m pip install -r requirements.txt`.
2. Put UCI HAR in `problem1/data/UCI HAR Dataset` (or set `HAR_ROOT`).
3. From the repository root, **Restart and Run All** on `problem1/problem1.ipynb`,
   then `problem2/problem2.ipynb`. Problem 2 uses Problem 1 results; Tiny
   Shakespeare downloads to `problem2/data/` if absent.

Mean laptop timings: HAR 8.56 s/epoch (RNN), 18.86 (LSTM), 21.04 (GRU);
encoder 4.37–4.67 s/epoch; language model 0.244 s/step. DirectML warned that
an Adam `lerp` operation falls back to CPU.

## Files and notes

- [REPORT.pdf](./REPORT.pdf): results, figures, analysis, AI-use statement, and references.
- [generated_samples.txt](./generated_samples.txt): three labelled 500-character samples (temperatures 0.2, 0.8, 1.5).
- Results: `problem1/results.csv`, `problem2/results.csv`; figures are under each problem's `figures/`.
- Context memory is an attention-score estimate, not measured GPU peak memory. The no-mask result leaks future tokens and is not a valid language-model score.
