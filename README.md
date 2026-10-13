# on-policy-distillation-ceiling

Starter code for **"Does On-Policy Distillation Have a Capability Ceiling?"**

Author: Pritam Shingavi (Vizuara SLM Bootcamp, 8 week research project)

## The question

Crosscoder model diffing suggests that post-training largely *reweights*
features a base model already carries rather than installing new ones.
If that is right, on-policy distillation should be unable to teach a
capability the student cannot already represent. This repo is the
machinery to test that, end to end, on a single GPU.

The experiment has three moving parts:

1. **Representability audit.** Before any training, score each target
   capability by how well the *base* student already represents it
   (linear probes on frozen activations, plus sparse autoencoder feature
   coverage). This gives `R(c)` per capability and splits the battery
   into a `REPRESENTED` set and an `ABSENT` set.
2. **Four training arms at equal token budget.** Off-policy sequence
   KD, on-policy distillation (the arm under test), RL with verifiable
   reward, and continued pretraining as a positive control that is
   *expected* to install new features.
3. **Three readouts.** Behavioral (pass@1 vs pass@k coverage),
   representational (crosscoder novel-latent fraction after
   latent-scaling correction), and causal (ablate or steer the candidate
   new latents and remeasure).

The hypothesis is confirmed if on-policy distillation gains nothing on
the `ABSENT` set, the corrected novel-latent fraction stays near zero,
and gain correlates with `R(c)`. It is refuted if any of those fail.

## Layout

```
src/
  data/
    capabilities.py      # the capability battery: REPRESENTED + ABSENT tasks
    synthetic.py         # generators for base-7 arithmetic, cipher, grammar
    loaders.py           # public dataset loading (GSM8K etc) via datasets
  models/
    registry.py          # student and teacher loading, dtype/device handling
    sae.py               # BatchTopK sparse autoencoder
    crosscoder.py        # sparse crosscoder + latent-scaling correction
  audit/
    probes.py            # linear probes on frozen activations -> probe AUC
    representability.py  # combines probe AUC x SAE coverage -> R(c)
  train/
    offpolicy_kd.py      # arm 1: sequence-level KD on teacher samples
    onpolicy_distill.py  # arm 2: GKD-style on-policy distillation
    rlvr.py              # arm 3: RL with verifiable reward (GRPO-style)
    continued_pretrain.py# arm 4: positive control
  eval/
    passk.py             # pass@1 / pass@k with the unbiased estimator
    diffing.py           # train crosscoder, report NLF before/after correction
    causal.py            # latent ablation and steering
  cli.py                 # single entry point: audit / train / eval / diff
configs/
  default.yaml           # the one config you edit
scripts/
  run_audit.sh
  run_arm.sh
  run_eval.sh
  run_full_pipeline.sh
experiments/
  log_template.md        # one row per table row in the paper
```

## Setup

Requires Python 3.10+ and a GPU with at least 24GB for the 1.7B student
with a 7B teacher in bf16. The 0.6B student fits comfortably in 16GB.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for teacher inference throughput:

```bash
pip install "vllm>=0.8.0"
```

Set up Hugging Face access (some teachers are gated):

```bash
huggingface-cli login
```

## How to run

Everything goes through `src/cli.py`. Start small: the default config
uses `Qwen/Qwen3-0.6B` as the student and a tiny token budget so you can
get a full pipeline run in under an hour.

**1. Smoke test the battery** (no GPU training, just checks the tasks
generate and verify correctly):

```bash
python -m src.cli battery --config configs/default.yaml
```

**2. Run the representability audit** on the base student. This writes
`runs/audit/representability.json` with `R(c)` per capability and the
`REPRESENTED` / `ABSENT` split. Freeze this before you train anything.

```bash
bash scripts/run_audit.sh
```

**3. Train one arm** and evaluate it:

```bash
bash scripts/run_arm.sh onpolicy_distill
bash scripts/run_eval.sh runs/onpolicy_distill
```

Then the whole thing:

```bash
bash scripts/run_full_pipeline.sh
```

## What to watch for

- **The audit must be frozen first.** If you adjust the split after
  seeing training results, the ceiling test stops being falsifiable and
  a reviewer will say so.
- **Verify the teacher can do the ABSENT tasks.** `cli.py audit` prints
  teacher accuracy per capability. A teacher at 5% on base-7 arithmetic
  cannot tell you anything about whether the student could have learned
  it.
- **Latent scaling is not optional.** The uncorrected novel-latent count
  is inflated by the sparsity penalty. `eval/diffing.py` reports both,
  and the corrected number is the one that goes in the paper.
- **Equal budgets.** `configs/default.yaml` sets a single
  `token_budget`; every arm reads it. Do not tune one arm harder than
  another.

## References

Key papers this builds on are in `../paper/references.bib`. The three to
read first: Agarwal et al. 2024 (on-policy distillation / GKD), Lindsey
et al. 2024 (sparse crosscoders), Minder et al. 2025 (sparsity artifacts
and latent scaling).
