# CLAUDE.md

Instructions for the coding agent working on this repository with
Pritam. Read this before touching code.

## Project goal

Test a single falsifiable claim: **on-policy distillation cannot teach a
small language model a capability it cannot already represent.**

The claim comes from crosscoder model diffing, which reports that
post-training reweights existing features rather than acquiring new
ones. If that is true, there is a capability ceiling. We are building
the experiment that either confirms it or breaks it.

Three parts, in strict order:

1. `audit/` scores each capability by how well the **base** student
   already represents it, producing `R(c)`, and splits the battery into
   `REPRESENTED` and `ABSENT`.
2. `train/` runs four arms at **equal token budget**: off-policy KD,
   on-policy distillation (the arm under test), RL with verifiable
   reward, continued pretraining (positive control).
3. `eval/` reads out behavior (pass@1 vs pass@k), representation
   (crosscoder novel-latent fraction after latent-scaling correction),
   and causality (ablate or steer candidate new latents).

## Non-negotiable invariants

Violating any of these invalidates the result, so treat them as tests,
not preferences.

- **The audit is frozen before training.** `runs/audit/representability.json`
  is written once and read thereafter. Never regenerate it after a
  training arm has run. If you need to change the audit, bump
  `audit_version` in the config and rerun everything downstream.
- **Equal budgets across arms.** Every arm reads `token_budget`,
  `optimizer_steps` and `seeds` from the same config block. If you add a
  knob to one arm, add the matched default to all four.
- **Latent scaling is mandatory.** Any novel-latent number reported
  without the latent-scaling correction is wrong. `eval/diffing.py`
  must always return both `nlf_raw` and `nlf_corrected`; the paper uses
  the corrected one.
- **ABSENT tasks are generated, not scraped.** Their absence from
  pretraining is the whole point. Do not substitute a natural
  low-resource dataset for a generated symbol system without saying so
  loudly.
- **Teacher competence gate.** Before an arm runs on a capability,
  teacher accuracy on that capability must clear
  `min_teacher_accuracy`. Otherwise the arm is skipped and logged as
  skipped, not as a zero.
- **Seeds.** Three seeds minimum for anything that lands in a table.
  Single-seed numbers are for debugging only.

## Code conventions

- Python 3.10+, type hints on public functions, no classes where a
  function will do.
- Config is a single YAML loaded into a dataclass (`src/config.py`
  pattern: `load_config(path) -> Config`). No reading env vars for
  experiment parameters.
- Every entry point is a subcommand of `src/cli.py`. Do not add
  top-level scripts that bypass it.
- All run artifacts go to `runs/<arm_name>/` with a `meta.json` holding
  the full resolved config plus git SHA. Never overwrite a run
  directory; add a suffix.
- Logging via `logging`, not `print`, except in `cli.py` user output.
- Keep modules short enough to read in one sitting. If a file passes
  roughly 300 lines, split it.
- No new dependencies without adding them to `requirements.txt` with a
  version floor.

## Things that will bite you

- Teacher and student usually have **different tokenizers**. On-policy
  distillation needs teacher log-probs on *student* tokens.
  `train/onpolicy_distill.py` has a `_align_tokenizers` path; if you
  switch models, re-check it. Easiest safe move: keep teacher and
  student in the same family (Qwen3 student, Qwen3 teacher).
- Reverse KL needs the teacher's full next-token distribution, which is
  memory heavy. The code top-$k$ truncates; the truncation level is a
  config knob and is worth an ablation row.
- Crosscoders need paired activations on the **same inputs** from both
  models. Any divergence in prompt formatting between base and trained
  student silently corrupts the diff.
- `datasets` caching will happily eat the disk. `HF_HOME` is set in the
  run scripts.

## Next steps, in order

1. Fill in `src/data/synthetic.py` verifiers and confirm every ABSENT
   task has exact-match grading plus a partial-credit signal.
2. Get `audit/probes.py` running end to end on the 0.6B student and
   sanity check against the label-shuffled control. A probe that beats
   chance on shuffled labels means a leak.
3. Run the audit, inspect `R(c)` by hand, and freeze the split. Commit
   the JSON.
4. Get arm 2 (on-policy distillation) training stably on a REPRESENTED
   capability first, so you know the implementation works before you
   point it at an ABSENT one.
5. Add arm 4 (continued pretraining) next, not last. It is the positive
   control, and if it fails the whole experiment is uninterpretable.
6. Then arms 1 and 3, then `eval/diffing.py`, then `eval/causal.py`.
7. Map every result into `experiments/log_template.md`, whose rows
   match the paper's Table 1 exactly.

## When asked to write results

Numbers go into `experiments/log_template.md` first, then into the
paper. Do not write a number into the paper that is not in the log with
a run directory next to it. If a number is a placeholder, it stays as
`XX.X` so it cannot be mistaken for a measurement.
