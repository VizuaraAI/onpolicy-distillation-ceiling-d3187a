# Experiment log

One row per cell of Table 1 in the paper. A number may not appear in the
paper unless it appears here first, with a run directory next to it.
Placeholders stay as `XX.X` so they can never be mistaken for a
measurement.

---

## 0. Frozen audit

Fill this in once, from `runs/audit/representability.json`, before any
training arm runs. Then commit it.

| Capability | Prior group | Probe score | SAE coverage | R(c) | Assigned group |
|---|---|---|---|---|---|
| gsm8k_math | represented | X.XXX | X.XXX | X.XXX | |
| base10_multistep_arith | represented | X.XXX | X.XXX | X.XXX | |
| format_compliance | represented | X.XXX | X.XXX | X.XXX | |
| extractive_qa | represented | X.XXX | X.XXX | X.XXX | |
| base7_arith_glyphs | absent | X.XXX | X.XXX | X.XXX | |
| script_transliteration | absent | X.XXX | X.XXX | X.XXX | |
| substitution_cipher | absent | X.XXX | X.XXX | X.XXX | |
| synthetic_grammar_validity | absent | X.XXX | X.XXX | X.XXX | |

Audit version: `v1`  ·  Student: ____  ·  Date frozen: ____  ·  Commit: ____

Any variable that tripped the shuffled-label leak warning: ____

---

## 1. Teacher competence gate

An arm may not run on a capability the teacher cannot do. Record the
check, and record skips as skips, never as zeros.

| Capability | Teacher accuracy | Gate (min_teacher_accuracy) | Arms allowed? |
|---|---|---|---|
| | XX.X | 0.60 | |

---

## 2. Main results (maps directly to Table 1)

| Set | Arm | Seed | pass@1 | pass@256 | Δpass@256 | NLF raw | NLF corrected | Causal NLF | Run dir |
|---|---|---|---|---|---|---|---|---|---|
| REPRESENTED | base student | - | XX.X | XX.X | - | - | - | - | - |
| REPRESENTED | offpolicy_kd | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| REPRESENTED | onpolicy_distill | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| REPRESENTED | rlvr | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| REPRESENTED | continued_pretrain | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| ABSENT | base student | - | XX.X | XX.X | - | - | - | - | - |
| ABSENT | offpolicy_kd | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| ABSENT | onpolicy_distill | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| ABSENT | rlvr | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| ABSENT | continued_pretrain | 0 | XX.X | XX.X | XX.X | XX.X | XX.X | XX.X | |
| - | teacher (reference) | - | XX.X | XX.X | - | - | - | - | - |

Repeat the block for seeds 1 and 2, then report mean and standard
deviation in the paper. Single-seed numbers are for debugging only.

---

## 3. Coverage signature (sharpening or expansion?)

Reweighting predicts pass@1 up with pass@256 flat or down. Acquisition
predicts the whole curve lifts.

| Capability | Arm | Δpass@1 | Δpass@4 | Δpass@16 | Δpass@64 | Δpass@256 | Verdict |
|---|---|---|---|---|---|---|---|
| | | XX.X | XX.X | XX.X | XX.X | XX.X | sharpening / expansion |

---

## 4. The representability correlation (Figure 2b)

One row per capability, pooling all three terciles including the middle
one. This is the result that generalizes beyond these specific tasks.

| Capability | R(c) | On-policy gain (points) |
|---|---|---|
| | X.XXX | XX.X |

Fitted: r = ____, p = ____, n = ____

---

## 5. Ablations

| Ablation | Setting | ABSENT-set gain | NLF corrected | Breaks the ceiling? | Run dir |
|---|---|---|---|---|---|
| Divergence direction | forward KL (beta=0) | XX.X | XX.X | | |
| Divergence direction | reverse KL (beta=1) | XX.X | XX.X | | |
| Divergence direction | interpolated (beta=0.5) | XX.X | XX.X | | |
| Token budget | 4x matched budget | XX.X | XX.X | | |
| Student scale | 0.6B | XX.X | XX.X | | |
| Student scale | 1.7B | XX.X | XX.X | | |
| Student scale | 2B (Gemma 2) | XX.X | XX.X | | |
| Teacher gap | 4x / 8x / 16x | XX.X | XX.X | | |
| Parameterization | LoRA | XX.X | XX.X | | |
| Mixed recipe | cpt then on-policy | XX.X | XX.X | | |
| Teacher top-k | 32 / 128 / 512 | XX.X | XX.X | | |
| Crosscoder sparsity | k = 16 / 32 / 64 | - | XX.X | | |
| Crosscoder width | 8x / 16x / 32x | - | XX.X | | |
| Crosscoder variant | batchtopk / delta | - | XX.X | | |

---

## 6. Running notes

Date, what you ran, what surprised you, what you will do next. Keep this
honest: the surprises are the paper.

- **YYYY-MM-DD** ·

---

## 7. Open threads

Things that are unresolved and would weaken the paper if a reviewer
found them first.

- [ ]
