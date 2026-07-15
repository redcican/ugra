# UGRA — Uncertainty-Guided Reading Agent

Reference implementation of the manuscript **"An Uncertainty-Guided
Perception, Reasoning, and Action Agent for Text Recognition in Degraded
Documents"** (Pattern Recognition, in preparation).

A document region enters a budgeted perception–reasoning–action loop: two
recognizers of different families read the current view, their normalized
disagreement acts as the uncertainty signal, a planner spends a tool budget
on re-perception (binarize, restore, magnify) or verification (lexicon,
character language model, layout), and calibrated gates decide. Accepted
transcriptions carry a distribution-free PAC (α, δ) bound on their **mean**
character error rate; everything else is deferred to a human reader.

## Repository map (module → paper section)

| Module | Paper | Content |
|---|---|---|
| `ugra/types.py` | §3.1 | State s_t = (v_t, H_t, e_t), actions, trajectories |
| `ugra/metrics/edit.py` | Eqs. 1, 2, 5 | clipped CER, WER, symmetric disagreement, pooled CER |
| `ugra/tools/` | §3.2, Table 2 | restorers / readers / verifiers with cost tiers |
| `ugra/agent/state.py` | Eq. 3 | working hypothesis, z̄ = 0 convention, tie rules |
| `ugra/agent/score.py` | §3.4 | running score q_t with verifier floors |
| `ugra/agent/serialization.py` | §3.3 | planner summary (threshold-free by construction) |
| `ugra/agent/planner.py` | §3.3, §4.4 | LLM planner + rule/random ablation planners |
| `ugra/agent/loop.py` | Alg. 1 | gate-free logging mode and deployed mode |
| `ugra/calibration/bounds.py` | Eq. 8, Lemma 2 | Hoeffding and Bentkus upper confidence bounds |
| `ugra/calibration/gate.py` | Eqs. 6–11 | first-crossing pricing, fixed-sequence scan, λ̂ |
| `ugra/metrics/selective.py` | §4.1 | risk–coverage sweep, validity V̂, mean cost |
| `ugra/eval/stats.py` | §4.1 | paired bootstrap (seeds averaged), Holm, Wilson CIs |
| `ugra/eval/baselines.py` | §4.1 | single pass, TTA at fixed cost, restore-then-read |
| `ugra/degradation/operators.py` | §4.1 | five operators × five severities, pixel-reproducible |
| `ugra/distill/dataset.py` | §3.5 | accepted-prefix (summary, action) pairs → JSONL |
| `ugra/mock.py` | — | decorrelated noisy-channel readers for dry runs and tests |

## Install

```bash
cd ugra
pip install -e .            # core: numpy, scipy, Pillow, PyYAML
pip install -e .[models]    # + torch/transformers/requests for the real stack
pip install -e .[dev]       # + pytest
```

## Quickstart — full protocol without any model weights

```bash
python scripts/demo_mock.py
```

runs 600 gate-free episodes on the mock ecosystem, calibrates the gate on
one half, and evaluates on the other:

```
calibrated threshold lambda-hat = 0.03
realized selective risk  : 0.0000  (target alpha = 0.05)
certified coverage       : 32.0%
empirical validity V-hat : 1.00  (requires >= 0.90)
```

The mock channel is deliberately faithful to the paper's mechanism: two
readers with decorrelated errors read through a view-quality channel,
restorers raise quality, so disagreement falls as the agent re-perceives —
the signal the gate calibrates. Risk stays under α by construction of the
guarantee; coverage is whatever the evidence permits.

## The real stack

`configs/default.yaml` pins the instantiation of Section 4.1: TrOCR-large
(specialist), Qwen3-VL-30B-A3B-Instruct behind an OpenAI-compatible
endpoint (generalist), Qwen3-235B-A22B (planner), Qwen3-8B (distilled
student), DocRes/Real-ESRGAN (learned restorers), KenLM 5-gram (character
model). Readers and planners are constructed from the config; every wrapper
imports its heavy dependency lazily, so the calibration/metrics core runs
without any of them.

Two invariants matter more than any model choice:

1. **Calibration episodes run gate-free** (`gate_lambda=None`) and log the
   full score trajectory; one pass prices every candidate threshold via
   first crossings (`ugra.calibration.gate`). The planner summary never
   mentions the threshold, so the deployed episode at λ̂ is a truncation of
   the logged one — `tests/test_loop.py::TestDeployedTruncation` asserts
   exactly this.
2. **The calibrated loss is clipped at 1** (Eq. 1); the concentration
   bounds require it. Reported corpus error rates pool raw edit distances
   and are unclipped (`pooled_cer`).

## Tests

```bash
pytest            # 82 tests: unit + integration + a Prop. 1 simulation
pytest -m unit    # fast subset
```

`tests/test_gate.py::TestProposition1` simulates the PAC guarantee over
200 calibration draws and checks the violation frequency against δ.

## Reproducibility notes

- All stochastic components take explicit seeds; the degradation protocol
  is reproducible to the pixel (`tests/test_degradation.py`).
- The fixed-sequence scan implements strict Eq. (11) by default: an empty
  bottom grid cell vetoes certification. The preregistered scan-start
  floor of Section 3.4 (`calibrate(..., min_accept=...)`) is measurable in
  the scores alone and leaves the argument untouched.
- Cost tiers, floors, α, δ, grid, budget, seeds: `configs/default.yaml`,
  matching the paper's Section 4.1 (PROVISIONAL values marked there).

## Status

Research code accompanying a manuscript in preparation; the experiment
numbers in the paper are provisional until the full runs land. The
calibration core (pricing, bounds, scan) and the statistical protocol are
final and tested; model wrappers are thin and expect served endpoints.
