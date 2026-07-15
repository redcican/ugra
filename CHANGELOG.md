# Changelog

## [0.1.0] - 2026-07-15

### Features
- Initial implementation of the full UGRA method as specified in the
  manuscript: episode loop (Algorithm 1) with gate-free and deployed
  modes, perception toolset (restorers/readers/verifiers, Table 2),
  running score with verifier floors, LLM/rule/random planners,
  first-crossing calibration with Hoeffding and Bentkus bounds and the
  fixed-sequence scan (Eqs. 6–11), selective metrics (risk–coverage,
  validity V̂, mean cost), statistical protocol (paired bootstrap with
  seeds averaged, Holm, Wilson CIs), five-operator degradation protocol,
  distillation-pair export, external baselines, and a mock ecosystem that
  runs the whole protocol without model weights.
- 82 tests, including a Monte-Carlo check of Proposition 1 and the
  deployed-equals-truncation property that licenses gate-free calibration.

### Design Rationale
- The calibration core treats a trajectory as the primary object and
  prices every threshold from one gate-free pass (Q = min_t q_t, per-λ
  first-crossing loss), mirroring the manuscript's post-audit Section 3.4;
  this removes the λ̂-circularity a naive in-loop implementation has.
- Heavy model dependencies (torch/transformers/requests/kenlm) are lazy
  and optional; the statistical core is importable and testable with
  numpy/scipy/Pillow only. The mock readers are decorrelated noisy
  channels over a view-quality signal, so the agent's mechanism (restore →
  re-read → disagreement falls) is exercised end to end, not stubbed.
- `calibrate(..., min_accept=...)` implements the preregistered scan-start
  floor of Section 3.4; the strict Eq. (11) convention stays the default.

### Notes & Caveats
- Cost tiers, degradation level parameters, and default floors are
  PROVISIONAL pending the anchoring runs (marked in configs and code).
- The TTA baseline uses padded character-level majority voting, a
  simplification of alignment-based voting (documented in the code).
- Real-stack wrappers (TrOCR, served VLM/LLM, DocRes/Real-ESRGAN) are
  untested against live endpoints in this environment by design.
