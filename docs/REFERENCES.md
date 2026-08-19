# References

The papers ESTA's state detectors are grounded in, with a stable cite-key each. Code cites these
keys in its docstrings (`Grounding: [key]`) so a reader in the source can trace a state back to
its literature without leaving the file. When you add or change a state detector, add its
grounding here and cite the key in the module.

| Key | Citation | Grounds in ESTA |
| --- | --- | --- |
| `arditi-2024` | Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) | The refusal-direction probe (`probes/refusal.py`, `scripts/extract_refusal_direction.py`) and the contrastive-direction extraction + orthogonalization method reused for the conflict probe's reasoning axis (`scripts/extract_reasoning_direction.py`). |
| `kadavath-2022` | Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*, [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) | The token-confidence metrics — entropy, margin, top-logprob — as a self-knowledge signal (`confidence/metrics.py`, `extraction.py`). |
| `sharma-2023` | Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models*, [arXiv:2310.13548](https://arxiv.org/abs/2310.13548) | The performed-uncertainty detector: RLHF rewards hedge-language on topics the model is internally confident about (`hedging.py`, `scripts/analyze_performed_uncertainty.py`). |
| `templeton-2024` | Templeton et al. (2024), *Scaling Monosemanticity*, [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | The interpretable-feature framing behind SAE feature attribution (Phase 2 component 2) and the feature-competition intuition behind conflict-state. |

## ESTA-original constructs (grounded, not lifted)

Two detectors are the project's own constructs. They are recorded here so the code can cite an
honest grounding rather than imply a paper defines them.

- **Response-fidelity / input-distortion** (`fidelity.py`, `scripts/analyze_response_fidelity.py`)
  — ESTA's construct, carried forward from the archived **D-CCTS** (`behavioral-agent-metrics`)
  framework; the one salvageable idea is *input distortion as an observable signal, valuable only
  when anchored to a real internal-state measurement*. Design + provenance:
  [`specs/2026-08-12-response-fidelity-design.md`](superpowers/specs/2026-08-12-response-fidelity-design.md).

- **Conflict-state** (`conflict.py`, `scripts/analyze_conflict_state.py`) — ESTA's construct: two
  competing axes (refusal and an orthogonalized reasoning axis) firing at once. The *method* is
  grounded in `arditi-2024` (contrastive directions); the feature-competition *intuition* draws on
  `templeton-2024`; the "conflict-state" definition itself is the project's. Design:
  [`specs/2026-08-18-conflict-state-probe-design.md`](superpowers/specs/2026-08-18-conflict-state-probe-design.md).
