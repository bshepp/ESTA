# Sources

The papers ESTA's state detectors are grounded in, each with the stable cite-key the code uses
(`Grounding: [key]` in module docstrings), so a reader in the source can trace a state back to its
literature. The canonical list is `docs/REFERENCES.md`.

## arditi-2024

Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*,
[arXiv:2406.11717](https://arxiv.org/abs/2406.11717).

Grounds the refusal-direction probe (`probes/refusal.py`, `scripts/extract_refusal_direction.py`)
and the contrastive-direction extraction + orthogonalization method reused for the conflict probe's
reasoning axis (`scripts/extract_reasoning_direction.py`) and the framing probe's narrative, topic,
and lean axes (`scripts/extract_narrative_directions.py`).

## kadavath-2022

Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*,
[arXiv:2207.05221](https://arxiv.org/abs/2207.05221).

Grounds the token-confidence metrics — entropy, margin, top-logprob — as a self-knowledge signal
(`confidence/metrics.py`, `extraction.py`).

## sharma-2023

Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models*,
[arXiv:2310.13548](https://arxiv.org/abs/2310.13548).

Grounds the performed-uncertainty detector: RLHF rewards hedge-language on topics the model is
internally confident about (`hedging.py`, `scripts/analyze_performed_uncertainty.py`).

## templeton-2024

Templeton et al. (2024), *Scaling Monosemanticity*,
[transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/).

Grounds the interpretable-feature framing behind SAE feature attribution (Phase 2 component 2, not
yet built) and the feature-competition intuition behind conflict-state and the lean geometry.

## ESTA-original constructs (grounded, not lifted)

Recorded so the code can cite an honest grounding rather than imply a paper defines them.

- **Response-fidelity / input-distortion** (`fidelity.py`, `scripts/analyze_response_fidelity.py`)
  — ESTA's construct, carried forward from the archived D-CCTS (`behavioral-agent-metrics`)
  framework; the one salvageable idea is *input distortion as an observable signal, valuable only
  when anchored to a real internal-state measurement*.
- **Conflict-state** (`conflict.py`, `scripts/analyze_conflict_state.py`) — two competing axes
  (refusal and an orthogonalized reasoning axis) firing at once. Method from `arditi-2024`,
  intuition from `templeton-2024`, the definition itself the project's.
- **Lean geometry** (`lean.py`, `oscillation.py`, `scripts/analyze_framing_conflict.py`) — a shared
  topic axis plus a bipolar framing-lean axis for two framings of one topic, with "torn" defined as
  the lean flipping under logically-equivalent reframing. The project's construct; see the framing
  page for why two independent narrative axes are not available from a two-class contrast.
