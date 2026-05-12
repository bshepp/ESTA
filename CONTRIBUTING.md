# Contributing to ESTA

Thanks for your interest. ESTA is in active early development; contributions are welcome but please coordinate on substantive work before opening a large PR.

## Code of conduct

Be civil and assume good faith. The project targets high-assurance deployment contexts, so the bar for technical claims is high — disagreements about methodology are welcome, but should be grounded in citations or reproducible experiments.

## Developer Certificate of Origin

All commits must be signed off under the [Developer Certificate of Origin](https://developercertificate.org/). Add the sign-off line to every commit:

```
git commit -s -m "your message"
```

This adds a `Signed-off-by: Your Name <you@example.com>` trailer that certifies you have the right to submit the contribution under the project license.

A Contributor License Agreement may be required for substantive contributions once governance is formalized; this section will be updated when that happens.

## Development setup

```bash
git clone https://github.com/YOUR_ORG/esta.git
cd esta
python -m venv .venv
source .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -e .[dev]
```

For changes that touch the inference path (model loading, hooks, generation), you'll also need the model runtime:

```bash
pip install -e .[model]
```

## Pre-PR checks

Run both before pushing. CI runs the same commands:

```bash
ruff check src tests
pytest -q
```

Unit tests must remain green and runnable without the `[model]` extra installed. Integration tests that load model weights belong in `tests/integration/` and must be marked `@pytest.mark.requires_model`.

## What to work on

The full design document is at [`docs/epistemic-transparency-agent (1).md`](<docs/epistemic-transparency-agent (1).md>). Phase 1 is the current focus; Phase 2 (conflict detection, SAE feature attribution) and Phase 3 (federal integration) are tracked but not yet under active development.

Good first contributions:
- Improving validation prompt sets in `data/validation_cases/`
- Expanding architecture coverage in `src/esta/inference/hooks.py:resolve_residual_layer`
- Documentation gaps in `docs/`
- Test coverage for edge cases in the audit chain / confidence metrics

## Things to avoid

- **Don't introduce false precision in metadata.** If a probe is uncalibrated or experimental, the schema should say so. The project's value is honesty about what we can and can't claim.
- **Don't bypass the schema versioning.** Adding fields requires a schema version bump and a migration note.
- **Don't commit model weights, refusal-direction tensors, or calibration sets.** These belong in `data/` and are gitignored.

## Reporting security issues

See [SECURITY.md](SECURITY.md). Do not open a public issue for security bugs.
