# Security Policy

## Scope

This policy covers vulnerabilities in the ESTA codebase itself:
- The FastAPI server (`src/esta/api/`)
- The audit logger and integrity chain (`src/esta/audit/`)
- The schema validation layer (`src/esta/schema/`)
- The probe and inference modules (`src/esta/probes/`, `src/esta/inference/`)
- Build and deployment configuration (`pyproject.toml`, CI workflows)

**Out of scope:**
- Vulnerabilities in the underlying language models (Llama, Qwen, Mistral, etc.) — report those to the model authors.
- Vulnerabilities in upstream dependencies (PyTorch, transformers, FastAPI, pydantic) — report those to the respective projects.
- Prompts that produce undesirable model output. ESTA does not filter or block responses; it exposes state. Concerns about model alignment belong upstream.

## Reporting

Email reports privately to **security@TBD** (this placeholder will be replaced when project governance is formalized; until then, contact the maintainer directly via the address in the project's `pyproject.toml` `[project.authors]` block).

Please include:
- A description of the vulnerability and its impact
- Steps to reproduce, or a proof-of-concept
- The affected version (commit SHA or release tag)
- Your name and contact details for credit (optional)

Do **not** open public issues for security bugs.

## Process

- Acknowledgment within 5 business days
- Triage and severity assessment within 10 business days
- Coordinated disclosure: standard window is 90 days from acknowledgment; this can be shortened by mutual agreement (e.g., if a fix is straightforward and ships quickly) or extended for complex issues
- Public credit to reporters who request it, in the release notes for the patched version

## High-priority categories

ESTA is targeted at high-assurance deployments, so the following classes of bug are treated as critical:
- **Audit chain integrity bypass** — any way to silently rewrite history while passing `verify_chain()`
- **Metadata spoofing** — any way to produce an `epistemic_state` block whose values don't reflect the underlying generation state
- **Unauthenticated access in deployments where authentication is documented** — the default server has no auth; this section applies to documented hardened-deployment configurations
- **Activation/probe data leakage** — leaks of raw activation tensors via error messages, log files, or response fields in configurations where `return_activations` should be disabled
