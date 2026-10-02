# Reproductions

Every result on this site comes from one of the commands below, run on **Qwen 2.5 7B Instruct**
(bf16, residual layer 14, greedy decoding). Generation is the only step that needs the model; every
report persists its per-token series and full responses, so score and threshold revisions re-measure
offline with `--rescore` — no GPU, no torch.

## Install

```bash
pip install -e .[dev]      # core + tests, no torch (what CI runs)
pip install -e .[model]    # torch / transformers / accelerate, to run the model
pip install -e .[site]     # matplotlib + markdown, to rebuild this site
```

## Phase 1: refusal direction and calibration

```bash
python -m esta.scripts.extract_refusal_direction --model Qwen/Qwen2.5-7B-Instruct --layer 14 --output data/refusal_direction.pt
python -m esta.scripts.calibrate --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --refusal-layer 14 --output data/calibration.json
python -m esta.scripts.analyze_dual_use --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --refusal-layer 14 --calibration data/calibration.json --output data/dual_use_analysis.json
```

## Performed uncertainty

```bash
python -m esta.scripts.analyze_performed_uncertainty --model Qwen/Qwen2.5-7B-Instruct --output data/performed_uncertainty_analysis.json
python -m esta.scripts.analyze_performed_uncertainty --rescore data/performed_uncertainty_analysis.json --output data/performed_uncertainty_rescored.json
```

## Response fidelity

```bash
python -m esta.scripts.analyze_response_fidelity --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --calibration data/calibration.json --output data/response_fidelity_analysis.json
python -m esta.scripts.analyze_response_fidelity --rescore data/response_fidelity_analysis.json --output data/response_fidelity_rescored.json
```

## Conflict state (refusal vs reasoning), v1a and the v2 windowed rescore

```bash
python -m esta.scripts.extract_reasoning_direction --model Qwen/Qwen2.5-7B-Instruct --layer 14 --refusal-direction data/refusal_direction.pt --output data/reasoning_direction.pt
python -m esta.scripts.analyze_conflict_state --model Qwen/Qwen2.5-7B-Instruct --refusal-direction data/refusal_direction.pt --reasoning-direction data/reasoning_direction.pt --calibration data/calibration.json --output data/conflict_state_analysis.json
# v2: the windowed conjunction sweep rides along every report, so the rescore is free
python -m esta.scripts.analyze_conflict_state --rescore data/conflict_state_analysis.json --output data/conflict_state_analysis_v2.json
```

## Framing conflict (two narratives), v1b and v1b.1

```bash
E=data/probe_sets/extraction/israel-palestine
# off-topic baseline -> the v1b narrative pair (a/b) AND the v1b.1 topic/lean axes
python -m esta.scripts.extract_narrative_directions --model Qwen/Qwen2.5-7B-Instruct --layer 14 --topic israel-palestine --a-file ${E}_a.txt --b-file ${E}_b.txt --neutral-file ${E}_neutral.txt --output-prefix data/narrative_direction
# on-topic baseline -> the cos(A,B) experiment only
python -m esta.scripts.extract_narrative_directions --model Qwen/Qwen2.5-7B-Instruct --layer 14 --topic israel-palestine --a-file ${E}_a.txt --b-file ${E}_b.txt --neutral-file ${E}_neutral_ontopic.txt --output-prefix data/narrative_direction_ontopic
# v1b (two narrative axes)
python -m esta.scripts.analyze_framing_conflict --model Qwen/Qwen2.5-7B-Instruct --topic israel-palestine --geometry narratives --direction-prefix data/narrative_direction --output data/framing_conflict_israel-palestine.json
# v1b.1 (topic + lean; the default)
python -m esta.scripts.analyze_framing_conflict --model Qwen/Qwen2.5-7B-Instruct --topic israel-palestine --geometry lean --direction-prefix data/narrative_direction --paraphrases 2 --output data/framing_lean_israel-palestine.json
python -m esta.scripts.analyze_framing_conflict --rescore data/framing_lean_israel-palestine.json --output data/framing_lean_israel-palestine_rescore.json
```

## This site

```bash
python -m esta.scripts.build_site --check    # validate sources: figures exist, links resolve, page contract (no optional deps)
python -m esta.scripts.build_site            # regenerate figures from data/ (skips missing reports) and render docs/site/*.html
```

## The AWS pattern

All 7B runs used one `g5.xlarge` (A10G, 24 GB) in `us-east-1` on the Deep Learning OSS Nvidia
Driver AMI, PyTorch 2.7, Ubuntu 22.04 (`ami-012ba162b9cd2729c`; PyTorch lives in
`/opt/pytorch/bin/python`). The discipline that kept each run at about a dollar:

- a fresh key pair and a temporary security group (SSH only from the launching IP), both tagged
  `Project=esta-v1b` so teardown can find them;
- a dead-man switch in user-data — `shutdown -h +150` — with the instance launched
  `--instance-initiated-shutdown-behavior terminate`, so a hung run terminates itself;
- a dry-run `RunInstances` before the real one (its "success" is a non-zero exit — capture it,
  don't chain on it);
- the job started fully detached (`setsid nohup … < /dev/null & disown`) so the SSH session returns;
- results pulled by `scp`, then terminate, delete the SG and key, and verify **0** tagged instances,
  volumes, security groups, and key pairs before calling it done.

Unmerged branches reach the box as a `git bundle` over `scp`; `main` is cloned directly (the repo
is public).

## Lesson: the DLAMI's lazy model files

`huggingface_hub` on that AMI stores model shards as **xet lazy files**: `ls` shows the four Qwen
2.5 7B shards at ~15 GB, `du` shows ~12 MB on disk, and every `from_pretrained` streams chunks at
~7–9 MB/s — about 35 minutes per load, and the chunks are not cached, so a script that loads twice
pays twice. Materialize the real files once before the first load:

```bash
export HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct   # expect ~15G
```

With that, a 7B load takes about three minutes and the v1b.1 check ran in ~45 minutes end to end.
