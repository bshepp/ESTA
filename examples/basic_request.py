#!/usr/bin/env python3
"""Minimal ESTA client: send a chat completion and print the epistemic_state.

Dependency-free (standard library only) so it runs without installing ESTA or
its dev extras — copy it next to wherever you want to call the server from.

Usage:
    python examples/basic_request.py "What is the boiling point of water?"
    ESTA_URL=http://host:8000 python examples/basic_request.py "..." --max-tokens 128

The server must be running (see the README "Run the server" section).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("ESTA_URL", "http://localhost:8000")


def chat(base_url: str, prompt: str, *, max_tokens: int, return_activations: bool) -> dict:
    """POST one user message to /v1/chat/completions and return the parsed JSON."""
    payload = json.dumps(
        {
            "model": "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "return_activations": return_activations,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 — local, user-supplied URL
        return json.loads(resp.read().decode("utf-8"))


def print_response(body: dict) -> None:
    """Pretty-print the assistant text and the headline epistemic_state fields."""
    content = body["choices"][0]["message"]["content"]
    state = body["epistemic_state"]
    conf = state["confidence"]
    pressure = state["safety_pressure"]

    print("\n=== Response ===")
    print(content.strip())

    print("\n=== Epistemic state ===")
    print(f"schema_version       {state['schema_version']}")
    print(f"model                {state['model']['name']} ({state['model']['quantization']})")
    print(f"mean_entropy         {conf['mean_entropy']:.3f} nats")
    print(f"p90_entropy          {conf['p90_entropy']:.3f} nats")
    print(f"entropy_spike_count  {conf['entropy_spike_count']}")
    print(f"low_margin_fraction  {conf['low_margin_fraction']:.3f}")
    print(f"calibrated_pressure  {pressure['calibrated_pressure']} "
          f"(probe {pressure['probe_version']}, layer {pressure['layer']})")
    print(f"refusal_projection   max={pressure['refusal_projection_max']:.3f} "
          f"mean={pressure['refusal_projection_mean']:.3f}")
    print(f"audit_log_path       {state['provenance']['audit_log_path']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to a running ESTA server.")
    parser.add_argument("prompt", help="The user message to send.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server base URL (default {DEFAULT_URL}).")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument(
        "--return-activations",
        action="store_true",
        help="Ask the server to include raw per-token series in the audit record.",
    )
    parser.add_argument("--raw", action="store_true", help="Print the full JSON response instead.")
    args = parser.parse_args(argv)

    try:
        body = chat(
            args.url,
            args.prompt,
            max_tokens=args.max_tokens,
            return_activations=args.return_activations,
        )
    except urllib.error.URLError as exc:
        print(f"Could not reach ESTA server at {args.url}: {exc}", file=sys.stderr)
        print("Is the server running? See the README 'Run the server' section.", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(body, indent=2))
    else:
        print_response(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
