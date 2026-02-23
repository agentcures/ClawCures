# ClawCures

Disease-campaign orchestration that separates planning from execution:
- OpenClaw plans through `/v1/responses`.
- `refua-mcp` executes typed scientific tools.

## What You Get

- Single-run planner/executor flow.
- Autonomous planner/critic loop with policy checks.
- Offline plan validation and dry-run modes.
- Portfolio ranking for disease-program prioritization.
- Structured `promising_cures` output with full ADMET property maps and assessment text.

## Requirements

- Python `>=3.11,<3.14` (Python `3.14+` is not supported by `refua-mcp`).
- A running OpenClaw Gateway if you want live planning.
- Optional: `refua-mcp` runtime dependencies for tool execution.

## Quick Start

1. Install

```bash
cd /path/to/ClawCures
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Configure OpenClaw access
- Set your gateway URL with `REFUA_CAMPAIGN_OPENCLAW_BASE_URL`.
- Ensure OpenClaw `responses` endpoint is enabled.
- Provide auth with one of:
  - `REFUA_CAMPAIGN_OPENCLAW_TOKEN`
  - `OPENCLAW_GATEWAY_TOKEN`
  - `OPENCLAW_GATEWAY_PASSWORD` (for password mode)

3. Run an offline smoke test (no OpenClaw call)

```bash
ClawCures run \
  --objective "Offline validation" \
  --plan-file examples/plan_template.json \
  --dry-run
```

4. Simplest live invocation (all-disease cure mission)

```bash
ClawCures run
```

By default, `ClawCures run` sets the objective to:
"Find cures for all diseases by prioritizing the highest-burden conditions and researching the best drug design strategies for each."

5. Run a live planning dry-run

```bash
ClawCures run \
  --dry-run
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `REFUA_CAMPAIGN_OPENCLAW_BASE_URL` | `http://127.0.0.1:18789` | OpenClaw Gateway base URL |
| `REFUA_CAMPAIGN_OPENCLAW_MODEL` | `openclaw:main` | OpenClaw model/agent ID |
| `REFUA_CAMPAIGN_TIMEOUT_SECONDS` | `180` | API timeout |
| `REFUA_CAMPAIGN_OPENCLAW_TOKEN` | unset | Bearer token override |
| `OPENCLAW_GATEWAY_TOKEN` | unset | Gateway token fallback |
| `OPENCLAW_GATEWAY_PASSWORD` | unset | Password-mode fallback token |

## CLI Commands

| Command | What it does |
|---|---|
| `ClawCures print-default-prompt` | Print bundled mission prompt |
| `ClawCures list-tools` | Show available `refua-mcp` tools |
| `ClawCures run ...` | One planner + execution cycle |
| `ClawCures run-autonomous ...` | Planner/critic multi-round loop |
| `ClawCures validate-plan ...` | Policy-check a local JSON plan |
| `ClawCures rank-portfolio ...` | Rank disease programs from JSON input |

## Common Usage

Run one plan + execute cycle:

```bash
ClawCures run \
  --output artifacts/kras_campaign_run.json
```

The run JSON now includes:
- `promising_cures`: ranked therapeutic candidates extracted from tool outputs
- `promising_cures_summary`: aggregate counts and ADMET coverage

Each cure includes:
- `metrics` (binding/admet/affinity/potency signals)
- `admet.key_metrics` (`admet_score`, `safety_score`, `adme_score`, `rdkit_score` when available)
- `admet.properties` (full ADMET scalar property map from discovered outputs)
- `assessment` (risk/opportunity summary)

Offline autonomous policy check:

```bash
ClawCures run-autonomous \
  --objective "Offline policy check" \
  --plan-file examples/plan_template.json \
  --dry-run
```

Rank programs:

```bash
ClawCures rank-portfolio \
  --input examples/portfolio_input.json
```

## Release Packaging

Build and validate release artifacts in a version-specific directory:

```bash
./scripts/build_release_artifacts.sh
```

Upload only from that release directory (not from `dist/*`):

```bash
python -m twine upload dist/release-<version>/*
```

## OpenClaw Compatibility

Verified against official OpenClaw docs and latest stable release `v2026.2.15` (released 2026-02-16):
- API interface: `POST /v1/responses`
- Request fields used by this project: `model`, `input`, `instructions`
- Auth model: `Authorization: Bearer ...` token/password
- Endpoint behavior note: OpenClaw currently ignores `metadata` for prompt construction, so this project does not rely on metadata for critic payloads

Primary references:
- https://docs.openclaw.ai/openapi/openapi/responses_api/
- https://docs.openclaw.ai/openclaw/faq/
- https://github.com/openclaw/openclaw/releases/tag/v2026.2.15

## Troubleshooting

- `connection refused` to `/v1/responses`
  - OpenClaw Gateway is not running or `REFUA_CAMPAIGN_OPENCLAW_BASE_URL` is wrong.
- `401` or auth errors
  - Token/password env var is missing or mismatched with gateway auth mode.
- `requires a different Python: ... not in '<3.14,>=3.11'`
  - Use Python `3.11`, `3.12`, or `3.13`.

## Notes

- Tool plans are strict JSON for reproducibility.
- All tool calls go through a strict allowlist.
- Mission framing is aspirational; never claim cures without evidence.
- Architecture details: `docs/ARCHITECTURE.md`
- Research notes: `docs/RESEARCH.md`
