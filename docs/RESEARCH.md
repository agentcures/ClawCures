# Research Notes (2026-02-15)

This document captures the external research used to drive architecture decisions for `ClawCures`.

## 1. Orchestration Interface: OpenClaw

Primary sources:
- OpenClaw OpenResponses API docs:
  - https://docs.openclaw.ai/openapi/openapi/responses_api/
- OpenClaw system prompt docs:
  - https://docs.openclaw.ai/openclaw/core/system_prompt/
- OpenClaw official repository:
  - https://github.com/openclaw/openclaw

Design implications:
- The planner/critic should use OpenResponses as the canonical interface.
- Campaign identity and safety constraints should be encoded in a single default system prompt.
- Structured JSON outputs are required to keep planning deterministic and executable.

## 2. Scientific Execution: refua-mcp

Primary local sources:
- `../refua-mcp/README.md`
- `../refua-mcp/src/refua_mcp/server.py`

Design implications:
- Campaign executor must remain tool-first and typed.
- `refua_validate_spec` should be used early to reduce expensive run failures.
- Async run support (`refua_job`) is essential for long-running fold/design jobs.

## 3. Regulatory and Translation Constraints

Primary sources:
- FDA page on AI/ML in drug development:
  - https://www.fda.gov/drugs/science-and-research-drugs/artificial-intelligence-and-machine-learning-ml-drug-development
- FDA CDER AI strategy framework:
  - https://www.fda.gov/drugs/science-and-research-drugs/cders-artificial-intelligence-strategy

Design implications:
- Autonomous planning should avoid cure claims and instead produce evidence-linked milestones.
- Runs need reproducibility and explicit rationale trails.
- Portfolio decisions must be interpretable and reviewable.

## 4. Frontier Modeling Inputs

Primary sources:
- AlphaFold 3 (Nature, 2024):
  - https://www.nature.com/articles/s41586-024-07487-w
- Therapeutics Data Commons overview (Nature Chemical Biology, 2022):
  - https://www.nature.com/articles/s41589-022-01131-2
- AlphaFlow (Nature Communications, 2025):
  - https://www.nature.com/articles/s41467-025-58305-z

Design implications:
- The campaign system should be model-agnostic and benchmark-aware.
- Multi-objective ranking (potency, confidence, safety, novelty) is necessary.
- Closed-loop autonomy should combine exploration (novel hypotheses) and exploitation (tractable programs).

## 5. Mission Framing

The objective "solve all human disease" is treated as a long-horizon mission target.

Operationally this is decomposed into:
- portfolio prioritization,
- target/assay strategy,
- design and screening,
- translational evidence packaging.

This framing avoids immediate cure claims while still optimizing for maximal disease impact.
