# Architecture

## Goal

`ClawCures` orchestrates disease-scale discovery campaigns with a strict split:

- OpenClaw plans.
- `refua-mcp` executes.

## Components

- `refua_campaign.openclaw_client.OpenClawClient`
  - Calls OpenClaw Gateway OpenResponses endpoint (`/v1/responses`).
  - Returns raw response plus extracted text.

- `refua_campaign.refua_mcp_adapter.RefuaMcpAdapter`
  - Executes typed `refua-mcp` calls through `refua_mcp.server`.
  - Enforces an allowlist of tools.

- `refua_campaign.orchestrator.CampaignOrchestrator`
  - Builds planner instructions.
  - Parses JSON tool plans.
  - Dispatches execution via the adapter.

- `refua_campaign.autonomy.AutonomousPlanner`
  - Runs planner/critic loops across multiple rounds.
  - Applies policy checks (tool allowlist, max calls, validation-first preference).
  - Records critique traces for reproducibility and audits.

- `refua_campaign.portfolio`
  - Ranks disease programs by weighted factors.
  - Supports portfolio-level prioritization before expensive model execution.

- `refua_campaign.cli`
  - `run` for planning/execution.
  - `run-autonomous` for planner/critic/policy closed-loop planning.
  - `validate-plan` for offline policy checks.
  - `rank-portfolio` for disease portfolio ranking.
  - `run --dry-run` for plan-only mode.
  - `run --plan-file` to bypass planner and load a local plan.

## Data Contract

Planner output must be strict JSON:

```json
{
  "calls": [
    {"tool": "refua_validate_spec", "args": {"entities": []}}
  ]
}
```

No markdown or free-text wrappers are expected.

## Prompt Strategy

Default system prompt is mission-driven and includes:

- "Primary mission: solve all human disease."
- a long-horizon interpretation to prevent ungrounded cure claims.
- prioritization of highest-burden/highest-fatality diseases first.
- emphasis on researching the best practical drug-design strategies per disease.
- requirements for measurable milestones and typed tool calls.
