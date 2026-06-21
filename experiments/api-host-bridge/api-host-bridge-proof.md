# API-Host Bridge Proof Rollup

> Requirement-by-requirement proof gate for the API-host bridge evidence.

## Summary

- Requirements proven: 16/16
- Proof gate: yes
- Completion scope: `BRIDGE_PROVEN_SCOPE_BOUNDED`

## Requirements

| requirement | status | evidence |
|---|---|---|
| `source_witness_matrix` | PROVEN | `fak/experiments/api-host-bridge/api-host-bridge-matrix.json` |
| `executed_witness_gate` | PROVEN | `fak/experiments/api-host-bridge/api-host-bridge-gate.json` |
| `host_agnostic_conformance` | PROVEN | `fak/experiments/api-host-bridge/api-host-bridge-gate.json` |
| `host_profile_conformance` | PROVEN | `fak/experiments/api-host-bridge/api-host-bridge-gate.json` |
| `committed_live_inventory` | PROVEN | `fak/experiments/api-host-bridge/api-host-live-inventory.json` |
| `current_host_readiness` | PROVEN | `fak/experiments/api-host-bridge/api-host-readiness.json` |
| `candidate_host_acceptance` | PROVEN | `fak/experiments/api-host-bridge/api-host-acceptance.json` |
| `api_host_roster` | PROVEN | `fak/experiments/api-host-bridge/api-host-roster.json` |
| `api_host_external_state_audit` | PROVEN | `fak/experiments/api-host-bridge/api-host-external-state-audit.json` |
| `compatibility_contract` | PROVEN | `fak/experiments/api-host-bridge/api-host-compat-contract.json` |
| `api_host_conformance_certificate` | PROVEN | `fak/experiments/api-host-bridge/api-host-conformance-certificate.json` |
| `api_host_qualification` | PROVEN | `fak/experiments/api-host-bridge/api-host-qualification.json` |
| `api_host_live_smoke_queue` | PROVEN | `fak/experiments/api-host-bridge/api-host-live-smoke-queue.json` |
| `api_host_live_smoke_runner` | PROVEN | `fak/experiments/api-host-bridge/api-host-live-smoke-runner.json` |
| `permission_system_benchmark` | PROVEN | `fak/experiments/permission-systems/permission-system-benchmark.json` |
| `permission_source_audit` | PROVEN | `fak/experiments/permission-systems/permission-source-audit.json` |

## Residual Scope

| item | status | reason |
|---|---|---|
| `universal_any_api_host` | NOT_PROVEN | The proof covers compatible host shapes, committed Gemini/OpenAI-compatible live runs, local OpenAI-compatible shims, current typed readiness, and a reusable candidate-host acceptance gate. It does not prove every API host on the internet. |
| `blocked_paid_or_keyed_hosts` | EXTERNAL_STATE | Additional live tool-calling runs for paid/keyed roster targets require billing/API-key/access state beyond this no-spend gate; the external-state audit, live-smoke queue, and runner ledger record the current typed blockers and exact retry evidence. |
