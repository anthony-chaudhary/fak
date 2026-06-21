# API-Host Bridge Verify All

One-command verifier for the scope-bounded API-host bridge proof. A green scope_bounded_verification_gate confirms the bounded proof and reports residuals; it does not prove every API host on the internet.

## Summary

- Executed steps passed: 35/35
- Failed steps: 0
- Proof gate: yes
- Permission benchmark gate: yes
- Permission source-audit gate: yes
- Certificate gate: yes
- External-state audit gate: yes
- Qualification gate: yes
- Live-smoke queue gate: yes
- Live-smoke runner gate: yes
- Goal complete: no
- Residual requirements: 2
- Scope-bounded verification gate: yes

| step | phase | status | elapsed ms |
|---|---|---|---:|
| `permission_system_benchmark` | generate | passed | 136 |
| `permission_source_audit` | generate | passed | 5404 |
| `api_host_bridge_matrix` | generate | passed | 117 |
| `api_host_bridge_gate` | generate | passed | 12368 |
| `api_host_live_inventory` | generate | passed | 132 |
| `api_host_roster` | generate | passed | 212 |
| `api_host_readiness` | generate | passed | 665 |
| `api_host_acceptance` | generate | passed | 588 |
| `api_host_retry_packet` | generate | passed | 113 |
| `api_host_external_state_audit` | generate | passed | 111 |
| `api_host_compat_contract` | generate | passed | 146 |
| `api_host_conformance_certificate` | generate | passed | 111 |
| `api_host_qualification` | generate | passed | 126 |
| `api_host_live_smoke_queue` | generate | passed | 112 |
| `api_host_live_smoke_runner` | generate | passed | 126 |
| `api_host_bridge_proof` | generate | passed | 154 |
| `api_host_goal_audit` | generate | passed | 113 |
| `permission_source_audit_test` | test | passed | 729 |
| `permission_system_benchmark_test` | test | passed | 181 |
| `api_host_bridge_matrix_test` | test | passed | 153 |
| `api_host_bridge_gate_test` | test | passed | 408 |
| `api_host_live_inventory_test` | test | passed | 141 |
| `api_host_readiness_probe_test` | test | passed | 754 |
| `api_host_acceptance_probe_test` | test | passed | 901 |
| `api_host_roster_test` | test | passed | 641 |
| `api_host_retry_packet_test` | test | passed | 153 |
| `api_host_external_state_audit_test` | test | passed | 245 |
| `api_host_compat_contract_test` | test | passed | 186 |
| `api_host_conformance_certificate_test` | test | passed | 165 |
| `api_host_qualification_test` | test | passed | 186 |
| `api_host_live_smoke_queue_test` | test | passed | 147 |
| `api_host_live_smoke_runner_test` | test | passed | 462 |
| `api_host_bridge_proof_test` | test | passed | 531 |
| `api_host_goal_audit_test` | test | passed | 459 |
| `gateway_go_tests` | test | passed | 1452 |

## Residual Requirements

| requirement | status |
|---|---|
| `universal_any_api_host` | NOT_PROVEN |
| `paid_or_keyed_live_hosts` | EXTERNAL_STATE |
