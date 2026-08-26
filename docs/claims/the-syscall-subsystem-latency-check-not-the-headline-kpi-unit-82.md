# The syscall subsystem latency check (not the headline KPI — unit 82)

[← Claims index](../../CLAIMS.md)


- [SHIPPED] In-process adjudication p50 vs a spawned-hook baseline, measured on THIS machine, apples-to-apples (same `Fold` decide, two transports). Current report: 2.427 µs in-process vs 6.913 ms spawned `fak hook` (n=100) ⇒ ~2,849× (full-binary spawn). Witness: `report.json` `gate_primary=="pass"`; `report.json` `spawned_hook_baseline.p50_ns > 1ms`.
- [SHIPPED] The check is useful as a subsystem regression sentinel: it times the adjudication fold and confirms the decide path is not accidentally paying a per-call process boundary. It is deliberately **not** a production-readiness, model-quality, serving-throughput, or 45× fleet headline.

