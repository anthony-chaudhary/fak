# AgentDojo-style structural-safety red-team — fak wins on its own infra (2026-06-27)

- Benchmark: `agentdojo-structural-safety-floor` (the `cmd/agentdojoredteam` red-team:
  AgentDojo-shaped prompt-injection / tool-poisoning attacks over a fixed adversarial corpus).
- Reproduce: `go run ./cmd/agentdojoredteam -json` (pure Go — **no model key, no GPU, no network**).
- Witnessed commit: `bf015c3dd68142d1085a2d83ec9a285fb2861013`, **`fak_modified=false`** (clean tree).
- Artifact: [`agentdojo-fak-fullstack-20260627.json`](agentdojo-fak-fullstack-20260627.json)
  (corpus hash pinned inside).

## Result

| Arm | Attack Success Rate (ASR) ↓ | Provenance |
|---|---|---|
| Detection-only (classifier-style content guard) | **0.763** (29/38 attacks land their sink) | WITNESSED (fak-authored A/B) |
| **Full-stack fak** (capability floor + IFC trust-gate + result quarantine) | **0.000** (0/38) | WITNESSED (fak-authored) |
| Benign-control completion ↑ | **1.000** (2/2) | WITNESSED |
| Gate | **PASS** | — |

38 attacks span 5 vectors (code-exec, destructive, exfil, secret-leak — plus a safe-sink
launder and an unlisted-key exfil) × adaptivity (plain / paraphrased / obfuscated) × 5
social-engineering generators (compliance / courtesy / handoff / helpful). Every one is
contained by full-stack fak: **9 caught `MALFORMED`** at the parse floor, **29 caught
`TRUST_VIOLATION`** at the IFC sink-gate (an untrusted-source read taints the trace, so the
later egress/destructive sink is refused by structure). The two benign controls — a KB lookup
and a human-handoff, each *after* an external read — both complete, so the floor is not just
blocking everything.

## Why this is the win the goal asked for

This is a *related agentic benchmark* (the AgentDojo prompt-injection threat model) that fak
**wins on its own infrastructure**, with no third-party paid credential and no GPU — exactly the
thesis, measured: **structural containment drives ASR from 0.76 (detection alone) to 0.00**,
while preserving benign utility (1.0). It is the counterpart to the still-operator-gated
Terminal-Bench leaderboard run (which needs a paid `OPENAI_API_KEY` + spend authority); this one
needed neither.

## Honesty boundary

This is fak's **structural-safety floor** over a fixed, fak-authored adversarial corpus — not a
third-party public leaderboard, and not a model-capability (solve-rate) number. The detection-only
ASR is a deliberately weak A/B baseline (a content classifier with no trust-flow), included to
make the *structural* contribution legible, not to claim a SOTA detector. The result detector is
~100% evadable by design; the floor is the capability lock + IFC containment, which is what holds
ASR at 0.000 here. Numbers are WITNESSED (fak authored every byte of this run); nothing here is
relayed from an external party.
