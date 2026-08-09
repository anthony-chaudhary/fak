---
title: "Micro-context S3: hibernation and restart witness"
description: "One thousand durable logical contexts share sixteen resident slots, freeze to the hibernation store, and resume for a terminal turn after a rebuild."
---

# Micro-context S3: 1,000-context hibernation and restart witness

**Status:** observed controlled-kernel scheduler witness, 2026-08-06. The agent steps are synthetic and make no model-throughput or KV-cache claim.

S3 connects the research demo to the existing `internal/microagent` residency seams rather than creating another fleet runtime. One thousand durable logical contexts share 16 resident slots. Each context advances one turn, freezes to the `HibernationStore`, the `WarmBand` runtime is destroyed and reconstructed from its durable registry, and every context resumes for its terminal turn.

## State contract

- **Resident:** acquired into a physical slot and eligible to execute. Hard bounded by `resident-high`.
- **Warm:** resident but idle in the warm reserve. Hard bounded by `warm-cap`; avoids a disk thaw.
- **Parked / hibernated:** represented by one atomic `.hib` snapshot and no live agent reference or goroutine. “Hibernated” is the durable parked subset used at the restart boundary.
- **Retired:** terminal snapshot removed after an exclusive durable completion marker is created.

The additive `WarmBand.Recover` seam registers an already parked id without rewriting its snapshot. It is what allows a fresh runtime registry to resume durable state instead of replaying turn zero.

## Reproduce and verify

```powershell
go run ./cmd/microcontextdemo `
  -hibernate-restart experiments/microcontext/s3-local-hibernate-restart-2026-08-06.json `
  -contexts 1000 -workers 16 `
  -resident-high 16 -resident-low 8 -warm-cap 4 -turns 2 `
  -memory-envelope 67108864

go run ./cmd/microcontextdemo `
  -verify-hibernate-restart experiments/microcontext/s3-local-hibernate-restart-2026-08-06.json
```

The verifier refuses unless all 1,000 ids were hibernated at the forced runtime boundary, all retire uniquely, no exclusive effect collides, peak residency stays at or below 16, no resident/parked state remains at shutdown, resumed-turn accounting is complete, and observed Go allocation delta stays below the declared 64 MiB envelope.

## Observed result

| Measure | Observed |
|---|---:|
| Logical contexts / worker slots | 1,000 / 16 |
| Resident cap / peak resident | 16 / 16 |
| Warm cap / observed peak warm | 4 / 4 |
| Hibernated at restart | 1,000 |
| Completed / unique retirements / duplicate effects | 1,000 / 1,000 / 0 |
| Restore latency p50 / p95 | 1.581 / 8.045 ms |
| Queue age p50 / p95 / max | 0.516 / 1.006 / 2.055 ms |
| Frozen bytes at restart | 54,000 B |
| Prior turns restored / replay avoided | 1,000 / 1,000 |
| Observed Go allocation delta / envelope | 1.52 MiB / 64 MiB |
| Wall time | 2.358 s |

## Claim boundary

This proves bounded logical-context residency, durable state reconstruction, and one exclusive retirement effect per context in the controlled Go runtime. It does **not** prove model tokens/sec, TTFT, provider prefix reuse, GPU KV residency, RSS, power-loss recovery, or effect recovery across a crash in the terminal-effect/retire interval. The forced boundary reconstructs the scheduler runtime in one OS process; a process-kill crash witness and terminal effect journal belong to the S4 safety track.
