# Daily Git hygiene dogfood — 2026-08-06

Issue: #5581  
Spine: `internal/gitdaily r3+g6b61f1dea8` (`fak git-daily`)

## Verdict

The live repository run completed without an incident. It reaped one stale commit lock whose owner PID was dead, ran every non-destructive maintenance tier, and folded 22,820 loose objects to zero. Grace-prune remained explicitly disabled (`PRUNE_OFF`), so the run deleted no object by age.

## Captured command

```powershell
.\fak.exe git-daily -force -json -root .
```

Captured at `2026-08-06T12:08:52-07:00` on the shared trunk checkout.

## Readout

| Signal | Before | After / verdict |
|---|---:|---:|
| Loose objects | 22,820 | 0 |
| Packed objects | 703,272 | 748,918 |
| Packs | 4 | 4 |
| Stale commit locks | 1 | reaped; dead PID 64960 |
| Always-safe tiers | planned | multi-pack-index + commit-graph ran |
| Safe-with-grace tiers | planned | loose-objects + prune-packed + incremental-repack ran |
| Grace prune | off | `PRUNE_OFF` (expected default) |
| Incident | — | false |

The loose-object delta is a consolidation witness, not a deletion claim: objects moved into packs while the default age-based prune tier stayed off.

## Defect census

No product defect surfaced in this run. The dry-run immediately before apply correctly predicted the stale commit-lock reap and showed that the lock would hold the safe-with-grace tier back; the applied run reaped it first and then completed the tier, which is the spine's ordering contract.

## Reproduction

```powershell
.\fak.exe git-daily -dry-run -json -root .
.\fak.exe git-daily -force -json -root .
.\fak.exe git-daily -status 1 -json -root .
```
