# Session lifecycle reconciliation

`fak-dev sessiondiag --inventory [--json]` is the read-only lifecycle oracle for Codex/fak session artifacts. Its output is always a dry-run (`read_only: true`) and joins thread state, the latest turn, writer locks, guard launch receipts, spawn edges, child registrations, and independently observed PID plus process-start identity.

## Artifact contracts

| Artifact | Owner / creation | Update | Terminal / cleanup rule |
|---|---|---|---|
| Guard launch receipt (`fak-guarded-sessions/<thread>.json`) | the guarded launcher, once launch is admitted | none; it is an append-like launch receipt, not a heartbeat | retain as historical evidence; never infer liveness or delete solely by age |
| Writer lock (`thread-writer-locks/<thread>.lock`) | the Codex writer owning that thread | Codex ownership lifecycle | remove only after a terminal failed/interrupted turn is joined with **no** matching process tree; the inventory emits a reasoned `remove` proposal |
| Spawn edge (`thread_spawn_edges`) | Codex when parent spawns child | provider-owned status | an `open` edge without a live joined parent/child pair is proposed as `terminalize_unknown`; never count stale `open` as active |
| Child registration (`fak-child-registration/1`) | fak launch adapter, before child start | PID/start, heartbeat, terminal events | stale `active` with no exact PID/start match becomes proposed `lost`; stale pre-start registration without process identity becomes bounded `unknown(reason)` |

PID alone is never a lifecycle witness. Exact PID plus process-start time defeats reuse. A plan can be run repeatedly with the same evidence and returns the same actions; this makes failed-resume cleanup deterministic and idempotent.

The command does not mutate artifacts. Apply destructive filesystem or database cleanup only through an owner-specific guarded operation after previewing the JSON plan and re-reading process identity. Registration terminalization is append-only through `sessionregistry.Store.Transition`; rows are never edited in place. Cross-host integrity, authentication, bounded archive retention, and coverage metrics remain the operating-envelope work in #6459.
