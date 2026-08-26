# Codex logical-session continuity

`sessionctl.CodexSession` owns the durable fak identity, semantic cursor, execution epoch, input lease, and replay journal for a Codex-backed session. `codexsession.Adapter` owns only the provider wire protocol.

- `New`, `Resume`, and `Fork` are explicit operations. A failed resume never creates a hidden replacement thread.
- The Codex thread ID and adapter version are persisted as opaque provider coordinates beneath the fak logical-session ID.
- Every child-process reconstruction increments the execution epoch. Events from an older child are rejected as `stale_epoch`.
- Semantic deliveries are persisted before they enter the live tail and deduplicated by epoch-scoped delivery ID.
- `Attach(after)` captures replay and installs the live subscription under one lock, preventing a replay/live handoff gap.
- One input lease admits the writer; concurrent writers receive a typed `input_lease_held` recovery error.
- Missing and incompatible upstream state expose typed, recoverable choices rather than guessing.

The deterministic kill/restart witness is `TestCodexReconnectRestartOrderedWithoutDuplicateEffect`: two browser leases and two reopened adapter states retain one logical session, one opaque Codex thread, two ordered turns, and no duplicate effect.
