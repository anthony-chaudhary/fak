---
title: "Native Harness Database and Data-Slot Lifecycle Architecture"
description: "Architecture and lifecycle contract for internal session persistence and local target database discovery, query, and migration safety in the native fak harness."
---

# Native Harness Database and Data-Slot Lifecycle Architecture

**Date:** 2026-09-03  
**Status:** Architectural Decision & Research Input  
**Related Masters:** #10646 (removable capability slots), #10652 (dormant data slot)  
**Implementation Tickets:** #10834, #10835, #10836, #10837, #10838  

---

## 1. Executive Summary

This note defines the complete database architecture and lifecycle contract for the **native `fak` harness** (`fak agent`, `fak serve --native`, `fak chat`).

The architecture resolves two historically conflated responsibilities:
1. **The Internal Harness State & Memory Layer** (how `fak` persists sessions, trajectories, and memories).
2. **The Local Program / Target Workload Layer** (how the agent safely discovers, queries, tests, and verifies databases used by the user's application).

It explicitly answers how the system operates by default on **brand new machines**, across **fresh repository clones**, and through **subsequent updates and schema migrations** without requiring manual daemon setup, external C-libraries (CGo), or fragile configuration.

---

## 2. Internal Harness State: Zero-CGo, Zero-Daemon Floor

### The Day-0 Machine Contract
On a fresh machine with only the static `fak` binary:
* **Zero External DBMS Dependencies**: `fak` never requires a local Postgres, MySQL, Redis, or Docker daemon to run. The core Go module root stays stdlib-only plus `x/term` and `x/sys`.
* **Append-Only Event WAL (`sessionjournal`)**: All turns, lifecycle events, and tool records append to `session-journal.jsonl`.
* **Boot-Epoch Crash Reconciliation**: System-level crashes (e.g., OS reboot, terminal termination) are detected deterministically by comparing session start timestamps against the host boot epoch (`sessionjournal.Classify`), avoiding database lock recovery corruption.
* **Curated Memory (`memq`)**: Operates as a pure-Go in-memory CAS blob store and page table (`MemStore`), querying markdown fact files (`NotesBackend`, `MEMORY.md`) with read-time artifact re-verification (`recall.ArtifactVerifier`).
* **Offline Diagnostic Tooling**: Deep historical queries over past external sessions (e.g., inspecting Codex `logs_2.sqlite` or OpenCode SQLite DBs) are isolated in quarantined nested modules or Python tools, preventing CGo from leaking into the core binary.

### Updates & Schema Evolution
* **Schema-Tagged Event Ledgers**: Every durable JSONL line carries a typed `schema` identifier (e.g., `fak.sessionjournal.v1`).
* **Forward-Tolerant Decoders**: Newer binaries ignore additive unknown fields; unexpected schema collisions trigger fail-closed `SchemaCollisionError` rather than silently corrupting existing ledgers.
* **Atomic Self-Update (`fak self-update`)**: Rebuilds the candidate binary in a temporary build directory, verifies it against the test gate, and atomically replaces the binary on disk.

---

## 3. Local Program / Workload Layer: The Dormant Data-Slot

When an agent is dropped into a user's repository, the local program often depends on databases (SQLite, PostgreSQL, MySQL, Redis, etc.). The harness governs these through the **Data-Slot Lifecycle**.

### Progressive Lifecycle States

```text
               ┌──────────────┐
               │    Absent    │ (No database files or migration configs detected)
               └──────┬───────┘
                      │ Repository contains prisma/schema.prisma or alembic/
                      ▼
        ┌───────────────────────────┐
        │  Dormant: Unmaterialized  │ (Schema detected; DB file not yet created)
        └─────────────┬─────────────┘
                      │ Migration runs or local db exists (e.g. dev.sqlite)
                      ▼
        ┌───────────────────────────┐
        │      Dormant: Ready       │ (Zero sockets; 0 locks; read metadata cached)
        └─────────────┬─────────────┘
                      │ Agent requests db_schema or db_query
                      ▼
        ┌───────────────────────────┐
        │          Active           │ (Bounded read-only reflection & query)
        └───────────────────────────┘
```

1. **`absent`**: The repository contains no database files, migration configs, or container database services. Zero tools or prompts are injected.
2. **`dormant: unmaterialized`**: Migration configs exist (e.g., `prisma/schema.prisma`, `alembic/`), but the physical database file does not yet exist. The harness prompts the agent that a setup/migration step is required before querying.
3. **`dormant: ready`**: A local database file or local container service is detected. No network sockets are opened, and no file locks are held. The capability is advertised in the dormant catalog without wasting context tokens.
4. **`active`**: When the agent proposes a database action, the connection is validated against the policy gate and executed with bounded read-only constraints.

---

## 4. Default Shipped Spine: Local SQLite + Standard Migrations

To honor the spine-first doctrine ([`docs/spine-first-defaults.md`](../spine-first-defaults.md)), the native harness ships a minimal, end-to-end working slice out of the box:

### Components of the Spine
1. **Static Detection (#10834)**:
   * Discovers `*.db`, `*.sqlite`, `*.duckdb`, Prisma, Alembic, goose, Flyway, and Docker Compose database services.
   * Pure Go, local filesystem only, zero network calls.
2. **Bounded Reflection & Query (#10835)**:
   * In-process schema reflection (`db_schema`) reading `sqlite_master` into compact JSON (<500 tokens).
   * Bounded query tool (`db_query`): enforces `PRAGMA query_only=ON`, auto-appends `LIMIT 50`, truncates wide columns, and enforces a 5-second deadline.
   * Eliminates shell hangs from interactive CLI pagers (`less`, `sqlite3`, `psql`).
3. **Adjudicator Security Fence (#10836)**:
   * Default-deny block (`PRODUCTION_DB_BLOCK`) on any non-loopback connection string (`DATABASE_URL`).
   * Categorizes `DROP TABLE`, `TRUNCATE`, and `ALTER TABLE` as `ReversibilityIrreversible` with required rollback remediation.
4. **Pre-Turn Shadow Snapshots (#10837)**:
   * Automatically takes a copy-on-write / hardlink snapshot of local file databases before an agent executes migrations or test suites.
   * Provides zero-turn atomic rollback if a migration errors or corrupts the test database.
   * Constrained by an explicit retention ring buffer (max 3 snapshots per target file) and storage ceiling to prevent disk bloat.
5. **Schema Migration Witness (#10838)**:
   * Verification oracle checking migration tables (`_prisma_migrations`, `alembic_version`, `goose_db_version`) and `PRAGMA integrity_check` to witness database-touching commits.

---

## 5. Unbounded Growth & Storage Resource Bounds

Both internal harness state and target workload databases must operate under strict, bounded envelopes to prevent disk exhaustion, I/O thrash, and NAND flash wear (aligning with [#10964](../research/storage/ssd-lifespan-extension-and-high-volume-caching-strix.md) and `internal/growthgate`):

### 1. Internal Event WAL & Journal Bounding
* **`growthgate` Integration**: `session-journal.jsonl` and `.fak/toolproc/*.jsonl` adhere to `growthgate`'s class budgets (`ClassLedger` / `ClassToolproc`).
* **Batched Appends**: Eliminates naive fsync-per-line writes, batching flushes to reduce filesystem write amplification and avoid SSD wearout.
* **Compaction & Archival**: Long-running serves fold resolved turn state into compact checkpoints and prune cold segments beyond the active retention window.

### 2. Shadow Snapshot Sprawl Prevention (#10837, #10839)
* **Bounded Ring Buffer**: Retains a maximum of $K=3$ generations per tracked database file in `.fak/snapshots/<db_hash>/`.
* **CoW / Reflink Exploitation**: Leverages filesystem copy-on-write capabilities (APFS clonefile, Btrfs reflink, hardlinks) where available to avoid physical byte duplication.
* **Storage Quota & Session Teardown**: Hard ceiling (default 1 GB aggregate) on database snapshots; stale snapshots are reaped automatically on session close or under disk pressure.

### 3. Query & Output Context Bounding (#10835)
* **Clamped Query Semantics**: `db_query` automatically injects `LIMIT 50` (configurable up to a hard ceiling of 500) and truncates wide columns (text/blob/JSON capped at 256 bytes inline).
* **Managed Scratch Spill**: Query outputs exceeding 64 KB are diverted to managed scratch (`_scratch/dataslot/queries/<query_hash>.json`), returning a compact structural summary to the model context.
* **Lifecycle Reaping**: Query scratch is automatically tagged and reaped via `fak tree-doctor --reap-scratch dataslot`.

### 4. Target Database WAL & Ephemeral Cleanup
* **WAL Checkpoint Hygiene**: Agents interacting with local SQLite databases issue passive checkpoints (`PRAGMA wal_checkpoint(PASSIVE)`) on disconnect to prevent runaway `-wal` log expansion during test loops.
* **Ephemeral Test Artifact Cleanup**: Temporary databases instantiated during test runs are tracked and cleaned up on test cycle completion.

---

## 6. Comparison: OpenCode vs. fak Native Harness

| Dimension | OpenCode (`anomalyco/opencode`) | fak Native Harness (`fak agent`) |
|---|---|---|
| **Internal Session Store** | Embedded SQLite (`better-sqlite3` / `bun:sqlite`) | Append-only JSONL (`sessionjournal`) + In-memory Radix/CAS (Zero CGo) |
| **Output Bounding** | Spills raw text over line limits to temp files | Structural bounding: row limits, byte caps, and managed scratch spill |
| **Unbounded Growth Defense** | None (unbounded SQLite WAL, logs, and temp growth) | `growthgate` budgets, snapshot ring buffers ($K=3$), and scratch auto-reap |
| **Target Database Access** | Raw `bash` execution (`sqlite3`, `psql`) | First-class `db_schema` / `db_query` tools + connection security fence |
| **Mutation Safety** | User confirmation prompt on bash command | Reversibility classification (`ReversibilityIrreversible`) + Pre-turn CoW rollback |
| **Verification / Evidence** | Model self-reports completion in chat | `dos verify` schema witness inspecting migration tables and database integrity |

---

## 7. Implementation Program & Ticket Map

* **Research Master**: #10646 (removable capability slots), #10652 (dormant data slot).
* **Child Leaves**:
  * **#10834**: `feat(dataslot): dormant database artifact detector and connection descriptor spine`
  * **#10835**: `feat(dataslot): bounded read-only schema reflection and SQL query engine`
  * **#10836**: `feat(adjudicator): production connection fence and database mutation reversibility classification`
  * **#10837**: `feat(dataslot): copy-on-write shadow snapshotting for local file databases`
  * **#10838**: `feat(dataslot): database schema migration state witness for dos_verify`
  * **#10839**: `feat(dataslot): bounded snapshot ring buffer, growthgate storage budget, and query spill lifecycle`
