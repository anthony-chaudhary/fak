---
title: "Credential backup + restore — no login should be able to lose an account"
description: "Where the #3987 credential safety net lives and how to use it: the content-addressed backup store under ~/.claude/backups, the backup-on-write hook in the adopt/enroll path, and the `fak accounts backup` / `restore-credential` operator verbs. Documents which ops snapshot, which don't need to, and the one overwrite path the hook cannot see."
version: 1
---

# Credential backup + restore — no login should be able to lose an account (2026-07-13)

This documents the credential safety net shipped for
[#3987](https://github.com/anthony-chaudhary/fak/issues/3987). The triggering
near-miss (2026-07-10): an in-harness `/login` re-logged a seat's config dir into a
*different* account, and the previous account survived only because a stale copy of
its credentials happened to exist under another name. Recovery must be by design,
not by luck.

## The store — where backups live

`<home>/.claude/backups/<seat>/<stamp>-<sha12>-<file>`
(`accounts.BackupRoot`, `internal/accounts/credbackup.go`).

- **Under the gitignored home tree, never the repo.** Backups hold live OAuth
  tokens; the store is deliberately a subdir of `~/.claude` — and *not* a
  `~/.claude-*` sibling, so config-home discovery never mistakes it for a seat.
  `TestBackupRootUnderHomeClaude` pins the shape.
- **Content-addressed.** Each snapshot name carries a sortable UTC stamp and the
  first 12 hex of the blob's sha256. An unchanged blob is stored once no matter how
  often the hook fires, and a restore re-hashes the bytes against the name — a
  corrupted store file surfaces as an error, never as a silently-wrong credential.
- **Covered blobs:** `.credentials.json`, `.claude.json`, `.oauth-token` (missing /
  empty files are skipped). Store dirs are `0700`, blobs `0600`, written atomically
  (temp + rename), matching how live credentials are written.

## Operator surface

```
fak accounts backup             [--name <seat>] [--keep N] [--json]   # snapshot every live seat (or one)
fak accounts backup --list       --name <seat>  [--json]              # what is recoverable for a seat
fak accounts restore-credential  --name <seat>  [--at <stamp|sha>] [--file <blob>] [--json]
```

- `backup` snapshots each live seat's blobs and prunes to `--keep` (default 20)
  snapshots **per file per seat**; pruning never drops the newest of each kind, and
  `keep <= 0` never prunes at all. Content addressing makes a roster-wide `backup`
  cheap and idempotent — run it freely before login work.
- `restore-credential` restores the newest snapshot, or the newest one whose stamp
  **or** sha has `--at` as a prefix. The restore first snapshots the *current* blob,
  so a restore is itself reversible — it never destroys the state it replaces
  (`TestRestoreCredential_RoundTripAndReversible`).

## Backup-on-write — which ops snapshot, and which don't need to

`accounts.SnapshotBeforeOverwrite` is the pre-image hook: it runs immediately
before an accounts op overwrites a seat's credential blobs
(`TestBackupThenOverwrite_PreImageRecoverable` is the acceptance witness).

| Op | Overwrites credential blobs? | Coverage |
|---|---|---|
| `add --adopt` / `enroll-current` (incl. `--force` reconcile) | **Yes** — `copyLoginBundle` into the target dir | Snapshots first (`cmd/fak/accounts_add.go`); a backup miss warns, never blocks the enroll |
| `restore-credential` | **Yes** — writes the restored blob | Engine snapshots the current blob before writing |
| `add` (setup-token path) | No pre-image exists | The never-clobber guard refuses an existing dir outright; `seedClaudeJSON` never overwrites an existing `.claude.json` |
| `remove` / `remove --archive` | No — registry tombstone; `--archive` *renames* the dir | Blob bytes are preserved by the rename; `restore` reverses it |
| `rehome` | No — a live-gateway seat switch (`POST /v1/fak/account/rehome`), no disk write | Nothing to snapshot |

## The honest limit — the overwrite the hook cannot see

The event that motivated #3987 — a raw in-harness `/login` — is the **Claude binary
itself** rewriting `.credentials.json`. It does not pass through any `fak accounts`
op, so backup-on-write cannot intercept it today. The defenses on that path are:

1. **On-demand roster snapshots**: `fak accounts backup` before login work (cheap,
   idempotent — see above). This is the designed replacement for the manual
   27-seat copy taken on 2026-07-10.
2. The login-time identity-hijack guard (#3953), which refuses the wrong-dir
   login before it writes.

The invalidating assumption to watch: backup-on-write assumes credential overwrites
flow through `fak accounts` ops. If a login wrapper / guard pre-login hook lands
later, it should call `SnapshotBeforeOverwrite` so the raw-`/login` path gets the
same pre-image guarantee automatically.

## Witnesses

- Engine + tests: `internal/accounts/credbackup.go`, `credbackup_test.go`
  (commit `b8f67b285`). CLI verbs: `cmd/fak/accounts_backup.go`, wired in
  `cmd/fak/accounts.go` (landed via `a6c4e05f1`); backup-on-write hook in
  `cmd/fak/accounts_add.go`.
- Gates (2026-07-13): `go vet ./internal/accounts` clean,
  `go test ./internal/accounts -count=1` green, `go build ./cmd/fak` green.
