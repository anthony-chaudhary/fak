# Account-lifecycle runbook — enroll, serve, retire, purge (2026-06-26)

**Kind:** durable procedure + lessons log (operator-facing). **Lane:** `internal/accounts`.
**Keep updating:** append a dated entry to *Lessons log* every time the switcher bites,
so the lifecycle gets smoother instead of re-learning the same trap.

Companion to the diagnosis note
[ACCOUNT-SWITCHER-SINGLE-SOURCE-2026-06-25](ACCOUNT-SWITCHER-SINGLE-SOURCE-2026-06-25.md)
(why one rate-limit bucket shows as several "serving" seats) and the
[Resume & rehome runbook](RESUME-REHOME-RUNBOOK-2026-06-26.md) (restarting walled sessions).

## The one model to hold in your head

`~/.claude-accounts/registry.json` (`fak-config-homes/v1`) is the **single source of
truth** for account identity + policy. Everything else is a **generated view** of it:

| Surface | File | Written by | Read by |
| --- | --- | --- | --- |
| registry (truth) | `~/.claude-accounts/registry.json` | `fak accounts add/remove`, hand-edit | `fak accounts *` |
| dos view | `~/.claude/accounts.yaml` | `fak accounts sync` | `dos accounts`, `claude-as` |
| job view | `<job>/config/claude_accounts.yaml` | `fak accounts sync --job-view` | `u` = `job-search claude-accounts observe` |

Never hand-edit a view — regenerate it. A view that drifts from the registry is caught by
`fak accounts check` (RED, exit 1).

The shell entry points (the shared `shortcuts.ps1` profile):

- `c` / `f` / `fu` — launch `claude`, rotating the seat so no one Max window burns.
- `claude-as <name>` — launch on a named seat (resolves `CLAUDE_CONFIG_DIR` via the registry).
- `u` — `job-search claude-accounts observe`: the operator roster card (serving / walled /
  needs-enroll, 24h usage spark, rate-limit-bucket reconcile).

## Lifecycle

```
add ──► serve/rotate ──► tombstone (remove) ──► dir-rename .DELETED-<date> ──► purge
        (live seat)       registry only           reversible                  rm (frees disk)
```

Tombstone and dir-rename are **two separate steps** by design (the `remove` code says so):
`fak accounts remove` only flips the registry; moving or deleting the config dir is a deliberate
destructive follow-on.

## Runbook: retire one or more seats

1. **Look before you cut.** `fak accounts list` — note `dup -> <canonical>` (a seat that is
   really another account), `CREDS` (does it hold a unique login?), and whether a backup exists
   under `~/.claude-account-backups/<email>`.
2. **Refuse to retire the seat you are sitting on.** `echo $CLAUDE_CONFIG_DIR` — you cannot
   rename or delete the dir this session runs from. Retire it from another session.
3. **Tombstone in the registry (reversible):**
   ```
   fak accounts remove --name <seat> --rehome-to <live-seat> --reason "<why>"
   ```
   Rehome a duplicate to its identity twin; rehome the rest to the default seat. This
   regenerates the dos view automatically.
4. **Rename the dir to the house tombstone form (reversible):**
   `~/.claude-<seat>` → `~/.claude-<seat>.DELETED-<YYYY-MM-DD>`, then repoint the registry
   entry's `name` + `dir` to match (this is how prior retired seats are represented). Skip any
   seat that has no backup unless you accept losing its login.
5. **Regenerate every view and prove no drift:**
   ```
   fak accounts sync --job-view <job>/config/claude_accounts.yaml
   fak accounts check --job-view <job>/config/claude_accounts.yaml   # ok dos / ok job
   fak accounts validate                                             # registry valid
   ```
6. **Confirm the operator surface (`u`) updated:** retired seats should appear under
   `tombstoned_accounts`, not the active table.
7. **Purge later (irreversible):** a sweep can `rm` the `.DELETED-*` dirs to reclaim disk once
   you are sure nothing is pinned to them.

## Lessons log

### 2026-06-26 — retiring four old seats

- **A stale `fak` binary fails silently.** The installed `fak.exe` predated the `add`/`remove`
  verbs, so `fak accounts remove --name …` died with `flag provided but not defined: -name` and
  the OLD usage block — no hint that the binary was behind source. Worked around with
  `go run ./cmd/fak accounts …`. *This is the strongest argument for versioning the switcher
  (below): a visible tool version would have named the cause in one line.*
- **You can't retire your own seat.** This session's `CLAUDE_CONFIG_DIR` was one of the seats
  being retired. Its registry entry could be tombstoned, but its dir had to be left for another
  session. (It was later dropped from the registry entirely once another seat became the
  canonical default; the dir remains until a purge.)
- **Check for a backup before a hard delete.** One of the four held the only copy of its login
  (`has_creds: true`, no `~/.claude-account-backups` entry); the others had no creds and did have
  backups. We chose the reversible `.DELETED-<date>` rename over `rm` so the unique login
  survives.
- **`remove` syncs the dos view, not the job view.** `fak accounts remove` only regenerates the
  default dos view; the job roster (`u`'s source) stayed stale until an explicit
  `fak accounts sync --job-view …`. Always re-sync the job view, then `check`.
- **Re-sync after any external registry edit.** The registry was hand-edited mid-task (a new
  default seat, a dropped entry). Views must be re-synced or `check` goes RED.

## Versioning & visibility — proposal (think-about-starting)

The registry **data** is versioned (`fak-config-homes/v1`, with a family-prefix accept check in
`internal/accounts`). What is **not** visible is the **tooling** version and the live seat's
retirement state — the two gaps that cost time on 2026-06-26.

Proposed, smallest-first:

1. **Stamp the generated views with the producing `fak` build.** Add `fak <version>` +
   `registry <schema>` to the view header comment so `u` / `dos accounts` can show *who* rendered
   the roster and *when*. (Touch: `internal/accounts` view render.)
2. **`fak accounts version`** (or a header line on `fak accounts list`): print the binary
   build/version, the supported registry family, and the verb set — so an operator can compare
   against source and see "your binary is behind." Directly closes the stale-binary trap.
3. **Stale-binary guard in `claude-as` / `u`.** When the installed `fak.exe` lacks a verb the
   registry needs, warn with the fix (`go install …/cmd/fak@latest` or `make install`) instead of
   failing with a raw flag error.
4. **Session-start seat banner.** Show the active seat plus a warning when it is tombstoned or
   slated for removal (this session ran on a seat being retired with no banner). `c` already
   prints `[c] account: <name>`; extend it to flag a tombstoned seat.

First step to implement: (1) + (2) together — a version surface on the tool and its output. Low
risk, fully testable, and it is the line that would have explained the failure we hit.
