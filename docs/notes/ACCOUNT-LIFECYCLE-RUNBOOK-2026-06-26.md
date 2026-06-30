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
`fak accounts check` (RED, exit 1). Both generated views now carry the same
`login_status` + `can_serve` fields as `fak accounts status`, so switchers can consume the
closed login vocabulary directly instead of guessing from `enabled`, directory existence, or
credential files.

The shell entry points (the shared `shortcuts.ps1` profile):

- `c` / `f` / `fu` — launch `claude`, rotating the seat so no one Max window burns.
- `claude-as <name>` — launch on a named seat (resolves `CLAUDE_CONFIG_DIR` via the registry).
- `u` — `job-search claude-accounts observe`: the operator roster card (serving / walled /
  needs-enroll, 24h usage spark, rate-limit-bucket reconcile).

## Quickref — add / remove a seat (super easy)

```
# add: one command — isolated-dir login, identity probe, twin-check, registry + views
fak accounts add <name>

# remove: one command — tombstone + archive the dir + repoint the registry + resync views
fak accounts remove --name <seat> --archive

# inspect: the roster (tool/registry version stamped on top) and "is my binary current?"
fak accounts list
fak accounts status --json
fak accounts version
```

`remove --archive` refuses the live `CLAUDE_CONFIG_DIR` seat — retire that one from another
session. Export `FAK_JOB_ROSTER=<job>/config/claude_accounts.yaml` once from your shell profile so
add/remove/sync regenerate the `u` view too, no flag needed.

## Lifecycle

```
add ──► serve/rotate ──► tombstone (remove) ──► dir-rename .DELETED-<date> ──► purge
        (live seat)       registry only           reversible                  rm (frees disk)
```

`remove` flips the registry only; `remove --archive` ALSO does the dir-rename + registry repoint
in the same command. A final `rm` of the `.DELETED-*` dirs (to reclaim disk) stays a separate,
deliberate step.

## Runbook: retire one or more seats

1. **Look before you cut.** `fak accounts list` shows the human table; `fak accounts status
   --json` is the machine report (`fak.accounts.login.v1`) with one closed `status` per seat
   (`ready`, `needs_login`, `missing_dir`, `disabled`, `tombstoned`), `can_serve`, warnings, and
   the next action. Note `dup -> <canonical>` / `duplicate_account_bucket` (a seat that is really
   another account), `CREDS` / `needs_login` (does it hold a unique login?), and whether a backup
   exists under `~/.claude-account-backups/<email>`. A seat with a unique login and no backup loses
   that login on the eventual purge — keep its `.DELETED-*` dir until you are sure.
2. **Don't retire the seat you are sitting on.** `echo $CLAUDE_CONFIG_DIR` — `--archive` refuses
   it anyway (you cannot move the dir this session runs from); retire that one from another session.
3. **Retire it in one command (reversible):**
   ```
   fak accounts remove --name <seat> --archive --rehome-to <live-seat> --reason "<why>"
   ```
   This tombstones the registry, renames the dir to `~/.claude-<seat>.DELETED-<date>`, repoints
   the registry entry (name + dir + any rehome refs), and regenerates the views. Omit `--archive`
   to tombstone the registry only and leave the dir in place. Rehome a duplicate to its identity
   twin; rehome the rest to the anchor seat (the default `--rehome-to`).
4. **Prove it landed:** `fak accounts check` (ok dos / ok job) and `fak accounts validate`
   (registry valid). The retired seat should appear under `u`'s `tombstoned_accounts`, not the
   active table.
5. **Purge later (irreversible):** a sweep can `rm` the `.DELETED-*` dirs to reclaim disk once
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
- **`remove` synced the dos view, not the job view.** Back then `fak accounts remove` only
  regenerated the dos view; the job roster (`u`'s source) stayed stale until an explicit
  `fak accounts sync --job-view …`. *Fix:* export `FAK_JOB_ROSTER=<job>/config/claude_accounts.yaml`
  from your shell profile so the job view joins the default sync set — then add/remove/sync refresh
  `u` with no flag.
- **Re-sync after any external registry edit.** The registry was hand-edited mid-task (a new
  default seat, a dropped entry). Views must be re-synced or `check` goes RED.

## Versioning & visibility

The registry **data** is versioned (`fak-config-homes/v1`, family-prefix accept check in
`internal/accounts`). The gaps that cost time on 2026-06-26 were the invisible **tooling** version
and the multi-step retire — both now closed.

**Shipped (2026-06-26):**

- **`fak accounts version`** (text + `--json`) — prints the build, the registry schema/family it
  supports, and the verb set. Closes the stale-binary trap: a binary behind source prints an old
  version / short verb list instead of failing a newer verb with a raw flag error.
- **`fak accounts list` provenance header** — `# fak <ver> · registry <schema>` above the table,
  so any roster read shows the tool version inline. It now includes a `LOGIN` column derived from
  the same closed status vocabulary as `fak accounts status`.
- **`fak accounts status [--json]`** — the observable login report for all users of the account
  switcher: no ad hoc boolean guessing, just one status, `can_serve`, warnings, roles, and a next
  action per seat.
- **Generated view login fields** — `fak accounts sync` writes `login_status` and `can_serve` into
  the dos/job roster rows, so `claude-as`, `u`, and other switcher consumers can read the same
  readiness surface without reimplementing the registry's login rules.
- **`fak accounts remove --archive`** — the one-command retire (tombstone + dir-rename + registry
  repoint + resync) that replaces the manual dance this note was written about.

**Still open:**

- **Stamp the GENERATED views** (dos/job) with the producing build. Deferred on purpose: putting
  the binary version *inside* the rendered view makes `fak accounts check` version-sensitive (drift
  on every binary bump). Decide the determinism rule before shipping it.
- **Stale-binary guard in `claude-as` / `u`.** When the installed `fak.exe` lacks a verb the
  registry needs, warn with the fix (`go install …/cmd/fak@latest`) instead of a raw flag error.
- **Session-start seat banner.** `c` already prints `[c] account: <name>`; extend it to flag a
  tombstoned / slated-for-removal seat (this session ran on a retiring seat with no banner).
