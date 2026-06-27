# The account switcher's single-source problem, and why gem8 "disappeared" (2026-06-25)

**Kind:** diagnosis + consolidation path (operator-facing). **Lane:** `internal/accounts`.

## The symptom

`claude-accounts observe` showed a 3-account roster — `gem7-netra`, `day24-netra`,
`default-host` — and the operator asked: *where is gem8?* gem8 is a real, enrolled
account; its config dir (`~/.claude-gem8-netra`) exists and is logged in. Yet it is not
on the roster.

## What is actually true on the host

Reading disk truth (`.claude.json` `oauthAccount` for the interactive login, a one-way
fingerprint of `.oauth-token` for the setup token):

| config dir | interactive login | setup-token fingerprint |
| --- | --- | --- |
| `~/.claude` (`default-host`) | day24@ *(was gem8@ earlier same day)* | gem8's token |
| `~/.claude-gem8-netra` | gem8@ | gem8's token |
| `~/.claude-q-netra.DELETED` | gem8@ | gem8's token |
| `~/.claude-day24-netra` | day24@ | gem8's token |
| `~/.claude-gem7-netra` | gem7@ | gem7's token |

So **gem8 never disappeared — it is being served under other dir NAMES.** The gem8
account's setup token has been smeared across four dirs, and at least three dirs resolve
to one account at a time. The dir literally named `gem8-netra` is **tombstoned** in the
roster the observe tool reads, so the operator looking for "gem8" sees `default-host` /
`day24-netra` instead and assumes it is gone.

## The root cause: three roster files and a multivalued identity

Two independent failures compound:

1. **Three rosters, no single source.** `claude-accounts observe` lives in the **`job`**
   repo and reads `job/config/claude_accounts.yaml` (gem8 is in its
   `tombstoned_accounts:`). `dos accounts` reads `~/.claude/accounts.yaml` (gem8 active).
   `fak accounts` reads `~/.claude-accounts/registry.json`. They drift; gem8 is "active"
   in one and "tombstoned" in another. Account-switcher *code* is likewise duplicated
   across `job/`, `dos-private/tools/`, `fak/tools/`, the standalone `account-switcher/`
   repo, and `fleet/`.

2. **Identity is multivalued, and the rosters key on the dir NAME.** A config dir has
   *two* credentials that can name *different* accounts: the interactive
   `.credentials.json` / `.claude.json` login, and the setup `.oauth-token` a headless
   `claude -p` may honor. `day24-netra` is logged in as day24@ but carries gem8's setup
   token. Keyed on the name, the roster counts gem8/q/default/day24 as four independent
   "serving" windows when they are really **one rate-limit bucket** — so a fan-out spreads
   onto one account and they all wall together. This is the same class as the q-netra
   phantom the operator hand-tombstoned earlier the same day.

## What shipped (this pass)

The operator chose to make **fak's Go `internal/accounts` the single source** and have it
auto-handle the collapse rather than maintain the dirs as separate. Landed
(`20f3fd4`, `a9208d4`, `504e95a`):

- `DeriveIdentity` now also reads `.oauth-token` and stores a **one-way SHA-256
  fingerprint** (`Identity.TokenFP`) — never the secret — so dirs sharing a setup token
  are detectable even when their `.claude.json` logins disagree.
- `Registry.Reconcile()` groups active seats by resolved account (`AccountKey`: login
  UUID, else token fingerprint), elects one deterministic **canonical** per account, marks
  the rest **duplicate**, and flags a **token-twin** (a dir whose setup token belongs to a
  different login than its own).
- The canonical election ranks by how many name tokens match the login, and `NameLie` now
  flags only when *no* token matches — so an org-suffixed truthful name (`gem8-netra` →
  gem8@) reads clean instead of a false `WARN`.
- `fak accounts list` surfaces `dup -> <canonical>` / `token-twin -> <names>` per row plus
  a summary. On this host it now reports: **13 active seats resolve to 8 distinct
  accounts; 3 duplicates collapse onto their canonical; 4 carry another login's setup
  token** — `q-netra.DELETED → dup of gem8-netra`, `default → dup of day24-netra`,
  `c10 → dup of anthony-agent`, all visible at a glance.

## Also fixed in the tool the operator actually runs (`job` repo)

`claude-accounts observe` is the `u` view, so the gem8 symptom was fixed at that surface too
(`job` `6fd12f649`, `c485afc34`, on origin):

- **gem8 re-enabled** in `job/config/claude_accounts.yaml` — moved out of `tombstoned_accounts:`
  back to an active, non-reserved worker (it was the rehome target for the deleted q-netra
  phantom, so it must appear as its own seat). `observe` now lists gem8-netra serving.
- **Token-twin reconcile in `observe`** — `account_observability` fingerprints each seat's
  `.oauth-token` (the credential `make_live_probe_fn` actually probes; same one-way SHA-256
  as fak) and the card prints a reconcile line: *"4 seats → 2 distinct rate-limit bucket(s);
  one bucket (same .oauth-token): day24-netra, default-host, gem8-netra"*. So one bucket is
  no longer silently shown as three serving seats. Additive (counts/verdict unchanged); 22 tests green.

## What remains (operator decisions)

- **Credentials.** `~/.claude` and `day24-netra` carrying gem8's setup token is a leak, not
  a config: their headless launches burn gem8's bucket. Re-login (an interactive browser
  auth only the operator can run) is the durable fix; the tooling now *detects* the split.
- **Full cutover.** The reconcile LOGIC now lives in two places (fak Go `internal/accounts`
  and the `job` Python `observe`), mirrored. Collapsing to one source — `observe` (and the
  `dos`/`fak` rosters) resolving through fak's Go — is the remaining consolidation step.

**See also:** the [Account-lifecycle runbook](ACCOUNT-LIFECYCLE-RUNBOOK-2026-06-26.md) — the
one-command add / `remove --archive` procedure, the `fak accounts version` stale-binary surface,
and the keep-updating lessons log built on this single-source model.
