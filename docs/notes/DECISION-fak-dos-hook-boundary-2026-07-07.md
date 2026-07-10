---
title: "Decision — the clean fak↔dos hook boundary (2704)"
description: "Records why fak keeps its settings.json dos hooks as headless insurance for plugin-less seats, choosing the Option 2 stable-contract boundary."
---

# Decision — the clean fak↔dos hook boundary (#2704)

**Issue:** #2704 (part of epic #2702; consumes the #2703 audit, commit `a8ee3e6`).
**Host evidence captured:** 2026-07-07, this dev box (win/amd64), live fleet.
**Status:** DECIDED. This note changes no wiring; `.claude/settings.json` is untouched
(that fence is the issue's own acceptance gate). The "decouple + native fast-path"
sibling implements what is decided here.

## TL;DR

- **Headless availability answer: NO — the headless fleet does not reliably load the
  user-scope dos plugin.** Every headless Claude launcher pins `CLAUDE_CONFIG_DIR` to a
  per-account seat dir, and only **2 of 14** live seats on this host carry a dos plugin
  cache. The codex/opencode seats are not Claude Code at all, so a Claude plugin can
  *never* cover them.
- **dos enforcement DOES run headless today** — solely via fak's committed
  `.claude/settings.json` Python hooks (project scope loads from the workspace cwd,
  independent of the config dir). Journal-witnessed below.
- Therefore fak's settings hooks are **deliberate headless insurance, not redundant
  duplication** — and the interactive double-fire measured in #2703 is the cost of that
  insurance, not a reason to delete it.
- **Chosen boundary: Option 2 — one headless-safe fak hook via a stable contract**
  (versioned dos entrypoint / native binary, plugin-presence skip, plugin-aligned
  matchers). Options 1 and 3 are eliminated by the availability evidence.

## Evidence A — launcher HOME/`--settings` handling (the load-bearing question)

| Launcher | Config-dir handling | Plugin consequence |
|---|---|---|
| `tools/issue_dispatch.py --wave` | `worker_env()` sets `env["CLAUDE_CONFIG_DIR"] = account_dir` (the pinned seat) | user scope = the seat dir, **not** `~/.claude` |
| `tools/launch_wave_detached.ps1` | pins `CLAUDE_CONFIG_DIR` per lane `config_dir` (header line 19; `$lane.config_dir` plumbing) | same |
| `fak dispatch auto/wave` (Go) | `dispatchWorkerEnv`, `cmd/fak/dispatch_tick.go:799` — `env["CLAUDE_CONFIG_DIR"] = account.Dir` for the claude backend | same |
| codex seat (`fak dispatch`, Go) | **deletes** `CLAUDE_CONFIG_DIR`; sets `CODEX_HOME` | not Claude Code — Claude plugins can never apply; dos is wired via codex's own cached hook manifests (`tools/codex_dos_hook_doctor.py`, audited by `tools/codex_dos_recent_audit.py`) |
| opencode seat (`fak dispatch`, Go) | **deletes** `CLAUDE_CONFIG_DIR`; sets `XDG_CONFIG_HOME` | not Claude Code — runs `opencode run --agent dos-dispatch` |

Additionally, the guarded launch path (`fak guard`, default-on for dispatch workers)
injects its **own** `--settings` temp file — observed live in
`.dispatch-runs/dispatch-claude-20260707-102108.log`:

```
claude --mcp-config …\fak-mcp-config.json --settings …\claude-precompact-settings.json
       -p --permission-mode bypassPermissions /dos-dispatch-loop --lane claude
```

with `CLAUDE_CONFIG_DIR=C:\Users\<user>\.claude-july6-netra` (the seat). So a headless
worker's "user scope" is whatever the seat dir contains — and mostly it contains no
plugin.

`tools/dispatch_worker.py` already documents the consequence (comment above
`CLAUDE_AGENT_PROMPT`): the namespaced plugin skill form "fails closed ('Unknown
command') whenever a per-account `.claude-<acct>` plugin cache is missing/empty — which
it is for freshly-enrolled worker accounts." The fleet already routes around missing
plugins; this note makes that the decided premise.

## Evidence B — seat survey (this host, 2026-07-07)

Of the 14 non-DELETED `.claude-*-netra` seats:

- **2 have the dos plugin cache** (`plugins/cache/dos`): `adminbackup`, `gem7`.
- **5 have no `plugins/` dir at all**: `day24`, `day28`, `faklocal`, `july8`, `july9`.
- **7 have `plugins/` but no `cache/`** (marketplace metadata only): `day26`, `day26NEW`,
  `day27`, `gem8NEW`, `july4`, `july6`, `july7`.
- `day26` is the sharpest case: its `plugins/installed_plugins.json` claims
  `dos-kernel@dos` v0.28.0 installed at
  `…\.claude-day26-netra\plugins\cache\dos\dos-kernel\0.28.0` — **a path that does not
  exist**. A manifest without a cache cannot load.

The interactive `~/.claude` by contrast has `dos-kernel@dos` v0.28.0 fully cached and
installed — which is why #2703 measured a double-fire *interactively*.

## Evidence C — headless dos-journal grep (enforcement runs headless today)

The worker spawned by `dispatch-claude-20260707-102108.log` (seat `.claude-july6-netra`
— one of the no-cache seats) is Claude session `c3485c7f-cae5-40c7-83b8-4738537c1d40`
(matched via the seat's `projects/C--work-fak/` transcript). Its dos stream journal
exists and advanced for the whole run:

- `.dos/streams/c3485c7f-cae5-40c7-83b8-4738537c1d40.jsonl` — **49 STEP rows**,
  `2026-07-07T10:22:49Z → 10:59:58Z` (38 Bash, 4 Grep, 2 PowerShell, 1 each
  Write/Read/ToolSearch…).
- The **Grep/Read STEP rows are the fingerprint of fak's settings path**: the plugin is
  unloadable on this seat, and fak's `PreToolUse` matcher excludes Read/Grep — so those
  rows can only come from fak's no-matcher `PostToolUse` Python hook.
- `.dos/metrics/observations.jsonl` rows in the same window show Python-path latencies
  (pretool ~506 ms, posttool ~380–680 ms), matching #2703's Python column (native is
  ~37–189 ms). The Python path is the one running headless.

Two sibling headless sessions on the same seat (`d67b741f…`, `cca534c5…`) show the same
shape (62 and 25 STEP rows). Conclusion: **headless dos enforcement is real today and it
is 100% fak's `.claude/settings.json` wiring** on plugin-less seats.

Second independent witness, from a `fak dispatch` resolve seat rather than the
dos-dispatch-loop: the #2704 worker itself (`.dispatch-runs/resolve-2704-20260707-142623.log`,
seat `.claude-july9-netra` — one of the **no `plugins/` dir at all** seats) ran with
`CLAUDE_CONFIG_DIR=…\.claude-july9-netra` and an **empty `CLAUDE_PLUGIN_ROOT`** (no plugin
process ever spawned), under the same guard-injected `--settings …\claude-precompact-settings.json`
launch shape — and `.dos/metrics/observations.jsonl` records `{"dialect":"claude-code",
"verb":"pretool"}` / `posttool` hook-observation rows throughout its window
(e.g. `2026-07-07T14:30:28Z`, pretool ~348 ms). Plugin verifiably not loaded; dos hooks
verifiably firing — only fak's settings wiring can be the source.

## Decision — Option 2: one headless-safe fak hook via a stable contract

Keep a single fak-side dos hook set in `.claude/settings.json` (it is the only thing
standing between a plugin-less headless seat and zero dos enforcement), but re-found it:

1. **Versioned entrypoint, not internals.** The hook must invoke a stable dos hook
   contract — the plugin's `bin/dos-hook` launcher / native `dos-hook-<os>-<arch>`
   binary when resolvable, a versioned Python entrypoint as fallback — never
   `python -m dos.cli` argparse internals (the exit-code-2-on-flag-rename coupling the
   epic exists to remove; today only a `sys.exit(0)` wrapper prevents a fleet wedge).
2. **Skip when the plugin is actually loaded** (kills the interactive double-fire).
   Caveat for the implementer: the issue's proposed `$CLAUDE_PLUGIN_ROOT` guard is set
   in *plugin-sourced* hook processes, so a *settings-sourced* hook likely never sees
   it — validate before relying on it. The robust probe is exactly what this note
   measured: `dos-kernel@dos` enabled in `$CLAUDE_CONFIG_DIR` (else `~/.claude`)
   `plugins/installed_plugins.json` **and** its cache install path existing on disk.
   One stat + one small JSON read, decided per session, fail-open to firing.
3. **Align matchers with the plugin** — scope fak's `PostToolUse` to
   `Read|Bash|Grep|Glob` (today it is un-matched and fires the ~373 ms Python posttool
   on *every* tool, the most wasteful divergence #2703 found).
4. **Keep the non-dos `repoguard` PreToolUse entry untouched** — fak-specific, no
   plugin equivalent.

### Why not the alternatives

- **Option 1 (plugin-only)** — removes all dos enforcement from 12/14 headless Claude
  seats today (Evidence B+C), and can never cover the codex/opencode seats (Evidence A).
  Eliminated by the availability answer.
- **Option 3 (plugin + explicit headless install)** — cannot cover codex/opencode
  (not Claude Code); adds a marketplace/network dependency and a first-launch race to
  every ephemeral seat enrollment (seats are disposable — note the `.DELETED-*` churn);
  and still needs a fak-side fallback for the window where install hasn't happened. It
  converges on Option 2 with extra moving parts.

## Migration (for the "decouple + native fast-path" sibling)

**State at landing:** sibling #2705 (commit `80b38ecf`, same day) already rewired the
`PreToolUse`/`PostToolUse` dos entries to `tools/dos_hook.py` — a launcher implementing
§Decision 1's contract (buffer stdin once; native `tools/.bin/dos-hook[.exe]` first;
`python -m dos.cli` fallback; always exit 0). The `Stop` entry is still the bare
`python -c … dos.cli` wrapper, and §Decision 2 (plugin-presence skip) and §3 (matcher
alignment) are not implemented. Migration steps below are adjusted to that reality.

1. **dos side (separate repo, first):** confirm/ship the stable hook entrypoint the
   contract names — the `bin/dos-hook` launcher already shipped in the plugin cache is
   the shape; it needs a seat-independent resolution order (plugin cache → `dos-hook`
   on PATH → versioned Python entrypoint).
2. **fak side (one commit, only after this note):** finish moving `.claude/settings.json`
   onto the wrapper — route the remaining `Stop` entry through `tools/dos_hook.py` and add
   §Decision 2 (plugin-presence skip) and §3 (matcher alignment) to the wired entries;
   leave `repoguard` alone.
3. **Verify both modes:** re-run #2703's latency probe interactively (expect a single
   native-speed fire per event) AND one headless dispatch on a plugin-less seat (expect
   STEP rows still appearing in `.dos/streams/<session>.jsonl`, per Evidence C's method).

## Rollback

`git revert` the single settings commit — it restores today's known-good (slow but
working) Python wiring verbatim. The wrapper is stateless and the journal schema is
unchanged, so there is no seat-side or `.dos/` state to migrate back.
