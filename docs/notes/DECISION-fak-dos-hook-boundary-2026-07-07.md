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

Additionally, the guarded launch path (`fak manage`, default-on for dispatch workers)
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

## Ordering & fence adjudication (addendum, 2026-07-12, #2704)

This note was reopened after its first landing (comment on #2704): the deliverable
satisfies done-condition clauses (a)/(b)/(c), **but the issue's own acceptance-gate
fence — "No `.claude/settings.json` change lands before this note" (stated three times:
Deliverable, Confusion risks, Batch policy) — was falsified.** The Migration section
above *narrated around* the out-of-order landing (§"State at landing") instead of
flagging it as a fence violation. This addendum flags it explicitly and adjudicates it,
so the note is self-contained about its own precondition.

### The violation (git-witnessed)

| Fact | Evidence |
|---|---|
| Sibling **#2705** rewired `.claude/settings.json`'s dos `PreToolUse`/`PostToolUse` entries to `tools/dos_hook.py` | commit `80b38ecf`, subject `feat(tools): … wire the dos hooks to it (#2705)` |
| It landed **before** this decision note | `git merge-base --is-ancestor 80b38ecf b26c24deb` → true |
| **24 minutes before** | `80b38ecf` 2026-07-07 07:11:44 −0700 → note `b26c24deb` 07:35:21 −0700 |

The epic's strict sequence is measure (#2703) → **decide (this #2704)** → implement
(#2705). #2705 (implement) committed the settings wiring while this decide-note was
still unwritten. The fence's ordering was therefore violated, factually.

### Adjudication — procedural slip, not the fleet-wide incident the fence guards

The fence exists for one reason, stated in epic #2702 and this issue: *"a bad shared-hook
edit is a fleet-wide incident (every agent's every tool call)."* Its purpose is to stop a
**wedging** settings edit from landing on an undecided boundary. Judge the landing against
that purpose, not only its clock:

- **What landed is the safe thing, and exactly what this note decides.** #2705's
  `tools/dos_hook.py` is the Option-2 §Decision-1 versioned contract verbatim — buffer
  stdin once, native `tools/.bin/dos-hook[.exe]` first, `python -m dos.cli` fallback, and
  **always exit 0** (argparse `exit 2` is coerced to 0; `_rc` is discarded on purpose —
  `tools/dos_hook.py:136`, docstring `tools/dos_hook.py:121-126`). It **removes** the
  `dos.cli`-internals-wedge coupling the epic exists to kill; it cannot introduce it.
- **So the fence's spirit was honored; only its letter (ordering) was broken.** The edit
  that jumped the queue is not "a bad shared-hook edit" — it is the good one this note
  would have prescribed. The concrete harm the fence guards against (a wedge blocking
  every tool call) did not and cannot occur through an always-exit-0 launcher.
- **Recommendation for the maintainer's accept-vs-incident call:** accept the note
  post-hoc. Do **not** revert #2705 — reverting reinstates the slow, `dos.cli`-internals,
  exit-2-capable Python wiring (the actual latent wedge). Record the ordering slip as a
  process note against the epic (implement child should gate its start on the decide-note
  commit existing), not as a rollback.

### Current wiring (verified 2026-07-12, HEAD)

Since this note first landed, the wiring advanced *past* the §Migration "State at landing"
snapshot: **all three** dos entries — `PreToolUse`, `PostToolUse`, **and `Stop`** — now
route through `tools/dos_hook.py --workspace <root>` (Migration step 2's `Stop` move is
done); the non-dos `repoguard` `PreToolUse` entry is untouched, as decided. Still open for
the implement child (unchanged by this addendum): §Decision 2 (plugin-presence skip) and
§Decision 3 (`PostToolUse` matcher alignment to `Read|Bash|Grep|Glob` — it remains
un-matched today). This addendum touches **only this note**; `.claude/settings.json` is
not modified here, honoring the fence going forward.
