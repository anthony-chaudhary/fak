---
title: "Dogfooding DOS from a fresh install on the laptop: what value lands in the first 60 seconds"
description: "A from-zero dogfood of the dos-kernel CLI on the laptop — a genuine `pip install dos-kernel` into a clean venv, then the fresh-user value path (start-here, quickstart, doctor, init, commit-audit, verify, exit-codes). Captures the real install cost, the one-dependency footprint, the caught-lie + collision demo verbatim, and the honest friction a newcomer hits."
date: 2026-06-25
---

# Dogfooding DOS from a fresh install on the laptop

*Goal: experience DOS the way a brand-new user does — install it cold and see
what value it hands you before you've read any docs. Run on the laptop
(the laptop over Tailscale, Windows side: Python 3.12.3, git 2.46) into a throwaway
venv isolated from the machine's pre-existing miniconda `dos`, so the install is
genuinely fresh. Every line below is captured from that run, not paraphrased.*

## TL;DR

- **Install is one command and one dependency.** `pip install dos-kernel` pulls
  `dos-kernel==0.28.0` and `PyYAML==6.0.3` — nothing else. A 4.4 MB wheel,
  ~20 s with the wheels cached.
- **First value is `dos quickstart`: 1.4 s, no setup.** It tells one story — an
  agent claims it shipped two things; git says only one landed — and proves the
  difference from artifacts, then shows two agents avoiding a collision on one
  repo. That is the whole product in two screens.
- **The verdict is the exit code.** `dos verify` exits 0 for SHIPPED, 1 for
  NOT_SHIPPED. A CI step or shell branches on it without parsing prose.
- **`dos commit-audit` caught a real lie I planted** — a commit subjected "fix:
  handle nil pointer deref in the auth handler" that only touched a README came
  back `UNWITNESSED [subject-only]`, exit 1. An honest doc commit came back
  `witnessed`.
- **Friction worth fixing:** `dos --version` is not a flag (it errors, though it
  helpfully routes you to the menu); `dos init --hooks auto` only works from a
  repo that already has an agent-runtime marker.

## What "fresh install" actually costs

```
$ pip install dos-kernel
Collecting dos-kernel
  Using cached dos_kernel-0.28.0-py3-none-win_amd64.whl.metadata (30 kB)
Collecting pyyaml>=6.0 (from dos-kernel)
Using cached dos_kernel-0.28.0-py3-none-win_amd64.whl (4.4 MB)
Successfully installed dos-kernel-0.28.0 pyyaml-6.0.3
INSTALL_SECONDS=19.6

$ pip freeze
dos-kernel==0.28.0
PyYAML==6.0.3
```

The README's "PyYAML is the only runtime dep" claim holds on a clean venv: the
freeze is two lines. Install time was 19.6 s with the wheels already cached on
this host; a cold machine adds the 4.4 MB + 154 KB download on top, so call it
under a minute on a normal connection.

One distribution gotcha the tool flags itself: the PyPI package is
`dos-kernel`, and `dos doctor` prints a warning that the bare `dos` name on PyPI
is an unrelated squatter. Good that the tool says so out loud, because
`pip install dos` is the obvious wrong guess.

## The value path a newcomer actually walks

### `dos start-here` — the task-to-verb router

A reflexive `dos --version` errors (there is no top-level `--version` flag; dos
requires a subcommand). But the failure is soft: it prints the "what do you want
to do?" menu instead of a bare stack trace. The same menu is `dos start-here`:

```
DOS — what do you want to do?
  get started in 60 seconds       dos quickstart    — the caught-lie + collision demo
  adopt DOS in a repo             dos init          — scaffold dos.toml (+ --skills / --hooks)
  check a claim actually landed   dos verify        — did (plan,phase) ship? git ancestry, not self-report
  check a commit matches its diff dos commit-audit  — subject vs its own diff (subjects are forgeable)
  stop two agents colliding       dos arbitrate     — may a loop start on this lane?
```

This is the right first screen: every row is a real task in the user's words,
mapped to the verb that does it.

### `dos quickstart` — the 60-second value demo (ran in 1.4 s)

This is the part that earns the install. It runs end-to-end in a throwaway repo
and tells one concrete story. Part 1 — catch a false "done":

```
$ dos verify AUTH AUTH1
  SHIPPED AUTH AUTH1 601798e (via grep-subject)
  exit=0  (0 = the verdict is SHIPPED)

$ dos verify AUTH AUTH2
  NOT_SHIPPED AUTH AUTH2 (via none)
  exit=1  (1 = NOT_SHIPPED — the claim is contradicted by the artifacts)
```

The agent claimed it shipped both AUTH1 and AUTH2. git says only AUTH1 has a
commit. DOS believes the artifacts, not the transcript, and the verdict is the
exit code. Part 2 — two agents, one repo, no collision:

```
$ dos lease-lane acquire --lane src --owner agent-A   -> acquire 'src'
$ dos lease-lane acquire --lane src --owner agent-B   -> acquire 'docs'
   (B requested the busy 'src', saw A's journaled lease, was handed free disjoint work)
$ dos lease-lane acquire --lane src --owner agent-C   -> refuse  (all lanes held; exit 1)
```

A silent overwrite becomes a typed, scriptable refusal. That is the second half
of the pitch — concurrency safety for a fleet sharing one tree — shown rather
than asserted.

### `dos commit-audit` — the caught lie, on a commit I authored

The strongest single moment of the dogfood. I made two commits and audited each:

```
# subject claims a code fix; the diff only touches README.md
$ dos commit-audit HEAD
  UNWITNESSED 5b12fc9 [subject-only]  code-effect claim but the diff touches no
  SOURCE file (only: README.md) — the claim rests on the subject text
  commit-audit: 1/1 commit(s) make a claim their diff does not witness.   (exit 1)

# honest doc commit
$ dos commit-audit HEAD
  witnessed a9adbf4 [diff-witnessed]  doc claim, the commit touches files
  (doc scope, no code over-claim)
```

A subject is forgeable; the diff is not. This is the same distrust discipline as
`verify`, aimed at a single commit, and it needs no plan or config — point it at
any commit in any git repo.

### `dos exit-codes` — the contract that makes it scriptable

Every verb's verdict is an exit code from a closed set: `verify` is 0 shipped /
1 not_shipped / 2 contract_error; `arbitrate` is 0 acquire / 1 refuse;
`complete` is 0/3 INCOMPLETE/5 UNDERDECLARED; and so on. A CI gate or a fleet
supervisor branches on the integer and never parses a sentence. This is what
lets the verdict *act* instead of merely being available.

### `dos init` and the adoption surface

- `dos init` on the bare demo scaffolded a single-writer `main` lane (it derives
  lanes from top-level source dirs; there were none yet).
- `dos init --skills` dropped 4 editable `SKILL.md` screenplays into
  `.claude/skills/` (`dos-next-up`, `dos-dispatch`, `dos-dispatch-loop`,
  `dos-replan`) — the workflow layer on top of the raw verdict.
- `dos init --hooks auto` refused, but usefully: it scanned for ~20 runtime
  markers (`.claude/`, `.cursor/`, `.codex/`, `.gemini/`, ...), found none in the
  throwaway dir, and told me exactly how to proceed — *"run it from the repo your
  agent works in, or name the host: `dos init --hooks <host>`"* with the full
  supported list. A refusal with a way forward, which is the DOS house style.

## Honest friction (the "etc.")

1. **`dos --version` is dead on arrival.** A newcomer's first reflex errors.
   It recovers to the menu, but a `--version` alias that prints `0.28.0` would
   cost nothing and remove a paper cut. Today the version only surfaces via
   `dos doctor` ("DOS v0.28.0") and the env-print line.
2. **`--hooks auto` needs a real agent repo.** Reasonable, but it means the
   single most valuable adoption step (wire the verdict into the host so a false
   "done" is *denied*, not just *detectable*) can't be exercised from the
   quickstart's throwaway dir — you have to be standing in the repo your agent
   actually runs in.
3. **Install timing is cached-wheel.** 19.6 s reflects warm pip caches on this
   host; a truly cold machine pays the download. Still sub-minute, still one dep.
4. **PowerShell noise is not DOS failing.** Running through `ssh -> powershell`,
   every nonzero exit / stderr line surfaced as a red `NativeCommandError`.
   Those are PowerShell flagging the exit code, not bugs — `commit-audit` exit 1
   on the planted lie is the *intended* verdict, not an error.

## Bottom line

From a cold `pip install` to "DOS just caught an agent's false done from the
git history, in front of me" was about 20 seconds of install plus a 1.4-second
demo, against one dependency, on a machine that had never run this build. The
value proposition lands fast and lands honestly: it distrusts the self-report
and shows you the artifact that contradicts it. The friction is cosmetic
(`--version`) or inherent (`--hooks` needs the real repo), not a barrier to the
first "aha."

This is the outward, fresh-user counterpart to the inward dogfood loop fak
already scores — the launched-session guard/Stop-hook stack measured by
`internal/dogfoodscore` (see `docs/fak/dogfood-loop-scorecard.md`). That loop
asks "does our own guarded session report itself honestly?"; this note asks
"what does a stranger get in their first minute?" Both answers come from real
evidence, not narration.

---

*Method: `dos_dogfood.ps1` + `dos_dogfood2.ps1` (scp'd to the laptop, run under
`powershell -File`). Scripts must be ASCII-only — an em-dash in a PowerShell
string is mangled by the Windows console codepage and breaks the parse. The
laptop's WSL Ubuntu was a bare distro (no python/pip/git), so the Windows-side
Python 3.12 + git was the clean fresh-install host.*
