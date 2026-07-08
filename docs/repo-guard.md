---
title: "fak repo-guard: refuse out-of-tree destructive writes"
description: "How fak's repo-guard refuses destructive or out-of-tree writes that resolve outside the repo, stopping a sibling-repo rm -rf before it runs."
---

# repo-guard — refuse destructive / out-of-tree writes before they escape the repo

> **One incident, two independent gates.** On 2026-06-21 a build script
> (`dogfood-claude.ps1`/`.sh`) resolved its output path one level *above* the repo
> root and wrote `fak.exe` into a **sibling git repo** (`work/tools`, the real
> `anthony-chaudhary/tools` project); that sibling was then `rm -rf`'d while
> mistaken for build scratch, destroying it. The path bug is fixed at the source,
> but the deeper lesson is structural: **a tool operated on a path that resolved
> outside the workspace, into another project's tree, and nothing refused it.**
> This page is the guard that does — in both the FAK and DOS layers, the same way
> the project stacks two independent gates everywhere else.

`work/` on a fleet host is a *shared tree of many sibling repos* (`fak/`,
`fak-private/`, `fleet/`, `dos/`, `tools/`, …). A path that escapes the workspace
root by one level lands in a *different project*. The guard refuses a
**destructive or write** op whose target resolves outside the current repo — while
never touching in-repo work, ordinary scratch (`/tmp`, `~/.cache`, the null/std-stream
devices like `/dev/null`), or the **one** sibling it treats as a safe destination: the
same-named `fak-private` companion (the operator's private store — see
[Two layers](#two-layers-write-time-placement-vs-commit-time-content) below).

## Two layers: write-time placement vs commit-time content

repo-guard is the **write-time** gate, and it is *content-blind* — it judges only
*where* a path resolves, never *what* is written. That is one half of fak's
public/private model. The other half runs at **commit time**, on *content*, and keeps
the private parts of the project out of the public repo's forever history:

| Layer | When | Judges | Gate | Reason |
|---|---|---|---|---|
| **placement** (this guard) | write-time (PreToolUse) | *where* a path resolves | [`tools/repo_guard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/repo_guard.py) | `OUT_OF_TREE_WRITE` |
| **content** | commit-time (pre-commit / CI) | private-only *paths* (the lab GPU-connection subsystem) | [`tools/check_committed_files.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/check_committed_files.py) | `FILE_ADMISSION` |
| **content** | commit-time (pre-commit / CI) | operator-private *strings* (IPs, hosts, SSH user) | [`tools/scrub_public_copy.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/scrub_public_copy.py) | `PUBLIC_LEAK` |

`fak` is the canonical **public** repo; `fak-private` is the operator's paired
**private** repo — the designated home for everything that must never be public
(private memory/notes, the lab GPU-connection code, operator-private orchestration).
The two layers meet at `fak-private`: repo-guard **lets you write** private content
*into* it (it is the one allowed out-of-tree destination), while the commit-time gates
**stop that same content** from being committed *into* public `fak`. Private content
flows freely to the private repo and is structurally blocked from the public one.

The FAK + DOS pair described next ("the two gates") are two implementations of this
single **write-time** layer — a regex floor and a path-resolving backstop — not the
commit-time content gates above.

## The two gates

| Gate | Mechanism | Catches | Misses (by design) |
|---|---|---|---|
| **FAK** — [`examples/repo-guard-policy.json`](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/examples/repo-guard-policy.json) | `arg_rules` regex over the Bash `command`, evaluated by the kernel's capability floor (`fak preflight`/`serve`/`agent --policy …`) | the `../x` **relative-escape** family — `go build -o ../x`, `> ../x`, `cp .. ../x`, plus the blanket `rm -rf`/`sudo`/fork-bomb/`curl\|sh`/`git push` denies | an **absolute** sibling path (`/c/Users/.../work/tools`) — a regex can't resolve it against the repo root |
| **DOS** — [`tools/repo_guard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/repo_guard.py), the named floor of `dos.toml [reasons.OUT_OF_TREE_WRITE]` | a structural checker that **resolves every target** against the repo root (git-bash `/c/..` and Windows `C:\..` aware) and flags the ones outside it | the absolute sibling escapes the regex misses, **and** Write/Edit `file_path` escapes | unresolvable `$VAR`/glob targets fall back to a conservative textual check |

They overlap on purpose: the regex floor is the deployable, reviewable manifest;
the structural floor is the path-resolving backstop. An attacker (or a buggy
script) has to beat **both**.

## The live integration (what actually stops the call)

[`.claude/settings.json`](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/.claude/settings.json) wires `repo_guard.py` as a
Claude Code **PreToolUse** hook, matcher-scoped to the mutating tools
(`Bash|Write|Edit|MultiEdit|NotebookEdit`) so reads pay no spawn cost:

```json
"PreToolUse": [{ "matcher": "Bash|Write|Edit|MultiEdit|NotebookEdit",
                 "hooks": [{ "type": "command",
                             "command": "python",
                             "args": ["-c", "import os,subprocess,sys; root=os.environ.get('CLAUDE_PROJECT_DIR') or os.getcwd(); p=os.path.join(root,'tools','repo_guard.py'); subprocess.call([sys.executable,p,'--hook']); sys.exit(0)"] }] }]
```

Every classifier rung is a **best-effort heuristic** (it raises the floor, it is not
a sandbox), so a *classified* violation does not imply a *denied* call. What the hook
does about a finding is a **per-reason severity**, and the default posture is
**permissive** — the hard blocks exist and are wired, but they are an opt-in the
operator dials up, not a default everyone else must escape. This matters because
cross-repo work in a fleet host's `work/` tree of sibling repos is **routine**, not
anomalous, and even a *warning* injected into the agent's context can steer the model
or waste a turn — so the routine rungs are *silent* by default.

```
 model proposes  ──▶  PreToolUse hook  ──▶  resolve the target's reason ──▶  per-reason severity
   rm -rf ../tools        (the floor)        OUT_OF_TREE_WRITE               record (silent) by default
                                                                            └▶ deny only if dialed up
```

### Knobs (mirror the trunk-guard convention)

The guard is **fail-open** on any internal error — a guard bug must never wedge a live
multi-session fleet; it is defense-in-depth behind the FAK policy and human review, not
the sole gate. A journal-write failure for a *silent* (record-level) finding is
swallowed **without** touching stderr, so silent stays silent.

**Severity levels** (least → most strict), what each does to a classified finding:

| Level | stderr | journal | call | 
|---|---|---|---|
| `off` | — | — | allow |
| `record` | — (silent) | ✓ row | allow |
| `warn` | advisory + fix | ✓ row | allow |
| `deny` | DENY line | ✓ row | **blocked** (`permissionDecision: deny`) |

**Default posture (permissive; the blocks are an opt-in capability):**

| Reason | Default | Why |
|---|---|---|
| `OUT_OF_TREE_WRITE` | `record` | routine cross-repo work trips it; a placement convention; silent so it never perturbs the model |
| `LIVE_MONITOR_OUTPUT_READ` | `record` | niche, harmless-if-wrong anti-pattern |
| `INTERACTIVE_HANG` | `warn` | the non-interactive-form hint genuinely helps the agent avoid a wasted turn |
| `FOREGROUND_SLEEP` | `warn` | the background-wait hint helps |

A reason with no default entry resolves to `deny` (fail-safe: any refusal-class reason
added later denies until explicitly softened).

**Master switch** — `FAK_REPO_GUARD` overrides everything below it:

- `FAK_REPO_GUARD=off` — disable the hook entirely (skip every rung).
- `FAK_REPO_GUARD=warn` — **cap** every rung at advisory (softens `deny`, never escalates).
- default (unset / `enforce`) — apply the per-reason severity table above.

**Per-reason dial** — `FAK_REPO_GUARD_SEVERITY=REASON=level,REASON=level` sets the
severity for individual reasons (a comma list; malformed pairs are skipped). Precedence:
the master switch wins, then this per-reason override, then the default table.

```bash
# security-minded operator wants the old hard blocks back:
FAK_REPO_GUARD_SEVERITY=OUT_OF_TREE_WRITE=deny,LIVE_MONITOR_OUTPUT_READ=deny

# silence a rung entirely:
FAK_REPO_GUARD_SEVERITY=INTERACTIVE_HANG=off
```

Because `OUT_OF_TREE_WRITE` is silent-recorded (not denied) by default, a routine
cross-repo flow — e.g. the Confluence publish path that writes to the sibling
`confluence-helpers` repo — **no longer needs** the `FAK_REPO_GUARD=warn` escape it
used to require; it just proceeds, and the guard records the crossing for the audit
trail (`repoguard --summary`).

## Verify it

```bash
# FAK floor — the incident build flips ALLOW -> DENY:
fak preflight --tool Bash --args '{"command":"go build -o ../tools/x ./cmd/fak"}' \
  --policy examples/repo-guard-policy.json          # => verdict=DENY reason=POLICY_BLOCK
fak preflight --tool Bash --args '{"command":"go build -o fak.exe ./cmd/fak"}' \
  --policy examples/repo-guard-policy.json          # => verdict=ALLOW (in-repo)

# DOS floor — resolves the absolute sibling a regex can't:
python tools/repo_guard.py --selftest               # => 30/30 passed
python tools/repo_guard.py --check "rm -rf /c/Users/you/work/tools"   # => DENY OUT_OF_TREE_WRITE
python tools/repo_guard.py --check "rm -rf ./build"                   # => ALLOW
python tools/repo_guard.py --check "make ci > /dev/null 2>&1"         # => ALLOW (null sink)

# the reason is now in the DOS closed vocabulary:
#   dos_refuse_reasons  ->  ... OUT_OF_TREE_WRITE ...
```

`tools/repo_guard_test.py` is the hermetic unit suite (no filesystem / no
subprocess): `python tools/repo_guard_test.py`.

## Honest scope

- The **FAK** rules are regex over a command string: they catch the `../` family
  and the named destructive verbs, not every absolute path or every shell shape.
  That is why the DOS floor (which resolves paths) exists alongside it.
- The **DOS** checker is best-effort shell parsing: it splits on `; | && ||`,
  tokenizes with `shlex`, and recognizes a fixed verb set (`rm`/`cp`/`mv`/`tee`/
  redirections/build `-o`…). A sufficiently obfuscated command can evade it — it
  raises the floor, it is not a sandbox. The real containment for *capabilities*
  remains the FAK default-deny floor ([`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md)).
- The guard only ever *flags* a target **outside** the workspace (and outside the
  scratch allow-list); no in-repo work is ever flagged. And under the default posture
  even an out-of-tree write is **recorded, not refused** — a hard `deny` is an opt-in
  the operator dials up per reason. This is consistent with the `dos.toml` rule that a
  declared reason introduces no spontaneous refusal of legitimate work: the default
  refuses nothing, it only records.

## See also

- [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — the capability floor + the `arg_rules` schema.
- [`dos.toml`](https://github.com/anthony-chaudhary/fak/blob/main/dos.toml) — `[reasons.OUT_OF_TREE_WRITE]` and its sibling reasons.
- [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) — the policy's deny table in context.
- [`tools/proc_resource_guard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/proc_resource_guard.py) — the sibling
  guard this one is modeled on (runaway-process reaper).
