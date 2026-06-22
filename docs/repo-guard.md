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
never touching in-repo work or ordinary scratch (`/tmp`, `~/.cache`).

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
                             "command": "python \"$CLAUDE_PROJECT_DIR/tools/repo_guard.py\" --hook" }] }]
```

On a violation the hook returns `permissionDecision: deny`, so the tool call never
runs. This matters most under `permissions.defaultMode = bypassPermissions` (the
fleet default): there is **no human approval prompt**, so a structural PreToolUse
deny is the only thing between the model and a sibling-repo `rm -rf`.

```
 model proposes  ──▶  PreToolUse hook  ──▶  repo_guard resolves the target
   rm -rf ../tools        (the floor)         outside the repo?  ──▶  DENY, call never runs
```

### Knobs (mirror the trunk-guard convention)

`repo_guard` is **fail-open** on any internal error — a guard bug must never wedge
a live multi-session fleet; it is defense-in-depth behind the FAK policy and human
review, not the sole gate.

- `FAK_REPO_GUARD=warn` — advisory: log the violation to stderr, allow the call.
- `FAK_REPO_GUARD=off` — disable the hook entirely.
- default (unset) — **enforce** (deny out-of-tree destructive/write ops).

## Verify it

```bash
# FAK floor — the incident build flips ALLOW -> DENY:
fak preflight --tool Bash --args '{"command":"go build -o ../tools/x ./cmd/fak"}' \
  --policy examples/repo-guard-policy.json          # => verdict=DENY reason=POLICY_BLOCK
fak preflight --tool Bash --args '{"command":"go build -o fak.exe ./cmd/fak"}' \
  --policy examples/repo-guard-policy.json          # => verdict=ALLOW (in-repo)

# DOS floor — resolves the absolute sibling a regex can't:
python tools/repo_guard.py --selftest               # => 22/22 passed
python tools/repo_guard.py --check "rm -rf /c/Users/you/work/tools"   # => DENY OUT_OF_TREE_WRITE
python tools/repo_guard.py --check "rm -rf ./build"                   # => ALLOW

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
- The guard only ever refuses a target **outside** the workspace (and outside the
  scratch allow-list). No in-repo work is ever blocked — consistent with the
  `dos.toml` rule that a declared reason introduces no spontaneous refusal of
  legitimate work.

## See also

- [`POLICY.md`](https://github.com/anthony-chaudhary/fak/blob/main/POLICY.md) — the capability floor + the `arg_rules` schema.
- [`dos.toml`](https://github.com/anthony-chaudhary/fak/blob/main/dos.toml) — `[reasons.OUT_OF_TREE_WRITE]` and its sibling reasons.
- [`DOGFOOD-CLAUDE.md`](https://github.com/anthony-chaudhary/fak/blob/main/DOGFOOD-CLAUDE.md) — the policy's deny table in context.
- [`tools/proc_resource_guard.py`](https://github.com/anthony-chaudhary/fak/blob/main/tools/proc_resource_guard.py) — the sibling
  guard this one is modeled on (runaway-process reaper).
