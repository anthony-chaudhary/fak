---
title: "fak security visual: the capability floor (2026-06-18)"
description: "Shows how fak adjudicates every tool call before it runs, allowing everyday tools and denying destructive calls by argument value at the kernel boundary."
---

# Security visual — the capability floor: allow the useful, deny the dangerous

> **One picture of the trust boundary.** Every tool call a model proposes is
> adjudicated *before it runs*. The floor allows the everyday tool set so the agent
> is useful, and refuses destructive calls **by argument value** — `Bash` is allowed,
> but `Bash{command:"rm -rf /"}` is denied before the shell ever sees it. This is the
> deployable, file-authored form of fak's "permissions as the floor" thesis, shown on
> the real Claude Code dogfood (`fak/examples/dogfood-claude-policy.json`).

Companion to `COMPARE-security-model-vs-guardrails` (private research companion)
(why this layer, not a content guardrail) and [`fak/POLICY.md`](../../POLICY.md) (the
manifest schema + author/validate workflow). Generated 2026-06-18.

---

## The visual — where the deny happens

```
   model proposes a tool call                         the kernel adjudicates              effect
   (untrusted, ring-3)                                 (BEFORE dispatch)                   (only if ALLOWED)
 ┌──────────────────────────┐      tool_use      ┌──────────────────────────────┐
 │  "run: rm -rf /tmp/x"     │ ─────────────────▶ │  Deny[tool]?            ──┐    │
 │  "run: ls -la"            │                    │  self-modify glob?       │    │
 │  "Edit .git/config"       │                    │  arg-value predicate?    ├──▶ │  ALLOW ─▶ run the tool
 └──────────────────────────┘                    │  allow-list / prefix?    │    │  DENY  ─▶ refused, with a
            ▲                                     │  else DEFAULT_DENY     ──┘    │            closed-vocab reason
            │  refusal + reason (POLICY_BLOCK / SELF_MODIFY / DEFAULT_DENY)       │            (never reaches shell)
            └─────────────────────────────────────────────────────────────────┘
```

The deny is **structural**, not a model judgement: an injection that talks the model
into proposing `rm -rf` changes nothing — the call is refused at the boundary. The
refused call never reaches the shell, the filesystem, or the network.

## The chart — what the default dogfood floor allows vs denies

| Proposed call | Verdict | Reason | Why |
|---|---|---|---|
| `Bash ls -la`, `cat`, `grep`, `git commit` | ✅ **ALLOW** | — | everyday dev work |
| `Read` / `Edit` / `Write` a normal file | ✅ **ALLOW** | — | the agent has to do its job |
| `Bash rm -rf …` / `rm -f …` | ⛔ **DENY** | `POLICY_BLOCK` | destructive removal — denied by **argument value** |
| `Bash sudo …` | ⛔ **DENY** | `POLICY_BLOCK` | privilege escalation |
| `Bash git push …` | ⛔ **DENY** | `POLICY_BLOCK` | the agent can commit, but not publish |
| `Bash curl … \| sh` | ⛔ **DENY** | `POLICY_BLOCK` | remote-code-execution pipe |
| `Bash :(){ :\|:& };:` | ⛔ **DENY** | `POLICY_BLOCK` | fork bomb |
| `Bash dd if=… of=/dev/sd…` / `mkfs …` | ⛔ **DENY** | `POLICY_BLOCK` | disk wipe |
| `Edit`/`Write` into `.git/`, `.ssh/`, `internal/kernel/`, `VERSION` | ⛔ **DENY** | `SELF_MODIFY` | can't rewrite the kernel or secrets |
| any tool the floor never named | ⛔ **DENY** | `DEFAULT_DENY` | fail-closed — nothing is allowed by default |

The load-bearing idea: **the deny is on the argument, not just the tool name.** A
coarse tool like `Bash` is one allow-listed capability, but its `command` value is
gated by `arg_rules` (RE2 deny-patterns), so the floor distinguishes `ls` from
`rm -rf` without forking the kernel.

## Prove any verdict without launching a session

```bash
# one call, adjudicated against the shipped floor:
fak preflight --tool Bash --args '{"command":"rm -rf /tmp/x"}' \
  --policy fak/examples/dogfood-claude-policy.json
#   => verdict=DENY reason=POLICY_BLOCK by=monitor

fak preflight --tool Bash --args '{"command":"ls -la"}' \
  --policy fak/examples/dogfood-claude-policy.json
#   => verdict=ALLOW reason=NONE by=monitor

# eyeball the entire floor (allow-list, arg rules, self-modify globs, redactions):
fak policy --check fak/examples/dogfood-claude-policy.json
```

The same matrix is locked as a regression test
(`fak/internal/adjudicator/dogfood_manifest_test.go::TestDogfoodManifestVerdictMatrix`),
so a manifest edit that silently widens the floor fails CI.

## How to use / change it

- The Claude Code dogfood (`fak/scripts/dogfood-claude.sh`, see
  [`fak/DOGFOOD-CLAUDE.md`](../../DOGFOOD-CLAUDE.md)) loads this floor by default so
  interactive sessions work; override with `FAK_DOGFOOD_POLICY=<path>` or
  `FAK_DOGFOOD_POLICY=none` for the raw fail-closed kernel.
- The floor is a JSON manifest, not a code change: `fak policy --dump` to start from
  the built-in default, edit, `--check`, then `--policy FILE`. Full schema +
  closed refusal vocabulary in [`fak/POLICY.md`](../../POLICY.md).
- Honest scope: the floor bounds *which tools run with which argument values*. It is
  the call-side capability deny — paired with the result-side containment
  (poisoned tool results held out of context) described in the top-level
  [`README`](../../README.md) and `COMPARE-security-model-vs-guardrails` (private research companion).
