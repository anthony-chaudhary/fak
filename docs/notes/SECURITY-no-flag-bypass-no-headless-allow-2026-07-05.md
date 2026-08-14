---
title: "Security floor: no bypass-by-flag, no headless default-allow (2026-07-05)"
description: "fak's capability floor has no force escape hatch and no headless auto-approve — the same structural deny applies interactive or headless, flag-set or unset, proven by a regression matrix. Contrast with Hermes Agent's force-skips-all-guards + trust-by-config auto-approve."
---

# The floor has no escape hatch: no `force` flag, no headless default-allow

> **A security floor with a `force` bypass and a headless default-allow is not a
> floor.** fak's capability floor has neither. `Adjudicator.Adjudicate` is a *pure
> function of the reviewed policy and the tool call* — there is no `force`
> parameter, no interactive/headless/TTY/session-mode input, and no env-based
> auto-approve. The **only** way to admit a denied capability is to change the
> in-git `Policy`; nothing a caller sets at runtime moves the verdict. Companion to
> [`SECURITY-capability-floor-2026-06-18.md`](SECURITY-capability-floor-2026-06-18.md)
> (the allow/deny-by-argument visual). Resolves
> [#2921](https://github.com/anthony-chaudhary/fak/issues/2921), a child of epic
> [#2908](https://github.com/anthony-chaudhary/fak/issues/2908) track C
> (deny-by-structure safety).

## The contrast

[Hermes Agent](https://github.com/nousresearch/hermes-agent) fronts its terminal
tool with a 2,921-line, ~90-regex approval layer (`tools/approval.py`). Two holes
make it default-allow the moment you leave an interactive human at the keyboard:

| | Hermes Agent (`tools/approval.py`) | fak capability floor |
|---|---|---|
| **`force` flag** | `force=True` on the terminal tool **skips ALL guards** (hardline, tirith, dangerous-command detection). Meant for the gateway's post-approval replay, it is a full bypass for anything that can set the flag. | **No such flag exists.** The verdict is computed from the policy and the call; there is no argument, field, or Meta key that turns a `DENY` into an `ALLOW`. |
| **headless session** | A non-interactive / non-gateway local session **auto-approves with a warning** — "trust-by-config" (`approval.py:2029`). Effective posture: default-allow when headless. | **No headless mode.** `Adjudicate` has no session-mode input and reads no `CI` / TTY / "non-interactive" signal. The same structural deny applies interactive or headless. |

The difference is architectural, not a tuning knob: Hermes fights obfuscated
commands with a regex normalizer plus an LLM *in the deny path*, then carves out
escape hatches for the cases where a human is not watching. fak refuses
**by structure** at the tool-call boundary, so there is no path — flagged,
headless, or otherwise — that reaches the effect without crossing the floor.

## Why fak cannot be bypassed by a flag

The reference monitor's whole decision is:

```go
func (a *Adjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict
```

- **No escalation input.** The signature carries a context and the call — no
  `force`, no `interactive bool`, no session handle. The `abi.ToolCall` envelope
  (`internal/abi/types.go`) is a *frozen, additive-only* wire shape whose fields
  are `Op, Tool, Engine, Args, Caps, Spec, Txn, SeqNo, TraceID, Meta, Ext`. None is
  a permission override.
- **`Meta` cannot widen authority.** `Meta` is the one open string map, and the
  spec is explicit: *"unknown keys MUST be ignored."* `decide.go` states the
  security consequence directly — *"caller Meta is model-controlled and cannot
  widen authority."* A model (or an injection) can attach `force=true`,
  `approve=true`, `dangerously_skip_permissions=true`, `headless=true` — the floor
  ignores every one.
- **The only lever is the reviewed policy.** Softening a rung is possible, but
  only through the in-git `Policy` (an `advisory_reason`, a `complain` entry, an
  `admit_and_log` posture), and even those are *clamped*: the genuine-danger
  reasons (`POLICY_BLOCK`, `SECRET_EXFIL`, `EGRESS_BLOCK`) can never be
  blanket-softened. That is a reviewable config change in version control, not a
  runtime flag a caller flips.

### The harness's own bypass flag does not reach the floor

Claude Code's `--dangerously-skip-permissions` disables *Claude Code's own*
permission prompt. `fak manage` **passes it through to the child unchanged and never
consumes it** (asserted in `cmd/fak/guard_codex_test.go`), so a child launched with
its own permission bypass *still* has every proposed tool call cross fak's floor.
The harness can turn off its prompts; it cannot turn off the kernel.

## The regression witness

The [#2921](https://github.com/anthony-chaudhary/fak/issues/2921) acceptance — a
matrix of **(interactive | headless) × (flag-set | flag-unset)** proving the floor
holds identically — is locked as a Go regression test:

- **`internal/policy/flag_bypass_capfloor_conformance_test.go`**
  - `TestFloorHoldsAcrossFlagAndHeadlessMatrix` re-adjudicates every deny/allow
    rung (name-deny, self-modify, arg-value deny, the hardwired egress floor,
    default-deny, and one affirmative allow) in every matrix cell. The flag axis
    merges hostile `force` / `approve` / `bypass` Meta payloads onto each call; the
    headless axis sets every ambient `CI` / `FAK_HEADLESS` / `FAK_NONINTERACTIVE` /
    no-TTY signal a naive auto-approver would read. The verdict is byte-identical in
    all of them.
  - `TestToolCallEnvelopeHasNoBypassSeam` is the structural tripwire: it fails if a
    future edit adds a `Force` / `Interactive` / `SkipPermissions`-shaped typed
    field to the frozen `abi.ToolCall`, forcing any such change to reckon with the
    floor instead of quietly opening a bypass class.

This joins the sibling bypass-axis witnesses that pin the same #2018 floor:
`internal/policy/isolation_capfloor_conformance_test.go` (a stronger
process-isolation tier never bypasses the floor) and
`internal/adjudicator/dogfood_manifest_test.go` (the shipped floor's verdict
matrix). Prove any single verdict yourself without a session:

```bash
fak preflight --tool Bash --args '{"command":"rm -rf /tmp/x"}' \
  --policy examples/dogfood-claude-policy.json
#   => verdict=DENY reason=POLICY_BLOCK   (there is no --force to add that changes this)
```

## Honest scope

This note and its test cover the **call-side** capability deny: no flag, env var,
or headless mode escalates a tool call past the floor. It is paired with — not a
substitute for — the result-side containment (poisoned tool results held out of
context) and the policy loader's clamp on which reasons may be softened. The
guarantee is structural for the adjudicator's own verdict; a floor an operator
deliberately widens in-git is a reviewed decision, which is exactly the point —
authority changes are visible in version control, never a runtime bypass.
