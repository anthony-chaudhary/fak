# fak playground — a browser-free, guided REPL for the tool-call-as-syscall idea

**Try this first.** One command, no API key, no model, no GPU, no network. You *propose*
tool calls; the fak kernel adjudicates each one — the way an OS adjudicates a syscall —
and narrates the verdict. It walks you through four calls against one tiny policy, then
hands you the keyboard to poke at your own.

```bash
bash examples/playground/play.sh
```

In a real terminal it is interactive: press Enter to fire each proposed call, then take
the wheel in a free REPL at the end (`playground> tool name:` — type anything, watch it
get judged). Piped or in CI it runs the whole guided tour on its own, so the same
verdicts print without a human at the keyboard:

```bash
bash examples/playground/play.sh | cat   # non-interactive: the full tour runs start to finish
```

Expected runtime: the non-interactive guided tour completes in seconds once `fak` is
on `PATH`.

## Prerequisites

A built `fak` binary on `PATH` (or `FAK_BIN=/path/to/fak`) — one static Go binary,
nothing else. The verdict is a pure function of `(policy, the proposed call)`, the
offline adjudication path `fak preflight` exposes, so the playground is deterministic and
CI-usable with no key, model, GPU, or network.

```bash
go build -o fak ./cmd/fak   # from this checkout — zero external deps
```

## The four calls it walks you through

The bundled floor is [`policy.json`](policy.json): a four-entry allow-list, one explicit
`deny`, and a `redact_fields` list. Four proposed calls draw out **three distinct
verdicts** (and two distinct *reasons* for the "no"):

| # | proposed call | verdict | why |
|---|---|---|---|
| 1 | `read_file {"path":"README.md"}` | **ALLOW** | on the allow-list — an affirmative rung admits it |
| 2 | `run_shell {"cmd":"rm -rf /"}` | **DENY · DEFAULT_DENY** | never *admitted* — nobody had to foresee it to refuse it |
| 3 | `send_email {…}` | **DENY · POLICY_BLOCK** | named in the manifest, but under `deny` — a *different* structured "no" |
| 4 | `create_ticket {…,"password":"hunter2"}` | **TRANSFORM** | allowed, but `password` is rewritten to `[REDACTED]` before dispatch |

Steps 2 and 3 are the point: a `DENY` for a call *nobody wrote a rule about*
(`DEFAULT_DENY`) is not the same as a `DENY` for a call *someone explicitly forbade*
(`POLICY_BLOCK`). Both are codes from the closed refusal vocabulary in
[`POLICY.md`](../../POLICY.md), so a loop can branch on the reason without another model
turn. Step 4 is the third verdict — a call that is *allowed* yet *sanitized* first.

## Why this popularizes the concept

Reading that "the tool call is a syscall" is abstract. *Proposing* an `rm -rf` and
watching it come back `DENY (DEFAULT_DENY)` — then editing one line of `policy.json` and
watching the same call flip — is not. Guided interactivity turns a passive reader into an
active one, which is how a concept sticks. Every verdict is the policy applied to the
call, never a model choosing to behave.

## Scope — what this does **not** claim

A **name-level** playground: it exercises the allow / default-deny / explicit-deny /
redact-transform verdicts on tool *names* and a redacted field. It does not drive a live
model into the gate ([`adjudication-demo/`](../adjudication-demo/README.md) does),
exercise argument-value rules ([`sql-analyst-policy.json`](../sql-analyst-policy.json)
does), or quarantine untrusted tool *results*
([`quarantine-demo/`](../quarantine-demo/README.md) does). No adoption or benchmark claim
is made — the playground proves one thing you can feel with your own hands: the verdict is
the policy file, not the model.

## Files

| file | what it is |
|---|---|
| `README.md` | this walkthrough |
| `play.sh` | the guided REPL — four narrated calls, then a free prompt of your own |
| `policy.json` | the tiny floor every verdict is read against — edit it and the verdicts change |

Related: [`deny-in-60s/`](../deny-in-60s/README.md) (the same default-deny boundary as a
single one-shot witness), [`deny-as-value/`](../deny-as-value/README.md) (what a loop
*does* with a structured deny), and [`../../POLICY.md`](../../POLICY.md) (the manifest
format and the closed refusal vocabulary).
