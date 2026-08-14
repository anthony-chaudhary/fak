---
title: "Is fak just a firewall? The boundary FAQ, expanded"
description: "Firewall is the metaphor people reach for when they first see fak, and it is close enough to mislead. This page concedes what is genuinely firewall-like about the default-deny tool-call gate, then draws the real line: a firewall filters traffic by rules on packets or HTTP it inspects, while fak adjudicates effects by capability on a path the model does not control and understands tool-call semantics a packet filter cannot. Same instinct, different layer."
slug: fak-vs-firewall
keywords:
  - is fak a firewall
  - agent tool firewall
  - default-deny capability gate
  - tool call is a syscall
  - WAF vs fak
  - prompt injection quarantine
  - fail closed
date: 2026-07-03
---

# Is fak just a firewall?

Short answer: no, but the instinct behind the question is right, and it is worth taking
seriously rather than waving away. "Firewall" is the word most people reach for the first
time they see fak sit in front of an agent and start refusing things. fak even borrows the
phrase itself in a few places (`agent tool firewall`). So before drawing the line, this
page concedes what is genuinely firewall-shaped about it.

If you want the one-line version, it is on the [objections card](../objections.md) (item 7).
This page is the long form: what a firewall does, what fak does, where the analogy holds,
and where it quietly breaks.

## What a firewall actually does

A network firewall inspects traffic and decides whether to pass it, using rules written
against what it can see. A classic packet filter works at the network and transport layers:
source and destination address, port, protocol, connection state. A web application
firewall (WAF) goes higher and inspects HTTP payloads against signatures and rules, looking
for the shapes of known attacks. In both cases the unit of control is *traffic the firewall
inspects*, and the rules are matched against content or metadata on the wire.

That model has earned fifty years of trust for good reasons, and several of those reasons
are exactly why the comparison to fak feels natural.

## Where the analogy genuinely holds

Concede these first, because they are real:

- **A single chokepoint.** A firewall works by being the one thing every packet passes
  through. fak works the same way: every tool call passes through one adjudicator, so there
  is a single place where a decision gets made and recorded.
- **An allow-list you can review.** A well-run firewall is default-deny with an explicit
  list of what is permitted. fak's floor is the same shape: a reviewable allow-list where a
  tool or action that was never granted cannot run.
- **An audit trail.** A firewall logs what it passed and dropped. fak emits a per-call
  verdict ledger — one line at exit, e.g. `131 kernel decisions; 121 allowed / 5 denied /
  2 repaired / 0 quarantined / 3 deferred`.
- **An appliance you put in front.** You do not rewrite an application to sit it behind a
  firewall; you route through it. fak is the same drop-in shape: repoint one base URL, or
  run `fak manage -- claude`. You keep your model, IDE, and keys.

If someone calls that posture "a firewall for agents," they are not wrong about the
instinct. The disagreement is about the layer and the unit of control.

## Where the analogy breaks

### 1. The boundary is a tool call, not a packet

A firewall reasons about traffic. fak reasons about a **tool call** — the structured
"run this tool with these arguments" message the model emits. That is a different boundary,
and the mental model fak actually uses is the OS one: the tool call is treated like a
**syscall** that must cross into a small kernel the caller does not control. A packet filter
cannot see that a call named `delete_temp_file` is really asking to erase a home directory,
because that meaning lives in tool-call structure, not in packet headers. fak judges the
ask, with its arguments, at the semantic layer where the ask exists. The full mental model
is [The tool call is a syscall](../../explainers/tool-call-is-a-syscall.md).

### 2. It gates effects by capability, not traffic by rules

A WAF tries to *recognize* bad requests by matching patterns. That is pattern recognition,
and it is the part fak deliberately does not lean on. fak's floor is a **capability lock**:
the dangerous lever is simply not wired up, so refusing does not depend on catching an
attack. fak ships an injection detector too, but it is measured roughly 100% evadable by a
determined attacker and is labeled a bonus for exactly that reason. The security does not
rest on the detector; it rests on the action never being available. That distinction is the
whole point, covered in [Policy in the kernel](../../explainers/policy-in-the-kernel.md).

### 3. It fails closed, on the same call path

A network firewall screens traffic from the outside, and many deployments fail **open** when
the box crashes or times out — traffic keeps flowing. fak puts the check on the *same call
path* as the tool call, in one address space, with no inter-process hop. It is something the
call passes *through*, and it is default-deny, so a break does not silently open the gate. The
measured cost of that in-path check is about **362 ns per decision** (Apple M3 Pro), which is
free at the timescale of any real tool. The fail-open-vs-fail-closed contrast is spelled out
in the [FAQ](../../FAQ.md#how-is-fak-different-from-a-normal-firewall-or-api-gateway).

### 4. It judges the result coming back, not just the call going out

A firewall's job ends when it passes or drops a request. fak also looks at what a tool call
*returns*. A suspicious tool result can be **quarantined** — held out of the model's context
by structure — so poisoned output (a prompt injection smuggled into a fetched page) never
becomes the model's next set of instructions. A firewall has no equivalent of "let this
through, but keep the reply from becoming the operator's next command."

### 5. It can refuse a false "done"

This is the furthest thing from a firewall. fak's trust substrate can refuse a claimed
completion from git evidence rather than the agent's word, and every refusal carries a reason
from a closed vocabulary (`DEFAULT_DENY`, `POLICY_BLOCK`, `SECRET_EXFIL`, …) instead of free
text. A firewall neither verifies that claimed work happened nor speaks a shared refusal
vocabulary across a fleet. The capability-by-capability version of all this is the
[capability matrix](./matrix.md).

## So what is fak, if not a firewall?

The honest framing is: same *instinct*, different *layer*. A firewall guards the network
boundary and belongs there; you should still run one. fak guards the tool-call boundary,
which no firewall was built to see, using the oldest idea in operating systems — a boundary
the caller cannot cross by being persuasive. The two do not compete. A firewall and fak can
sit in the same stack governing different boundaries.

## Honest scope

- fak does not replace your network firewall or your WAF, and this page makes no such claim.
  Those layers guard traffic; fak guards tool calls. Keep both.
- None of this is a new invention. A 29-claim prior-art audit scored **0/29 novel**; every
  primitive here is established. The contribution is the assembly into one in-process,
  default-deny, fail-closed gate where the tool call is the checkpoint. See
  [`CLAIMS.md`](../../../CLAIMS.md).
- No market-adoption claim is made. The numbers on this page (the ~362 ns guard tax, the
  evadable detector, the ledger line) are witnessed and traced to the explainers linked above;
  no benchmark is quoted here that was not already run.

## Where to go next

- [Objections & one-line answers](../objections.md) — the thread-ready one-liner (item 7) and
  the nine sibling rebuttals.
- [Capability matrix across the category](./matrix.md) — fak scored cell-by-cell against
  guardrails libraries, gateways, and inference servers.
- [The tool call is a syscall](../../explainers/tool-call-is-a-syscall.md) — the mental model
  this whole comparison rests on.
- [FAQ](../../FAQ.md#how-is-fak-different-from-a-normal-firewall-or-api-gateway) — the short
  firewall/gateway answer in the machine-facing FAQ.

## Verify

```
test -f docs/adoption/compare/vs-firewall.md         # this artifact exists
fak score seo                                        # new doc does not red the SEO scorecard
```
