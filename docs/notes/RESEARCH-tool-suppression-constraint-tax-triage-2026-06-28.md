---
title: "idea-scout triage: 'Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints' — a serving-side phenomenon (open-weight models STOP invoking tools when Tool Calling and a JSON-Schema output constraint are enabled JOINTLY, while keeping high schema compliance; each works alone), directly on fak's guided-decode + adjudication seam. Verdict: prior art to cite + empirical validation of fak's #907 'one canonical internal shape' decision (model tool-calling AS the schema — grammar.Grammar -> oneOf-of-tools — so tool-selection and structured output are ONE constraint, never two competing ones; Tool Suppression is the failure mode of STACKING an independent response_format schema on a tool-calling turn, the configuration fak's canonical shape avoids) + a named SELF-INFLICTED failure mode fak can induce two ways (native guided-decode internal/model/constraint.go behind FAK_NATIVE_GUIDED_DECODE=1, which DEFAULTS OFF = accidental protection; or the #907 passthrough forwarding a client response_format to an upstream alongside tools, where the suppression is the UPSTREAM's behavior, OBSERVED not caused). The load-bearing trust-floor point: a suppressed tool call is INVISIBLE to a default-deny adjudication floor — internal/adjudicator/decide.go Adjudicate(*abi.ToolCall) only ever sees a call that WAS emitted, so a suppressed call is a capability FALSE-NEGATIVE below the floor (no call = nothing to deny = trivially 'safe'), a silent capability regression an adjudicator structurally cannot catch by adjudicating calls; catching it needs OBSERVING THE ABSENCE (tool offered + joint schema active + zero calls + schema-valid prose), an observability ask fak does not measure today. Not adopted as a new mechanism. One residual FILED not built: a gateway tool-suppression-suspected signal, gated on a real served-open-weight reproduction this dev box cannot run. No code change; no change to tools/idea_scout.py (surfaced + scored correctly). (2026-06-28)"
description: "Triage of the idea-scout candidate arXiv:2606.25605 (Fangzheng Li, Aimin Zhang, Chen Lv — 'Constraint Tax in Open-Weight LLMs: An Empirical Study of Tool Calling Suppression Under Structured Output Constraints', submitted 2026-06-24): Tool Calling and Structured Output are two core capabilities of modern Agent systems whose interaction under JOINT deployment is under-studied; the paper reports a reproducible production phenomenon — when Tool Calling and a JSON-Schema constraint are simultaneously enabled, multiple open-weight model families CEASE invoking tools while still producing high-schema-compliance output, a behavior it names Tool Suppression, reproduced across model families and deployment settings, with tool execution and schema compliance both remaining functional when enabled SEPARATELY. The framing — a 'Constraint Tax' the structured-output constraint silently levies on the tool-calling capability. Verdict: prior art to cite, and a phenomenon that lands squarely on fak's serving seam (unlike the recent off-mission training/pedagogy hits #1008/#1009). (1) Empirical validation of fak's #907 structured-output design decision: fak's canonical internal shape is grammar.Grammar -> a oneOf-of-tools JSON Schema -> EITHER the response_format carrier (ride mode) OR a native logit mask (#929) — i.e. fak already models tool-CALLING AS the schema, one unified constraint, so the two-competing-constraints configuration that produces Tool Suppression (an independent output response_format stacked on top of a tool-calling turn) is the one fak's 'one canonical internal shape' explicitly avoids. (2) A self-inflicted failure mode fak can INDUCE two ways: the native guided-decode logit mask (internal/model/constraint.go, behind FAK_NATIVE_GUIDED_DECODE=1) applied on a tool-calling turn, which DEFAULTS OFF so a schema mask is never stacked unless an operator opts in (accidental protection); and the #907 passthrough (internal/gateway/http.go:464, stream_proxy.go, wire.go) forwarding a client's response_format to an upstream vLLM/SGLang engine alongside tools, where any resulting suppression is the UPSTREAM engine's behavior — OBSERVED/relayed, not a fak action (the conflation-honest controlled-vs-observed boundary). (3) The load-bearing trust-floor insight (why the issue is labelled trust-floor): fak's default-deny adjudication floor — internal/adjudicator/decide.go Adjudicate(ctx, *abi.ToolCall) — only ever receives a tool call that WAS emitted; a SUPPRESSED tool call never reaches it, so Tool Suppression is a capability FALSE-NEGATIVE that lives BELOW the floor (no call to deny = trivially 'safe' to a deny-bad-calls gate) while being a silent capability regression. An adjudicator that grades emitted calls structurally cannot detect a call that was never emitted; detection requires OBSERVING THE ABSENCE — a turn where a tool was available and a joint output schema was active but the completion is schema-valid prose with zero tool calls. Not adopted as a new kernel mechanism (the right design is already shipped as #907's unified shape). Not a classic adversarial threat (a constraint-interaction defect, no attacker). One recorded residual, FILED not built: a gateway 'tool-suppression-suspected' observability counter (tools offered + joint response_format active + zero emitted calls + schema-compliant completion), gated behind a real reproduction on a served open-weight model under joint constraints — a GPU/model-gated witness this win32 dev box cannot run, so the metric's 'expected-a-tool-call' calibration is unproven and the counter is not shipped. No capability code change. Scout calibration: surfaced under topic tool-call-adjudication (score 44) on a genuine tool call (title) term + freshness (4d, <=30d) — correctly on-topic AND close-to-mission (a serving-side tool-calling/structured-output interaction touching fak's exact guided-decode + adjudication surfaces), the scout working as designed; no change to tools/idea_scout.py."
---

# idea-scout triage — Tool Suppression under joint tool-call + schema constraints (issue #1121)

> Closes the daily idea-scout candidate [#1121](https://github.com/anthony-chaudhary/fak/issues/1121)
> (`tools/idea_scout.py`, filed 2026-06-28). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a capability,
> defend against as a threat, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + empirical validation of fak's [#907](https://github.com/anthony-chaudhary/fak/issues/907)
> "one canonical internal shape" design decision. fak already models tool-CALLING *as*
> the schema (`grammar.Grammar` → a `oneOf`-of-tools JSON Schema → the `response_format`
> carrier OR a native logit mask), so the two-competing-constraints configuration that
> produces Tool Suppression — an independent output `response_format` stacked on a
> tool-calling turn — is exactly the one fak's unified shape avoids. It is also a named
> *self-inflicted* failure mode fak can induce two ways: the native guided-decode mask
> (`internal/model/constraint.go`, behind `FAK_NATIVE_GUIDED_DECODE=1`, which **defaults
> off** — accidental protection), and the #907 passthrough forwarding a client
> `response_format` to an upstream alongside tools (where the suppression is the
> **upstream's** behavior — OBSERVED, not a fak action). The load-bearing trust-floor
> point: a suppressed tool call is **invisible** to fak's default-deny adjudication floor
> — `Adjudicate(*abi.ToolCall)` only ever sees a call that *was* emitted — so Tool
> Suppression is a capability FALSE-NEGATIVE *below* the floor (no call = nothing to deny
> = trivially "safe"), a silent regression an adjudicator cannot catch by adjudicating
> calls; catching it needs OBSERVING THE ABSENCE. Not adopted as a new mechanism (the
> right design ships as #907). Not a classic threat. One residual — a gateway
> "tool-suppression-suspected" observability counter — is FILED not built (it needs a
> served-open-weight reproduction this dev box cannot run). No code change.**

**Source:** https://arxiv.org/abs/2606.25605 — "Constraint Tax in Open-Weight LLMs: An
Empirical Study of Tool Calling Suppression Under Structured Output Constraints",
Fangzheng Li, Aimin Zhang, Chen Lv (submitted 2026-06-24). Read from the arXiv abstract
as surfaced by the scout on 2026-06-28; this is a surface read of the abstract, not a
paper audit.

## What it is

A **serving / deployment** result, not a training method, an attack, or a protocol. Two
core Agent capabilities — **Tool Calling** and **Structured Output** (a JSON-Schema
constraint on the generation) — are usually studied in isolation; the paper studies their
**joint** deployment and reports a reproducible production phenomenon it names **Tool
Suppression**:

- when Tool Calling and a JSON-Schema constraint are **simultaneously** enabled, multiple
  **open-weight** model families **stop invoking tools** — while still emitting **high
  schema-compliance** output;
- each capability is **functional in isolation** — tool execution works when only tools
  are on, schema compliance works when only the schema is on; the regression appears
  **only under the joint constraint**;
- reproduced **across model families and deployment settings** (the abstract's claim;
  per-model numbers are not in the surfaced text).

The paper's framing is a **"Constraint Tax"**: the structured-output constraint silently
*taxes* the tool-calling capability — the model satisfies the constraint it can see (emit
schema-valid text) by abandoning the capability the constraint did not explicitly demand
(call a tool). This is a constraint-**interaction** defect of the decoding/prompting
stack, not a property of either capability alone.

## Where this lands on fak (the surfaces, verified against the tree)

fak is an **agent kernel** at the tool-call seam, and it has *both* sides of the exact
interaction this paper describes:

- **The structured-output constraint** — `internal/model/constraint.go` (#929): the
  native guided-decode path is a per-step **JSON-schema / grammar logit mask** over the
  canonical `oneOf`-of-tools shape, plus the OpenAI `logit_bias` map. It is gated by
  `GuidedDecodeEnabled()` → `FAK_NATIVE_GUIDED_DECODE=1` and **defaults OFF** ("a schema
  mask is never applied unless an operator opts in"), so in-kernel it is **bit-exact-off**
  by default.
- **The ride-mode passthrough** — `internal/gateway/http.go:464`, `stream_proxy.go`,
  `wire.go` (#907): fak forwards the client's `response_format` / `logit_bias` to the
  upstream engine **verbatim** (vLLM `guided_json`/`response_format`, SGLang `json_schema`)
  and adjudicates the constrained candidate **after** generation. When fak fronts an
  upstream, the constraint is enforced **there**, not in fak.
- **The tool-call floor** — `internal/adjudicator/decide.go`:
  `Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict`. The floor adjudicates a
  tool call **that was emitted**. It has no input for a call that was never produced.
- **The canonical shape** — per the [#907 structured-output readout](RESEARCH-structured-output-decoding-2026-06-26.md):
  every constraint surface (OpenAI `response_format`/`logit_bias`, vLLM/SGLang guided
  fields, Anthropic/MCP tool schema, fak `grammar.Grammar`) is mapped onto **one canonical
  internal shape** — `grammar.Grammar` → a `oneOf`-of-tools JSON Schema → the
  `response_format` carrier (ride) **or** a logit mask (native). fak models tool-CALLING
  **as** the schema.

## The three triage questions

### Adopt as a capability? — adopt the *design constraint*, which fak already shipped (#907). No new mechanism.

The paper is **empirical validation** of a decision fak already made. Tool Suppression is
the failure mode of running **two competing constraints** on one turn: a tool-calling
decision *and* an independent output `response_format` schema, where the model resolves
the conflict by satisfying the schema and dropping the tool call. fak's #907 design
**collapses the two into one**: tool-calling *is* the schema (`grammar.Grammar` →
`oneOf`-of-tools), so there is a single constraint to satisfy, not two to trade off. The
honest reading is "fak's canonical shape avoids the configuration that produces Tool
Suppression" — so the paper **strengthens** the #907 / #929 design rather than asking fak
to build anything. The design **rule it crystallizes**: do not stack an independent
output schema on top of a tool-calling turn; express the tool selection *in* the schema.

### Defend against as a threat? — a *self-inflicted* failure mode, not an adversary; fak can induce it two ways, and the default-off posture is the protection.

There is no attacker. But Tool Suppression is a real regression fak's own stack can
**cause**:

1. **Native, in-kernel** — turning on `FAK_NATIVE_GUIDED_DECODE=1` and applying the
   schema mask on a tool-calling turn. fak's **default-off** posture (`constraint.go`)
   means a schema mask is never stacked on a tool-calling decision unless an operator
   explicitly opts in — an **accidental but real protection** against inducing the
   phenomenon in-kernel.
2. **Passthrough, upstream** — the #907 ride path forwarding a client `response_format`
   to a vLLM/SGLang engine **alongside** the tool list. Here any suppression is the
   **upstream engine's** behavior. fak **relays** the constraint; it does not enforce it.
   This is a **controlled-vs-observed** boundary (the conflation discipline): a fak
   exit-summary or metric must label a suppressed-tool outcome on this path as
   **OBSERVED** (the provider's joint-constraint behavior), never as a fak adjudication
   outcome.

### The trust-floor blind spot (why the issue is labelled `trust-floor`) — the load-bearing point.

A default-deny adjudication floor sees **only the calls that are emitted**.
`Adjudicate(ctx, *abi.ToolCall)` is handed a `*abi.ToolCall`; a **suppressed** call is
never constructed, so it never reaches the floor. To a gate whose job is to **DENY bad
calls**, a suppressed call is **trivially safe** — there is nothing to deny — even though
it is a silent **capability** regression. So Tool Suppression is a **false-negative below
the floor**: not a safety event the adjudicator should have caught, but a capability event
the adjudicator **structurally cannot** catch by adjudicating calls.

Detecting it is therefore an **observability** ask, not an **adjudication** ask: it
requires **observing the absence** — a turn where (a) a tool was available, (b) a joint
output schema was active, and (c) the completion is schema-valid prose with **zero** tool
calls. fak does not measure this today. This is the sharp, citable distinction the paper
sharpens: *an adjudication floor that grades emitted calls is blind to a capability that
was suppressed before any call existed.*

## Recorded residual (FILED, not built)

The smallest honest follow-on is a **gateway "tool-suppression-suspected" signal**: a
per-turn counter that fires when tools were offered **and** a joint output `response_format`
was active **and** the completion carried zero tool calls **and** was schema-compliant —
surfaced as an L2 metric next to the existing structured-output passthrough seam
(`internal/gateway`), labelled **OBSERVED** on the ride path (it is the upstream's
behavior). It is **not shipped here** because the "expected a tool call" half of the
predicate needs **calibration against a real reproduction** — a served **open-weight**
model run under joint constraints to confirm a tool *should* have fired — and that witness
is **GPU/model-gated**: this win32 dev host serves no model and cannot reproduce the
phenomenon (see [`docs/notes/AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md`](AVOID-TESTING-ON-THIS-MACHINE-2026-06-25.md)).
Shipping the counter without that calibration would be an unwitnessed "expected" claim, so
it is filed as the next checkable step, not built.

## Scout calibration

Surfaced under topic **`tool-call-adjudication`** (score 44) on a genuine `tool call`
(title) term + freshness (4 days old, ≤30d) — **correctly on-topic, and close to
mission**: unlike the recent off-mission training/pedagogy hits
([#1008 CoT-training](RESEARCH-cot-training-gains-agents-triage-2026-06-27.md),
[#1009 SE-education](RESEARCH-llm-mcp-se-education-triage-2026-06-27.md)), this is a
**serving-side** tool-calling / structured-output interaction that touches fak's **exact**
guided-decode (`internal/model/constraint.go`) and adjudication
(`internal/adjudicator/decide.go`) surfaces. The scout judged new-and-on-topic and handed
the worth-pursuing call to human triage — **working as designed**. No change to
`tools/idea_scout.py`.

## Disposition

**Cite as prior art** (empirical validation of fak's #907 unified-constraint shape) and
**record the self-inflicted failure mode + the trust-floor blind spot** above. No
capability adopted (the right design is already shipped), no threat to defend (no
adversary), one observability residual filed behind a reproduction gate fak cannot reach
on this host. The issue is resolved by this triage note; the follow-on metric is the
named next step.
