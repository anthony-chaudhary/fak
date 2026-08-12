# The port is the agent: gateway UI/UX plan

> Status: design proposal, 2026-08-11. Companion to
> [`docs/fleet-ui-generation-plan.md`](fleet-ui-generation-plan.md) (the terminal fleet
> workspace, #6477/#6476). That plan owns the **cross-fleet** operator surface in the TUI.
> This plan owns the **per-agent** surface that each gateway already serves on its own port,
> and the addressing scheme that connects the two.

## The claim in one line

Every `fak guard` session already binds its own private loopback port, already publishes
that port to a durable index, and already serves an HTML page and a ~25-block live read
model on it. **The per-agent visual surface is not a new frontend — it is an origin that
exists and renders four facts where it could render the whole session.** The work is to
promote that origin into a console, and to define the address space beneath it so 10,000
micro-contexts are reachable without 10,000 listeners.

## Value frame

- **For:** an operator running several guarded agents (and, under them, micro-context
  fleets) who needs to answer *what is this agent doing, what did the kernel refuse, and
  what is it costing* without attaching to its terminal.
- **Problem today:** the answer exists — it is in `/debug/vars` on that agent's own port —
  but the only rendered form is a static link hub. An operator either curls JSON or reads
  the TUI's session-local blocks, which is one seam away from the agent itself.
- **Better because:** the read model, the auth posture, the port, and the discovery index
  are all already built. Rendering them is the smallest remaining step, and it is the
  *only* human surface possible for a micro-context, which by design has no terminal.

The real next-best alternative is "curl `/debug/vars` and read the JSON." A console earns
its keep only where it removes an operator join — attention ordering, the base-fingerprint
break, the exception rows under an aggregate. Pretty-printing the same JSON does not.

## What already exists (verified in this tree)

| Capability | Where | State |
|---|---|---|
| One private loopback port per guarded agent | `cmd/fak/guard.go:796` — `127.0.0.1:0`, OS-picked, bound *before* the child is wired | **Shipped** |
| Port published to a durable, prefix-resolvable index | `internal/guardsessions` — `Row.GatewayURL` + read-scoped `Bearer`, stamped by `publishGuardSessionGateway` (`cmd/fak/guard_sessions.go:106`, #5400) | **Shipped** |
| "Missing URL" is diagnosable, not ambiguous | `guardsessions.GatewayPublishEpoch` / `PredatesGatewayPublish` — separates *the build had no publisher* from *this session published nothing* | **Shipped** |
| An HTML origin on every port | `internal/gateway/home.go` — self-contained `html/template`, strict CSP, `no-store`, no credentials, no asset pipeline | **Shipped, 4 facts + 6 links** |
| A rich live read model on the same port | `internal/gateway/debug.go:19` — ~25 blocks: adjudication verdicts, inference TTFT/prefill/decode, inflight max age, upstream errors, cache attribution, token savings, shrink levers, KV memory, harness CPU/RSS/IO, sessions, assumptions, context queries, endpoints, fleet, watchdog | **Shipped** |
| Browser-reachable without a token, on loopback only | `internal/gateway/http.go:417` — `/`, `/a2a/v1/agent-card`, `/metrics`, `/debug/vars` are auth-exempt **iff** the peer IP is loopback (`requestFromLoopback`, fail-closed, ignores `X-Forwarded-For`) | **Shipped** |
| A per-agent identity/skills/task card | `/a2a/v1/agent-card` + the A2A task, cancel, progress, and push-notification surface (`internal/gateway/a2a*.go`) | **Shipped** |
| Cross-machine fleet aggregate | `gateway.SessionFleet` ← `cmd/fak/guard_fleet.go` (30s TTL, 8-row bounded sample, snapshot-only, no subprocess on the poll path) | **Shipped** |
| Micro-context identity | `microagent.Descriptor` — `base_id + task_delta + tools + budget + continuation + output_contract` | **Shipped** |
| Micro-context runtime tiers | `internal/microagent` — scheduler, warm band, hibernation, compat-batch, fair-sched | **Shipped** |

The consequence worth stating plainly: **a console on the agent's own port needs no new
auth, no new port, no new daemon, and no token in the browser.** The server already holds
the data and the loopback exemption already admits the browser. Off-box, the same page
stays bearer-gated by the existing `ReadBearer`. That is an unusually clean security story
for a dashboard, and it comes from decisions already made.

## The two-tier address space

The user-facing question — *should every agent have its own addressable port?* — resolves
into two answers, because there are two populations with cardinalities four orders of
magnitude apart.

```text
TIER A — PROCESS-ADDRESSABLE                    TIER B — DESCRIPTOR-ADDRESSABLE
one loopback port per guard session             many logical contexts per port
cardinality: 10^0 – 10^2 per box                cardinality: 10^2 – 10^4 per port
address: http://127.0.0.1:<port>                address: <that port>/v1/fak/ctx/<id>
identity: guardsessions.Handle (g1a2b3c4)       identity: Descriptor.ID over BaseID
lifecycle: process alive / crashed              lifecycle: queued/warm/parked/hibernated/done
discovery: guard_sessions.jsonl                 discovery: a list route under its owner
already real                                    proposed

  fak://<machine>/<handle>              →  http://127.0.0.1:51234/
  fak://<machine>/<handle>/<ctx-id>     →  http://127.0.0.1:51234/v1/fak/ctx/8f2a…
```

One resolver, two tiers. The handle is already stable and prefix-matchable; the descriptor
id is already stable and content-seeded. Nothing new needs inventing at the identity layer
— only the routing beneath the port and one URI form above it.

### Why ports stop at the master agent

Three independent reasons, any one of which is sufficient:

1. **A hibernated context has no runtime to listen.** The micro-context fabric's whole
   economics depend on warm/parked/hibernated tiers
   ([`micro-context-fabrics.md`](research/micro-context-fabrics.md): *"10k logical contexts
   cannot imply 10k resident KV allocations"*). Addressability must survive hibernation.
   A path survives it; a socket does not. This is the decisive argument — the others are
   merely true.
2. **Ports are OS-scarce and fd-scarce.** 10k listeners per box is an ephemeral-range and
   descriptor-table problem before it is a UX problem, and it buys nothing: the contexts
   already share one process, one kernel, and one base.
3. **A port implies an independent trust boundary that does not exist.** Micro-contexts
   are deliberately *inside* one guard's capability floor and budget. Giving each its own
   listener would model them as peers of their own supervisor, which is exactly wrong and
   would invite someone to route to one directly, around the floor.

So the rule: **a port per thing that owns a process and a policy; a path per thing that
owns only a context.**

## How this interacts with micro-context windows

Beyond addressing, three interactions matter, and two of them are constraints on the UI
rather than features of it.

### 1. The console is the only human surface a micro-context can have

The fabrics plan names the blocker directly: *"many harnesses assume one process, terminal,
transcript, cwd, credential set, and human approval channel per agent,"* and the lightweight
descriptor deliberately **omits those surfaces**. A micro-context has no TTY, no transcript
pane, and no approval channel. If a human ever needs to inspect one, approve an effect for
one, or cancel one, the port console is the only place that can happen. That makes it
load-bearing infrastructure for the micro-context program, not decoration on the guard
program. This is the strongest reason to build it.

### 2. Observability must be rendered from counters, never from a model turn

If an agent's own UI is produced by asking the model to summarize itself, it burns the
context budget it is reporting on — and at 10k contexts it burns 10k of them. **Invariant:
every pixel in the console is derived from an expvar, a counter, or a durable row. No
console render ever costs a model turn.** The existing `/debug/vars` contract already holds
the sibling invariant (payload-free, no subprocess on the poll path); this extends it to the
renderer.

### 3. At 10k contexts, rows are the wrong primitive

The fabrics plan flags *"per-context events can overwhelm telemetry"* and prescribes
*"sampled spans plus aggregate queue/cache/quality metrics."* The UI consequence is a
rendering rule that inverts the usual dashboard instinct:

> **Aggregates for the mass, rows for the exceptions.**

A 10,000-row table is not an interface. A state histogram, a queue-age distribution, a
compatibility-class breakdown, and then *only* the contexts that are failing, starving,
budget-exhausted, or quarantined — that is an interface. This is the same discipline the
fleet plan states as "attention first, not activity first," applied one tier down.

### 4. The base fingerprint is the single highest-value visual in the system

Prefix-identity drift is the named silent killer: *"timestamps, reordered tools, per-agent
IDs, provider wrappers, and sampling metadata can destroy cache sharing… expose a
fingerprint plus miss reason."* A fleet whose cache sharing broke looks **completely
healthy** on every other axis — same throughput shape, same success rate, 20× the bill.
Nothing in a conventional dashboard catches it.

The console can, in one strip:

```text
BASE  sha256:9c4f…21e  installed 14:02  contexts on this base 8,412 / 9,001
      hit 91% ──────────────────────────▁▁▁▁  4%   ← break at 14:02
      miss reason: tool order changed (3 tools reordered vs base)
```

That is the visual that pays for the whole plan.

## Generation map

Ordered read model → address space → density → control, mirroring the fleet plan's
discipline. Each generation must land a witness before the next is treated as real.

| Gen | Operator question | Scope | Exit witness |
|---|---|---|---|
| **P0 — Reach it** | Which agents are live and what is each one's URL? | `fak session ls --urls` renders handle → gateway URL → liveness, with the three missing-URL causes distinguished (`PredatesGatewayPublish`, published-nothing, port dead). `fak session open <prefix>` resolves a prefix and prints/opens the origin. | Fixture over `guard_sessions.jsonl` covering all four row shapes; a live run resolving a prefix to a reachable origin. |
| **P1 — Read it** | What is *this* agent doing, refusing, and costing? | Promote `handleHome` from link hub to server-rendered console over the read model it already holds. Attention-ordered. No JS build, no asset server, same CSP. | Captured render (golden HTML) for healthy / deny-heavy / stalled-inflight / cold fixtures; loopback-vs-offbox auth assertion. |
| **P2 — Address the contexts** | Which of the 9,001 micro-contexts needs me? | `GET /v1/fak/ctx` (bounded, paginated, state-filtered) and `/v1/fak/ctx/{id}` under the owning port; console renders the density view + exception rows + the base-fingerprint strip. | Fixture at 10k descriptors asserting bounded response size, aggregate correctness, and that exception rows are complete (not sampled) while the mass is aggregated. |
| **P3 — Join the fleet** | What needs attention across every agent on every box? | The fleet console is a **fan-out join over ports**, not a new store: walk the index, read each live origin, fold. Feeds the TUI Fleet workspace (#6476) and the external endpoint (#4231). | A fold across ≥3 sessions including one dead port and one unpublished row, rendering distinct outcomes rather than a shared "unknown". |
| **P4 — Act on it** | Can I cancel/steer this without racing another operator? | Control via the **A2A verbs that already exist** — task cancel, progress, push-notify — plus capability, precondition/CAS, receipt, independent read-back. | Race, denial, and replay witnesses. No optimistic "clicked means done." |

P0–P2 are additive to shipped code. P3 is a fold, not a service. P4 is gated on the same
control semantics the fleet plan defers to #2753 — and should reuse A2A's vocabulary rather
than invent a second one.

## Wireframes

The information order is the contract; glyphs and layout are implementation choices. The
renderer degrades by dropping detail, never by hiding state.

### P1 — the per-agent console (`GET /` on the agent's own port)

```text
 fak · g1a2b3c4          claude          C:\work\fak          pid 24188      up 3h42m
 ────────────────────────────────────────────────────────────────────────────────────
 KERNEL          1,204 allow · 61 deny · 12 transform          top deny: EGRESS_TAINTED
                 deny-all turns 2 (auto-resumed)               tool-prune 4 · compact 1

 WIRE            TTFT 0.84s   prefill 2,190 tok/s   decode 61 tok/s
 !               oldest in-flight 94s  ← a request has been open since 15:31
                 upstream: 3 · 429 (last 15:22)

 CONTEXT         budget 61% used   savings 148k tok   cache attribution: base 92%
 FOOTPRINT       rung 3 · 14 tools admitted, 6 withheld (prerequisite unset)

 CONTEXTS        9,001 total     see the density pane →

 source: this process's own counters · read-only · loopback · observed just now
```

`!` marks the one row that needs attention; everything above and below is context. The
in-flight-age row is the hung-request detector that no completion histogram can show until
the request ends — it is already in the read model and is currently rendered nowhere.

### P2 — the micro-context density pane

```text
 CONTEXTS on base sha256:9c4f…21e            9,001 logical · 64 workers · 1 base install

 warm     ████▏                420      queued   ██▏          198   age p95 41s
 parked   ██████████▎        1,940      done     ████████████████████████▎  6,301
 hibernated ███████▊         1,140      failed   ▏             2      quarantined 0

 compat classes  4  ·  padding tax 11%  ·  head-of-line: 1 tool wait > 60s

 NEEDS ATTENTION                                                    (rows are complete)
 ! ctx 8f2a1c   FAILED       output contract "contains" unmet, 3 retries    base 9c4f…
 ! ctx 41b9de   STARVING     queued 14m, class C, deadline in 2m
 ! ctx 7c02aa   BUDGET       max_output_tokens exhausted at turn 6/8

 6,301 done rows are aggregated, not listed. Exceptions above are exhaustive.
```

The last line is not decoration. The fleet plan's rule — *"if a workflow bounds coverage,
log what was dropped"* — applies to renderers too: an operator must never mistake an
aggregate for a complete list, and must never mistake a bounded exception list for one.

### P3 — the fleet fold

Stays terminal-first, in the TUI Fleet workspace per #6476. Its only new job here is the
**link**: each row carries the resolvable handle, so `fak session open <handle>` is the
documented step from the fleet view into the per-agent console. The fleet view does not
re-render the per-agent read model — it points at it.

## Borrowed techniques

### Hermes (NousResearch)

Already a first-class integration here (`docs/integrations/hermes.md`; `fak guard -- hermes`
over its OpenAI Chat Completions wire), and already a borrowed source: rung 3 of the Hermes
**"Footprint Ladder"** is re-derived in `internal/policy/serviceadmit.go` as a kernel rule
over the emitted catalog rather than a per-tool `check_fn` convention.

Two things worth borrowing further, both UI-shaped:

- **The ladder as a visible posture.** The footprint rung is currently a policy behavior
  with no operator-facing rendering. The console's `FOOTPRINT` row above makes it legible:
  which rung this agent is at, how many tools are admitted, how many are withheld, and
  *why* (prerequisite unset). Because fak re-derives admission in the kernel, that row is a
  fact, not a self-report — which is precisely the claim upstream's per-tool convention
  cannot make.
- **`base_url`-repoint onboarding.** Hermes' `~/.hermes/config.yaml` `model.base_url` is the
  whole integration: point one URL, you're guarded. The console should mirror that
  affordance in the other direction — the origin page is where an operator *copies* the URL
  that a sibling harness needs. One field, one click, no docs round-trip.

### Chrome DevTools Protocol — the closest structural prior art

CDP binds **one** debug port and enumerates **many** targets under it (`/json/list`), each
with its own inspector URL. That is exactly the two-tier split proposed here, validated at
scale in a system everyone already trusts: one process-scoped listener, many
path-addressable logical inspectables, a machine-readable list route, and a human-readable
inspector per target. Borrow the shape, including the well-known list path.

### Jupyter kernel gateway, Ray dashboard

- **Jupyter:** many kernels under one gateway port, addressed by id, with lifecycle states
  in the URL space — the direct precedent for hibernate/restore being visible at the address
  layer rather than hidden behind a socket that comes and goes.
- **Ray:** actors and tasks under one dashboard port, and the hard-won lesson that at high
  cardinality the default view must be aggregate; per-entity rows are a drill-in, never the
  landing page. This is where the "rows for exceptions" rule comes from empirically.

### Erlang/OTP observer, systemd

Per-node introspection **attached to a live runtime**, with no external store between the
operator and the truth. That posture is already fak's (`/debug/vars` is the process's own
state, with provenance and freshness) and should not be traded away for convenience.

### What not to borrow

**The Grafana shape** — scrape into a time-series DB, render from the DB. It is the obvious
move and it is wrong here for two reasons: it inserts a store between the operator and the
process's own truth (so "the dashboard says X" and "the kernel decided X" can diverge, with
no way to tell which lied), and it cannot render the things that matter most — a live
in-flight age, a base-fingerprint break at the instant it happens, an exhaustive exception
list. Prometheus is already exported for those who want that pipeline; the console is
deliberately not it.

## Invariants

1. **No token in the browser.** Rendering is server-side on the process that owns the data;
   loopback exemption admits the browser; off-box stays bearer-gated. The console never
   ships a credential to a client.
2. **No model turn costs a pixel.** Every rendered value comes from a counter, expvar, or
   durable row.
3. **No poll-path subprocess, no payload.** The existing `/debug/vars` contract extends
   unchanged to the renderer: display metadata only, never a token or a request payload.
4. **The renderer never reclassifies.** It may order, shorten, and aggregate. It may not
   invent health, progress, or recovery states — those come from the typed backends
   (`fleetpane`, `microagent`, `AdjudicationSummary`), exactly as the fleet plan requires.
5. **Aggregation is disclosed.** Any bounded or folded view states what it dropped.
6. **Read-only until receipts exist.** P1–P3 mutate nothing; the row says so.

## Decisions and non-goals

1. **This is not a web application, and it is not a second fleet dashboard.** The fleet
   plan's decision — *extend the TUI, do not start a web app* — holds for the **cross-fleet**
   surface, and this plan does not touch it. What it touches is an origin that already
   exists on a port that already exists, serving the process's own state. Those are
   different objects, and conflating them is how a project ends up with two read models.
2. **One read model, two renderers.** The TUI and the console must render the *same*
   `/debug/vars` blocks. If a value exists in one and not the other, that is a bug in the
   renderer, not a feature of the surface.
3. **No frontend build, ever.** `html/template`, inline CSS, strict CSP, no asset server —
   the constraint `home.go` already lives under, kept deliberately.
4. **Ports for processes, paths for contexts.** Restated as a decision because it is the one
   most likely to be relitigated at the first "but it would be easier if…".
5. **A2A is the control plane.** When P4 arrives, cancel/progress/push-notify come from the
   protocol already served on the port, not from a new verb family.
6. **Aggregate-first at scale.** The landing view for 10k contexts is a distribution, not a
   table. Non-negotiable.

## What would falsify this

- If `/debug/vars` turns out to be too expensive to render at console refresh rates on a
  loaded box, the render must move to a snapshot with an explicit staleness stamp — and the
  console must say "observed 8s ago," not pretend to be live. (The fleet provider already
  pays a 30s TTL for exactly this reason.)
- If operators in practice never open a per-agent page and only ever want the fleet roll-up,
  P1 is decoration and should be cut to P0 + P3 (make the URL reachable, fold in the TUI).
  The cheapest test of this is P0 alone: ship `fak session open`, and see whether anyone uses it.
- If micro-context exception rows are routinely in the hundreds rather than the ones, the
  "rows for exceptions" rule breaks and the pane needs a second aggregation tier (cluster
  exceptions by cause) rather than a longer list.

## Relationship to existing work

| Issue / doc | Relationship |
|---|---|
| [#6476](https://github.com/anthony-chaudhary/fak/issues/6476) / [#6477](https://github.com/anthony-chaudhary/fak/issues/6477) — TUI Fleet workspace | Owns the cross-fleet surface. This plan feeds it the per-agent link and shares its read model. No overlap in surface. |
| [#5400](https://github.com/anthony-chaudhary/fak/issues/5400) — gateway URL + read bearer in the session index | The producer half of P0. **Already shipped**; P0 is its consumer. |
| [#4231](https://github.com/anthony-chaudhary/fak/issues/4231) — external fleet-status endpoint | P3's off-box form. Should be the fold, not a new store. |
| [#4227](https://github.com/anthony-chaudhary/fak/issues/4227)/[#4229](https://github.com/anthony-chaudhary/fak/issues/4229)/[#4230](https://github.com/anthony-chaudhary/fak/issues/4230) — mobile read/steer/approve | Downstream of P3/P4; the console's read model is what they render. |
| [#2753](https://github.com/anthony-chaudhary/fak/issues/2753) — out-of-band control vocabulary | Gates P4. Reconcile with A2A's existing verbs before adding a second vocabulary. |
| [#5785](https://github.com/anthony-chaudhary/fak/issues/5785) — micro-context fabric epic | P2 is its missing human surface. The descriptor list route is the smallest piece of it. |
| [#5789](https://github.com/anthony-chaudhary/fak/issues/5789) — harness-assumption inventory | Names the absent terminal/approval channel that P2 supplies. |
| [`docs/explainers/gateway.md`](explainers/gateway.md) | The concept page. This plan adds nothing to the gateway's guarantees — it renders them. |

## Planning freshness rule

Refresh when `debugVarsResponse`, `guardsessions.Row`, `microagent.Descriptor`, the route
table, or the auth-exemption list changes. A refresh must reconcile the verified-inventory
table against the tree; adding a mockup without re-verifying those five contracts is not a
refresh.
