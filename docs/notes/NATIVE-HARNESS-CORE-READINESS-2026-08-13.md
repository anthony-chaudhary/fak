---
title: "Native harness — core readiness (2026-08-13)"
description: "Evidence-mapped update on the fak-owned agent loop's core items: what shipped since the June survey, the one structural gap that blocks every coding use, how the fleet epic reframes it, and what is actually worth importing from Claude Code."
---

# Native harness — core readiness

Date: 2026-08-13. Supersedes the June evidence map
([native-harness-progress-tracking-1315](native-harness-progress-tracking-1315.md))
and the Crush benchmark ([NATIVE-HARNESS-NEXT-STEPS-VS-CRUSH](NATIVE-HARNESS-NEXT-STEPS-VS-CRUSH-2026-06-30.md)),
both of which are now ~6 weeks stale and wrong in both directions — they understate
what shipped and they miss the gap that matters.

## The one-liner

**The loop is owned. The work surface is not.** `fak serve --native` drives fak's own
`RunArm`, streams, adjudicates every call, and reports typed progress — but the loop it
drives is wired to the *travel-demo* tool fixture, and the served request's `tools[]` and
message history are dropped on the floor. Every mechanism the program set out to build is
built; none of them can currently touch a real coding task.

June's honest one-liner was "the engine is owned, the loop is not." August's is:
**the loop is owned, the work is not.**

## What shipped since June (the June note is stale here)

June's Tier-1 "the two places Crush is ahead" are both closed:

| June gap | State now | Witness |
|---|---|---|
| Streaming native serve (no issue yet — FILE IT) | **SHIPPED** | `internal/agent/loop.go:360 RunArmStream`; `internal/gateway/messages.go:202 serveNativeMessagesStream`; `internal/gateway/native_serve.go:89` |
| Durable session state / WAL turn checkpoint (#1363) | **SHIPPED** (#1363 CLOSED) | #1365 (durable-by-default + doctor posture) still OPEN |

And a second wave landed on top of it, all on the live owned loop:

| Capability | Witness |
|---|---|
| Structured loop-progress SSE (typed lifecycle → named SSE events) | `native_serve.go:150-170` `onProgress`; `agent.WithProgressObserver` (#5148) |
| Mid-flight verb mailbox — a control op reaches an in-flight owned run | `87beabadab` (#2403) |
| Terminal-tool loop wake (event-driven, not poll-shaped) | `3f0a8cb573` (#2400) |
| Classified non-querying steer bus with scheduler classes | `4749de8ab4` (#2402) |
| Per-call routing + model-account roster + tenancy gate on in-loop calls | `native_serve.go:200-210` (#5644 / #5706 / #5707 / #5708) |
| Tool-call runaway budget wired into the owned loop | `66e132fbfb` (#5235) |
| `RunGovernedArm` — one governed arm with its full call trace | `internal/agent/governedrun.go:22` |
| Stop-gate: a rejected final answer never leaks as an SSE delta | `native_serve.go:236-243` |

So the *loop mechanics* are in good shape. The June framing ("borrow Crush's durability
and streaming") is done.

## The structural gap: the owned loop has no work surface

This is the finding. Three facts, each grep-confirmed:

1. **The tool catalog is the airline-support demo fixture.**
   `internal/agent/loop.go:401` — `tools := ToolCatalog()`, and
   `internal/agent/tools.go:25-33` defines that catalog as
   `get_user_details` / `search_direct_flight` / `calculate` / `convert_currency` /
   `fetch_policy` / `book_flight` / `delete_account`. There is no
   Read / Write / Edit / Bash / Grep / Glob anywhere in the owned loop.

2. **The kernel is bound to the demo engine.**
   `internal/agent/loop.go:386` — `k = kernel.New("localtools")`;
   `internal/agent/tools.go:266` registers that engine. `internal/agent/readengine.go`
   says so in its own header comment: *"The demo `localtools` engine implements only the
   travel-domain toolset (no real file I/O)."* The one real filesystem engine
   (`FakReadEngineID = "fakread"`, `readengine.go:35`) is **read-only** and exists to back
   the `fak_read` **MCP** tool — i.e. it serves *Claude Code on the proxy path*, not the
   native loop.

3. **The served request's tools and history never reach the loop.**
   `internal/gateway/native_serve.go:216` and `:228` —
   `task := lastUserText(req.Messages)`. That is the entire seed. `req.Tools` is not read
   anywhere in `native_serve.go`; prior conversation turns are discarded.

The consequence is blunt and worth stating plainly: **`fak serve --native` cannot serve a
real client conversation, and `fak chat` cannot edit a file.** Not "slowly" — there is no
code path. `--native` is correctly still off by default (`cmd/fak/serve.go:244`), though
that flag's help text is now stale (it still claims "a streaming request falls through to
the proxy path", which stopped being true when `serveNativeMessagesStream` landed).

This is tracked as **#3265** (BYO-tools / MCP registration seam, OPEN) under epic #3256,
and named as out-of-scope-for-now in **#3258**'s body ("BYO-tools (#B7)"). It is not
tracked as a *native-serve* gap, and it should be — #3265 is scoped to the agent-runtime
endpoint, not to `native_serve.go`'s wire path.

### #1380 is still open

The definition-of-done witness run — an AgentDojo or coding task driven entirely by the
native path, with ≥1 vDSO-materialized tool result and ≥1 speculated turn that
suspends/promotes — has been OPEN since 2026-06-29. Fact 1 above explains why it has not
moved: the "or coding task" half is unreachable, and the AgentDojo half is the only shape
the current catalog can express. **Until the tool surface generalizes, #1380 can only ever
be witnessed on the toy domain**, which is a much weaker artifact than the epic intends.

## Core-item scorecard

Read "core" as: what must exist before a person can do real work in the native harness.

| Core item | State | Note |
|---|---|---|
| Owned turn loop drives dispatch | **SHIPPED** | `native_serve.go:47/89` → `RunArm`/`RunArmStream` |
| Kernel is the sole tool path | **SHIPPED** | every in-loop call crosses `k.Syscall` + adjudication |
| Streaming | **SHIPPED** | closed the June gap |
| Structured progress / interruptibility | **SHIPPED** | #2400/#2402/#2403 |
| Session control (pause/resume/budget/pace/terminate) | **SHIPPED** | `loop_session.go`, terminate seam at `loop.go:409-434` |
| Durable turn checkpoint | **SHIPPED** | #1363; on-by-default is #1365 (OPEN) |
| Suspend/resume + write barrier + consistency enum | **SHIPPED, default-off** | #1318/#1319/#1317 |
| **Real tool surface (file/shell/search)** | **NOT YET** | the gap above — nothing exists |
| **BYO / inbound `tools[]` on the served turn** | **NOT YET** | #3265 OPEN, not scoped to native serve |
| **Conversation history seeding** | **NOT YET** | `lastUserText` only |
| DoD witness run | **NOT YET** | #1380 OPEN since June |
| Permission *regimes* (modes, typed grant, canonical match) | **NOT YET** | #2389 + 4 children, all OPEN |
| Hooks as adjudication rungs | **NOT YET** | #2396 OPEN |
| Subagent registry / task graph | **NOT YET** | #2397 OPEN |
| Progressive disclosure (skills/tool schemas as paging) | **NOT YET** | #2398 OPEN |
| Fleet resource governance | **NOT YET** | #6552, filed today, six rungs, nothing shipped |

The twelve child epics of **#2387** ("distill the spirit of Claude Code into the fak-owned
loop") are **all twelve OPEN**, though three of #2388's children (#2400/#2402/#2403) have
shipped code — the epics are lagging their commits.

## How the fleet approach reframes this

**#6552** (filed 2026-08-13) is the newest and most important context shift. Its argument:
owning `RunArm` means fak now owns the loop's *local resource footprint*, and the measured
fleet on the maintainer box is 87 processes / 12.90 GiB working set, of which 20
near-identical MCP server processes burn 4 CPU-seconds total — a 0.2% duty cycle paid 20
times. Seat cost ≈ 600 MiB, mostly duplicated tool catalogs and idle runtimes.

Two things follow for the core-items question:

1. **The fleet epic assumes a per-seat native harness that does not yet exist.** Rung
   #6562 ("co-host N native arms in one process over a shared tool catalog") is pooling
   *the very tool catalog that is currently the travel demo*. The pooling design is sound;
   its subject is a fixture. Sequencing matters: generalizing the tool surface first makes
   #6560/#6562 pool something real; doing them in the other order optimizes a demo.

2. **It supplies the real reason to want the fixed tool surface.** The seat-cost argument
   ("the per-seat substrate becomes shareable only once fak owns `RunArm`") is the
   strongest available case for finishing the native harness, and it is *economic*, not
   architectural — which makes it much easier to defend than "ownership is nicer."

Also relevant: **#5953** (guards should join the fleet bus and apply steer by default) and
the shipped fleet control bus (`--fleet-bus`, `cmd/fak/serve.go:248`) mean cross-process
fan-out already exists — but its `steer` directive **requires `--native`** and refuses with
`STEER_NO_OWNED_LOOP` otherwise. So the fleet control plane is already gated on the native
harness being usable.

## Permissions: where fak is ahead, and where it is not

We should not import Claude Code's permission *model*. fak's is structurally stronger and
that is well-evidenced in the tree:

- `internal/adjudicator/decide.go` is a real reference monitor — restrictiveness lattice,
  rank-100 authoritative rung, deny-as-value with bounded-disclosure witnesses, fail-to-
  abstain on unprovable cases, `ArgPredicate` gating argument *values* not just tool names,
  and TRANSFORM verdicts that redact before dispatch.
- The closed refusal vocabulary (`internal/abi/reasons.go`) makes refusals queryable.
- `internal/policy/policy.go:64` `fak-policy/v1` is a fail-loud manifest with advisory
  posture, secret posture, and RE2 secret patterns.
- The command-analysis rungs are genuinely sophisticated and have no Claude Code
  counterpart: `cli_grammar.go`, `shell_dialect.go`, `rce_pipe.go`, `rm_rf.go`,
  `heredoc_inert.go`, `argcanon.go`, `secretposture.go`, `runas_elevation.go`,
  `terraform_destroy.go`, `outoftree.go`.

What fak lacks is the **ergonomic layer**, which is exactly what #2389 scopes: named
regimes (one declarative pivot instead of N rule edits), "always allow this" as a typed,
scoped, expiring, journaled syscall, tiered settings compiled to a content-hashed policy
epoch, and — the important one — **canonicalization before matching** (#2407).

## What is actually worth importing from Claude Code

Bluntly: **not the regexes.** The instinct to port Claude Code's pattern matching is the
wrong read of where the value is. Its permission matching is string-pattern based and its
changelog history is a long tail of bypass repairs (path spellings, env aliases, quoting) —
#2389's own thesis says this. fak's adjudicator already *beats* that layer. Porting the
patterns would import the bug class along with the feature.

Four things are genuinely worth taking, in value order:

1. **The tool contract and its invariants — the highest-value import, and it is the gap.**
   Not the code, the *contract*: absolute-path requirement; `cat -n` line-numbered read
   output with an offset/limit window and a truncation ceiling; read-before-edit as a
   hard precondition; edit-by-exact-string with a uniqueness requirement and an explicit
   `replace_all`; write-refuses-unread-overwrite. These are the "symbolic filters" that
   make a coding agent reliable, and every one of them is a *state invariant across calls*
   — which is precisely the class of thing that is convention-and-prompt in Claude Code and
   can be a **kernel contract** in fak. This is #2387's stated doctrine ("convention →
   contract") applied to the one surface that does not exist yet. The read-before-edit
   precondition in particular is a natural `ArgPredicate` + vDSO epoch check: fak already
   tracks per-path epochs (`internal/vdso/pathscope.go: files:<path>`), so "this edit is
   based on a stale read" is *provable* here and merely hoped-for there.

2. **Canonicalization before rule evaluation** (#2407). This is the real lesson of the
   regex history — the conclusion to draw is "a string-pattern permission language needs a
   canonicalizer underneath, failing closed to `MALFORMED`", not "copy these patterns."
   Build it once, structurally.

3. **A path/glob rule grammar for file-shaped tools.** fak's manifest keys on tool *name*
   (`Allow`, `AllowPrefix`, `Deny`) plus `ArgPredicate`s and `SelfModifyGlobs`. There is no
   `Tool(path-specifier)` rule language — because there are no file tools to write rules
   for. This lands with item 1, not before it.

4. **ripgrep-backed search as a kernel-mediated tool.** fak's `internal/codesearch` is a
   single small file; there is no rg integration, no type filters, no multiline mode. A
   Grep/Glob pair behind the kernel is where vDSO caching should pay off most obviously
   (search results are the most re-requested, most cacheable tool output in a coding
   session), so this is a capability *and* a demo of the thesis.

Hook matchers map to #2396 (hookbus) and should be built as adjudication rungs, not as a
matcher table — same reasoning as above.

## Ordered next steps

Worst-regret-first. The first item unblocks the other four.

1. **Generalize the owned loop's tool surface.** Two separable halves: (a) accept the
   served request's `tools[]` and seed the loop with real conversation history rather than
   `lastUserText` (`native_serve.go:216`, `:228`); (b) ship a real, kernel-mediated coding
   toolset (Read/Write/Edit/Bash/Grep/Glob) behind engines the way `readengine.go` already
   does for reads — with the item-1 invariants above enforced as kernel contracts, not
   prompt text. **File this against `native_serve.go`**; #3265 is scoped to the
   agent-runtime endpoint and does not cover the wire path.
2. **Then witness #1380** — with the coding-task shape the DoD actually names, not the
   AgentDojo fallback.
3. **Fix the stale `--native` help text** (`cmd/fak/serve.go:244` still says streaming
   falls through to the proxy). Cheap, and it currently misinforms every operator who
   reads `--help`.
4. **#2389 regimes + #2407 canonicalization** — the permission ergonomics gap, once there
   are file-shaped tools to write rules against.
5. **#6552 rungs #6560/#6562** — pool MCP hosts and co-host arms, after (1) so the pooled
   substrate is the real tool catalog.

### 2026-08-13 execution update

A guarded three-worker wave converted the first ordered step into explicit, witnessed work:

- **#6657 CLOSED** — the Messages wire now preserves ordered conversation history and the request-scoped `tools[]` catalog for both buffered and streaming native handlers. Witnessed by commits `5847cd1901` and `561f6ff7d1`; malformed declarations and unsupported roles fail closed at the wire conversion seam.
- **#6659 CLOSED** — `serve --help` no longer claims streaming falls through to proxy; commit `18942abfe7` added the captured help regression test (later trunk reconciliation preserved that contract).
- **#6658 OPEN** — the real kernel-mediated Read/Write/Edit/Bash/Grep/Glob engines remain the binding prerequisite. The first guarded worker exhausted its managed-context restart budget without a witnessed ship; no coding-engine commit was found during reconciliation.

The dependency order below therefore remains unchanged after the wire half: **#6658 → coding-shaped #1380 witness → #2405/#2408/#2409 permission ergonomics → #6560/#6562 pooled/co-hosted real catalog**. #2407 canonicalization is already CLOSED and should not be reimplemented.

## Honesty note

Everything marked SHIPPED above has a `file:line` or commit witness in this document and
was re-checked against the tree on 2026-08-13. Nothing here is closed on a self-report.
Two claims are deliberately *not* made: no assertion that the native path has served a
real coding turn (it has not), and no assertion that #2387's epics are further along than
their OPEN state suggests — three of #2388's children shipped code, and that is stated as
a lag between commits and epic bookkeeping, not as progress on the other eleven.
