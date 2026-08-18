# Public system-prompt control patterns for the fak-native harness (2026-08-18)

## Decision frame

- **Centrality:** Core. The native harness owns the model request; instruction composition is part of that core loop, not a UI-only preference.
- **P1 managed context:** typed instruction fragments make residency, provenance, and replacement explicit.
- **P2 net-true efficiency:** preserve a stable base prefix and measure changed bytes/tokens/cache impact; do not claim savings from configurability alone.
- **P3 bounded adaptation:** recompute at a declared boundary (run/turn/role), validate output, and retain a deterministic last-good snapshot.
- **P4 integrated operations:** preview, diff, fingerprint, audit, replay, and deny unsafe mutations through the same public contract.
- **For / Problem / Today / Better because / Witness:** for native-harness builders; prompt control is currently an advertised `instructions` extension plane without a public typed runtime contract; today they must hide composition in private code or replace opaque strings; a typed composer makes dynamic changes inspectable and cache-aware; witness a harness run where one turn/role change updates only the intended fragment and emits a deterministic composition record.

## Pinned public sources (observed 2026-08-18)

| Source | Revision / release | Mechanism observed | Transferable lesson | License / route |
|---|---|---|---|---|
| OpenAI Agents Python | `ebb746dc00b0dd6a90c30bc5ccb7e9c445e55493`, v0.21.1 (2026-08-16) | `Agent.instructions` accepts a string or callable over run context + agent, evaluated when the agent runs (`src/agents/agent.py`; `docs/agents.md`). | A first-class callback is the minimum viable dynamic-instruction API; context and agent identity are useful inputs. | MIT; adapt in Go. |
| OpenAI Codex | `3df5087f754af3794f4b414c78921b5f07af1ace`, v0.147.0 (2026-08-07) | `TurnStartParams` carries typed, per-turn `additional_context`, plus sticky turn settings such as personality and collaboration mode (`codex-rs/app-server-protocol/src/protocol/v2/turn.rs`). Entries distinguish untrusted from application context. | Dynamic prompt inputs need trust/type metadata and explicit turn-vs-sticky semantics, not one undifferentiated text field. | Apache-2.0; adapt types/semantics. |
| LangChain | `b28d8c4630d4a58a8a3fdc6f2266d35860e3776a`, langchain-core 1.5.6 (2026-08-17) | `dynamic_prompt` middleware transforms a model request and returns a system message; tests cover sync/async middleware and request context. | Composition can be middleware, but fak should constrain it to deterministic fragments and expose the realized result rather than permit invisible arbitrary mutation. | MIT; adapt concept. |
| Anthropic Claude Code | `354757e5b2d9aa1ebb62e5d05ecd384f0e11c0f7`, v2.1.234 (2026-08-17) | Public changelog documents replace (`--system-prompt-file`), append (`--append-system-prompt[-file]`), an `agent` setting combining prompt + tool restrictions, removal of built-in git instructions, custom-prompt context reporting, and moving date out of the system prompt to improve cache hits. | Operators need replace/append/profile controls and truthful introspection; volatile fields must remain outside the stable prefix. | No OSI license found in checkout; inspire only. |

Pinned source checkouts were allocated under `_scratch/system-prompt-control-research/` and are intentionally not committed.

## Current fak witness

**Verdict: PARTIAL.**

Present:

- `pkg/harnesskit` declares `PlaneInstructions` in the public extension-plane list.
- `internal/syspromptmmu` already implements cache-aware stable base + overlays, editing gates, styles/work profiles, steering, fingerprints, and audit witnesses.
- The owned loop previously gained a live system-prompt overlay caller in closed issue #1322.
- Open issue #6673 covers role-selective child fragments; #4057 covers idempotent live skill injection; #7110 covers dynamic semantic skill loading.

Absent on the requested native-harness axis:

- no exported harnesskit instruction composer/provider contract;
- no typed fragment fields for source, trust, precedence, lifetime, role/audience, or cache residency;
- no atomic preview/apply/rollback protocol carrying realized digest and prefix-change impact;
- no harness-facing render/diff explaining exactly what the model receives and why.

This makes the public `instructions` plane nominal rather than usable. The smallest useful spine is not another prompt profile; it is the typed bridge from a harness request into the existing MMU realization and audit path.

## Candidate portfolio

| Candidate | Fak status | Disposition | Why |
|---|---|---|---|
| Runtime instruction callback with run/agent inputs | ABSENT publicly | **DEFAULT** | Minimum dynamic-control seam; deterministic inputs and snapshot output. |
| Typed fragments: trust, provenance, priority, lifetime, audience | PARTIAL internally | **DEFAULT** | Prevents conflating app policy, user context, untrusted retrieval, and transient steering. |
| Stable-base + volatile-tail composition with prefix fingerprint | PRESENT internally, absent publicly | **DEFAULT** | Preserves fak's cache advantage while allowing dynamic control. |
| Preview/diff/apply/rollback with realized digest | ABSENT | **DEFAULT** | A controllable prompt must also be inspectable and reversible. |
| Role-scoped fragments | Open #6673 | **OPTIONAL-MODULE** initially | Useful for fleets; keep the first public spine single-run/turn and compose this later. |
| Free-form arbitrary middleware mutation | ABSENT | **EXCLUDE** | Too difficult to audit/replay; accept typed fragment output instead. |
| Full prompt replacement | PARTIAL via internal editing | **RECIPE** | Compatibility escape hatch, but not the safest default because it can erase security/cache invariants. |
| Model-generated self-edit of system policy | ABSENT | **WATCH** | High leverage but requires authority separation, evaluation, and rollback before exposure. |

## Recommended spine and follow-ons

1. Export `harnesskit.InstructionProvider` and typed request/result/fragment structures, including explicit lifetime and trust.
2. Add one adapter that realizes those fragments through `internal/syspromptmmu`, retaining stable-prefix audit data.
3. Extend `harnesswebdemo -selfcheck` (or a dedicated LCD demo) to change one turn-scoped fragment and prove the base prefix digest stays stable.
4. Follow with preview/diff/apply/rollback, role scoping (#6673), idempotent skill fragments (#4057/#7110), and policy-authority separation.

## Honest limits

This study compares public APIs and source-visible behavior, not undisclosed provider system prompts. It does not establish which prompt produces better model quality. The proposed witness proves deterministic control, provenance, and cache-stable composition; workload quality and token economics need separate measured experiments.
