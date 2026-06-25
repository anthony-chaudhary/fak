// Package agent is the HOST-SIDE agentic loop and the wire servers that expose it.
//
// The name is a trap this doc exists to spring: "agent" here does NOT mean the
// untrusted program fak guards. fak (the kernel — internal/adjudicator,
// internal/ctxmmu, internal/vdso) is the reference monitor; the GUEST is the
// external AI program whose tool calls are gated. This package lives on the HOST
// side of that line — it is the machinery that drives a model turn-by-turn and
// serves the wire a guest client speaks. Naming it "agent" must not be read as
// "the agent fak guards"; it is "the host loop + wire servers that run one."
//
// # The trust line this package sits on
//
// Every tool call the loop below emits is mediated by the kernel before it runs —
// vDSO -> adjudicate -> grammar repair -> dispatch -> context-MMU admit — so the
// host loop is never a bypass. On the fused ("fak") arm the in-kernel model drives
// the planner and its calls go through the same gate; on the "now" baseline arm the
// same conversation runs with tool calls wired directly, so the two arms differ
// only by the kernel. The guest never escapes the gate by being hosted here.
//
// # What the package is (the host-side pieces)
//
//   - loop.go / loop_session.go: the agentic loop — a live model (the planner)
//     drives a multi-turn, tool-calling conversation.
//   - anthropic_server.go / gemini_server.go / anthropic_stream.go: the wire
//     servers. They expose Anthropic- and Gemini-compatible endpoints so an
//     external guest client (Claude Code, an SDK, anything base-URL-swappable)
//     drops in with no guest-side code change.
//   - inkernel_planner.go: fak's OWN model driving the loop on the fused arm.
//   - chat.go: the planner seam — provider transcript adapters plus the Planner
//     interface both the live client and the offline mock satisfy.
//   - tools.go / toolcall_fallback.go: the tool-call surface every call of which
//     the kernel gates.
//
// # What turns the static benchmark into a live one
//
// This loop is what makes the project's A/B latency benchmark a LIVE,
// turn-counting one: it measures model round-trips (turns) and tokens with the
// kernel ON vs OFF, against a real OpenAI-compatible endpoint (or a deterministic
// offline MockPlanner for CI).
//
// For the companion vocabulary split across the whole tree — the five senses of
// the bare word "session" and the four senses of "agent" — see the worklist at
// docs/notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md. The canonical drive-state
// "session" is internal/session; the model.Session decoder and the recall.Session
// core image are the non-canonical senses documented at their types.
package agent
