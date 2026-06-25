# Token streaming TTFT with whole-turn tool gating (2026-06-25)

Issue #47 changes the streaming contract from "finish the turn, adjudicate it, then
synthesize SSE" to "stream safe prose while the turn is still decoding, but hold every
tool-call byte until the kernel has seen the complete proposed call set." The split is
intentional: prose deltas are ordinary assistant text, while `tool_calls` /
`tool_use.input` bytes are executable intent and stay behind `k.Decide`.

## Wire behavior

- OpenAI `/v1/chat/completions`: a streaming-capable planner (`agent.StreamingPlanner`,
  currently the OpenAI-compatible `HTTPPlanner`) receives `stream:true`. Each content
  fragment becomes a downstream `delta.content` chunk immediately. Native
  `delta.tool_calls` fragments are accumulated inside the planner and emitted only after
  gateway adjudication, as surviving `tool_calls` deltas.
- Anthropic `/v1/messages`: the real Anthropic passthrough relays upstream text/thinking
  events live and holds upstream `tool_use` blocks. The generic planner path now maps
  the same `StreamingPlanner` content callback to Anthropic `text_delta` events, then
  emits surviving `tool_use` blocks after adjudication.
- Non-streaming planners still use the buffered fallback. That includes the offline mock
  and any engine that does not implement `StreamingPlanner`.

Both live paths use the lift guard before writing prose. If a model emits a known
text-form tool-call dialect inside content, the guard withholds that span, the normalizer
lifts it into a structured call, and the call is adjudicated before any surviving
structured tool block is written.

## Perceived TTFT

For prose-first turns, perceived TTFT now tracks the upstream model's first content
fragment plus gateway framing overhead. It no longer waits for the full turn and the
tool-call gate.

For tool-heavy turns, perceived TTFT depends on whether the model emits leading prose:

- Leading prose before tools: that prose streams immediately; tool bytes remain gated.
- Pure tool-call turn: there is no safe prose to show, so the first meaningful tool
  output is still delayed until the whole proposed call set is adjudicated.
- All-denied tool turn: no denied arguments are shown; the client receives the normal
  in-band fak refusal note after adjudication.

This preserves the trust floor while making TTFT/TPOT/ITL measurable for the parts of a
turn that are safe to expose incrementally.
