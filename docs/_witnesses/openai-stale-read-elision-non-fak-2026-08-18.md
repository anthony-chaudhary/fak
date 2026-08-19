# Default OpenAI stale-read elision in a non-FAK repository — 2026-08-18

**Verdict:** the default stale-read mechanism now reaches decoded OpenAI-compatible transcripts, not only Anthropic request bytes.

- `TestDecodedStaleReadElisionReachesOpenAIPlannerInput` constructs an OpenAI function-call transcript containing `Read(src/app.go)`, its large tool result, and a later `Edit(src/app.go)`. Before `completeServed` invokes the planner, FAK replaces the old snapshot with a bounded superseded-read marker.
- The caller's message slice remains unchanged, and the exact original tool-result bytes are content-addressed in the existing `fak_context_restore` stash.
- `TestDecodedStaleReadElisionProtectsRecentAndUneditedReads` proves the recent working-set tail and reads without a later same-path edit stay byte-identical.
- Guard and serve already default `--elide-stale-reads` on through `gateway.DefaultElideStaleReads`; `--elide-stale-reads=false` remains the independent opt-out.

Structured artifact: [`openai-stale-read-elision-non-fak-2026-08-18.json`](openai-stale-read-elision-non-fak-2026-08-18.json).

This retires stale-read elision for decoded OpenAI-compatible traffic under #8089. Cold-tool deferral and cross-backend vCache signaling remain.
