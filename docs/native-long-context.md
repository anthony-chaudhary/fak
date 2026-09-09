# Native context windows

`fak serve` uses one resolved native context window for model loading, request
admission, and model discovery. `fak up` exposes the same native context control
for turnkey launches. The window counts the complete rendered prompt, including
the chat template, plus the request's reserved output tokens (`max_tokens`).

## Serve with automatic sizing

```bash
fak serve --gguf MODEL.gguf
```

The default `--native-context-tokens 0` reads the model's declared position
limit and selects a context window that fits the available memory headroom. The
resolved value is used consistently for loader sizing, native admission, and the
model catalog.

This resolved capacity is a configuration and memory-sizing result. It does not
qualify throughput, latency, or maximum usable context on physical hardware; those
claims still require a separate hardware receipt.

For a SafeTensors directory, auto mode reads `config.json` before tensor load and
uses its declared limit. That path is identified as `model-metadata` because it
does not yet have the GGUF load arm's header-only weight estimator and therefore
is not memory-qualified by the auto-sizer.

Inspect the resolved plan without loading the model weights:

```bash
fak serve --gguf MODEL.gguf --plan-json
```

## Set an explicit native window

```bash
fak serve --gguf MODEL.gguf --native-context-tokens 131072
```

A positive value requests an exact upper bound. It must not exceed the model's
declared position limit or, for a GGUF model, the selected arm's memory-backed
load plan. The model-declaration check happens before the weight payload is
loaded. SafeTensors directories use only `config.json` for this early check.

## Turnkey `fak up`

```bash
fak up --context 131072
```

`--context` requests the turnkey profile's native context ceiling, which is
resolved against the selected model's metadata. The actual planner window is
passed to native request admission and exposed by the turnkey model catalog.

To name a local model source and its native ceiling directly:

```bash
fak up --gguf MODEL.gguf --native-context-tokens 131072
```

## Request boundary

A request that reaches the native planner is within its model window when:

```text
rendered_prompt_tokens + max_tokens <= resolved_native_context_tokens
```

The rendered prompt includes system and user messages, tool definitions, and
the model's chat-template tokens. An over-limit request that reaches the native
planner returns HTTP 400 with the error code `context_length_exceeded` before
model execution or device allocation begins. An explicitly smaller
`--native-admission-token-budget` is an independent scheduler cap and can reject
the request first with HTTP 429. When that scheduler flag is omitted, local
serve starts from the resolved native window; measured warmup capacity may
subsequently refine the scheduler resource cap without changing the model
window.

Query the effective known capacity through the standard model endpoint:

```bash
curl http://127.0.0.1:8080/v1/models
```

Native rows report the planner's effective `ContextWindow()` as
`data[].context_length`, including the minimum of a configured ceiling and the
model declaration. The Codex-compatible model shape also reports
`context_window` and `max_context_window`. Rows whose capacity is unknown omit
these fields.

## Related limits

The context window is separate from these resource controls:

- `--native-admission-token-budget` limits the scheduler's concurrently admitted
  native token work. Setting it explicitly continues to override automatic
  scheduler sizing; it does not change the model's context window.
- `--context-budget-tokens` limits cumulative managed-session context over its
  lifetime. Its `--ctx` alias has the same session-lifetime meaning. Neither
  spelling changes the per-request native model window.
- `--max-total-tokens` limits the request envelope (rendered prompt plus reserved
  output); `--max-batch-prefill-tokens` is its batch-prefill companion alias.
  The effective envelope must fit the scheduler admission cap. An invalid
  explicit scheduler envelope is rejected immediately; with automatic local
  sizing it is checked against the final resolved cap after header inspection
  and before weight payload loading.

Use `--native-context-tokens` for the model's per-request window,
`--native-admission-token-budget` for concurrent scheduler pressure, and
`--context-budget-tokens` for the cumulative managed-session budget.
