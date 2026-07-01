# FrontierSWE fak-routing shim

The one-line seam that makes fak's value stack bite on a 20-hour
[FrontierSWE](https://github.com/Proximal-Labs/frontier-swe) trial: it points the
harness's model traffic at a `fak serve` gateway and changes nothing else.

This is epic [#1706](../../) child **C6** — the live-wiring keystone. Without it, fak
never sits on the trial's request path, so KV reuse / in-kernel adjudication / vDSO
call-elimination can't touch the wall-clock.

## Why a shim

FrontierSWE drives each trial with a pluggable agent named in `job.yaml` by `import_path`:

```yaml
agents:
  - name: claude-code-api-key-no-search
    import_path: harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch
    model_name: anthropic/claude-opus-4-6
    override_timeout_sec: 72000
    kwargs: { effort_level: max }
```

Every `harbor_ext` agent class wraps a CLI coding harness (claude-code, codex, gemini-cli,
qwen-code, …) pointed at an OpenAI-compatible model endpoint through a `base_url`. The
natural seam, therefore, is to keep the agent exactly as-is and **swap only its base URL**
to a co-resident `fak serve` gateway. Turn *k* then reuses the resident KV of turns
`1..k-1` instead of re-prefilling them, and the per-tool-call gate runs in-kernel instead
of spawning a hook — the two costs that dominate a thousand-turn, 20-hour trajectory.

`FakRoutedAgent` does that and only that. `model_name`, `override_timeout_sec`, and every
other kwarg pass through to the real agent untouched. That single-dial property is what
keeps the raw-vs-fak comparison honest (the score-parity gate, C11): same model, same
prompts, same budget — one changed thing.

## What it changes (and what it doesn't)

| | raw agent | `FakRoutedAgent`-wrapped |
| --- | --- | --- |
| model `base_url` | vendor endpoint | the fak gateway |
| `model_name` | unchanged | **unchanged** |
| `override_timeout_sec` | unchanged | **unchanged** |
| `kwargs` (e.g. `effort_level`) | unchanged | **unchanged** |
| base-URL env (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, …) | vendor | the fak gateway |

Under `allow_internet = false` (the FrontierSWE default) the shim **refuses** a gateway URL
that isn't an in-sandbox host — loopback, a private address, or `localhost`. The gateway
must be co-resident with the trial, never an external call; a misconfiguration raises
instead of silently leaking the trial's traffic off the sandbox.

## Register it in a FrontierSWE checkout

The shim is resolved by FrontierSWE as `harbor_ext.fak_routed:FakRoutedAgent`, so make this
module importable under that name (a namespace-package drop-in — it never imports
`harbor_ext` itself, so it coexists with the real package):

```bash
# From your FrontierSWE checkout, put this dir on the import path as harbor_ext.fak_routed:
cp path/to/fak/examples/frontierswe-harbor-shim/fak_routed.py $HARBOR_EXT/harbor_ext/fak_routed.py
```

Then point the agent at the shim and name the real agent to wrap:

```yaml
agents:
  - name: claude-code-fak-routed
    import_path: harbor_ext.fak_routed:FakRoutedAgent
    model_name: anthropic/claude-opus-4-6
    override_timeout_sec: 72000
    kwargs:
      wrapped: harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch   # the real agent, unchanged
      fak_base_url: http://127.0.0.1:8080/v1                     # the co-resident gateway
      allow_internet: false                                     # mirror the task env
      effort_level: max                                         # passes through to `wrapped`
```

`fak_base_url` defaults to `$FAK_GATEWAY_URL` or `http://127.0.0.1:8080/v1`. Standing up the
gateway inside the task's Docker/Modal sandbox is child **C7**; scraping its `/metrics`
mid-trial to witness the prefix-reuse this routing unlocks is **C8**.

## Test

```bash
python3 fak_routed_test.py     # stdlib only — no fak, no harbor_ext, no GPU, no network egress
```

The test stands up a mock OpenAI-compatible endpoint and a stub harbor_ext agent, then
asserts the shim (a) reroutes traffic to the gateway, (b) leaves every non-base-URL field
byte-identical to an un-routed instance, and (c) refuses an external gateway under
`allow_internet = false`. `make ci` runs the same test through a Go driver
(`cmd/fak/frontierswe_shim_test.go`) wherever python3 is available.

See [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) for a captured successful run.

## Honesty boundary

This shim is the **routing seam only**. It does not, by itself, prove any time-to-solution
win — that is the live measurement in C8 (per-turn cache-witness) and C14
(wall-clock-to-`correctness==1.0`), gated-until-witnessed like every other fak benchmark
claim. What is witnessed here is narrow and exact: the agent's model traffic goes through
the gateway, and nothing but the base URL changed.

What this demo does not claim: it does not run the FrontierSWE harness, grade a task,
measure wall-clock time, or prove a cache-reuse speedup. Those claims require the official
FrontierSWE scorer, score-parity gate, and live cache-witness artifacts named above.
