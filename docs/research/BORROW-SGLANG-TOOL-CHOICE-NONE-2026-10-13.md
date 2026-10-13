# SGLang active-tool prompt decision

Fak adapts the active-tool decision from SGLang PR [#43006](https://github.com/sgl-project/sglang/pull/43006), merged October 13, 2026 at 03:41:51 UTC. Source: `python/sglang/srt/entrypoints/openai/serving_chat.py:1593` at revision `4d5692b6cd80aca2f42eca6cac3346652a206de9`, licensed under [Apache-2.0](https://github.com/sgl-project/sglang/blob/4d5692b6cd80aca2f42eca6cac3346652a206de9/LICENSE).

Upstream passes the normalized active-tool selection to custom DeepSeek encoders instead of always forwarding the request's offered tools. Fak adapts this mechanism in Go at `inKernelRenderPrompt`: the explicit JSON string `none` clears the local tools argument before choosing a prompt format. This covers ChatML, Ornith Qwen and DeepSeek V4.1 without introducing an encoder or changing caller-owned data. No upstream Python source or test text is copied.

Historical tool calls and results remain in the conversation. Active choices and response formats retain their existing behavior. The independent offered-tool output parsing contract is unchanged; this correction does not guarantee that a model cannot spontaneously generate tool syntax.

Reproduce the request-contract witness:

```text
go test ./internal/agent -run 'TestInKernelRenderPromptToolChoiceNone|TestRenderDeepSeekV41ChatMatchesGGUFTemplate|TestV41RawPromptOnlyForRawDeepSeekV41' -count=1 -timeout=5m
```

The tests compare tool-disabled prompts with tool-free requests through native prompt encoding and a synthetic native completion. This is software evidence for request preparation, not a physical model-quality or throughput benchmark.
