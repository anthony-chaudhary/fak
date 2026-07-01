// Package vllmcompile is the tuned-baseline gate for served-engine benchmarks:
// it records an engine's torch.compile / CUDA-graph / warmup state as a
// `vllm_compile` artifact block and refuses to let a cold or accidentally
// misconfigured engine masquerade as a tuned baseline (#1731).
//
// vLLM enables torch.compile by default, caches the compiled artifacts, and
// finishes compilation before serving so a request never pays compile latency
// (docs/design/torch_compile.md). A benchmark that measures a window where the
// compile cache was cold or disabled, warmup had not finished, or a request
// triggered compilation is a cold-start reading — not the tuned baseline the
// net-true-value standard (docs/standards/net-true-value.md) requires as the
// honest comparison point. This leaf makes that distinction mechanical:
// Classify folds the recorded state into tuned / cold-start / diagnostic, and
// Gate fails closed on anything but tuned.
//
// It is pure and stdlib-only, off the request path: the compile/graph state is
// captured by the engine adapter or bench harness and handed here as data. The
// same block applies to raw vLLM, fak-fronted vLLM, and other compiled engines
// (SGLang, llama.cpp graph builds) when comparable, so no compared row is
// silently cold.
package vllmcompile
