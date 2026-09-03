// Package servingsim provides a high-fidelity, discrete-event simulation engine
// for LLM serving architectures. It models continuous batching, chunked prefill,
// paged KV cache allocation, speculative decoding, and hardware latency profiles,
// while emitting Chrome / Perfetto execution traces.
package servingsim
