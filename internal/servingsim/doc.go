// Package servingsim provides a high-fidelity, discrete-event simulation engine
// for LLM serving architectures. It models continuous batching, chunked prefill,
// paged KV cache allocation, speculative decoding, and hardware latency profiles,
// while emitting Chrome / Perfetto execution traces.
//
// Invariant: Simulation timestamps advance monotonically non-decreasingly throughout execution.
// Guard: Workload requests with non-positive tokens, negative arrival times, or invalid
// simulator configurations are rejected fail-closed before execution.
package servingsim
