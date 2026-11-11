// Inter-token latency (ITL) for streaming chat completions.
//
// Source: adapted from ai-dynamo/aiperf (Apache-2.0),
// src/aiperf/metrics/types/inter_token_latency_metric.py @
// ab610add4e8876fca223197596c728e17365f381; ADAPT verdict, re-expressed in idiomatic
// Go, no code copied.
//
// Upstream formula: Inter Token Latency = (Request Latency - Time to First Token) /
// (Output Sequence Length - First Content Chunk Tokens). The decode window (request
// latency - TTFT) covers tokens arriving after the first content chunk, so the
// divisor subtracts that chunk's token count rather than assuming one token. Servers
// that bundle several tokens into the first streamed chunk (e.g. TRT-LLM
// stream-interval) otherwise over-count decode tokens and inflate per-user
// throughput.
//
// Deviations from upstream, forced by loadgen's observation shape:
//   - Upstream's explicit-empty-first-chunk rule (first streamed chunk carries no
//     content, so the divisor becomes osl) has no analog here: loadgen sets TTFT
//     only on non-empty content chunks, so the first content chunk always carries
//     content by construction of the observation.
//   - Upstream warns once on inconsistent per-chunk usage; loadgen keeps metric
//     paths log-free and instead counts fallback divisors in
//     PointResult.PerChunkUsageDivisorRequests.
package main

import "time"

// ITLDivisor is the decode-token divisor for one request's ITL.
type ITLDivisor struct {
	Tokens   int  // decode-token count: tokens that arrived after the first content chunk
	PerChunk bool // true when Tokens came from server per-chunk usage; false = legacy osl-1 fallback
}

// DecodeTokenDivisor maps aiperf's ITL first-chunk divisor to loadgen's inputs.
// osl is the server-reported output sequence length (final usage.completion_tokens);
// firstChunkTokens is usage.completion_tokens reported on the first content chunk
// (per-chunk usage), nil when not reported. ok=false only when no ITL is computable
// (osl < 2); every fallback path returns ok=true. PerChunk=true exactly when the
// divisor came from server per-chunk usage.
//
// Divisor rules, mirroring upstream _decode_token_count: first-chunk token count not
// reported -> legacy osl-1 fallback; reported but non-positive, or
// osl-firstChunkTokens < 1 (lagging usage, or the whole response in one chunk) ->
// osl-1 fallback; otherwise osl-firstChunkTokens.
func DecodeTokenDivisor(osl int, firstChunkTokens *int) (ITLDivisor, bool) {
	if osl < 2 {
		return ITLDivisor{}, false
	}
	if firstChunkTokens == nil {
		return ITLDivisor{Tokens: osl - 1}, true
	}
	if *firstChunkTokens <= 0 {
		return ITLDivisor{Tokens: osl - 1}, true
	}
	decode := osl - *firstChunkTokens
	if decode < 1 {
		return ITLDivisor{Tokens: osl - 1}, true
	}
	return ITLDivisor{Tokens: decode, PerChunk: true}, true
}

// ITLMS returns one request's inter-token latency in milliseconds:
// (latency - ttft) / divisor.Tokens. Callers pass a divisor with ok=true.
func ITLMS(latency, ttft time.Duration, d ITLDivisor) float64 {
	return float64((latency - ttft).Nanoseconds()) / 1e6 / float64(d.Tokens)
}
