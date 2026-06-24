package model

// batch.go — MULTI-USER batched decode: the aggregate-throughput lane the baseline doc
// (MODEL-BASELINE-RESULTS.md) explicitly scoped OUT as "vLLM's regime, not fak's claim".
//
// The lever, stated precisely. Batch-1 decode is MEMORY-BANDWIDTH-bound at 0.50 flop/byte
// (the profiler's verdict, Act 1): per generated token the kernel re-streams ALL the weight
// bytes (~537 MB f32 / ~150 MB Q8), and time ≈ weight_bytes ÷ bandwidth, almost independent
// of how much arithmetic those weights drive. So serving ONE user wastes the machine: the
// weights are streamed to compute a single token's worth of MACs, then thrown away.
//
// Multi-user batching fixes exactly that. Stack ONE decode token from each of B independent
// users into a [B, *] panel and run each of the seven weight matmuls + the LM head as ONE
// GEMM over that panel (matMulBatch / qGemm8). Each weight row is now read ONCE and reused
// across all B users — the same arithmetic-intensity move that makes prefill fast, applied
// to the batch dimension instead of the token dimension. The bottleneck byte-stream is
// amortised B-fold, so AGGREGATE throughput (tokens/sec across all users) scales ~linearly
// with B until the GEMM becomes compute-bound, then plateaus at the compute roofline. This
// is "continuous batching" — the single biggest throughput multiplier in LLM serving — done
// in-kernel over kernel-OWNED per-user KV caches.
//
// What is per-user and what is shared. The seven projection GEMMs + the head are SHARED
// (one weight stream, B rows). Attention is PER-USER: user b's query attends only to user
// b's own KVCache (its own history, its own length), so there is zero cross-user mixing —
// the caches stay independent objects the context-MMU can still Evict/Clone per user.
//
// The bit-identity contract (f32). matMulBatch's row b is, by construction, bit-for-bit
// equal to parMatRows(weight, panel_row_b) — same fdot, same i-order (TestParallelMatchesSerial
// already pins this). The per-user attention here replays tokenHidden's EXACT scalar
// arithmetic (dot for scores, in-order V accumulation). So StepBatch's per-user logits are
// bit-for-bit identical to running each user through the serial Session.Step — proven by
// TestBatchedDecodeMatchesSerial (Float32bits equality on logits AND every user's KV cache).
// Batching changes only WHICH tokens share a weight load, never a single rounding.
//
// The Q8 path (stepBatchQ) reuses the register-blocked tile GEMM (qGemm8) the prefill path
// uses, so — exactly like prefill — it is NOT bit-identical to the serial qdot8 decode
// kernel (the tile reduces in a different lane order) but clears the same honest Q8 gate:
// argmax-exact + logit-cosine vs the f32 path (TestBatchedDecodeQMatchesF32). The f32 KV
// cache it builds is the same object either way, so Evict/Clone are untouched.

import (
	"os"
	"strconv"
)

const batchRectPrefillMaxTokens = 512

var attnGQAFuse = initAttnGQAFuse()

func initAttnGQAFuse() bool {
	switch os.Getenv("FAK_QATTN_GQA") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

var attnSaxpy3SIMDMinPos = initAttnSaxpy3SIMDMinPos()

func initAttnSaxpy3SIMDMinPos() int {
	if s := os.Getenv("FAK_SAXPY3_SIMD_MINPOS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 0 {
			return n
		}
	}
	return 1
}

var attnSaxpy3SIMDMinBatch = initAttnSaxpy3SIMDMinBatch()

func initAttnSaxpy3SIMDMinBatch() int {
	if s := os.Getenv("FAK_SAXPY3_SIMD_MINB"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n
		}
	}
	// Long-context batch decode spends most of its non-GEMM time accumulating V for the
	// three GQA query heads; the amd64 helper wins even at small B and can still be
	// disabled/tuned with FAK_SAXPY3_SIMD_MINB.
	return 1
}

var attnFdot3SIMD = initAttnFdot3SIMD()

func initAttnFdot3SIMD() bool {
	switch os.Getenv("FAK_FDOT3_SIMD") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

var attnFdot3SIMDMinBatch = initAttnFdot3SIMDMinBatch()

func initAttnFdot3SIMDMinBatch() int {
	if s := os.Getenv("FAK_FDOT3_SIMD_MINB"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n
		}
	}
	return 64
}

var qFastSwiGLU = initQFastSwiGLU()

func initQFastSwiGLU() bool {
	switch os.Getenv("FAK_Q_FAST_SWIGLU") {
	case "0", "false", "False", "FALSE", "off", "OFF":
		return false
	default:
		return true
	}
}

func swigluQInPlace(g, u []float32) {
	if qFastSwiGLU {
		swigluFastInPlace(g, u)
		return
	}
	swigluInPlace(g, u)
}

func batchRectFastPathOK(cfg Config, quant bool) bool {
	return batchPreNormFastPathOK(cfg, quant)
}

func batchDecodeFastPathOK(cfg Config, quant bool) bool {
	return batchPreNormFastPathOK(cfg, quant)
}

// batchPreNormFastPathOK reports whether the shared-weight panel GEMM lane (rect
// prefill + decode) covers this config, or whether it must fall back to the per-user
// serial Session path. The fast lane hardcodes the standard PreNorm q/k/v attention +
// Llama-shaped FFN; every excluded axis below is an attention/FFN/RoPE/topology shape
// that lane does not model. isGLMMoeDsa is excluded because GLM-5.2 attention is MLA +
// Dynamic Sparse Attention (q_a/q_b/kv_a/kv_b + a learned indexer), NOT q_proj/k_proj/
// v_proj — routing it here reads tensors it does not have. The real GLM-5.2 is also MoE
// (already excluded), but a DENSE glm_moe_dsa (the synthetic / pipelinegen form) is only
// caught by this explicit guard, so the exclusion keys on the attention arch, not on the
// incidental MoE property.
func batchPreNormFastPathOK(cfg Config, quant bool) bool {
	if cfg.BlockTopology != PreNorm ||
		cfg.IsMoE() ||
		cfg.isGLMMoeDsa() ||
		cfg.DenseMLP ||
		cfg.Alibi ||
		cfg.IsQwen35Hybrid() ||
		cfg.AttnOutputGate ||
		cfg.AttnSoftcap != 0 ||
		cfg.hasLayerSpecificRopeTheta() {
		return false
	}
	if quant {
		return q8FastPreNormOK(cfg)
	}
	return true
}

// BatchSession decodes B independent user sequences in lockstep, sharing one weight stream
// per layer across all B. Each user is a full Session with its OWN kernel-owned KVCache, so
// per-user prefill, eviction, and prefix-clone all still work; the batch only fuses the
// decode-step matmuls. Users may sit at different absolute positions (different history
// lengths) — attention reads each user's own cache length.
type BatchSession struct {
	M    *Model
	Seqs []*Session // B per-user sessions, each owns its KVCache and absolute position

	// Quant routes the batched step through the Q8_0 tile-GEMM lane (stepBatchQ); the model
	// must have had Quantize() called. Mirrors Session.Quant. The f32 default is byte-for-byte
	// the proven path.
	Quant bool

	// scratch is one reused Q8 activation panel for the quantized path: the per-step panels
	// (Xn → q/k/v, attnOut → o, Xn2 → gate/up, G → down, Xnorm → head) are consumed
	// sequentially, so a single growable buffer serves them all and avoids re-allocating a
	// panel every decode step (the same hygiene prefillBatchedQ uses). nil in the f32 path.
	scratch *q8Panel

	// dbuf holds the reused decode-step output/intermediate buffers. Every
	// projection output and panel is a fixed [B, width] shape each step, so one grown-once
	// buffer per role replaces the ~B-scaled MB of per-step allocation that BenchmarkStepBatchQ
	// measured (34 MB/step at B=32, 133 MB at B=128) — pure GC pressure, no numerics change.
	// The result is bit-identical to the allocating path. nil until the first batched decode step.
	dbuf *batchDecodeBuf

	// pbuf holds reused buffers for rectangular Q8 prefill (the repeated private-result
	// ingestion phase in fleetserve). PrefillEach still returns fresh final logits; these
	// buffers only replace per-layer temporaries that are fully consumed before return.
	pbuf *batchRectPrefillBuf

	// lastStepMACs is the exact MAC count of the B-proportional projection GEMMs (QKV +
	// attention-output + SwiGLU gate/up/down + the LM head) of the most recent StepBatch /
	// StepBatchActive — the weight-streaming decode work the batch dimension amortises and the
	// ragged (active-lane-masked) path compacts. It scales EXACTLY with the active batch size,
	// so a fleet with K of C lanes idle reports (C−K)/C of the full-batch count (#520). Closed
	// form from the model shape, set once per step; does NOT include attention (per-user,
	// cache-length-dependent). See LastStepMACs / recordStepMACs.
	lastStepMACs int64
}

// batchDecodeBuf is the per-BatchSession reused-buffer set for batched decode. All fields
// are grown once to the batch's shape and overwritten every step. Logits is the one buffer the
// step RETURNS (out[b] aliases it), so per the StepBatch contract a caller must consume the
// returned logits before the next StepBatch call (every in-tree caller — GenerateBatch and the
// benchmarks — does).
type batchDecodeBuf struct {
	X, Xn, Q, K, V, attn, O, Xn2, G, U, Down, Xnorm, Logits []float32
	scores                                                  [][]float32
	pos                                                     []int
	cos, sin                                                [][]float32
	caches                                                  []*KVCache
	out                                                     [][]float32
}

type batchRectPrefillBuf struct {
	X, Xn, Q, K, V, attn, O, Xn2, G, U, Down, Xnorm []float32
	base                                            []int
	caches                                          []*KVCache
	cos, sin                                        [][]float32
	scores                                          [][]float32
}

// grow returns b resliced to length n, reallocating only when cap is short. The returned slice
// is NOT zeroed (every use below fully overwrites it before reading).
func grow(b []float32, n int) []float32 {
	if cap(b) < n {
		return make([]float32, n, growCap(n))
	}
	return b[:n]
}

func growCap(n int) int {
	return n + n/8 + 64
}

func grow2D(b [][]float32, rows, cols int) [][]float32 {
	if cap(b) < rows {
		b = make([][]float32, rows)
	} else {
		b = b[:rows]
	}
	for i := range b {
		b[i] = grow(b[i], cols)
	}
	return b
}

func growInts(b []int, n int) []int {
	if cap(b) < n {
		return make([]int, n)
	}
	return b[:n]
}

func growCaches(b []*KVCache, n int) []*KVCache {
	if cap(b) < n {
		return make([]*KVCache, n)
	}
	return b[:n]
}

func growLogitRows(b [][]float32, n int) [][]float32 {
	if cap(b) < n {
		return make([][]float32, n)
	}
	return b[:n]
}

// NewBatchSession starts a B-user batch, each user with a fresh KV cache.
func (m *Model) NewBatchSession(n int) *BatchSession {
	bs := &BatchSession{M: m, Seqs: make([]*Session, n)}
	for i := range bs.Seqs {
		bs.Seqs[i] = m.NewSession()
	}
	return bs
}

// NewBatchFromPrefix starts an n-user batch where every user's KV cache is a CLONE of an
// already-computed prefix — a shared system prompt + tool schemas prefilled ONCE, then
// spliced into all n agents. This is the cross-agent KV-reuse path the fleet exists for:
// n agents that share a long prefix pay the prefix prefill a SINGLE time plus n cheap
// deep-copies, where a per-slot serving engine with no cross-request KV sharing (llama.cpp)
// must prefill that prefix n times to decode the n agents concurrently. Each clone is exact
// (KVCache.Clone), so every user is bit-identical to one that prefilled the prefix itself —
// the same R14 prefix-reuse property TestKVPrefixReuseMatchesRecompute proves for one
// session, now fanned out across the batch. The returned users sit at the prefix's length,
// ready for StepBatch on their first generated token.
func (m *Model) NewBatchFromPrefix(prefix *KVCache, n int) *BatchSession {
	return m.NewBatchFromPrefixReserve(prefix, n, 0)
}

// NewBatchFromPrefixReserve is NewBatchFromPrefix with per-user cache capacity reserved
// for the known decode/result tail. It preserves the same exact cloned prefix but avoids
// append-triggered prefix re-copies during the measured fleet run.
func (m *Model) NewBatchFromPrefixReserve(prefix *KVCache, n, extraPositions int) *BatchSession {
	bs := &BatchSession{M: m, Seqs: make([]*Session, n)}
	for i := range bs.Seqs {
		bs.Seqs[i] = &Session{M: m, Cache: prefix.CloneWithReserve(extraPositions)}
	}
	return bs
}

// SetQuant turns on the Q8_0 lane for every user in the batch (call after Model.Quantize()).
func (bs *BatchSession) SetQuant(q bool) {
	bs.Quant = q
	for _, s := range bs.Seqs {
		s.Quant = q
	}
}

// Reserve grows every user's KV cache for extra future positions without changing contents.
func (bs *BatchSession) Reserve(extraPositions int) {
	for _, s := range bs.Seqs {
		s.Cache.Reserve(extraPositions)
	}
}

// N is the number of users in the batch.
func (bs *BatchSession) N() int { return len(bs.Seqs) }
