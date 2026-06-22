//go:build cuda

package compute

import (
	"runtime"
	"testing"
)

// cuda_graph_test.go — the `-tags cuda` witness for issue #483 (CUDA Graphs / capture for the
// batch-1 decode step). At batch-1 autoregressive decode the per-op kernel-launch overhead
// dominates; the per-token decode op sequence (RMSNorm→QKV→RoPE→Attention→o_proj→FFN per layer,
// then final RMSNorm→head) is FIXED, so it can be captured ONCE into a cudaGraph_t on g_stream
// and replayed each step as a single cudaGraphLaunch instead of N kernel launches.
//
// This witness compares the SAME op chain (decodeChain, in cuda_test.go) run two ways on the
// SAME cuda backend:
//   - EAGER     (stepDev)      — each op launched individually on g_stream;
//   - CAPTURED  (stepDevGraph) — GraphBegin → the op chain (recorded by the open capture) →
//                                GraphEndLaunch (end capture, instantiate-or-ExecUpdate, launch,
//                                fence) — one graph launch per step.
// and asserts the captured path is numerically UNCHANGED under the cuda backend's Approx gate
// (argmax-exact + logit cosine ≥ 0.999), token for token across a prompt + greedy generation.
// It also pins the advertise/fallback contract: Caps.GraphCompile tracks graphEnabled, so a
// consumer that reads it false cleanly falls back to the synchronous per-op core.
//
// HARDWARE: the actual capture+replay RUN, the Approx verdict, and the capture-vs-no-capture
// tok/s delta need a CUDA node — they are the explicit residual of this build+commit handoff.
// On the win32 dev host (no CUDA toolkit) these skip cleanly; run them on a GPU node via
// tools/run_483_acceptance_on_gpu.sh. The Go+cgo here type-checks under `go vet -tags cuda`.

// withGraphEnabled flips the process-global graphEnabled gate (set once from FAK_CUDA_GRAPH at
// init) and returns a restore func. The witness/benchmarks opt in directly — regardless of the
// env — because graphEnabled gates BOTH GraphBegin's consent AND NewKV preallocating the fixed-
// capacity cache capture requires (a cudaMalloc mid-capture is illegal). Must be called BEFORE
// newSynth so the model's KVStore is created fixed-capacity.
func withGraphEnabled(on bool) func() {
	prev := graphEnabled
	graphEnabled = on
	return func() { graphEnabled = prev }
}

// stepDevGraph runs one token's decodeChain wrapped in a CUDA-graph capture+replay (#483), and
// returns the device-resident logits plus whether capture actually engaged. The input upload
// stays OUTSIDE the captured region (cudaMalloc/H2D mid-capture is illegal), exactly as the HAL
// does; the goroutine is OS-thread-pinned across the captured token so the open stream capture
// sees a single consistent submitter. Preconditions the caller guarantees: the buffer pool is
// warm (every devTr is served from the free list, no cudaMalloc) and the KV is fixed-capacity
// (graphEnabled true ⇒ NewKV preallocated, so AppendKV never reallocs).
func (m *synthModel) stepDevGraph(id int) (Tensor, bool) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	gr := m.be.(*cudaBackend)
	x := m.embedDev(id) // input stage OUTSIDE capture
	captured := gr.GraphBegin()
	logits := m.decodeChain(x, m.pos)
	if captured {
		gr.GraphEndLaunch() // end capture, instantiate (or ExecUpdate the kept exec), launch, fence
	}
	m.pos++
	return logits, captured
}

// TestCUDAGraphCompileCapGated pins the #483 advertise/fallback contract: the cuda backend
// advertises Caps.GraphCompile exactly when the graph path is live (graphEnabled), so a consumer
// that reads it false falls back to the synchronous per-op core. Runs whenever a cuda backend is
// registered; needs no graph execution.
func TestCUDAGraphCompileCapGated(t *testing.T) {
	cb := cudaOrSkip(t)
	restore := withGraphEnabled(false)
	if cb.Caps().GraphCompile {
		t.Fatal("#483: Caps.GraphCompile must be false when graphs are disabled — a consumer must then fall back to the synchronous per-op core")
	}
	restore()
	restore = withGraphEnabled(true)
	if !cb.Caps().GraphCompile {
		t.Fatal("#483: Caps.GraphCompile must be true when graphs are enabled (capture+replay path live)")
	}
	restore()
}

// TestCUDAGraphDecodeParity is the #483 acceptance witness (see file header): graph-replayed
// decode is numerically unchanged vs the eager device path under the Approx gate.
func TestCUDAGraphDecodeParity(t *testing.T) {
	cb := cudaOrSkip(t)
	defer withGraphEnabled(true)() // GraphBegin consents + NewKV preallocates the fixed-cap cache
	if !cb.Caps().GraphCompile {
		t.Fatal("#483: cuda backend must advertise Caps.GraphCompile=true when graphs are enabled")
	}
	cb.GraphReset() // drop any kept exec from a prior test (it is bound to stale buffer addresses)

	cfg, host := asyncWitnessCfg()
	mEager := newSynth(cb, cfg, host) // eager device path: each op a separate launch
	mGraph := newSynth(cb, cfg, host) // captured path: one graph launch per token

	// Warm-up: run the first `warm` tokens eagerly on BOTH models so each backend's buffer pool
	// holds every op-output size before mGraph's first capture (the pooled allocator must serve
	// every devTr from the free list — a cudaMalloc mid-capture is illegal). Mirrors the HAL's
	// halStep>=2 warm-up gate before it opens capture.
	const warm = 2
	warmStep := func(m *synthModel, tok int) { cb.Read(m.stepDev(tok)); cb.Recycle() }

	prompt := []int{5, 17, 42, 3, 88, 11}
	for i := 0; i < warm; i++ {
		warmStep(mEager, prompt[i])
		warmStep(mGraph, prompt[i])
	}

	capturedAny := false
	check := func(tag string, tok int) int {
		// Eager first, then Recycle so the pool is full again before the capture allocates from
		// it (mEager and mGraph share the pool and need identical buffer sizes).
		lE := cb.Read(mEager.stepDev(tok))
		cb.Recycle()
		lgG, captured := mGraph.stepDevGraph(tok)
		lG := cb.Read(lgG)
		cb.Recycle()
		capturedAny = capturedAny || captured

		// Approx gate: argmax must be EXACT (same next token) and the logit cosine ≥ 0.999. The
		// graph replays the identical kernels in the identical order — only the launch mechanism
		// differs — so on a correct device this is essentially bit-identical; the gate is the
		// cuda backend's Approx contract, not bit-identity (cuda is never the Reference).
		aE, aG := argmaxF32(lE), argmaxF32(lG)
		if aE != aG {
			t.Fatalf("%s: graph-replayed argmax %d != eager device argmax %d", tag, aG, aE)
		}
		if c := cosine(lE, lG); c < 0.999 {
			t.Fatalf("%s: graph-replayed logit cosine %.6f < 0.999 (Approx gate)", tag, c)
		}
		return aE
	}

	var next int
	for i := warm; i < len(prompt); i++ {
		next = check("prompt"+itoaC(i), prompt[i])
	}
	const nGen = 8
	for s := 0; s < nGen; s++ {
		next = check("gen"+itoaC(s), next)
	}
	if !capturedAny {
		t.Fatal("#483: graph capture never engaged (GraphBegin declined every step) — the witness would be vacuous; a node that cannot capture is a failure, not a pass")
	}
	t.Logf("#483 graph witness: %d prompt + %d greedy steps, graph-replayed==eager argmax-exact, logit cosine ≥ 0.999; device=%s tier=%s class=%s",
		len(prompt)-warm, nGen, cb.Name(), cb.Tier(), cb.Class())
}

// BenchmarkCUDADecodeNoCapture measures batch-1 greedy decode with every op launched
// individually (N kernel launches/token). graphEnabled is on so the KV geometry (fixed-capacity)
// matches the capture benchmark — the only isolated variable is the graph replay; stepDev simply
// never opens a capture. Recycle() at the token boundary keeps allocation at steady state.
func BenchmarkCUDADecodeNoCapture(b *testing.B) {
	cb := cudaTBOrSkip(b)
	defer withGraphEnabled(true)()
	cfg, host := asyncWitnessCfg()
	m := newSynth(cb, cfg, host)
	next := cb.Argmax(m.stepDev(5)) // warm: populate buffer pool + weight cache
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		next = cb.Argmax(m.stepDev(next))
		cb.Recycle()
		if m.pos >= benchCtxCap {
			b.StopTimer()
			m.resetKV()
			b.StartTimer()
		}
	}
	b.StopTimer()
	_ = next
}

// BenchmarkCUDADecodeCapture measures the SAME decode with the per-token op stream captured ONCE
// and replayed as a single cudaGraphLaunch (#483). The no-capture vs capture tok/s delta is what
// tools/run_483_acceptance_on_gpu.sh reports. GraphReset after resetKV drops the kept exec (bound
// to the freed KV addresses) so the next capture cleanly re-instantiates rather than relying on
// the C-side ExecUpdate self-heal.
func BenchmarkCUDADecodeCapture(b *testing.B) {
	cb := cudaTBOrSkip(b)
	defer withGraphEnabled(true)()
	cb.GraphReset()
	cfg, host := asyncWitnessCfg()
	m := newSynth(cb, cfg, host)
	cb.Read(m.stepDev(5)) // warm the pool eagerly so the first capture allocates only from it
	cb.Recycle()
	lg, _ := m.stepDevGraph(6) // warm the captured exec (first instantiate)
	next := cb.Argmax(lg)
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lg, _ := m.stepDevGraph(next)
		next = cb.Argmax(lg)
		cb.Recycle()
		if m.pos >= benchCtxCap {
			b.StopTimer()
			m.resetKV()
			cb.GraphReset()
			b.StartTimer()
		}
	}
	b.StopTimer()
	_ = next
}
