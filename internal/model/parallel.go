package model

// parallel.go — the performance lane: parallelize the matmuls across cores and batch
// the prefill GEMM, WITHOUT perturbing a single bit of the proven numerics.
//
// The whole in-kernel-model lane's value is that correctness is PROVEN (oracle-witnessed
// rung by rung), and several rungs assert *exact* fak-vs-fak bit-identity (R2 cached
// decode == prefill at max|Δ|=0; R14 prefix-reuse == recompute at d==0). So the speedups
// here are constrained to be **bit-identical to the serial reference**: every output
// element y[o] is still sum_i w[o,i]*x[i] accumulated in the SAME i-order; only the
// assignment of *which core computes which output row* (and, for the batched GEMM, the
// loop nest order over (token,row)) is parallel/reordered. Float addition is
// non-associative, so we never split a single dot-product's reduction across workers —
// that would drift ~1e-6 and break the max|Δ|=0 rungs. Row-parallel + same-inner-order
// keeps it exact, which TestParallelMatchesSerial and the oracle suite both enforce.

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// numWorkers caps matmul parallelism. Default = GOMAXPROCS, overridable two ways (see
// budget.go for the full precedence): FAK_WORKERS=<n> pins an ABSOLUTE count (so a
// benchmark can measure serial-vs-parallel in the SAME environment, and set 1 to
// reproduce the serial reference exactly), while FAK_BUDGET=<fraction> pins a
// machine-PORTABLE share (0.75 -> 75% of the cores on whatever box this is). Batch-1
// decode is memory-bandwidth-bound, so this saturates well below core count.
var numWorkers, workerBudgetSource = resolveBudgetWorkers(
	os.Getenv("FAK_WORKERS"), os.Getenv("FAK_BUDGET"), runtime.GOMAXPROCS(0),
	func(note string) { fmt.Fprintln(os.Stderr, "[fak] "+note) },
)

// NumWorkers reports the worker count for the next dispatch. The default follows the
// runtime GOMAXPROCS value; explicit environment and flag overrides remain pinned.
func NumWorkers() int { return currentWorkerCount() }

// WorkerBudget reports HOW the worker count was resolved — "FAK_WORKERS=8",
// "FAK_BUDGET=0.75", or "default(GOMAXPROCS)" — so a recorded run states the budget it
// was taken at (a number at 75% of a 32-core box is a different regime than 100% of an
// 8-core box, and only the source makes that legible in the JSON report).
func WorkerBudget() string {
	_, source := currentWorkerBudget()
	return source
}

// parThreshold is the minimum work (output elements × inner dim) below which a matmul
// runs serially — goroutine dispatch isn't worth it for tiny ops (norms, small projs).
const parThreshold = 1 << 16

// The matmul worker pool is a SPIN-then-park barrier rather than a channel of jobs. The decode
// path issues ~200 tiny matmuls per token (q/k/v/o/gate/up/down × layers), and a channel/WaitGroup
// pool parked and re-woke every worker on each one — a CPU profile of a batch-1 Qwen-1.5B decode
// spent ~62% of all samples in pthread_cond_wait/signal with only ~3.5 of 12 cores busy. Because
// the matmuls arrive microseconds apart, the workers now SPIN on a per-worker atomic sequence
// number (no futex) and only park after parSpinBudget idle spins (between requests), collapsing the
// wake/park churn. parFor slices [0,n) into contiguous chunks that every participant grabs
// DYNAMICALLY from a shared atomic cursor (work-stealing), so a fast P-core pulls more chunks than a
// slow E-core — static even chunking left the E-cores the makespan bottleneck (prefill scaled only
// 5.16× on the 12-core M3 Pro). Each row is still computed exactly once, in index order within its
// chunk, by one goroutine, so every output is bit-identical to the serial reference
// (TestParallelMatchesSerial and the exact decode==prefill rungs are unaffected — the dynamic
// assignment changes only WHICH core computes a row, never the per-row reduction). The caller joins
// the steal loop. parDispatchMu serializes concurrent parFor callers so the shared work queue stays
// single-owner (the pool is one shared resource either way).
// parSpinBudget is how many idle spins a worker tolerates before parking. It must exceed the
// SERIAL gap between consecutive decode matmuls (activation quantize + norm + attention), or the
// worker parks mid-token and the next matmul must re-wake it through the runtime scheduler — a
// profile showed ~40% of decode CPU in runtime.stealWork/goready from exactly that park/wake
// churn. ~1ms (1<<20) bridges the gaps for batch-1 Qwen-1.5B — a spin sweep on M3 Pro peaked there
// (decode 22.4 -> 34.3 tok/s vs 1<<15; 1<<22 over-spins and regresses) — while still parking when
// a request truly ends. Tunable via FAK_PAR_SPIN for A/B.
var parSpinBudget = envInt64Default("FAK_PAR_SPIN", 1<<20)

func envInt64Default(name string, def int64) int64 {
	if s := os.Getenv(name); s != "" {
		var n int64
		if _, err := fmt.Sscanf(s, "%d", &n); err == nil && n >= 0 {
			return n
		}
	}
	return def
}

type parSlot struct {
	seq      atomic.Uint64 // dispatch generation; release-signals the worker to join the steal loop
	wake     chan struct{}
	sleeping atomic.Bool
	_        [40]byte // pad to a cache line, avoid false sharing between slots
}

type parPoolGeneration struct {
	width int
	slots []parSlot
	stop  chan struct{}
	done  sync.WaitGroup
}

var parWorkerLabelContext = pprof.WithLabels(
	context.Background(),
	pprof.Labels("fak.component", "model.parallel"),
)

var (
	parDispatchMu sync.Mutex
	parPool       *parPoolGeneration

	// Per-dispatch shared work queue (single-owner under parDispatchMu). parBody/parN/parChunkSize/
	// parNumChunks are published before the per-slot seq bump (release) and read by workers after
	// observing the bump (acquire). parNextChunk is the dynamic chunk cursor every participant
	// fetch-adds; parActive counts dispatched workers still in the steal loop.
	parBody      func(lo, hi int)
	parN         int
	parChunkSize int
	parNumChunks int64
	parNextChunk atomic.Int64
	parActive    atomic.Int64
)

// parChunkGranularity is how many chunks-per-worker the queue is sliced into: >1 so a fast P-core
// grabs several chunks while a slow E-core grabs one (dynamic balance across heterogeneous cores).
// Too fine and the per-chunk atomic fetch-add dominates tiny matmuls; 4 is the compromise (a sweep
// of 4/8/16/32 on M3 Pro was flat within noise — the prefill scaling limit is the E-cores' ~3×
// slower throughput + serial glue, not chunk size).
const parChunkGranularity = 4

// ensureParPool installs the generation required by the next dispatch. parDispatchMu
// is held, so the previous generation has no in-flight work and can be stopped before
// its replacement is published. At most one generation is live after this boundary.
func ensureParPool(width int) {
	if width < 1 {
		width = 1
	}
	if parPool != nil && parPool.width == width {
		return
	}
	if parPool != nil {
		close(parPool.stop)
		for i := range parPool.slots {
			select {
			case parPool.slots[i].wake <- struct{}{}:
			default:
			}
		}
		parPool.done.Wait()
	}
	g := &parPoolGeneration{width: width, stop: make(chan struct{})}
	if width > 1 {
		g.slots = make([]parSlot, width-1)
		for i := range g.slots {
			g.slots[i].wake = make(chan struct{}, 1)
		}
		for i := range g.slots {
			g.done.Add(1)
			go parWorkerLoop(g, i)
		}
	}
	parPool = g
}

// parGrab is the work-stealing loop: fetch-add the chunk cursor and run each contiguous chunk until
// the queue drains. Run by every dispatched worker AND by the caller.
func parGrab() {
	for {
		c := parNextChunk.Add(1) - 1
		if c >= parNumChunks {
			return
		}
		lo := int(c) * parChunkSize
		hi := lo + parChunkSize
		if hi > parN {
			hi = parN
		}
		parBody(lo, hi)
	}
}

func parWorkerLoop(g *parPoolGeneration, w int) {
	defer g.done.Done()
	// Persistent workers must not retain request-, tenant-, or secret-bearing
	// labels from the goroutine that first initializes the process-wide pool.
	pprof.SetGoroutineLabels(parWorkerLabelContext)

	sl := &g.slots[w]
	var last uint64
	var spun int64
	for {
		select {
		case <-g.stop:
			return
		default:
		}
		s := sl.seq.Load()
		if s != last {
			last = s
			parGrab()
			parActive.Add(-1)
			spun = 0
			continue
		}
		if spun < parSpinBudget {
			spun++
			continue
		}
		// Idle past the spin budget: park until woken (or until work raced in).
		sl.sleeping.Store(true)
		if sl.seq.Load() == last {
			select {
			case <-sl.wake:
			case <-g.stop:
				return
			}
		}
		sl.sleeping.Store(false)
		spun = 0
	}
}

func fdot(r, x []float32) float32 { return mathx.FDot(r, x) }

func fdot3(r0, r1, r2, x []float32) (float32, float32, float32) {
	return fdot3scalar(r0, r1, r2, x)
}

func fdot3SIMD(r0, r1, r2, x []float32) (float32, float32, float32) {
	if a, b, c, ok := fdot3Fast(r0, r1, r2, x); ok {
		return a, b, c
	}
	return fdot3scalar(r0, r1, r2, x)
}

func fdot3scalar(r0, r1, r2, x []float32) (float32, float32, float32) {
	var a0, a1, a2, a3, a4, a5, a6, a7 float32
	var b0, b1, b2, b3, b4, b5, b6, b7 float32
	var c0, c1, c2, c3, c4, c5, c6, c7 float32
	n := len(x)
	i := 0
	for ; i+8 <= n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		a0 += r0[i] * x0
		a1 += r0[i+1] * x1
		a2 += r0[i+2] * x2
		a3 += r0[i+3] * x3
		a4 += r0[i+4] * x4
		a5 += r0[i+5] * x5
		a6 += r0[i+6] * x6
		a7 += r0[i+7] * x7
		b0 += r1[i] * x0
		b1 += r1[i+1] * x1
		b2 += r1[i+2] * x2
		b3 += r1[i+3] * x3
		b4 += r1[i+4] * x4
		b5 += r1[i+5] * x5
		b6 += r1[i+6] * x6
		b7 += r1[i+7] * x7
		c0 += r2[i] * x0
		c1 += r2[i+1] * x1
		c2 += r2[i+2] * x2
		c3 += r2[i+3] * x3
		c4 += r2[i+4] * x4
		c5 += r2[i+5] * x5
		c6 += r2[i+6] * x6
		c7 += r2[i+7] * x7
	}
	a := ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7))
	b := ((b0 + b1) + (b2 + b3)) + ((b4 + b5) + (b6 + b7))
	c := ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7))
	for ; i < n; i++ {
		v := x[i]
		a += r0[i] * v
		b += r1[i] * v
		c += r2[i] * v
	}
	return a, b, c
}

// parFor splits [0,n) into up to `workers` contiguous chunks and runs body(lo,hi) on
// each concurrently, returning when all finish. Every index is handled by exactly one
// chunk, so any per-index work stays independent and order-preserving.
func parFor(n, workers int, body func(lo, hi int)) {
	if n <= 1 {
		body(0, n)
		return
	}
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	parDispatchMu.Lock()
	poolWidth := currentWorkerCount()
	if poolWidth < workers {
		// The caller may have retained an older default width at its operation boundary.
		// Keep that in-flight choice intact even if GOMAXPROCS changed before this dispatch.
		poolWidth = workers
	}
	ensureParPool(poolWidth)
	if workers <= 1 {
		parDispatchMu.Unlock()
		body(0, n)
		return
	}
	// Slice [0,n) into ~workers*granularity contiguous chunks so a fast core can grab several while
	// a slow core grabs one. Only the requested worker budget is dispatched; this lets decode paths
	// deliberately use fewer workers than the global prefill budget.
	chunkSize := (n + workers*parChunkGranularity - 1) / (workers * parChunkGranularity)
	if chunkSize < 1 {
		chunkSize = 1
	}
	numChunks := int64((n + chunkSize - 1) / chunkSize)
	nDisp := workers - 1
	if nDisp > len(parPool.slots) {
		nDisp = len(parPool.slots)
	}
	parBody = body
	parN = n
	parChunkSize = chunkSize
	parNumChunks = numChunks
	parNextChunk.Store(0)
	// parActive MUST be set before any seq bump so a worker that drains the queue immediately can
	// never decrement a not-yet-published counter.
	parActive.Store(int64(nDisp))
	for w := 0; w < nDisp; w++ {
		sl := &parPool.slots[w]
		sl.seq.Add(1) // release: publishes parBody/parN/parChunkSize/parNumChunks
		if sl.sleeping.Load() {
			select {
			case sl.wake <- struct{}{}:
			default:
			}
		}
	}
	// The caller is the numWorkers'th worker: join the steal loop, then wait for the rest to drain.
	parGrab()
	for parActive.Load() != 0 {
	}
	parBody = nil
	parDispatchMu.Unlock()
}

// parForRange runs body over [0,n) exactly once per index, deciding serial-vs-parallel the
// way every row-parallel quant kernel does: run inline when parallelism is disabled
// (numWorkers <= 1) or the work estimate is below parThreshold (goroutine dispatch isn't
// worth it for a tiny op), otherwise split across numWorkers via parFor. Both paths call
// body with the same [lo,hi) contract, so the result is bit-identical to the serial reference
// regardless of which branch runs. work is the caller's cost estimate (typically out×in,
// or out×in×P for a batched GEMM).
func parForRange(n, work int, body func(lo, hi int)) {
	parForRangeWorkers(n, work, currentWorkerCount(), body)
}

// parForRangeWorkers is parForRange with an explicit worker count. The batch-1 decode GEMV
// paths cap workers BELOW the global numWorkers to dodge the parFor barrier collapse across
// NUMA nodes (see Q4KDecodeWorkers / Q8DecodeWorkers). The [lo,hi) contract is identical, so
// the result stays bit-identical to the serial reference regardless of the worker count.
func parForRangeWorkers(n, work, workers int, body func(lo, hi int)) {
	if workers <= 1 || work < parThreshold {
		body(0, n)
		return
	}
	parFor(n, workers, body)
}

// parMatRows is matRows parallelized across OUTPUT ROWS. y[o] = sum_i w[o*in+i]*x[i] is
// computed by exactly one worker in the SAME i-order as the serial matRows, so the
// result is BIT-IDENTICAL regardless of worker count. This is the decode-path speedup:
// the agent-loop regime is memory-bound, and 1 core used ~41% of single-thread bandwidth,
// so spreading rows across cores taps the machine's much larger aggregate bandwidth.
func parMatRows(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	row := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			y[o] = mathx.FDot(w[o*in:o*in+in], x)
		}
	}
	if out*in < parThreshold {
		row(0, out)
		return y
	}
	parFor(out, currentWorkerCount(), row)
	return y
}

// matMulBatch computes Y[t*out+o] = sum_i W[o*in+i]*X[t*in+i] for all t in [0,P), o in
// [0,out) — the BATCHED-GEMM access pattern that is the prefill speedup. By holding one
// weight row in cache and sweeping all P input rows, each weight is read once and reused
// P times, raising arithmetic intensity from GEMV's 0.5 flop/byte toward compute-bound
// (which is exactly why HF/llama.cpp prefill is ~20-150x faster — they batch, fak did
// GEMV-per-token). The inner reduction over i is in-order, so Y[t*out+o] is bit-identical
// to parMatRows(W, X[t], out, in)[o] — the per-token path's result, just computed faster.
// Output is row-major [P, out]. Parallelized across output rows.
func matMulBatch(w, X []float32, out, in, P int) []float32 {
	Y := make([]float32, P*out)
	body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			r := w[o*in : o*in+in]
			for t := 0; t < P; t++ {
				Y[t*out+o] = mathx.FDot(r, X[t*in:t*in+in])
			}
		}
	}
	if out*in*P < parThreshold {
		body(0, out)
		return Y
	}
	parFor(out, currentWorkerCount(), body)
	return Y
}
