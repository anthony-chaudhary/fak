package model

// decode_roofline.go — the decode-gap diagnosis seam.
//
// Decode (batch=1 autoregressive generation) is memory-bandwidth-bound: each generated
// token streams EVERY weight matrix exactly once with no reuse, so per-token latency is
// floored by (weight bytes streamed) / (achieved memory bandwidth). This file makes the
// NUMERATOR of that roofline exact: q8DecodeStreamBytes sums the Q8_0 bytes the fast Q8
// decode path (tokenHiddenQ + headQ) actually reads per token. A measured ms/tok then
// converts to an achieved decode GB/s, which compared against the machine's STREAM ceiling
// (MeasureMemBandwidthGBps) answers the question the decode gap turns on with evidence
// instead of a guess: is fak's decode AT the bandwidth roofline (so the only lever left is
// streaming fewer bytes — Q4 — not a faster kernel), or is it well under the ceiling (so
// there is kernel / forward-path headroom to recover)?
//
// This is arch-neutral Go: the same accounting runs on amd64 (AVX path) and arm64 (the NEON
// path the 41-vs-71.9-tok/s gap is measured on), so the structural breakdown it feeds is
// portable and the absolute GB/s is whatever the box under test extracts.

import (
	"sync"
	"time"
)

// bytes returns the number of bytes this Q8_0 tensor streams when read once end-to-end:
// out*in int8 codes + out*nblk f32 block scales — i.e. 1 + 4/32 = 1.125 B/weight, the same
// footprint quant.go's header documents for the ~3.6× decode-bandwidth win over f32.
func (t *q8Tensor) bytes() int64 {
	if t == nil {
		return 0
	}
	return int64(len(t.q)) + 4*int64(len(t.d))
}

// q8DecodeStreamBytes is the exact Q8_0 weight footprint a single fast-path decode token
// streams: every layer's q/k/v/o attention projections + gate/up/down MLP projections, plus
// the LM head. These are precisely the operands tokenHiddenQ/headQ feed to qMatRowsInto, so
// the sum is the all-of-the-model byte stream that dominates decode latency.
//
// It deliberately EXCLUDES: the f32 KV cache (read by attention, but L2-resident at these
// sequence lengths — profile.go measures attn as latency-bound, ~2 GB/s, not bandwidth-bound),
// the f32 RMSNorm gains, and the projection biases — all negligible next to the projections.
// Requires Model.Quantize() to have built the resident Q8 cache.
func (m *Model) q8DecodeStreamBytes() int64 {
	var n int64
	if m.q8layers != nil {
		for l := range m.q8layers {
			ql := &m.q8layers[l]
			n += ql.qProj.bytes() + ql.kProj.bytes() + ql.vProj.bytes() + ql.oProj.bytes()
			n += ql.gateProj.bytes() + ql.upProj.bytes() + ql.downProj.bytes()
		}
	}
	if m.q8head != nil {
		n += m.q8head.bytes()
	}
	return n
}

// DecodeRoofline is the bandwidth-roofline verdict for one measured decode run: the exact
// per-token weight byte stream, the achieved decode bandwidth derived from the measured
// ms/tok, the machine's STREAM-triad ceiling, and the utilization between them.
type DecodeRoofline struct {
	StreamBytes  int64   `json:"stream_bytes"`              // Q8 weight bytes per decode token
	PerTokenMS   float64 `json:"per_token_ms"`              // measured (best-of-reps) decode latency
	TokPerSec    float64 `json:"tok_per_sec"`               // 1000 / PerTokenMS
	AchievedGBps float64 `json:"achieved_gbps"`             // StreamBytes / (PerTokenMS/1000)
	CeilingGBps  float64 `json:"stream_ceiling_gbps"`       // MeasureMemBandwidthGBps
	BWUtilPct    float64 `json:"bandwidth_utilization_pct"` // 100 * Achieved / Ceiling
}

// DecodeRooflineFor builds the roofline for a measured per-token decode latency (ms) on this
// model. The decode GEMV runs multi-core (parFor over numWorkers), so its achieved GB/s is
// AGGREGATE — and the ceiling it must be judged against is therefore the AGGREGATE STREAM
// bandwidth (all cores hammering memory at once), NOT one core's share. Comparing aggregate
// achieved against a single-core ceiling produces a nonsense >100% utilization; this uses the
// multi-core ceiling so the util is the honest "fraction of the memory system decode taps."
func (m *Model) DecodeRooflineFor(perTokenMS float64) DecodeRoofline {
	bytes := m.q8DecodeStreamBytes()
	r := DecodeRoofline{StreamBytes: bytes, PerTokenMS: perTokenMS}
	if perTokenMS > 0 {
		r.TokPerSec = 1000.0 / perTokenMS
		r.AchievedGBps = float64(bytes) / (perTokenMS / 1e3) / 1e9
	}
	r.CeilingGBps = measureAggregateStreamGBps(int(bytes), numWorkers)
	if r.CeilingGBps > 0 {
		r.BWUtilPct = 100 * r.AchievedGBps / r.CeilingGBps
	}
	return r
}

// measureAggregateStreamGBps runs a STREAM-triad (a[i]=b[i]+scalar*c[i]) concurrently across
// `workers` goroutines over `modelBytes`-worth of f32 buffers total, and returns the aggregate
// bytes-touched / wall-time — the multi-core memory ceiling the multi-threaded decode is
// actually bounded by. Best of 5 reps; zero-duration reps are skipped (a coarse timer would be
// +Inf and poison the max). workers<1 is treated as 1.
func measureAggregateStreamGBps(modelBytes, workers int) float64 {
	if workers < 1 {
		workers = 1
	}
	nPer := modelBytes / 4 / 3 / workers // three f32 buffers per worker ~ one model's traffic
	if nPer < 1<<18 {
		nPer = 1 << 18
	}
	a := make([][]float32, workers)
	b := make([][]float32, workers)
	c := make([][]float32, workers)
	for k := 0; k < workers; k++ {
		a[k] = make([]float32, nPer)
		b[k] = make([]float32, nPer)
		c[k] = make([]float32, nPer)
		for i := 0; i < nPer; i++ {
			b[k][i] = float32(i % 7)
			c[k][i] = float32(i % 5)
		}
	}
	const scalar = 3.0
	run := func() {
		var wg sync.WaitGroup
		for k := 0; k < workers; k++ {
			wg.Add(1)
			go func(ak, bk, ck []float32) {
				defer wg.Done()
				for i := range ak {
					ak[i] = bk[i] + scalar*ck[i]
				}
			}(a[k], b[k], c[k])
		}
		wg.Wait()
	}
	run() // warm
	best := 0.0
	for rep := 0; rep < 5; rep++ {
		t := time.Now()
		run()
		secs := time.Since(t).Seconds()
		if secs <= 0 {
			continue
		}
		gbps := float64(12*nPer*workers) / secs / 1e9 // triad touches 12 bytes/elem
		if gbps > best {
			best = gbps
		}
	}
	return best
}
