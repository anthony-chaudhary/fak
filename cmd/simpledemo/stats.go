// stats.go — the demo's self-inspection layer. The chat loop measures four honest
// things per reply (prefill tokens + wall time, decode tokens + wall time, the
// KV-prefix cache hit, and the device it ran on) and this file turns them into the
// numbers a curious user actually wants: prefill vs decode tok/s reported SEPARATELY
// (folding them into one number hides that a short prompt's "slow" tok/s is really
// prefill), the EXACT prefill operation count from compute.Profile (an analytic FLOP
// roofline — counted, never timed, so it cannot fabricate throughput), the decode
// bandwidth stream that sets the tok/s ceiling, and the cache-hit % that prefix reuse
// actually achieved. Everything here is either measured on the real run or counted
// from the model shape; nothing is assumed.
package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// modelStats is the run-invariant "model card": the shape + residency facts gathered
// once after load and reused to label every per-turn panel. None of it changes during
// the chat, so it is computed exactly once.
type modelStats struct {
	name        string
	cfg         model.Config
	resident    *model.ResidentReport
	weightDtype compute.Dtype // the dtype the resident weights actually stream as
	device      string        // human label for the active compute path
	backends    []string      // compute.Registered() — the accelerators this build carries
	threads     int           // GOMAXPROCS — the CPU parallelism the forward pass may use
}

// gatherModelStats reads the loaded model's shape + resident report once. weightDtype
// is taken from what is actually resident (Q4_K majority vs Q8_0) rather than assumed,
// so the prefill roofline charges the real per-weight byte cost.
func gatherModelStats(m *model.Model, name, device string) modelStats {
	rep := m.ResidentReport()
	dt := compute.Q8_0
	if rep.Q4KBytes > rep.Q8Bytes {
		dt = compute.Q4_K
	}
	return modelStats{
		name:        name,
		cfg:         m.Cfg,
		resident:    rep,
		weightDtype: dt,
		device:      device,
		backends:    compute.Registered(),
		threads:     runtime.GOMAXPROCS(0),
	}
}

// geometry maps the model config into the compute package's prefill cost-model shape
// for a prompt of length p. It is the bridge between "what the model is" and "how much
// arithmetic a p-token prefill costs."
func (s modelStats) geometry(p int) compute.PrefillGeometry {
	c := s.cfg
	return compute.PrefillGeometry{
		DModel:      c.HiddenSize,
		NHeads:      c.NumHeads,
		NKVHeads:    c.NumKVHeads,
		HeadDim:     c.HeadDim,
		DFF:         c.IntermediateSize,
		NLayers:     c.NumLayers,
		Vocab:       c.VocabSize,
		P:           p,
		WeightDtype: s.weightDtype,
	}
}

// kvBytesPerToken is the resident KV-cache cost of one cached position: the kernel-owned
// cache holds K, the pre-RoPE Kraw (kept so eviction can re-rotate survivors), and V —
// three NumKVHeads*HeadDim f32 rows per layer. This is why long context, not weights, is
// what eventually fills memory.
func (s modelStats) kvBytesPerToken() int64 {
	kvStride := int64(s.cfg.NumKVHeads) * int64(s.cfg.HeadDim)
	return int64(s.cfg.NumLayers) * 3 * kvStride * 4
}

// printModelCard prints the one-time card to stderr: the model shape, the residency +
// bandwidth facts, the device, and the sampling assumptions in force. It deliberately
// surfaces the "assumptions" the goal asked for (quant, temperature, token budget,
// device, threads) so nothing about the run is implicit.
func printModelCard(s modelStats, temp float64, maxNew int, sampling string) {
	c := s.cfg
	gqa := ""
	if c.NumKVHeads > 0 && c.NumHeads > c.NumKVHeads {
		gqa = fmt.Sprintf(" (GQA %d×)", c.NumHeads/c.NumKVHeads)
	}
	ctx := "—"
	if c.MaxPositionEmbeddings > 0 {
		ctx = commafy(int64(c.MaxPositionEmbeddings))
	}
	fmt.Fprintf(os.Stderr, "🧠 %s · %d layers · d_model %s · %d heads / %d KV%s · head_dim %d · ffn %s · vocab %s · ctx %s\n",
		s.name, c.NumLayers, commafy(int64(c.HiddenSize)), c.NumHeads, c.NumKVHeads, gqa,
		c.HeadDim, commafy(int64(c.IntermediateSize)), commafy(int64(c.VocabSize)), ctx)

	r := s.resident
	fmt.Fprintf(os.Stderr, "💾 weights %s resident (%s) · KV %s/token · decode stream %.2f GiB/token\n",
		humanBytes(r.TotalResidentBytes), s.weightDtype.String(), humanBytes(s.kvBytesPerToken()), r.DecodeGiBPerToken)

	fmt.Fprintf(os.Stderr, "⚙️  device %s · %d threads · backends available: [%s]\n",
		s.device, s.threads, strings.Join(s.backends, ", "))
	if !hasGPUBackend(s.backends) {
		fmt.Fprintln(os.Stderr, "    └─ no GPU backend in this build — rebuild `-tags cuda` on an NVIDIA box (or `-tags fakmetal` on Apple) and pass `-backend cuda` to run on the GPU")
	}

	fmt.Fprintf(os.Stderr, "🎯 sampling: %s · temp %.2f · max %d tok/reply · %s weights\n",
		sampling, temp, maxNew, s.weightDtype.String())

	// A reference prefill cost model at P=512 — the analytic "operations running" for a
	// representative prompt, so the structural shape (GEMM-dominated, attention's O(P²)
	// memory-bound tail) is visible before the first chat turn.
	prof := compute.Profile(s.geometry(512))
	fmt.Fprintf(os.Stderr, "🧮 prefill @ P=512 ≈ %.1f GFLOP (counted, not timed) · heaviest %s %.1f GFLOP · attention intensity %.2f FLOP/B (memory-bound)\n",
		gflop(prof.TotalFLOPs), prof.Dominant.Name, gflop(prof.Dominant.FLOPs), attnIntensity(prof))
}

// turnStats is one reply's measured + counted numbers — what the chat loop hands to the
// renderer after a turn completes.
type turnStats struct {
	turn         int
	promptTokens int     // the logical prompt length (resident prefix + newly fed)
	newTokens    int     // tokens this turn actually prefilled (the suffix after cache reuse)
	reusedTokens int     // resident-prefix tokens reused from prior turns (the cache hit)
	prefillS     float64 // wall time spent prefilling the suffix
	genTokens    int     // tokens decoded this reply
	decodeS      float64 // wall time spent decoding
	kvPositions  int     // KV cache occupancy after the turn

	cumReused int // session-cumulative reused tokens
	cumPrompt int // session-cumulative logical prompt tokens
}

// printTurnStats renders the per-reply panel: prefill and decode reported as distinct
// throughputs, the cache hit prefix reuse achieved, the analytic prefill work + achieved
// compute rate, the decode bandwidth, and the KV occupancy. Every rate is measured this
// run; every FLOP/byte count is exact for the model shape.
func printTurnStats(ms modelStats, ts turnStats) {
	fmt.Fprintf(os.Stderr, "\n📊 turn %d · %s\n", ts.turn, ms.device)

	// Prefill line — tok/s over the tokens we actually computed, plus the cache hit.
	preTPS := rate(ts.newTokens, ts.prefillS)
	hit := pct(ts.reusedTokens, ts.promptTokens)
	if ts.reusedTokens > 0 {
		fmt.Fprintf(os.Stderr, "   prefill  %d new tok · %.3fs · %.0f tok/s · cache hit %.0f%% (%d/%d reused, %d recomputed)\n",
			ts.newTokens, ts.prefillS, preTPS, hit, ts.reusedTokens, ts.promptTokens, ts.newTokens)
	} else {
		fmt.Fprintf(os.Stderr, "   prefill  %d tok in · %.3fs · %.0f tok/s · cache cold (0%%, %d recomputed)\n",
			ts.newTokens, ts.prefillS, preTPS, ts.newTokens)
	}

	// Decode line — tok/s plus the bandwidth that bounds it. Decode reads every matmul
	// weight once per token, so achieved GiB/s against the per-token stream is the real
	// ceiling. Both are binary GiB so the ratio reads consistently with the stream figure.
	decTPS := rate(ts.genTokens, ts.decodeS)
	decGiBps := 0.0
	if ts.decodeS > 0 {
		decGiBps = float64(ms.resident.DecodeBytesPerToken) * float64(ts.genTokens) / ts.decodeS / (1 << 30)
	}
	fmt.Fprintf(os.Stderr, "   decode   %d tok out · %.3fs · %.1f tok/s · %.1f GiB/s of the %.2f GiB/token stream\n",
		ts.genTokens, ts.decodeS, decTPS, decGiBps, ms.resident.DecodeGiBPerToken)

	// Compute line — the analytic prefill operation count. "full N-tok prefill" is what a
	// cache-less prefill of the whole prompt would cost (exact, counted). The work ACTUALLY
	// done this turn is the cost-model delta between the full prompt and the reused prefix:
	// the suffix's GEMMs (linear in new tokens) plus its attention over the whole context
	// (the P²−reused² causal-pair growth) — exactly what prefix reuse leaves to recompute.
	// Dividing that by the measured prefill time is the honest achieved compute rate.
	prof := compute.Profile(ms.geometry(ts.promptTokens))
	workThisTurn := prof.TotalFLOPs
	if ts.reusedTokens > 0 {
		// The cost model charges one LM-head GEMM (last token only) regardless of P, so the
		// delta cancels both profiles' head — but the suffix prefill still runs the head once
		// for the next-token logits. Add that single head back so the count isn't understated.
		workThisTurn -= compute.Profile(ms.geometry(ts.reusedTokens)).TotalFLOPs
		workThisTurn += stageFLOPs(prof, "lm_head")
	}
	achGFLOPs := 0.0
	if ts.prefillS > 0 {
		achGFLOPs = gflop(workThisTurn) / ts.prefillS
	}
	fmt.Fprintf(os.Stderr, "   compute  full %d-tok prefill ≈ %.1f GFLOP · heaviest %s %.1f GFLOP · this turn %.1f GFLOP @ %.1f GFLOP/s\n",
		ts.promptTokens, gflop(prof.TotalFLOPs), prof.Dominant.Name, gflop(prof.Dominant.FLOPs), gflop(workThisTurn), achGFLOPs)

	// Footer — end-to-end latency, time-to-first-token (prefill), and KV footprint.
	kvMiB := float64(int64(ts.kvPositions)*ms.kvBytesPerToken()) / (1 << 20)
	cumHit := pct(ts.cumReused, ts.cumPrompt)
	fmt.Fprintf(os.Stderr, "   total    %.3fs · TTFT %.3fs · KV %s pos / %.1f MiB · session cache hit %.0f%%\n\n",
		ts.prefillS+ts.decodeS, ts.prefillS, commafy(int64(ts.kvPositions)), kvMiB, cumHit)
}

// ---- small honest formatters (no external deps) ---------------------------------

// rate is tokens-per-second, guarded against a zero interval.
func rate(tokens int, seconds float64) float64 {
	if seconds <= 0 {
		return 0
	}
	return float64(tokens) / seconds
}

// pct is a/b as a percentage, guarded against a zero denominator.
func pct(a, b int) float64 {
	if b <= 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

// gflop converts a multiply-add FLOP count to GFLOP.
func gflop(flops int64) float64 { return float64(flops) / 1e9 }

// humanBytes renders a byte count at the largest unit that keeps it readable, so a
// per-token KV cost (~tens of KiB) and a multi-GiB weight set both read sensibly instead
// of collapsing to "0 MiB".
func humanBytes(b int64) string {
	const ki = 1 << 10
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GiB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.0f MiB", float64(b)/(1<<20))
	case b >= ki:
		return fmt.Sprintf("%.0f KiB", float64(b)/ki)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

// attnIntensity pulls the attention stage's arithmetic intensity out of a roofline. Naive
// attention is a ~0.5 FLOP/B constant (deeply memory-bound) — the long-sequence bottleneck
// flash/paged attention exists to cut — so surfacing it explains where prefill time goes.
func attnIntensity(p compute.PrefillRoofline) float64 {
	for _, s := range p.Stages {
		if s.Name == "attn" {
			return s.Intensity
		}
	}
	return 0
}

// stageFLOPs returns the FLOP count of a named stage in a roofline, or 0 if absent.
func stageFLOPs(p compute.PrefillRoofline, name string) int64 {
	for _, s := range p.Stages {
		if s.Name == name {
			return s.FLOPs
		}
	}
	return 0
}

// hasGPUBackend reports whether any registered backend is an accelerator (not the CPU
// reference floor), so the card can tell the user how to light one up when none is built.
func hasGPUBackend(names []string) bool {
	for _, n := range names {
		if n != "cpu-ref" {
			return true
		}
	}
	return false
}

// commafy groups an integer with thousands separators for readability (151936 -> 151,936).
func commafy(n int64) string {
	neg := n < 0
	if neg {
		n = -n
	}
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		if neg {
			return "-" + s
		}
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre > 0 {
		b.WriteString(s[:pre])
		if len(s) > pre {
			b.WriteByte(',')
		}
	}
	for i := pre; i < len(s); i += 3 {
		b.WriteString(s[i : i+3])
		if i+3 < len(s) {
			b.WriteByte(',')
		}
	}
	out := b.String()
	if neg {
		return "-" + out
	}
	return out
}
