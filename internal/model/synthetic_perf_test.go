package model

import (
	"fmt"
	"os"
	"testing"
	"time"
)

// qwen25_1_5bConfig is the Qwen2.5-1.5B-Instruct shape — used to build a synthetic (random-weight)
// model so the end-to-end decode/prefill tok/s the parity goal targets can be measured on a box
// with no HF export. Perf is weight-VALUE-independent (same matmul/attention/quant work), so the
// throughput is faithful even though the logits are meaningless.
func qwen25_1_5bConfig() Config {
	return Config{
		HiddenSize: 1536, NumLayers: 28, NumHeads: 12, NumKVHeads: 2, HeadDim: 128,
		IntermediateSize: 8960, VocabSize: 151936, RMSNormEps: 1e-6, RopeTheta: 1000000,
		TieWordEmbeddings: true, EOSTokenID: 151643, HiddenAct: "silu", ModelType: "qwen2",
	}
}

func perfLCG(n, vocab int) []int {
	ids := make([]int, n)
	st := uint64(2463534242)
	for i := range ids {
		st = (st*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(st % uint64(vocab))
	}
	return ids
}

// TestSyntheticQwen15BPerf reports decode tok/s and prefill@256 tok/s for the Qwen2.5-1.5B Q8 path,
// the contention-robust MIN over reps (the same headline q8bench prints). Gated on FAK_PERF=1.
func TestSyntheticQwen15BPerf(t *testing.T) {
	if os.Getenv("FAK_PERF") == "" {
		t.Skip("set FAK_PERF=1 to run the synthetic Qwen2.5-1.5B perf measurement")
	}
	t0 := time.Now()
	m := NewSynthetic(qwen25_1_5bConfig())
	m.Quantize()
	vocab := m.Cfg.VocabSize
	fmt.Fprintf(os.Stderr, "[perf] built+quantized synthetic Qwen2.5-1.5B in %.1fs, workers=%d\n",
		time.Since(t0).Seconds(), NumWorkers())

	// warm both paths (page in weights, grow scratch) — untimed.
	{
		s := m.NewSession()
		s.Quant = true
		s.Prefill(perfLCG(8, vocab))
		s.Step(7)
	}

	decodeReps := envIntDefault("FAK_PERF_DECODE_REPS", 15)
	steps := envIntDefault("FAK_PERF_DECODE_STEPS", 32)
	promptLen := envIntDefault("FAK_PERF_PROMPT", 16)
	best := time.Duration(1) << 62
	for r := 0; r < decodeReps; r++ {
		s := m.NewSession()
		s.Quant = true
		s.Prefill(perfLCG(promptLen, vocab))
		id := 7
		tt := time.Now()
		for i := 0; i < steps; i++ {
			_ = s.Step(id)
			id = (id*48271 + 1) % vocab
		}
		if d := time.Since(tt) / time.Duration(steps); d < best {
			best = d
		}
	}
	decMS := float64(best.Nanoseconds()) / 1e6
	decTokS := 1000.0 / decMS

	// Decode bandwidth roofline: per-token weight byte stream / best-case decode latency =
	// achieved decode GB/s, vs the aggregate (multi-core) STREAM ceiling. This is the number
	// that resolves the decode gap: near-ceiling => bandwidth-bound (lever = stream fewer
	// bytes); well under => kernel/forward-path headroom. See decode_roofline.go.
	roof := m.DecodeRooflineFor(decMS)

	// One opt-in profiled decode rep splits the per-token time across the fast Q8 path's
	// phases (q8_qkv_proj / q8_attn / q8_o_proj / q8_mlp / q8_norm_quant + head) — the
	// breakdown tokenHiddenQ formerly left unattributed. Kept OUT of the timed loop above.
	phaseSteps := envIntDefault("FAK_PERF_PHASE_STEPS", 24)
	var phase *PhaseProfile
	{
		s := m.NewSession()
		s.Quant = true
		s.Prefill(perfLCG(promptLen, vocab))
		pp := NewPhaseProfiler()
		s.PhaseProfiler = pp
		id := 7
		pt := time.Now()
		for i := 0; i < phaseSteps; i++ {
			_ = s.Step(id)
			id = (id*48271 + 1) % vocab
		}
		phase = pp.Snapshot("decode", promptLen, phaseSteps, time.Since(pt).Nanoseconds())
	}

	preMS, preTokS, prefP, prefReps := 0.0, 0.0, 0, 0
	if os.Getenv("FAK_PERF_NO_PREFILL") == "" {
		prefP = envIntDefault("FAK_PERF_PREFILL", 256)
		prefReps = envIntDefault("FAK_PERF_PREFILL_REPS", 6)
		pbest := time.Duration(1) << 62
		pids := perfLCG(prefP, vocab)
		for r := 0; r < prefReps; r++ {
			s := m.NewSession()
			s.Quant = true
			tt := time.Now()
			s.Prefill(pids)
			if d := time.Since(tt); d < pbest {
				pbest = d
			}
		}
		preMS = float64(pbest.Nanoseconds()) / 1e6
		preTokS = float64(prefP) / (preMS / 1e3)
	}

	fmt.Fprintf(os.Stderr, "\n==== Qwen2.5-1.5B Q8 (synthetic) ====\n")
	fmt.Fprintf(os.Stderr, "DECODE       %.1f tok/s  (%.2f ms/tok, min of %d reps)\n", decTokS, decMS, decodeReps)
	if prefReps > 0 {
		fmt.Fprintf(os.Stderr, "PREFILL@%-3d  %.1f tok/s  (%.1f ms, min of %d reps)\n", prefP, preTokS, preMS, prefReps)
	}
	fmt.Fprintf(os.Stderr, "goal: decode 71.9 tok/s · prefill 547 tok/s (llama.cpp CPU)\n")
	fmt.Fprintf(os.Stderr, "\n---- decode bandwidth roofline ----\n")
	fmt.Fprintf(os.Stderr, "weight stream  %.0f MB/tok\n", float64(roof.StreamBytes)/1e6)
	fmt.Fprintf(os.Stderr, "achieved       %.1f GB/s   aggregate STREAM ceiling %.1f GB/s   util %.0f%%\n",
		roof.AchievedGBps, roof.CeilingGBps, roof.BWUtilPct)
	fmt.Fprintf(os.Stderr, "verdict: %s\n", rooflineVerdict(roof.BWUtilPct))
	if phase != nil {
		fmt.Fprintf(os.Stderr, "\n---- decode phase breakdown (%d steps, bottleneck=%s) ----\n", phaseSteps, phase.Bottleneck)
		for _, ph := range phase.Phases {
			fmt.Fprintf(os.Stderr, "  %-16s %7.1f ms  %5.1f%%  (%d calls)\n", ph.Phase, ph.MS, ph.TimePct, ph.Calls)
		}
	}
}

// rooflineVerdict turns the measured bandwidth utilization into the decode-gap call: at the
// ceiling the kernel is doing its job and the only lever left is streaming fewer bytes (Q4);
// well under it there is real kernel / forward-path headroom to recover before reaching for
// a smaller quant.
func rooflineVerdict(utilPct float64) string {
	switch {
	case utilPct >= 80:
		return "BANDWIDTH-BOUND at the STREAM ceiling — kernel saturates memory; the lever to beat llama.cpp is streaming fewer bytes (Q4), not a faster Q8 kernel"
	case utilPct >= 55:
		return "near bandwidth-bound — modest kernel/MLP headroom; check the phase breakdown for non-matmul (attn/overhead) share before a kernel rewrite"
	default:
		return "UNDER the ceiling — real headroom; the gap is kernel memory-level-parallelism and/or forward-path overhead, not the byte count (see phase breakdown)"
	}
}

func envIntDefault(name string, def int) int {
	if s := os.Getenv(name); s != "" {
		var n int
		if _, err := fmt.Sscanf(s, "%d", &n); err == nil && n > 0 {
			return n
		}
	}
	return def
}
