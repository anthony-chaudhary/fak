// Command sessionbench measures the NET VALUE-ADD of the fused agent kernel on a
// realistic long, multi-agent session — the regime the whole fusion exists for, and the
// one a per-call / per-turn naive setup pays for dearly.
//
// THE WORKLOAD. C concurrent agents share one long prefix (system prompt + tool schemas,
// P tokens). Each runs T turns; a turn decodes D assistant tokens, then (between turns)
// ingests R private tool-result tokens. So an agent's context grows P → P + T·(D+R).
// This is the "50+ turns, 5+ agents" shape: a big shared preamble, short answers, and a
// per-agent context that grows every turn.
//
// THREE ARMS — same model, same bit-identical kernels, so the delta is PURE work-reuse,
// never a numerics shortcut (the f32 batched/clone paths are proven == serial Step):
//
//	A. naive-stateless  — the common local pattern (call a stateless API / `llama-cli -p
//	   <full prompt>` each turn). Every (agent,turn) re-prefills the ENTIRE context so far
//	   and decodes serially. Prefill work is QUADRATIC in T (and the prefill's own
//	   attention is quadratic in context length) and ×C. No KV persistence, no batching.
//	B. per-agent-KV     — a careful single-tenant setup (prompt cache / persistent KV per
//	   agent, but no cross-agent sharing and no batching). Prefix prefilled C times;
//	   incremental result ingestion; serial decode. Turn-tax eliminated, still single-tenant.
//	C. fak fused        — prefix prefilled ONCE and cloned into C agents; batched decode
//	   (one weight stream serves all C); incremental result ingestion. Per-agent KV is
//	   preserved (Evict/Clone still work), which a shared-slot engine cannot offer.
//
// HEADLINE = B/C — fak vs a WARM per-agent-KV cache (arm B: prompt cache / persistent KV per
// agent, the real serving baseline vLLM / SGLang / provider prompt-caching give you). That is
// the honest number: fak's cross-agent prefix sharing ON TOP of a warm cache. A/C — fak vs the
// COLD naive re-prefill pattern (arm A) — is a worst-case REFERENCE only, NOT a serving
// baseline anyone ships. The two decompose: A→B is the turn-tax (KV persistence vs re-prefill);
// B→C is prefix reuse (P once not ×C) + decode batching (one weight stream serves C).
//
// METHODOLOGY (live where tractable, measured-rate where not — never modeled). Arms B and
// C are run END-TO-END LIVE: every prefill and decode is a real kernel call, so attention's
// growth with context length is captured exactly (the per-turn cost rises through the
// session, as it must). Arm A's re-prefill is QUADRATIC in T and intractable to run at 50×5
// (~hours); it is computed from prefillCost(L) — prefill wall-clock MEASURED at sampled
// lengths spanning the session (so the prefill's own O(L^2) attention term is captured, not
// assumed linear) — summed over the EXACT per-turn context lengths, plus arm A's decode,
// which is byte-identical serial work to arm B's (so A_decode := B's measured live decode).
// A `-validate` run executes arm A fully live at a small scale and confirms the computed
// arm-A wall-clock matches. fak's incremental prefill is charged live (no discount).
//
// Usage:
//
//	sessionbench -hf <snapshot> -lean -prefix 2048 -turns 50 -agents 5 \
//	  -decode 32 -result 64 -out experiments/.../session-qwen.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/intlist"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// bestLiveB / bestLiveC run the live arms `reps` times and keep the least-contended (min)
// wall-clock per component — the same least-contended-sample methodology the model-baseline
// docs use for the bandwidth-sensitive decode numbers, so a transient fleet-load spike on one
// rep cannot inflate an arm and skew the ratio.
func bestLiveB(m *model.Model, quant, vocab, P, T, C, D, R, reps int) (px, inc, dc float64) {
	px, inc, dc = math.MaxFloat64, math.MaxFloat64, math.MaxFloat64
	for r := 0; r < reps; r++ {
		p, i, d := liveB(m, quant, vocab, P, T, C, D, R)
		px, inc, dc = math.Min(px, p), math.Min(inc, i), math.Min(dc, d)
		runtime.GC()
	}
	return
}

func bestLiveC(m *model.Model, quant, vocab, P, T, C, D, R, reps int) (pf, cl, dc float64) {
	pf, cl, dc = math.MaxFloat64, math.MaxFloat64, math.MaxFloat64
	for r := 0; r < reps; r++ {
		p, c, d := liveC(m, quant, vocab, P, T, C, D, R)
		pf, cl, dc = math.Min(pf, p), math.Min(cl, c), math.Min(dc, d)
		runtime.GC()
	}
	return
}

// ---- model loading (mirrors cmd/modelbench) ----

func readHFConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

// syntheticShape maps a named model size to its HF Config, so sessionbench can run the
// 3-arm value-stack on a box with NO HuggingFace export (no -hf/-dir/-lean). model.NewSynthetic
// fills the layout with deterministic random weights; the logits are meaningless but the
// throughput is FAITHFUL — the matmul/attention/quant work is weight-VALUE-independent (the same
// rationale internal/model/synthetic_perf_test.go uses to report tok/s on a weightless box). So
// the work-elimination ratios (B/C, A/C) and batched-vs-serial decode are measured on the real
// kernel at the real model shape; only the absolute wall-clock is this-box, not the target host.
func syntheticShape(name string) (model.Config, bool) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "tiny":
		// A WIRING shape (the radixbench synthetic-llama: 64h/4L/8q-2kv, HeadDim 8,
		// vocab 256). The 135M+ shapes run the live B/C arms end-to-end in f32, which
		// dominates CPU wall-clock and times out unattended nightrun (#967). At this
		// shape the full live arm finishes in seconds, so the work-elimination ratios
		// are measurable on a no-GPU box; only the absolute numbers are this-box.
		return model.Config{
			HiddenSize: 64, NumLayers: 4, NumHeads: 8, NumKVHeads: 2, HeadDim: 8,
			IntermediateSize: 128, VocabSize: 256, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 255,
			TieWordEmbeddings: true, HiddenAct: "silu", ModelType: "llama",
		}, true
	case "smollm2-135m", "135m", "smollm2":
		return model.Config{
			HiddenSize: 576, NumLayers: 30, NumHeads: 9, NumKVHeads: 3, HeadDim: 64,
			IntermediateSize: 1536, VocabSize: 49152, RMSNormEps: 1e-5, RopeTheta: 10000,
			TieWordEmbeddings: true, HiddenAct: "silu", ModelType: "llama",
		}, true
	case "qwen25-1.5b", "1.5b", "qwen25-1_5b", "qwen2.5-1.5b":
		return model.Config{
			HiddenSize: 1536, NumLayers: 28, NumHeads: 12, NumKVHeads: 2, HeadDim: 128,
			IntermediateSize: 8960, VocabSize: 151936, RMSNormEps: 1e-6, RopeTheta: 1000000,
			TieWordEmbeddings: true, EOSTokenID: 151643, HiddenAct: "silu", ModelType: "qwen2",
		}, true
	case "qwen25-7b", "7b", "qwen2.5-7b":
		return model.Config{
			HiddenSize: 3584, NumLayers: 28, NumHeads: 28, NumKVHeads: 4, HeadDim: 128,
			IntermediateSize: 18944, VocabSize: 152064, RMSNormEps: 1e-6, RopeTheta: 1000000,
			TieWordEmbeddings: false, EOSTokenID: 151643, HiddenAct: "silu", ModelType: "qwen2",
		}, true
	}
	return model.Config{}, false
}

func loadModel(dir, hf, synthetic string, lean bool) (*model.Model, string, error) {
	switch {
	case synthetic != "":
		cfg, ok := syntheticShape(synthetic)
		if !ok {
			return nil, "", fmt.Errorf("unknown -synthetic shape %q (tiny|smollm2-135m|qwen25-1.5b|qwen25-7b)", synthetic)
		}
		return model.NewSynthetic(cfg), synthetic + " [synthetic]", nil
	case lean:
		if hf == "" {
			return nil, "", fmt.Errorf("-lean requires -hf")
		}
		cfg, err := readHFConfig(hf)
		if err != nil {
			return nil, "", err
		}
		m, err := model.LoadSafetensorsQuantDir(hf, cfg)
		if err != nil {
			return nil, "", fmt.Errorf("safetensors(lean): %w", err)
		}
		return m, filepath.Base(hf) + " [lean]", nil
	case hf != "":
		cfg, err := readHFConfig(hf)
		if err != nil {
			return nil, "", err
		}
		m, err := model.LoadSafetensors(filepath.Join(hf, "model.safetensors"), cfg)
		if err != nil {
			return nil, "", fmt.Errorf("safetensors: %w", err)
		}
		return m, filepath.Base(hf), nil
	default:
		m, err := model.Load(dir)
		return m, filepath.Base(dir), err
	}
}

func lcgIDs(n, vocab int, seed uint64) []int {
	ids := make([]int, n)
	state := 2463534242 + seed
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

func minDur(ds []time.Duration) time.Duration {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return cp[0]
}
func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

// ---- prefill cost samples: prefill wall-clock measured at sampled lengths ----
// Captures the prefill's O(L) weight/compute term AND its O(L^2) attention term, so the
// quadratic re-prefill of arm A can be summed over exact context lengths without running it.

type prefillModel struct {
	Lens []int     `json:"lens"`
	MS   []float64 `json:"ms"` // best-of-reps prefill wall-clock at each Lens[i]
}

func measurePrefill(m *model.Model, quant, vocab int, lens []int, reps int) prefillModel {
	pm := prefillModel{}
	for _, L := range lens {
		ids := lcgIDs(L, vocab, uint64(1000+L))
		var ds []time.Duration
		for r := 0; r < reps; r++ {
			s := m.NewSession()
			s.Quant = quant != 0
			t0 := time.Now()
			s.Prefill(ids)
			ds = append(ds, time.Since(t0))
		}
		pm.Lens = append(pm.Lens, L)
		pm.MS = append(pm.MS, ms(minDur(ds)))
	}
	return pm
}

// cost returns the prefill wall-clock for a context of L tokens by piecewise-linear
// interpolation between samples (linear extrapolation from the top two beyond the range —
// avoided in practice by sampling up to the session's max context).
func (pm prefillModel) cost(L int) float64 {
	n := len(pm.Lens)
	if n == 0 {
		return 0
	}
	if L <= pm.Lens[0] {
		return pm.MS[0] * float64(L) / float64(pm.Lens[0])
	}
	for i := 1; i < n; i++ {
		if L <= pm.Lens[i] {
			lo, hi := pm.Lens[i-1], pm.Lens[i]
			frac := float64(L-lo) / float64(hi-lo)
			return pm.MS[i-1] + frac*(pm.MS[i]-pm.MS[i-1])
		}
	}
	// extrapolate from the top two
	lo, hi := pm.Lens[n-2], pm.Lens[n-1]
	slope := (pm.MS[n-1] - pm.MS[n-2]) / float64(hi-lo)
	return pm.MS[n-1] + slope*float64(L-hi)
}

// ---- live arms (every prefill/decode a real kernel call) ----

// liveB runs arm B (per-agent persistent KV, serial decode). It splits the PREFIX prefill
// (paid once per agent — the live anchor for prefillCost(P)) from the incremental result
// prefill and from decode, so the caller can re-scale the sampled prefill model to arm B's
// live timebase (removing cross-time contention drift in the computed arm-A projection).
func liveB(m *model.Model, quant, vocab, P, T, C, D, R int) (prefixMS, incMS, decodeMS float64) {
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)
	var px, inc, dc time.Duration
	for a := 0; a < C; a++ {
		s := m.NewSession()
		s.Quant = quant != 0
		t0 := time.Now()
		s.Prefill(prefix)
		px += time.Since(t0)
		tok := ids0[a]
		for t := 0; t < T; t++ {
			t1 := time.Now()
			for d := 0; d < D; d++ {
				s.Step(tok)
				tok = (tok*48271 + 1) % vocab
			}
			dc += time.Since(t1)
			if t < T-1 {
				rr := lcgIDs(R, vocab, uint64(50000+t*1000+a*97))
				t2 := time.Now()
				s.Prefill(rr)
				inc += time.Since(t2)
			}
		}
	}
	return ms(px), ms(inc), ms(dc)
}

// liveC runs arm C (fak fused: prefix once + clone + batched decode + incremental).
func liveC(m *model.Model, quant, vocab, P, T, C, D, R int) (prefillMS, cloneMS, decodeMS float64) {
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)
	var pf, cl, dc time.Duration

	base := m.NewSession()
	base.Quant = quant != 0
	t0 := time.Now()
	base.Prefill(prefix)
	pf += time.Since(t0)

	t1 := time.Now()
	bs := m.NewBatchFromPrefix(base.Cache, C)
	cl += time.Since(t1)
	bs.SetQuant(quant != 0)

	ids := append([]int(nil), ids0...)
	for t := 0; t < T; t++ {
		t2 := time.Now()
		for d := 0; d < D; d++ {
			bs.StepBatch(ids)
			for j := range ids {
				ids[j] = (ids[j]*48271 + 1) % vocab
			}
		}
		dc += time.Since(t2)
		if t < T-1 {
			prompts := make([][]int, C)
			for a := range prompts {
				prompts[a] = lcgIDs(R, vocab, uint64(50000+t*1000+a*97))
			}
			t3 := time.Now()
			bs.PrefillEach(prompts)
			pf += time.Since(t3)
		}
	}
	return ms(pf), ms(cl), ms(dc)
}

// liveA runs arm A fully (only for small-scale validation — quadratic re-prefill).
func liveA(m *model.Model, quant, vocab, P, T, C, D, R int) (prefillMS, decodeMS float64) {
	prefix := lcgIDs(P, vocab, 1)
	ids0 := lcgIDs(C, vocab, 991)
	var pf, dc time.Duration
	for a := 0; a < C; a++ {
		ctx := append([]int(nil), prefix...)
		tok := ids0[a]
		for t := 0; t < T; t++ {
			s := m.NewSession()
			s.Quant = quant != 0
			t0 := time.Now()
			s.Prefill(ctx) // re-prefill the WHOLE context so far
			pf += time.Since(t0)
			t1 := time.Now()
			for d := 0; d < D; d++ {
				s.Step(tok)
				ctx = append(ctx, tok)
				tok = (tok*48271 + 1) % vocab
			}
			dc += time.Since(t1)
			if t < T-1 {
				ctx = append(ctx, lcgIDs(R, vocab, uint64(50000+t*1000+a*97))...)
			}
		}
	}
	return ms(pf), ms(dc)
}

// computeAPrefill sums prefillCost over arm A's exact per-turn context lengths, ×C agents.
func computeAPrefill(pm prefillModel, P, T, C, D, R int) float64 {
	var total float64
	for t := 0; t < T; t++ {
		ctx := P + t*(D+R) // context length at the start of turn t (0-indexed)
		total += pm.cost(ctx)
	}
	return total * float64(C)
}

// prefillTokens returns the EXACT prefill-token counts each arm processes — pure arithmetic
// from the session structure, independent of any timing, so the work-elimination ratio it
// yields is CONTENTION-FREE (the deterministic floor under the measured time ratio).
//
//	A (naive):     C · Σ_{t=0..T-1}(P + t·(D+R))   — re-prefill the whole context every turn
//	B (per-agent): C · (P + (T-1)·R)                — prefix once per agent + incremental
//	C (fak):           P + C·(T-1)·R                — prefix ONCE total + incremental
func prefillTokens(P, T, C, D, R int) (a, b, c int) {
	for t := 0; t < T; t++ {
		a += P + t*(D+R)
	}
	a *= C
	b = C * (P + (T-1)*R)
	c = P + C*(T-1)*R
	return
}

func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	cp := append([]float64(nil), xs...)
	sort.Float64s(cp)
	return cp[len(cp)/2]
}

// ---- per-cell result ----

type arm struct {
	Name      string  `json:"name"`
	PrefillMS float64 `json:"prefill_ms"`
	CloneMS   float64 `json:"clone_ms,omitempty"`
	DecodeMS  float64 `json:"decode_ms"`
	TotalMS   float64 `json:"total_ms"`
	Live      bool    `json:"live"` // true if run end-to-end; false if prefill computed from samples
}

// prefillTokCounts are the EXACT prefill-token counts each arm processes (from prefillTokens) —
// pure arithmetic, independent of any timing, so the work-elimination ratios are CONTENTION-FREE:
// the deterministic floor under the measured wall-clock ratios. On a busy fleet box the timed
// A/C can drift; a_over_c / b_over_c cannot — they are fixed by the session structure alone.
type prefillTokCounts struct {
	A      int     `json:"a"`        // naive: C·Σ(P+t·(D+R)) — re-prefill whole context every turn
	B      int     `json:"b"`        // per-agent: C·(P+(T-1)·R) — prefix ×C + incremental
	C      int     `json:"c"`        // fak: P+C·(T-1)·R — prefix ONCE total + incremental
	AOverC float64 `json:"a_over_c"` // exact prefill work-elimination vs naive (timing-free)
	BOverC float64 `json:"b_over_c"` // exact prefill work-elimination vs tuned single-tenant
}

type cell struct {
	Turns       int              `json:"turns"`
	Agents      int              `json:"agents"`
	Prefix      int              `json:"prefix"`
	Decode      int              `json:"decode"`
	Result      int              `json:"result"`
	A           arm              `json:"arm_A_naive_stateless"`
	B           arm              `json:"arm_B_per_agent_kv"`
	C           arm              `json:"arm_C_fak_fused"`
	Anchor      float64          `json:"prefill_anchor"`         // live prefix-prefill / sampled prefillCost(P)
	APrefillRaw float64          `json:"arm_A_prefill_raw_ms"`   // before anchor correction
	PrefillTok  prefillTokCounts `json:"prefill_tokens"`         // EXACT token counts — contention-free floor under the timed ratios
	NetVsNaive  float64          `json:"net_value_add_vs_naive"` // A/C — fak vs COLD no-cache re-prefill (worst-case REFERENCE only, anchored)
	NetVsTuned  float64          `json:"net_value_add_vs_tuned"` // B/C — HEADLINE: fak vs a WARM per-agent KV cache (the serving baseline)
	TurnTax     float64          `json:"turn_tax_A_over_B"`      // A/B
}

func main() {
	dir := flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir (-dir mode)")
	hf := flag.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors)")
	synthetic := flag.String("synthetic", "", "run weightless on a synthetic model at a named shape (tiny|smollm2-135m|qwen25-1.5b|qwen25-7b) — no -hf/-dir needed; ratios faithful, absolute wall-clock is this-box; tiny is the CPU-tractable wiring shape for unattended nightrun")
	lean := flag.Bool("lean", false, "memory-lean quantize-at-load (requires -hf; implies -quant)")
	quantF := flag.Bool("quant", false, "use the Q8_0 quantized lane (else f32) — opt-in, matching batch/model/radixbench")
	prefix := flag.Int("prefix", 2048, "shared prefix tokens (system prompt + tool schemas)")
	turnsArg := flag.String("turns", "50", "comma-separated turn counts to sweep")
	agentsArg := flag.String("agents", "5", "comma-separated agent counts to sweep")
	decode := flag.Int("decode", 32, "assistant tokens decoded per turn")
	result := flag.Int("result", 64, "tool-result tokens ingested per turn")
	reps := flag.Int("reps", 2, "best-of-N (min) wall-clock per arm component — least-contended sampling")
	validate := flag.Bool("validate", true, "run arm A fully live at a small scale to anchor the computed arm-A wall-clock")
	valScale := flag.String("val-scale", "256,8,3,16,32", "live-validate P,T,C,D,R")
	countsOnly := flag.Bool("counts-only", false, "emit the deterministic prefill-token floor only; no live timing, no model load")
	out := flag.String("out", "", "write JSON here (default stdout)")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)
	*hf = pathutil.ExpandTilde(*hf)

	quant := 0
	if *quantF || *lean {
		quant = 1
	}
	turns := intlist.Parse(*turnsArg)
	agents := intlist.Parse(*agentsArg)
	if *countsOnly {
		if strings.TrimSpace(*synthetic) == "" {
			fmt.Fprintln(os.Stderr, "-counts-only requires -synthetic so the report identity is explicit")
			os.Exit(2)
		}
		if _, ok := syntheticShape(*synthetic); !ok {
			fmt.Fprintf(os.Stderr, "unknown -synthetic shape %q (tiny|smollm2-135m|qwen25-1.5b|qwen25-7b)\n", *synthetic)
			os.Exit(2)
		}
		writeSessionReport(deterministicReport(*synthetic+" [synthetic]", quant != 0, turns, agents, *prefix, *decode, *result), *out)
		return
	}

	m, name, err := loadModel(*dir, *hf, *synthetic, *lean)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	if quant != 0 {
		m.Quantize()
	}
	vocab := m.Cfg.VocabSize
	// warm
	ws := m.NewSession()
	ws.Quant = quant != 0
	ws.Prefill(lcgIDs(8, vocab, 77))
	ws.Step(1)

	// prefill samples spanning the session's context range (max ctx = P + maxT*(D+R))
	maxT := 0
	for _, t := range turns {
		if t > maxT {
			maxT = t
		}
	}
	maxCtx := *prefix + maxT*(*decode+*result)
	lens := sampleLens(*prefix, maxCtx)
	fmt.Fprintf(os.Stderr, "measuring prefill cost at lengths %v ...\n", lens)
	pm := measurePrefill(m, quant, vocab, lens, 2)
	for i, L := range pm.Lens {
		fmt.Fprintf(os.Stderr, "  prefill(%d) = %.0f ms (%.1f tok/s)\n", L, pm.MS[i], float64(L)/(pm.MS[i]/1e3))
	}

	var cells []cell
	for _, T := range turns {
		for _, C := range agents {
			if T < 1 || C < 1 {
				continue
			}
			fmt.Fprintf(os.Stderr, "cell T=%d C=%d P=%d: running arms B,C live (best of %d) ...\n", T, C, *prefix, *reps)
			bPx, bInc, bDc := bestLiveB(m, quant, vocab, *prefix, T, C, *decode, *result, *reps)
			cPf, cCl, cDc := bestLiveC(m, quant, vocab, *prefix, T, C, *decode, *result, *reps)
			bPf := bPx + bInc
			// Anchor the sampled prefill model to arm B's LIVE prefix prefill (same time window
			// as arms B,C) so contention drift between the sampling phase and the arm runs does
			// not bias the computed arm-A projection. anchor=1 means sampling-time load == run-time.
			anchor := 1.0
			if base := pm.cost(*prefix); base > 0 && C > 0 {
				anchor = (bPx / float64(C)) / base
			}
			aPfRaw := computeAPrefill(pm, *prefix, T, C, *decode, *result)
			aPf := aPfRaw * anchor
			aDc := bDc // arm A decode is byte-identical serial work to arm B's

			a := arm{Name: "A_naive_stateless", PrefillMS: aPf, DecodeMS: aDc, TotalMS: aPf + aDc, Live: false}
			b := arm{Name: "B_per_agent_kv", PrefillMS: bPf, DecodeMS: bDc, TotalMS: bPf + bDc, Live: true}
			cc := arm{Name: "C_fak_fused", PrefillMS: cPf, CloneMS: cCl, DecodeMS: cDc, TotalMS: cPf + cCl + cDc, Live: true}
			// Exact, timing-free prefill work-elimination — the contention-immune floor under the
			// measured ratios above (cannot drift with fleet load; fixed by the session structure).
			ta, tb, tc := prefillTokens(*prefix, T, C, *decode, *result)
			ptok := prefillTokCounts{A: ta, B: tb, C: tc}
			if tc > 0 {
				ptok.AOverC = float64(ta) / float64(tc)
				ptok.BOverC = float64(tb) / float64(tc)
			}
			cl := cell{
				Turns: T, Agents: C, Prefix: *prefix, Decode: *decode, Result: *result,
				A: a, B: b, C: cc, Anchor: anchor, APrefillRaw: aPfRaw, PrefillTok: ptok,
				NetVsNaive: a.TotalMS / cc.TotalMS,
				NetVsTuned: b.TotalMS / cc.TotalMS,
				TurnTax:    a.TotalMS / b.TotalMS,
			}
			cells = append(cells, cl)
			fmt.Fprintf(os.Stderr,
				"  C_fak %.1fs  B_warmKV %.1fs  A_nocache %.1fs (pf %.0f[anchor %.2f] + dc %.0f) | HEADLINE fak vs WARM per-agent KV %.2f×  (turn-tax %.1f×) | no-cache ref %.1f× | exact prefill-tok B/C %.1f× (warmKV)  A/C %.1f× (no-cache ref)\n",
				cc.TotalMS/1e3, b.TotalMS/1e3, a.TotalMS/1e3, aPf, anchor, aDc, cl.NetVsTuned, cl.TurnTax, cl.NetVsNaive, ptok.BOverC, ptok.AOverC)
			runtime.GC()
		}
	}

	var liveVal map[string]any
	if *validate {
		vs := intlist.Parse(*valScale)
		for len(vs) < 5 {
			vs = append(vs, 1)
		}
		P, T, C, D, R := vs[0], vs[1], vs[2], vs[3], vs[4]
		fmt.Fprintf(os.Stderr, "validate: arm A FULLY LIVE at P=%d T=%d C=%d D=%d R=%d (best of %d) ...\n", P, T, C, D, R, *reps)
		aPfLive, aDcLive := math.MaxFloat64, math.MaxFloat64
		for r := 0; r < *reps; r++ {
			p, d := liveA(m, quant, vocab, P, T, C, D, R)
			aPfLive, aDcLive = math.Min(aPfLive, p), math.Min(aDcLive, d)
			runtime.GC()
		}
		bPx, _, bDcSmall := bestLiveB(m, quant, vocab, P, T, C, D, R, *reps)
		aPfComputedRaw := computeAPrefill(pm, P, T, C, D, R)
		anchor := 1.0
		if base := pm.cost(P); base > 0 && C > 0 {
			anchor = (bPx / float64(C)) / base
		}
		aPfComputedAnchored := aPfComputedRaw * anchor
		liveVal = map[string]any{
			"scale":                             map[string]int{"P": P, "T": T, "C": C, "D": D, "R": R},
			"armA_prefill_live_ms":              aPfLive,
			"armA_prefill_computed_raw_ms":      aPfComputedRaw,
			"prefill_anchor":                    anchor,
			"armA_prefill_computed_anchored_ms": aPfComputedAnchored,
			"anchored_computed_over_live":       aPfComputedAnchored / aPfLive,
			"raw_computed_over_live":            aPfComputedRaw / aPfLive,
			"armA_decode_live_ms":               aDcLive,
			"armB_decode_live_ms":               bDcSmall,
			"decode_A_over_B":                   aDcLive / bDcSmall,
		}
		fmt.Fprintf(os.Stderr,
			"  arm-A prefill: live %.0f ms | computed raw %.0f (raw/live %.2f) | anchor %.2f → anchored %.0f (anchored/live %.2f); decode A/B %.2f\n",
			aPfLive, aPfComputedRaw, aPfComputedRaw/aPfLive, anchor, aPfComputedAnchored, aPfComputedAnchored/aPfLive, aDcLive/bDcSmall)
	}

	report := map[string]any{
		"app_version": appversion.Current(),
		"engine":      "fak sessionbench (multi-agent session value stack, Q8=" + boolStr(quant != 0) + ")",
		"model":       name,
		"timing_mode": "live_BC_sampled_A",
		"go_threads":  runtime.GOMAXPROCS(0),
		"headline": "fak vs a WARM per-agent-KV cache (B/C = net_value_add_vs_tuned) — the honest serving baseline; " +
			"fak's cross-agent prefix sharing on top of a warm cache. The cold no-cache re-prefill arm (A, net_value_add_vs_naive) " +
			"is a worst-case REFERENCE only, not a serving baseline.",
		"methodology": "arms B,C run end-to-end LIVE (attention growth captured); arm A re-prefill computed " +
			"from prefillCost(L) measured at sampled lengths (O(L^2) prefill-attention captured) summed over exact " +
			"per-turn contexts; arm A decode := arm B live decode (byte-identical serial work); validate runs arm A fully live",
		"prefill_model":   pm,
		"prefix":          *prefix,
		"decode_per_turn": *decode,
		"result_per_turn": *result,
		"cells":           cells,
		"live_validate":   liveVal,
	}
	writeSessionReport(report, *out)
}

func writeSessionReport(report map[string]any, out string) {
	blob, _ := json.MarshalIndent(report, "", "  ")
	if out != "" {
		if err := os.WriteFile(out, blob, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write %s: %v\n", out, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "wrote %s\n", out)
	} else {
		fmt.Println(string(blob))
	}
}

func deterministicReport(name string, quant bool, turns, agents []int, prefix, decode, result int) map[string]any {
	var cells []cell
	for _, T := range turns {
		for _, C := range agents {
			if T < 1 || C < 1 {
				continue
			}
			ta, tb, tc := prefillTokens(prefix, T, C, decode, result)
			ptok := prefillTokCounts{A: ta, B: tb, C: tc}
			if tc > 0 {
				ptok.AOverC = float64(ta) / float64(tc)
				ptok.BOverC = float64(tb) / float64(tc)
			}
			turnTax := 0.0
			if tb > 0 {
				turnTax = float64(ta) / float64(tb)
			}
			cells = append(cells, cell{
				Turns: T, Agents: C, Prefix: prefix, Decode: decode, Result: result,
				A:          arm{Name: "A_naive_stateless", Live: false},
				B:          arm{Name: "B_per_agent_kv", Live: false},
				C:          arm{Name: "C_fak_fused", Live: false},
				PrefillTok: ptok,
				NetVsNaive: ptok.AOverC,
				NetVsTuned: ptok.BOverC,
				TurnTax:    turnTax,
			})
		}
	}
	return map[string]any{
		"app_version": appversion.Current(),
		"engine":      "fak sessionbench (multi-agent session value stack, Q8=" + boolStr(quant) + ", counts-only=true)",
		"model":       name,
		"timing_mode": "deterministic_prefill_token_counts_only",
		"headline": "deterministic prefill-token floor only; no wall-clock, no live B/C timing, and no model-quality claim. " +
			"Use the live mode for measured timing; this offline floor preserves the exact A/B/C token geometry.",
		"methodology": "counts-only mode uses prefillTokens(P,T,C,D,R): A=C*Σ(P+t*(D+R)), B=C*(P+(T-1)*R), C=P+C*(T-1)*R. " +
			"Ratios are exact token-work ratios, not measured seconds.",
		"prefix":          prefix,
		"decode_per_turn": decode,
		"result_per_turn": result,
		"cells":           cells,
		"live_validate":   nil,
	}
}

// sampleLens picks ~6 lengths spanning [256, maxCtx] (and includes prefix) for the prefill model.
func sampleLens(prefix, maxCtx int) []int {
	set := map[int]bool{}
	add := func(x int) {
		if x >= 16 {
			set[x] = true
		}
	}
	add(256)
	add(prefix)
	if maxCtx > 256 {
		for _, f := range []float64{0.25, 0.5, 0.75, 1.0} {
			add(256 + int(f*float64(maxCtx-256)))
		}
	}
	var out []int
	for k := range set {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
