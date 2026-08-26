// Command gdn-quant-length-sensitivity is the host-runnable, device-independent arm of issue
// #4273 — Qwen3.6-27B GGUF degenerates into a verbatim repetition loop on ~1.3k-token prompts
// while short prompts stay coherent. It is the length-axis sibling of gdn-divergence-sensitivity
// (which answers the *token-3 correctness* drift on the depth axis).
//
// The phenomenon (#4273, from the on-device real-artifact witnesses q36rawrepro / q36safe18 / q36tokenloop7):
// the completion opens coherent and grounded, then degrades ~40 tokens into decode and cycles to
// the token cap. It has been narrowed, by real-artifact witnesses, to LONG-CONTEXT REAL QUANTIZED
// inference — and specifically NOT to: the tokenizer/chat-template (HF tokenization byte-identical),
// generic f32 Gated-DeltaNet (GDN) recurrence (the Ornith oracle matches HF greedy at short and
// 300-token horizons), batched-vs-token-loop prefill (both collapse identically), recurrent state
// carry (the long-horizon self-consistency guard is green), or universal model corruption (`2+2=4`
// decodes cleanly at short context). Short coherent + long collapse + f32-fine + quant-fails is the
// signature this program measures.
//
// The variable it isolates: WEIGHT QUANTIZATION of the GDN projections. In the real loader
// (internal/model/safetensors_quant.go:isQuantWeight) all five linear_attn matmul weights —
// in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj — are quantized to Q8_0 at compute time
// in BOTH failing paths: the default lean-Q8 path quantizes everything, and the resident-Q4_K path
// refuses linear_attn from raw-Q4_K residency (quant_q4k.go:ResidentQ4KEligible returns false for
// .linear_attn.) then re-quantizes the dequant'd f32 to Q8 via the builder. That is exactly why the
// collapse is quant-INDEPENDENT (Q8 and Q4_K both fail): the GDN path runs Q8 in both. The coherent
// Ornith oracle, by contrast, is full f32. So the one difference between "coherent" and "collapses"
// on the GDN projections is weight quantization — the perturbation this program applies.
//
// It answers a falsifiable question with no Mac, no GPU, and no 27B artifact:
//
//	Does GDN-projection weight-quantization error, fed through the delta-rule recurrent scan,
//	COMPOUND with decode length — i.e. does the relative hidden divergence rho = ||Δh||/||h||
//	between the f32 and the Q8/Q4_K run GROW with the number of carried positions P, reaching a
//	decode-destabilizing magnitude by the ~1757-token failure horizon?
//
// Two outcomes, both useful and honest:
//   - rho grows with P and reaches a large magnitude (>= rho* ~ 0.09, the order at which the argmax
//     routinely moves) near P~1757: weight-quant magnitude, compounded over the recurrence, is a
//     SUFFICIENT mechanism for the length-onset collapse -> the fix direction is to keep the GDN
//     projections (and/or the recurrent state) in higher precision, as the reference runtimes do.
//   - rho stays small and roughly FLAT in P (like the f16-state bracket's ~3e-5 in the sibling):
//     weight-quant magnitude ALONE cannot explain #4273 -> the quantized long-context path has an
//     ALGORITHMIC defect (dequant/scale/tensor-mapping), not a precision-magnitude one, and the
//     on-artifact early-logit comparison vs llama.cpp/HF is the decisive next witness.
//
// The GDN layer math is copied verbatim from internal/model/metal_prefill_hybrid_core.go:202-246
// (the prefill twin of qwen35.go:linearAttnStep) via the sibling gdn-divergence-sensitivity, so the
// recurrence is faithful; only the projection weights are round-tripped through Q8_0 / Q4_K before
// the f32 matmul. Weight round-trip (dequant back to f32, then f32 matmul) models the WEIGHT-quant
// contribution; the real kernel additionally quantizes the activation vector per matmul, so the real
// path's total quant error is >= what this measures — this rho is a conservative lower bound on the
// quant-induced divergence.
//
// What this does NOT do: it does not run llama.cpp, does not load the 27B artifact, uses
// seeded-random (not trained) weights so it CANNOT reproduce the trained repetition attractor
// itself, and does not claim to have found a bug or a fix. Its claim is about the ORDER OF MAGNITUDE
// and the P-SLOPE of quant-induced recurrent divergence — the device-independent bound the current
// #4273 hypothesis demands. The on-artifact witness remains the honest `not yet`.
//
// Run:
//
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity                    # human table
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity -json              # machine result
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity -positions 16,128  # quick smoke
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity -hidden 5120       # real-H confirm (slow)
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/experiments/qwen36/gdn"
)

// The Qwen3.6-27B Gated-DeltaNet layer dims (nK/nV/kHd/vHd/K and the derived keyDim/valDim/
// convDim) live in the shared gdn package. H (hidden) is a flag HERE and not a constant: rho is
// relative (||Δh||/||h||), so it is ~independent of the matmul fan-in H, which only drives cost —
// the recurrence dynamics that #4273 lives in depend on nV/kHd/vHd/K and the carried length, all
// real here. -hidden 5120 reproduces the true H for a (slow) confirmatory run.
//
// hidden is set from -hidden in main; the gdn package's derived widths are H-independent.
var hidden = 1024

// aLogMean is the mean of the per-head A_log draw, set from -alog. It controls the recurrent decay
// g = exp(-exp(A_log)*dt): mean -2 -> g~0.88 (effective memory ~8 positions); mean -5 -> g~0.99
// (long memory, ~100+ positions). It is the load-bearing knob for whether per-step error can
// compound across a long decode, so the sweep MUST include a near-1-decay run to be honest.
var aLogMean = -2.0

// quantMode selects the weight-quantization applied to the GDN projections in the test run.
type quantMode int

const (
	modeQ8  quantMode = iota // per-row, per-32 block, symmetric int8 (GGML Q8_0; quant.go:quantizeQ8)
	modeQ4K                  // per-row, per-32 affine 4-bit sub-block (GGML Q4_K sub-block; min/max)
)

// q8RowRoundTrip round-trips w[out,in] through Q8_0 (block=32, symmetric, d=amax/127) and returns
// the dequantized f32 — bit-faithful to internal/model/quant.go:quantizeQ8 + its dequant.
func q8RowRoundTrip(w []float32, out, in int) []float32 {
	const blk = 32
	dq := make([]float32, len(w))
	nblk := in / blk
	tail := in % blk // in is a multiple of 32 for real GDN dims; handle a stray tail faithlessly-but-safely
	for o := 0; o < out; o++ {
		row := w[o*in : o*in+in]
		dst := dq[o*in : o*in+in]
		for b := 0; b < nblk; b++ {
			block := row[b*blk : b*blk+blk]
			var amax float32
			for _, v := range block {
				a := v
				if a < 0 {
					a = -a
				}
				if a > amax {
					amax = a
				}
			}
			d := amax / 127
			for i := 0; i < blk; i++ {
				if d == 0 {
					dst[b*blk+i] = 0
					continue
				}
				q := int32(math.Round(float64(block[i] / d)))
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				dst[b*blk+i] = float32(q) * d
			}
		}
		for i := nblk * blk; i < nblk*blk+tail; i++ {
			dst[i] = row[i]
		}
	}
	return dq
}

// q4kRowRoundTrip round-trips w[out,in] through a per-row, per-32 affine 4-bit sub-block scheme
// (min/max, scale=(max-min)/15, 4-bit code, dequant q*scale+min). This models the GGML Q4_K
// sub-block (quant_q4k.go: 8 sub-blocks of 32 per 256 super-block); the real Q4_K additionally
// quantizes the per-sub-block (scale,min) pair to 6 bits, which only INCREASES error, so this is a
// lower bound on the Q4_K contribution.
func q4kRowRoundTrip(w []float32, out, in int) []float32 {
	const sub = 32
	dq := make([]float32, len(w))
	nsub := in / sub
	tail := in % sub
	for o := 0; o < out; o++ {
		row := w[o*in : o*in+in]
		dst := dq[o*in : o*in+in]
		for b := 0; b < nsub; b++ {
			block := row[b*sub : b*sub+sub]
			mn, mx := block[0], block[0]
			for _, v := range block {
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
			scale := (mx - mn) / 15
			for i := 0; i < sub; i++ {
				if scale == 0 {
					dst[b*sub+i] = mn
					continue
				}
				q := int32(math.Round(float64((block[i] - mn) / scale)))
				if q > 15 {
					q = 15
				} else if q < 0 {
					q = 0
				}
				dst[b*sub+i] = float32(q)*scale + mn
			}
		}
		for i := nsub * sub; i < nsub*sub+tail; i++ {
			dst[i] = row[i]
		}
	}
	return dq
}

// quantizeProjections returns a copy of lw with the five isQuantWeight projection matrices
// round-tripped through the given quant mode and the control tensors shared (f32) — exactly the
// tensor split the real loader applies.
func quantizeProjections(lw *gdn.LayerWeights, mode quantMode) *gdn.LayerWeights {
	rt := q8RowRoundTrip
	if mode == modeQ4K {
		rt = q4kRowRoundTrip
	}
	q := *lw // copy the slice headers; the control tensors are shared read-only
	q.Wqkv = rt(lw.Wqkv, gdn.ConvDim, lw.Hidden)
	q.Wz = rt(lw.Wz, gdn.ValDim, lw.Hidden)
	q.Wb = rt(lw.Wb, gdn.NV, lw.Hidden)
	q.Wa = rt(lw.Wa, gdn.NV, lw.Hidden)
	q.WOut = rt(lw.WOut, lw.Hidden, gdn.ValDim)
	return &q
}

// runStackForLength runs the 48-GDN-layer residual stack over P positions TWICE — a reference run
// with f32 weights and a test run with the same weights but the projections quantized (mode) — and
// returns the relative hidden divergence rho of the LAST token after the whole stack, plus the
// per-layer rho curve. Both runs share identical inputs and identical (pre-quantization) weights, so
// the only source of divergence is projection weight quantization.
//
// The layer itself is gdn.Layer in gdn.ModeForward: this experiment perturbs only the WEIGHTS, so
// the scan keeps the trunk's ascending-i f32 order in BOTH runs and contributes no divergence of its
// own (the depth-axis sibling gdn-divergence-sensitivity is the one that varies the scan).
func runStackForLength(P, layers int, mode quantMode) (perLayerRho []float64, finalRho float64) {
	eps := float32(1e-6)
	X := make([]float32, P*hidden)
	Y := make([]float32, P*hidden)
	r := rand.New(rand.NewSource(0xC0FFEE))
	for i := range X {
		v := float32(r.NormFloat64())
		X[i] = v
		Y[i] = v
	}
	lw := gdn.NewLayerWeights(hidden)
	last := (P - 1) * hidden
	for l := 0; l < layers; l++ {
		lw.Fill(l, aLogMean)
		qw := quantizeProjections(lw, mode)
		oRef := gdn.Layer(lw, X, P, gdn.ModeForward, eps)
		oTst := gdn.Layer(qw, Y, P, gdn.ModeForward, eps)
		for i := range X {
			X[i] += oRef[i]
			Y[i] += oTst[i]
		}
		perLayerRho = append(perLayerRho, gdn.RelDiv(X[last:last+hidden], Y[last:last+hidden]))
	}
	finalRho = gdn.RelDiv(X[last:last+hidden], Y[last:last+hidden])
	return
}

type lengthPoint struct {
	Positions int     `json:"positions"`
	FinalRho  float64 `json:"final_rho_last_token"`
	ImpliedDL float64 `json:"implied_logit_delta_at_logit_scale_20"`
}

type modeResult struct {
	Mode         string        `json:"mode"`
	Points       []lengthPoint `json:"points"`
	GrowthRatio  float64       `json:"rho_growth_ratio_maxP_over_minP"`
	MaxRho       float64       `json:"max_rho"`
	Destabilizes bool          `json:"reaches_destabilizing_rho_at_max_P"`
}

type result struct {
	Schema     string       `json:"schema"`
	Issue      string       `json:"issue"`
	Host       string       `json:"host"`
	Model      string       `json:"model"`
	Hidden     int          `json:"hidden"`
	ALogMean   float64      `json:"a_log_mean_decay_knob"`
	Layers     int          `json:"gdn_layers"`
	Positions  []int        `json:"positions_swept"`
	LogitScale float64      `json:"assumed_logit_scale"`
	RhoStar    float64      `json:"rho_destabilize_threshold"`
	Modes      []modeResult `json:"modes"`
	Verdict    string       `json:"verdict"`
}

func parsePositions(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if n < 1 {
			return nil, fmt.Errorf("position %d must be >= 1", n)
		}
		out = append(out, n)
	}
	sort.Ints(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("no positions parsed")
	}
	return out, nil
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	positionsFlag := flag.String("positions", "16,128,512,1757", "comma-separated carried-position counts to sweep (the ~1757 failure horizon by default)")
	layers := flag.Int("layers", 48, "number of GDN layers in the residual stack")
	hiddenFlag := flag.Int("hidden", 1024, "hidden size H (rho is ~H-independent; 5120 is the real 27B H, slower)")
	alogFlag := flag.Float64("alog", -2.0, "mean of the per-head A_log draw (decay strength): -2 -> g~0.88 (~8-pos memory); -5 -> g~0.99 (long memory)")
	flag.Parse()
	aLogMean = *alogFlag

	if *hiddenFlag%256 != 0 {
		fmt.Fprintf(os.Stderr, "hidden must be a multiple of 256 (Q8_0/Q4_K block alignment); got %d\n", *hiddenFlag)
		os.Exit(2)
	}
	hidden = *hiddenFlag

	positions, err := parsePositions(*positionsFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bad -positions: %v\n", err)
		os.Exit(2)
	}

	const logitScale = 20.0           // |top logit| order from the witnessed 27B decode
	const rhoStar = 1.75 / logitScale // ~0.0875: the relative hidden move at which a near-tie argmax flips

	res := result{
		Schema: "qwen36-gdn-quant-length-sensitivity/v1",
		Issue:  "#4273 (long-context quantized repetition collapse); length sibling of gdn-divergence-sensitivity",
		Host:   "windows/amd64 (CGO_ENABLED=0); device-independent, no Mac / no GPU / no 27B artifact",
		Model: fmt.Sprintf("Qwen3.6-27B GDN shapes (H=%d nK=%d nV=%d kHd=%d vHd=%d K=%d), projections round-tripped Q8_0 / Q4_K",
			hidden, gdn.NK, gdn.NV, gdn.KHd, gdn.VHd, gdn.K),
		Hidden:     hidden,
		ALogMean:   aLogMean,
		Layers:     *layers,
		Positions:  positions,
		LogitScale: logitScale,
		RhoStar:    rhoStar,
	}

	modes := []struct {
		m    quantMode
		name string
	}{
		{modeQ8, "Q8_0 (per-32 symmetric int8; the compute dtype of the GDN projections in BOTH failing paths)"},
		{modeQ4K, "Q4_K (per-32 affine 4-bit sub-block; lower bound — real Q4_K also quantizes the sub-scales)"},
	}

	for _, md := range modes {
		mr := modeResult{Mode: md.name}
		for _, P := range positions {
			_, final := runStackForLength(P, *layers, md.m)
			mr.Points = append(mr.Points, lengthPoint{Positions: P, FinalRho: final, ImpliedDL: final * logitScale})
			if final > mr.MaxRho {
				mr.MaxRho = final
			}
		}
		first := mr.Points[0].FinalRho
		lastPt := mr.Points[len(mr.Points)-1]
		if first > 0 {
			mr.GrowthRatio = lastPt.FinalRho / first
		}
		mr.Destabilizes = lastPt.FinalRho >= rhoStar
		res.Modes = append(res.Modes, mr)
	}

	// Verdict from whether either mode's rho GROWS with P and REACHES the destabilizing order at the
	// failure horizon. Both outcomes route the investigation (see file header).
	anyDestab, anyGrowth := false, false
	for _, m := range res.Modes {
		if m.Destabilizes {
			anyDestab = true
		}
		if m.GrowthRatio >= 3 {
			anyGrowth = true
		}
	}
	switch {
	case anyDestab:
		res.Verdict = "SUFFICIENT: GDN-projection weight-quantization error, compounded through the delta-rule " +
			"recurrence, reaches a decode-destabilizing relative hidden divergence (rho >= rho*) by the ~1757-token " +
			"horizon. Weight-quant magnitude is a plausible mechanism for #4273's length-onset collapse -> fix " +
			"direction: keep the GDN projections (and/or recurrent state) in higher precision, as llama.cpp/HF do."
	case anyGrowth:
		res.Verdict = "PARTIAL: quant-induced rho GROWS materially with decode length (>=3x from min to max P) but " +
			"stays below the near-tie flip order at the horizon. Compounding is real and length-dependent (consistent " +
			"with #4273's onset) but its magnitude alone is sub-threshold on random weights; a trained repetition " +
			"attractor could amplify it. The on-artifact early-logit comparison remains the decisive witness."
	default:
		clauses := []string{
			"INSUFFICIENT: quant-induced rho stays small and roughly FLAT in decode length, without material compounding.",
			"Weight-quantization magnitude alone cannot explain #4273; the quantized long-context path has an algorithmic dequant, scale, or tensor-mapping defect rather than a precision-magnitude one.",
			"The decisive next witness is the on-artifact early token/logit comparison against llama.cpp or HF on the real 27B Q4_K_M.",
		}
		res.Verdict = gdn.Verdict(clauses...)
	}

	if *asJSON {
		_ = gdn.EmitJSON(os.Stdout, res)
		return
	}

	fmt.Printf("Qwen3.6-27B GDN weight-quant / decode-length sensitivity — #4273 (device-independent)\n")
	fmt.Printf("  shapes: H=%d nK=%d nV=%d kHd=%d vHd=%d convDim=%d valDim=%d K=%d, %d GDN layers\n",
		hidden, gdn.NK, gdn.NV, gdn.KHd, gdn.VHd, gdn.ConvDim, gdn.ValDim, gdn.K, *layers)
	fmt.Printf("  swept carried positions: %v   (rho* to flip a near-tie ~ %.4f at |logit|~%.0f)\n\n", positions, rhoStar, logitScale)
	for _, m := range res.Modes {
		fmt.Printf("  mode: %s\n", m.Mode)
		fmt.Printf("    %-10s %-16s %-16s\n", "positions", "final_rho", "implied|Δlogit|")
		for _, p := range m.Points {
			fmt.Printf("    %-10d %-16.4e %-16.4e\n", p.Positions, p.FinalRho, p.ImpliedDL)
		}
		fmt.Printf("    growth rho(maxP)/rho(minP) = %.2fx ; max rho = %.4e ; destabilizes at horizon: %v\n\n",
			m.GrowthRatio, m.MaxRho, m.Destabilizes)
	}
	fmt.Printf("  VERDICT: %s\n", res.Verdict)
}
