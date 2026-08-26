// Command gdn-divergence-sensitivity is the host-runnable, device-independent arm of the
// Qwen3.6-27B *correctness* parity blocker — the "token-3 drift" — described in
// experiments/qwen36/token3-drift-investigation-2026-06-28.md (§4 third bullet, §5 step 2).
//
// The phenomenon: on the fixed 22-token ChatML prompt, greedy decode, fak and llama.cpp
// agree for two tokens (`248068, 198`) then disagree on the third — fak picks `8160`
// (logit 23.18, 2nd-place `90700` at 21.43, a ~1.75-logit near-tie) while llama.cpp picks
// `90700`. The arch math is bit-exact vs HF on the tiny fixture, and the drift survives the
// Q8->q4_k change on BOTH engines, so it is localized to a kernel-numerics divergence at 27B
// scale on the hybrid Gated-DeltaNet (GDN) path. Hypothesis H1 (the top-ranked one) is that
// the delta-rule recurrent scan's fixed serial reduction order rounds differently from
// llama.cpp's kernel, and — because the scan is the only STATEFUL op — that per-step
// difference COMPOUNDS across tokens (state carry) and layers (residual stream) until it tips
// the near-tie at token 3.
//
// This program supplies the DEVICE-INDEPENDENT measurement that hypothesis demands: it runs a
// faithful 48-GDN-layer residual stack TWICE, where the two runs are bit-identical in every op
// EXCEPT the reduction order (or accumulation precision) of the recurrent scan, and measures
// how fast the resulting hidden-state divergence compounds with depth and tokens. The two runs
// share the same (seeded-random) weights and inputs, so the ONLY source of divergence is the
// scan numerics — exactly the "same math, different rounding" H1 posits. The recurrence math is
// copied verbatim from internal/model/metal_prefill_hybrid_core.go:202-246 (the prefill twin of
// qwen35.go:linearAttnStep); only the i-loop direction (mode "reorder") or a per-step f16 state
// round-trip (mode "f16state") differs between the two runs.
//
// It answers a falsifiable question with no Mac and no 27B artifact:
//
//	Is a reduction-order-only (or f16-state) numeric difference, compounded over 48 GDN
//	layers and ~24 carried positions, LARGE ENOUGH on its own to flip a ~1.75-logit
//	near-tie at the token-3 decode step?
//
// If the measured final relative hidden divergence rho >= rho* (~ margin / |logit| ~ 1.75/20
// ~ 0.09), pure rounding suffices and the fix is to match llama.cpp's reduction order. If
// rho << rho* even in the f16-state bracket, then numerics-as-rounding CANNOT explain the
// flip — the token-3 divergence is anomalously large and an algorithmic/ordering mismatch
// (not mere accumulation) is implicated, which sharpens the per-layer probe's job (find the
// layer where cosine drops ANOMALOUSLY, not merely below 1).
//
// What this does NOT do: it does not run llama.cpp, does not load the 27B artifact, and does
// not claim to have found the diverging (layer, op). Those are the Mac/artifact-gated steps
// (token3-drift-investigation §5 steps 4-5). This is the host-independent sensitivity bound.
//
// Real Qwen3.6-27B GDN dims (H=5120; LinearNumKeyHeads 16, LinearNumValueHeads 48,
// LinearKeyHeadDim 128, LinearValueHeadDim 128, ssm.conv_kernel 4) match the in-tree fixture
// internal/model/quant_q4k_resident_test.go and the sibling gdn-recurrence-bench.
//
// Run:
//
//	go run ./experiments/qwen36/gdn-divergence-sensitivity                 # human table, default modes
//	go run ./experiments/qwen36/gdn-divergence-sensitivity -json           # machine result (result.json)
//	go run ./experiments/qwen36/gdn-divergence-sensitivity -tokens 8 -layers 12   # quick smoke
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"

	"github.com/anthony-chaudhary/fak/experiments/qwen36/gdn"
)

// aLogMean is the mean of the per-head A_log draw this experiment pins: A_log ~ -2 -> decay
// exp(-exp(A_log)*dt) ~ 0.88, an effective memory of ~8 positions. The depth axis measured here
// holds it FIXED (the length-axis sibling gdn-quant-length-sensitivity sweeps it instead), so the
// only thing varying between the reference and the test run stays the scan numerics.
const aLogMean = -2.0

// runStackTraced records the relative divergence of the last token's hidden
// state against a reference run after EACH layer (the compounding curve). It runs the reference
// (forward) and the test (mode) in lockstep so per-layer snapshots align. `hidden` is a parameter
// (main passes the real gdn.Hidden27B) so the curve can be exercised at a small H without
// allocating the 27B-scale projection matrices.
func runStackTraced(hidden, P, layers int, mode gdn.ScanMode) (perLayerRho []float64, finalRho float64) {
	eps := float32(1e-6)
	X := make([]float32, P*hidden)
	Y := make([]float32, P*hidden)
	r := rand.New(rand.NewSource(0xC0FFEE))
	for i := range X {
		v := float32(r.NormFloat64())
		X[i] = v
		Y[i] = v
	}
	lwRef := gdn.NewLayerWeights(hidden)
	lwTst := gdn.NewLayerWeights(hidden)
	last := (P - 1) * hidden
	for l := 0; l < layers; l++ {
		lwRef.Fill(l, aLogMean)
		lwTst.Fill(l, aLogMean)
		oRef := gdn.Layer(lwRef, X, P, gdn.ModeForward, eps)
		oTst := gdn.Layer(lwTst, Y, P, mode, eps)
		for i := range X {
			X[i] += oRef[i]
			Y[i] += oTst[i]
		}
		perLayerRho = append(perLayerRho, gdn.RelDiv(X[last:last+hidden], Y[last:last+hidden]))
	}
	finalRho = gdn.RelDiv(X[last:last+hidden], Y[last:last+hidden])
	return
}

type modeResult struct {
	Mode              string    `json:"mode"`
	PerLayerRho       []float64 `json:"per_layer_rho_last_token"`
	FinalRho          float64   `json:"final_rho_last_token"`
	ImpliedLogitDelta float64   `json:"implied_logit_delta_at_logit_scale_20"`
	FlipsNearTie      bool      `json:"flips_175_logit_near_tie"`
}

type result struct {
	Schema        string       `json:"schema"`
	Issue         string       `json:"issue"`
	Host          string       `json:"host"`
	Model         string       `json:"model"`
	Tokens        int          `json:"tokens_carried"`
	Layers        int          `json:"gdn_layers"`
	LogitScale    float64      `json:"assumed_logit_scale"`
	NearTieMargin float64      `json:"observed_near_tie_margin_logits"`
	RhoThreshold  float64      `json:"rho_threshold_to_flip"`
	Modes         []modeResult `json:"modes"`
	Verdict       string       `json:"verdict"`
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	tokens := flag.Int("tokens", 24, "positions carried before the measured (token-3) step (22 prompt + 2 agreed)")
	layers := flag.Int("layers", 48, "number of GDN layers in the residual stack")
	flag.Parse()

	const logitScale = 20.0 // |top logit| from the witnessed token-3 data (fak 23.18 / 21.43)
	const nearTie = 1.75    // observed fak margin between {8160, 90700}
	rhoStar := nearTie / logitScale

	modes := []struct {
		m    gdn.ScanMode
		name string
	}{
		{gdn.ModeReverse, "reorder (reverse-i reduction; pure rounding, models a different SIMD/threadgroup order)"},
		{gdn.ModeF16State, "f16state (forward order + f16 state round-trip each step; models f16 state storage)"},
	}

	res := result{
		Schema:        "qwen36-gdn-divergence-sensitivity/v1",
		Issue:         "token-3 drift (correctness parity); see token3-drift-investigation-2026-06-28.md",
		Host:          "windows/amd64 (CGO_ENABLED=0); device-independent, no Mac / no GPU / no 27B artifact",
		Model:         "Qwen3.6-27B q4_k_m GDN shapes (H=5120 nK=16 nV=48 kHd=vHd=128 K=4)",
		Tokens:        *tokens,
		Layers:        *layers,
		LogitScale:    logitScale,
		NearTieMargin: nearTie,
		RhoThreshold:  rhoStar,
	}

	for _, md := range modes {
		perLayer, final := runStackTraced(gdn.Hidden27B, *tokens, *layers, md.m)
		implied := final * logitScale
		res.Modes = append(res.Modes, modeResult{
			Mode:              md.name,
			PerLayerRho:       perLayer,
			FinalRho:          final,
			ImpliedLogitDelta: implied,
			FlipsNearTie:      implied >= nearTie,
		})
	}

	// Verdict from the strongest bracket (f16state).
	strongest := res.Modes[len(res.Modes)-1]
	if strongest.FlipsNearTie {
		res.Verdict = "numerics-as-rounding is SUFFICIENT: even/at-least one modeled kernel-numerics " +
			"difference compounds to >= the observed 1.75-logit near-tie over the GDN stack, so the token-3 " +
			"flip is consistent with pure accumulation/precision divergence -> match llama.cpp's scan " +
			"reduction order / state dtype. Confirm the actual (layer,op) with the Mac per-layer probe."
	} else {
		insufficient := []string{
			"numerics-as-rounding is INSUFFICIENT on its own: even the f16-state bracket stays far below the rho needed to flip the 1.75-logit near-tie,",
			"so accumulated reduction-order/precision divergence ALONE cannot explain token-3; the flip implies an ANOMALOUS algorithmic or ordering divergence, not mere rounding.",
			"The per-layer probe must find the real op mismatch where cosine drops anomalously; a 1-ULP-floor threshold would miss it.",
		}
		res.Verdict = gdn.Verdict(insufficient...)
	}

	if *asJSON {
		_ = gdn.EmitJSON(os.Stdout, res)
		return
	}

	fmt.Printf("Qwen3.6-27B GDN reduction-order / precision sensitivity — token-3 drift (H1), device-independent\n")
	fmt.Printf("  shapes: H=%d nK=%d nV=%d kHd=%d vHd=%d convDim=%d valDim=%d K=%d\n",
		gdn.Hidden27B, gdn.NK, gdn.NV, gdn.KHd, gdn.VHd, gdn.ConvDim, gdn.ValDim, gdn.K)
	fmt.Printf("  stack:  %d GDN layers, %d carried positions (22 prompt + 2 agreed, predicting token 3)\n", *layers, *tokens)
	fmt.Printf("  near-tie: observed fak margin %.2f logits at |logit|~%.0f -> need rho >= %.4f (%.2f%%) of ||hidden|| to flip\n\n",
		nearTie, logitScale, rhoStar, rhoStar*100)
	for _, m := range res.Modes {
		fmt.Printf("  mode: %s\n", m.Mode)
		fmt.Printf("    per-layer relative hidden divergence (last token), layers 1..%d:\n      ", *layers)
		for i, v := range m.PerLayerRho {
			fmt.Printf("%.2e ", v)
			if (i+1)%8 == 0 {
				fmt.Printf("\n      ")
			}
		}
		fmt.Printf("\n    final rho = %.4e  -> implied |Δlogit| ~ %.4e  (flips 1.75 near-tie: %v)\n\n",
			m.FinalRho, m.ImpliedLogitDelta, m.FlipsNearTie)
	}
	fmt.Printf("  VERDICT: %s\n", res.Verdict)
}
