// Command gdn-recurrence-bench is the host-runnable, device-independent arm of the
// "benchmark both" ask in issue #65 (Gated-DeltaNet recurrence — GPU kernel vs
// CPU-hybrid decision).
//
// The #65 decision (experiments/qwen36/metal-gdn-recurrence-decision-2026-06-28.md)
// is: keep the GDN recurrence on the CPU (the CPU-hybrid arm); do NOT write a Metal
// GDN-scan kernel speculatively. Its load-bearing quantitative claim is that the
// recurrent scan is a tiny fraction of the work — "≈0.5% of prefill" — measured once
// on an M3 Pro via FAK_QPROFILE and recorded on disk. That single on-device number is
// not reproducible on a non-Apple box.
//
// This program supplies the DEVICE-INDEPENDENT half of that claim: the recurrence-vs-
// projection COMPUTE RATIO at the real Qwen3.6-27B GDN shapes. FLOP counts are exact
// arithmetic over the layer dimensions (independent of CPU/GPU and of weight dtype), so
// the ratio is reproducible on any box. It bounds what a perfect, zero-cost GPU scan
// kernel (arm B / #92) could ever save: at most the recurrence's share. The projections
// — which the CPU-hybrid already routes to the GPU — are the ~97% lever, so a GPU scan
// kernel chases the wrong few percent. A faithful CPU timing of the same scan and an
// equivalent projection matmul corroborates that the FLOP ratio also holds in wall-time.
//
// What this does NOT measure: the on-device Mac hybrid serialization / CPU<->GPU
// round-trip fraction (the recurrence fraction AFTER the projections move to the GPU).
// That stays the §6 gate of the decision doc — it needs Apple Silicon + -tags fakmetal.
//
// Real Qwen3.6-27B GDN dims are sourced from the in-tree fixture
// internal/model/quant_q4k_resident_test.go (HiddenSize 5120; LinearNumKeyHeads 16,
// LinearNumValueHeads 48, LinearKeyHeadDim 128, LinearValueHeadDim 128) and the GDN
// recurrence math is copied verbatim from internal/model/qwen35.go:linearAttnSeq
// (lines 320-383). Conv kernel K=4 (Qwen3-Next ssm.conv_kernel); it is a negligible term.
//
// Run:  go run ./experiments/qwen36/gdn-recurrence-bench           # human table
//
//	go run ./experiments/qwen36/gdn-recurrence-bench -json     # machine result
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"time"
)

// Qwen3.6-27B Gated-DeltaNet layer dims (the 48 linear_attn layers).
const (
	H   = 5120 // hidden size
	nK  = 16   // LinearNumKeyHeads
	nV  = 48   // LinearNumValueHeads
	kHd = 128  // LinearKeyHeadDim
	vHd = 128  // LinearValueHeadDim
	K   = 4    // ssm.conv_kernel (Qwen3-Next); negligible term
	pp  = 22   // prefill length the #65 table benchmarks (pp22)
)

const (
	keyDim  = nK * kHd          // 2048
	valDim  = nV * vHd          // 6144
	convDim = 2*keyDim + valDim // 10240
)

// flopCounts returns the exact per-token multiply-accumulate counts for the three
// cost classes of one GDN layer. Counts are dtype- and device-independent.
func flopCounts() (projMACs, convMACs, recurMACs int64) {
	// Five input/output projections (the GEMMs the CPU-hybrid routes to the GPU).
	projMACs = int64(convDim)*H + // in_proj_qkv [convDim x H]
		int64(valDim)*H + // in_proj_z   [valDim x H]
		int64(nV)*H + // in_proj_b   [nV x H]
		int64(nV)*H + // in_proj_a   [nV x H]
		int64(H)*valDim // out_proj    [H x valDim]

	// Causal depthwise conv1d (kernel K, no bias).
	convMACs = int64(convDim) * K

	// Per v-head delta-rule recurrent scan (qwen35.go:320-383). State per head is
	// kHd*vHd. Per head per token: decay (kHd*vHd) + kv_mem accumulate (kHd*vHd) +
	// delta (vHd) + state update & readout (2*kHd*vHd). Plus q/k L2-norm (2*keyDim).
	perHead := int64(kHd*vHd) + int64(kHd*vHd) + int64(vHd) + int64(2*kHd*vHd)
	recurMACs = perHead*int64(nV) + int64(2*keyDim)
	return
}

// ---- faithful CPU reproductions, timed at pp22 ----

func silu(x float32) float32 { return x / (1 + float32(math.Exp(float64(-x)))) }

func sigmoidf(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }

func softplus(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }

func l2normInto(dst, src []float32, eps float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss/float32(len(src))+eps)))
	for i := range src {
		dst[i] = src[i] * inv
	}
}

// naiveMatmul is the CPU shape of mat.mul: out[i] = sum_j W[i*in+j]*x[j].
func naiveMatmul(out, w, x []float32, outDim, inDim int) {
	for i := 0; i < outDim; i++ {
		var acc float32
		base := i * inDim
		for j := 0; j < inDim; j++ {
			acc += w[base+j] * x[j]
		}
		out[i] = acc
	}
}

// timeProjections runs the five projection GEMMs for every position of a pp-token
// prefill and returns the elapsed wall-time. Weights are f32 (the FLOP count is the
// same for Q8/Q4K; f32 if anything OVERSTATES projection time, so the recurrence
// fraction reported here is a conservative LOWER bound).
func timeProjections(x []float32) time.Duration {
	wqkv := make([]float32, convDim*H)
	wz := make([]float32, valDim*H)
	wb := make([]float32, nV*H)
	wa := make([]float32, nV*H)
	wo := make([]float32, H*valDim)
	fill(wqkv, 0.0007)
	fill(wz, 0.0011)
	fill(wb, 0.0013)
	fill(wa, 0.0017)
	fill(wo, 0.0009)
	qkv := make([]float32, convDim)
	z := make([]float32, valDim)
	b := make([]float32, nV)
	a := make([]float32, nV)
	o := make([]float32, H)
	core := make([]float32, valDim)
	fill(core, 0.01)

	start := time.Now()
	for t := 0; t < pp; t++ {
		naiveMatmul(qkv, wqkv, x, convDim, H)
		naiveMatmul(z, wz, x, valDim, H)
		naiveMatmul(b, wb, x, nV, H)
		naiveMatmul(a, wa, x, nV, H)
		naiveMatmul(o, wo, core, H, valDim)
	}
	sink = qkv[0] + z[0] + b[0] + a[0] + o[0]
	return time.Since(start)
}

// timeRecurrence runs the delta-rule recurrent scan (verbatim from qwen35.go) over a
// pp-token sequence and returns the elapsed wall-time.
func timeRecurrence() time.Duration {
	// Synthetic post-conv q/k/v and per-head gates, shaped exactly like convOut.
	convOut := make([][]float32, pp)
	gDecay := make([][]float32, pp)
	beta := make([][]float32, pp)
	for t := 0; t < pp; t++ {
		row := make([]float32, convDim)
		for c := range row {
			row[c] = silu(0.01 * float32((c%7)-3))
		}
		convOut[t] = row
		g := make([]float32, nV)
		bt := make([]float32, nV)
		for h := 0; h < nV; h++ {
			a := float32(math.Exp(float64(-0.5)))
			dt := softplus(0.1)
			g[h] = float32(math.Exp(float64(-a * dt)))
			bt[h] = sigmoidf(0.2)
		}
		gDecay[t] = g
		beta[t] = bt
	}

	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	state := make([][]float32, nV)
	for h := range state {
		state[h] = make([]float32, kHd*vHd)
	}
	qNorm := make([]float32, keyDim)
	kNorm := make([]float32, keyDim)
	kvmem := make([]float32, vHd)
	delta := make([]float32, vHd)

	start := time.Now()
	for t := 0; t < pp; t++ {
		q := convOut[t][0:keyDim]
		k := convOut[t][keyDim : 2*keyDim]
		for h := 0; h < nK; h++ {
			l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
			l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
			for i := h * kHd; i < (h+1)*kHd; i++ {
				qNorm[i] *= scale
			}
		}
		v := convOut[t][2*keyDim : 2*keyDim+valDim]
		out := make([]float32, valDim)
		for h := 0; h < nV; h++ {
			kh := h / repeat
			qn := qNorm[kh*kHd : (kh+1)*kHd]
			kn := kNorm[kh*kHd : (kh+1)*kHd]
			vh := v[h*vHd : (h+1)*vHd]
			g := gDecay[t][h]
			bt := beta[t][h]
			st := state[h]
			for i := range st {
				st[i] *= g
			}
			for d := range kvmem {
				kvmem[d] = 0
			}
			for i := 0; i < kHd; i++ {
				ki := kn[i]
				base := i * vHd
				for d := 0; d < vHd; d++ {
					kvmem[d] += st[base+d] * ki
				}
			}
			for d := 0; d < vHd; d++ {
				delta[d] = (vh[d] - kvmem[d]) * bt
			}
			od := out[h*vHd : (h+1)*vHd]
			for i := 0; i < kHd; i++ {
				ki := kn[i]
				qi := qn[i]
				base := i * vHd
				for d := 0; d < vHd; d++ {
					st[base+d] += ki * delta[d]
					od[d] += st[base+d] * qi
				}
			}
		}
		sink += out[0]
	}
	return time.Since(start)
}

var sink float32

func fill(s []float32, v float32) {
	for i := range s {
		s[i] = v * float32((i%13)-6)
	}
}

type result struct {
	Schema           string  `json:"schema"`
	Issue            string  `json:"issue"`
	Host             string  `json:"host"`
	Model            string  `json:"model"`
	Layer            string  `json:"layer"`
	Pp               int     `json:"pp"`
	ProjMACsPerTok   int64   `json:"proj_macs_per_token"`
	ConvMACsPerTok   int64   `json:"conv_macs_per_token"`
	RecurMACsPerTok  int64   `json:"recur_macs_per_token"`
	RecurFracOfProj  float64 `json:"recur_frac_of_projections"`
	RecurFracOfLayer float64 `json:"recur_frac_of_layer"`
	ProjMillis       float64 `json:"proj_wall_ms_pp22"`
	RecurMillis      float64 `json:"recur_wall_ms_pp22"`
	RecurWallFrac    float64 `json:"recur_wall_frac_of_layer_compute"`
	Iters            int     `json:"iters"`
	Verdict          string  `json:"verdict"`
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	iters := flag.Int("iters", 50, "timing repetitions (median taken)")
	flag.Parse()

	projMACs, convMACs, recurMACs := flopCounts()
	layerMACs := projMACs + convMACs + recurMACs
	recurFracProj := float64(recurMACs) / float64(projMACs)
	recurFracLayer := float64(recurMACs) / float64(layerMACs)

	// Median over -iters runs to damp scheduler noise.
	x := make([]float32, H)
	fill(x, 0.02)
	projTimes := make([]time.Duration, *iters)
	recurTimes := make([]time.Duration, *iters)
	for i := 0; i < *iters; i++ {
		projTimes[i] = timeProjections(x)
		recurTimes[i] = timeRecurrence()
	}
	projMs := medianMs(projTimes)
	recurMs := medianMs(recurTimes)
	recurWallFrac := recurMs / (projMs + recurMs)

	res := result{
		Schema:           "fak-gdn-recurrence-compute-ratio/1",
		Issue:            "65",
		Host:             "windows/amd64 (CGO_ENABLED=0); device-independent FLOP ratio + native CPU timing",
		Model:            "Qwen3.6-27B q4_k_m",
		Layer:            "linear_attn (Gated-DeltaNet), one of 48",
		Pp:               pp,
		ProjMACsPerTok:   projMACs,
		ConvMACsPerTok:   convMACs,
		RecurMACsPerTok:  recurMACs,
		RecurFracOfProj:  recurFracProj,
		RecurFracOfLayer: recurFracLayer,
		ProjMillis:       projMs,
		RecurMillis:      recurMs,
		RecurWallFrac:    recurWallFrac,
		Iters:            *iters,
		Verdict: "recurrence compute is a low-single-digit % of the linear_attn projections; a " +
			"perfect zero-cost GPU scan kernel (arm B/#92) could save at most that share, while the " +
			"projections (already GPU-routed by the CPU-hybrid) are the ~97% lever -> keep recurrence on CPU",
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return
	}

	fmt.Printf("Gated-DeltaNet recurrence vs projections — Qwen3.6-27B linear_attn layer (#65 arm A, device-independent)\n")
	fmt.Printf("  shapes: H=%d nK=%d nV=%d kHd=%d vHd=%d convDim=%d valDim=%d K=%d pp=%d\n\n", H, nK, nV, kHd, vHd, convDim, valDim, K, pp)
	fmt.Printf("  per-token MACs (exact, dtype/device-independent):\n")
	fmt.Printf("    projections (5 GEMMs, GPU-routed by CPU-hybrid): %15d\n", projMACs)
	fmt.Printf("    conv1d (depthwise, K=%d):                         %15d\n", K, convMACs)
	fmt.Printf("    recurrence (delta-rule scan, %d heads):          %15d\n", nV, recurMACs)
	fmt.Printf("    ---------------------------------------------------------------\n")
	fmt.Printf("    recurrence / projections                       = %6.3f%%\n", recurFracProj*100)
	fmt.Printf("    recurrence / whole linear_attn layer compute   = %6.3f%%\n\n", recurFracLayer*100)
	fmt.Printf("  native CPU wall-time, pp%d, median of %d runs (corroboration; f32 overstates projections):\n", pp, *iters)
	fmt.Printf("    projections: %8.3f ms   recurrence: %8.3f ms   recurrence/total = %6.3f%%\n\n", projMs, recurMs, recurWallFrac*100)
	fmt.Printf("  VERDICT: %s\n", res.Verdict)
}

func medianMs(ds []time.Duration) float64 {
	cp := make([]time.Duration, len(ds))
	copy(cp, ds)
	// insertion sort (small n)
	for i := 1; i < len(cp); i++ {
		for j := i; j > 0 && cp[j] < cp[j-1]; j-- {
			cp[j], cp[j-1] = cp[j-1], cp[j]
		}
	}
	mid := cp[len(cp)/2]
	return float64(mid.Microseconds()) / 1000.0
}
