//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// kquantMulMMReceiptPathEnv directs TestKQuantMulMMReceiptVsScalar to measure and write its JSON
// receipt. Unset, the test skips: the receipt is [HW-WITNESSED] and only meaningful when it is
// deliberately captured on an uncontended GPU.
const kquantMulMMReceiptPathEnv = "FAK_KQUANT_MULMM_RECEIPT"

const (
	kquantMulMMReceiptSchema  = "fak.metalgemm.kquant_mulmm_receipt/v1"
	kquantMulMMReceiptRepeats = 5
	kquantMulMMReceiptDate    = "2026-10-04"
	kquantMulMMReceiptCommit  = "37e7d7293+fak#13692"
	kquantMulMMReceiptGate    = 1.10 // fak#9937 routing gate
	kquantMulMMReceiptTFLOPsP = 512
)

// kquantMulMMReceiptPrompts is the prompt-length sweep. Every P at or above a format's floor
// (Q4KMulMMMinPrompt / Q6KMulMMMinPrompt) is gated; the smaller ones are recorded for context.
var kquantMulMMReceiptPrompts = []int{2, 4, 8, 16, 32, 64, 128, 512, 2048}

// kquantMulMMShape is one Qwen2.5-7B Q4_K_M projection shape (out,in) in one format, with its
// per-layer weight. Q4_K_M stores attn_v and ffn_down alternately as Q4_K and Q6_K across layers,
// so those shapes appear once per format at weight 0.5.
type kquantMulMMShape struct {
	Name   string  `json:"name"`
	Format string  `json:"format"`
	Out    int     `json:"out"`
	In     int     `json:"in"`
	Weight float64 `json:"layer_weight"`
}

var kquantMulMMReceiptShapes = []kquantMulMMShape{
	{"gate_up", "q4_k", 18944, 3584, 2},
	{"o_proj", "q4_k", 3584, 3584, 1},
	{"v_proj", "q4_k", 512, 3584, 0.5},
	{"v_proj", "q6_k", 512, 3584, 0.5},
	{"down", "q4_k", 3584, 18944, 0.5},
	{"down", "q6_k", 3584, 18944, 0.5},
}

var kquantMulMMReceiptFormats = []string{"q4_k", "q6_k"}

// kquantSmallPApplies reports whether the fak#13694 small-P GEMV arm exists at P (its 2..20 band).
func kquantSmallPApplies(P int) bool { return P >= 2 && P <= 20 }

// kquantMulMMPoint is one shape at one P. Per-shape ratios are recorded but NOT gated: the gate is
// per format on the per-layer weighted time (kquantMulMMFormatPoint).
type kquantMulMMPoint struct {
	P          int       `json:"p"`
	ControlMS  float64   `json:"control_gpu_ms_median"`
	SmallPMS   float64   `json:"small_p_gpu_ms_median,omitempty"`
	MulMMMS    float64   `json:"mul_mm_gpu_ms_median"`
	BestPreMS  float64   `json:"best_pre_13692_gpu_ms_median"`
	RatioCtrl  float64   `json:"ratio_control_over_mul_mm"`
	RatioBest  float64   `json:"ratio_best_pre_over_mul_mm"`
	ControlS   []float64 `json:"control_gpu_ms_samples"`
	SmallPS    []float64 `json:"small_p_gpu_ms_samples,omitempty"`
	MulMMS     []float64 `json:"mul_mm_gpu_ms_samples"`
	BestPreArm string    `json:"best_pre_13692_arm"`
}

type kquantMulMMShapeResult struct {
	kquantMulMMShape
	Control string             `json:"control_kernel"`
	Points  []kquantMulMMPoint `json:"points"`
}

// kquantMulMMFormatPoint is the gated quantity: the per-layer weighted on-GPU time of one format
// at one P for each arm, and mul_mm against the best pre-#13692 arm at that P.
type kquantMulMMFormatPoint struct {
	Format     string  `json:"format"`
	P          int     `json:"p"`
	Gated      bool    `json:"gated"`
	ControlMS  float64 `json:"control_weighted_ms"`
	SmallPMS   float64 `json:"small_p_weighted_ms,omitempty"`
	MulMMMS    float64 `json:"mul_mm_weighted_ms"`
	BestPreArm string  `json:"best_pre_13692_arm"`
	BestPreMS  float64 `json:"best_pre_13692_weighted_ms"`
	Ratio      float64 `json:"ratio_best_pre_over_mul_mm"`
}

type kquantMulMMReceipt struct {
	Schema        string                   `json:"schema"`
	ContractIssue string                   `json:"contract_issue"`
	GeneratedAt   string                   `json:"generated_at"`
	CaptureDate   string                   `json:"capture_date"`
	EvidenceKind  string                   `json:"evidence_kind"`
	Verdict       string                   `json:"verdict"`
	DeviceName    string                   `json:"device_name"`
	OSVersion     string                   `json:"os_version"`
	SourceCommit  string                   `json:"source_commit"`
	SourceTest    string                   `json:"source_test"`
	Metric        string                   `json:"metric"`
	Repeats       int                      `json:"repeats"`
	GateMargin    float64                  `json:"gate_margin"`
	Floors        map[string]int           `json:"mul_mm_min_prompt"`
	FormatGates   []kquantMulMMFormatPoint `json:"format_gates"`
	Shapes        []kquantMulMMShapeResult `json:"shapes"`
	TFLOPsP       int                      `json:"tflops_prompt"`
	TFLOPsMulMM   float64                  `json:"tflops_mul_mm_layer_weighted"`
	TFLOPsControl float64                  `json:"tflops_control_layer_weighted"`
	Notes         []string                 `json:"notes"`
}

type kquantMulMMArm struct {
	name    string
	run     func(x []float32, P int, y []float32) bool
	samples *[]float64
}

// TestKQuantMulMMReceiptVsScalar is the on-silicon speed receipt that licenses the fak#13692
// mul_mm default. For every Qwen2.5-7B Q4_K_M projection shape it times three arms over warmed
// pipelines, interleaved with rotating order, taking the median LastGEMMGPUMs over 5 repeats:
//   - control: Q4_K the scalar q4k_gemm as shipped with fak#13692 (one 2D dispatch, padded xbuf),
//     not the pre-change per-tile-dispatch kernel; Q6_K the naive q6k_gemm;
//   - small-P: the fak#13694 batched multi-token GEMV, for 2 <= P <= 20 only;
//   - mul_mm.
//
// The gate is per FORMAT on the per-layer weighted time (Q4_K: gate_up*2 + o*1 + v*0.5 +
// down*0.5; Q6_K: v*0.5 + down*0.5): mul_mm must be >= 1.10x faster than the BEST pre-#13692 arm
// at that P (min of control and, inside its band, small-P) at every P >= that format's floor.
// Per-shape ratios are recorded, not gated. It also records the per-layer-weighted P=512 TFLOP/s
// (2 FLOP/MAC) and writes the JSON receipt only when every gate passed.
func TestKQuantMulMMReceiptVsScalar(t *testing.T) {
	path := os.Getenv(kquantMulMMReceiptPathEnv)
	if path == "" {
		t.Skipf("%s unset; the mul_mm receipt is [HW-WITNESSED] and only captured on request", kquantMulMMReceiptPathEnv)
	}
	if !Available() {
		t.Skip("Metal unavailable; the mul_mm receipt needs on-silicon execution")
	}
	if bits := kquantMulMMBits(); bits&3 != 3 {
		t.Fatalf("mul_mm pipelines unavailable: ready bits=%b", bits)
	}

	leaseOpts := gpulease.Options{Mode: gpulease.ModeExclusive, NoWait: true}
	if waitMS := os.Getenv(q4kCrossoverLeaseWaitEnv); waitMS != "" {
		ms, err := strconv.Atoi(waitMS)
		if err != nil || ms < 0 {
			t.Fatalf("%s=%q must be a non-negative integer of milliseconds", q4kCrossoverLeaseWaitEnv, waitMS)
		}
		if ms > 0 {
			leaseOpts = gpulease.Options{Mode: gpulease.ModeExclusive, Timeout: time.Duration(ms) * time.Millisecond}
		}
	}
	lease, err := gpulease.Acquire(leaseOpts)
	if err != nil {
		t.Skipf("mul_mm receipt is [HW-WITNESSED] only on an uncontended GPU; exclusive GPU lease unavailable (%v)", err)
	}
	defer lease.Release()
	defer ResetQ4K()

	receipt := kquantMulMMReceipt{
		Schema:        kquantMulMMReceiptSchema,
		ContractIssue: "13692",
		CaptureDate:   kquantMulMMReceiptDate,
		EvidenceKind:  Q4KReceiptEvidenceHW,
		DeviceName:    DeviceName(),
		OSVersion:     OSVersion(),
		SourceCommit:  kquantMulMMReceiptCommit,
		SourceTest:    "TestKQuantMulMMReceiptVsScalar",
		Metric:        "on_gpu_ms",
		Repeats:       kquantMulMMReceiptRepeats,
		GateMargin:    kquantMulMMReceiptGate,
		Floors:        map[string]int{"q4_k": Q4KMulMMMinPrompt, "q6_k": Q6KMulMMMinPrompt},
		TFLOPsP:       kquantMulMMReceiptTFLOPsP,
		Notes: []string{
			"gate: per format, per-layer weighted on-GPU time; ratio = best pre-#13692 arm / mul_mm, required >= 1.10 at every P >= that format's floor",
			"weights: q4_k gate_up*2 + o*1 + v*0.5 + down*0.5; q6_k v*0.5 + down*0.5 (Q4_K_M alternates v and down between Q4_K and Q6_K)",
			"best pre-#13692 arm = min(control, small-P GEMV) inside the small-P band 2..20, else control; per-shape ratios are recorded but not gated",
			"on_gpu_ms is LastGEMMGPUMs: cb.GPUEndTime-cb.GPUStartTime, excluding host encode/commit/wait/memcpy; medians of interleaved repeats with rotating arm order",
			"q4_k control is the scalar q4k_gemm as shipped with fak#13692 (single 2D dispatch, padded xbuf), not the pre-change per-tile-dispatch kernel",
			"q6_k control is the naive one-simdgroup-per-(row,token) q6k_gemm; small-P is the fak#13694 batched multi-token GEMV",
			"tflops uses 2 FLOP/MAC over the combined per-layer weights of both formats at P=512",
		},
	}

	type weighted struct{ control, smallP, mulmm float64 }
	byFormat := map[string]map[int]*weighted{}
	for _, f := range kquantMulMMReceiptFormats {
		byFormat[f] = map[int]*weighted{}
		for _, P := range kquantMulMMReceiptPrompts {
			byFormat[f][P] = &weighted{}
		}
	}
	var macsP float64

	for _, shape := range kquantMulMMReceiptShapes {
		res := kquantMulMMShapeResult{kquantMulMMShape: shape}
		var run func(x []float32, P int, y []float32, mode Q4KGEMMMode) Q4KGEMMExecution
		switch shape.Format {
		case "q4_k":
			res.Control = "q4k_gemm scalar (fak#13692 single-dispatch, padded xbuf)"
			w := UploadQ4K(q4kTestRaw(shape.Out, shape.In, 0x13692), shape.Out, shape.In)
			if w == nil {
				t.Fatalf("%s %s: UploadQ4K returned nil", shape.Name, shape.Format)
			}
			defer w.Release()
			run = func(x []float32, P int, y []float32, mode Q4KGEMMMode) Q4KGEMMExecution {
				return w.GEMMWithEventsMode(x, P, y, nil, mode).Executed
			}
		case "q6_k":
			res.Control = "q6k_gemm naive"
			w := UploadQ6K(q6kTestRaw(shape.Out, shape.In, 0x6692), shape.Out, shape.In)
			if w == nil {
				t.Fatalf("%s %s: UploadQ6K returned nil", shape.Name, shape.Format)
			}
			defer w.Release()
			run = func(x []float32, P int, y []float32, mode Q4KGEMMMode) Q4KGEMMExecution {
				return w.GEMMWithEventsMode(x, P, y, nil, mode)
			}
		default:
			t.Fatalf("unknown format %q", shape.Format)
		}
		armFn := func(mode Q4KGEMMMode, want Q4KGEMMExecution) func([]float32, int, []float32) bool {
			return func(x []float32, P int, y []float32) bool { return run(x, P, y, mode) == want }
		}

		for _, P := range kquantMulMMReceiptPrompts {
			x := kquantTestPanel(P, shape.In)
			y := make([]float32, P*shape.Out)
			pt := kquantMulMMPoint{P: P}
			arms := []kquantMulMMArm{
				{"control", armFn(Q4KGEMMModeScalar, Q4KGEMMExecutedScalar), &pt.ControlS},
				{"mul_mm", armFn(Q4KGEMMModeMulMM, Q4KGEMMExecutedMulMM), &pt.MulMMS},
			}
			if kquantSmallPApplies(P) {
				arms = append(arms, kquantMulMMArm{"small_p", armFn(Q4KGEMMModeSmallPGEMV, Q4KGEMMExecutedSmallPGEMV), &pt.SmallPS})
			}
			// Warm every pipeline once so compile/residency setup is outside the timed window.
			for _, arm := range arms {
				if !arm.run(x, P, y) {
					t.Fatalf("%s %s P=%d: %s warmup did not execute the requested kernel", shape.Name, shape.Format, P, arm.name)
				}
			}
			for rep := 0; rep < kquantMulMMReceiptRepeats; rep++ {
				// Interleave the arms with a rotating start so drift hits every arm equally.
				for i := range arms {
					arm := arms[(rep+i)%len(arms)]
					if !arm.run(x, P, y) {
						t.Fatalf("%s %s P=%d rep=%d: %s did not execute", shape.Name, shape.Format, P, rep, arm.name)
					}
					*arm.samples = append(*arm.samples, LastGEMMGPUMs())
				}
			}
			pt.ControlMS, pt.MulMMMS = medianFloat(pt.ControlS), medianFloat(pt.MulMMS)
			pt.BestPreMS, pt.BestPreArm = pt.ControlMS, "control"
			if len(pt.SmallPS) > 0 {
				pt.SmallPMS = medianFloat(pt.SmallPS)
				if pt.SmallPMS < pt.BestPreMS {
					pt.BestPreMS, pt.BestPreArm = pt.SmallPMS, "small_p"
				}
			}
			for _, v := range []float64{pt.ControlMS, pt.MulMMMS, pt.BestPreMS} {
				if !(v > 0) || !q4kFinite(v) {
					t.Fatalf("%s %s P=%d: non-positive on-GPU window control=%g small_p=%g mul_mm=%g",
						shape.Name, shape.Format, P, pt.ControlMS, pt.SmallPMS, pt.MulMMMS)
				}
			}
			pt.RatioCtrl, pt.RatioBest = pt.ControlMS/pt.MulMMMS, pt.BestPreMS/pt.MulMMMS
			t.Logf("[HW-WITNESSED] %s %s (%dx%d) P=%d control_ms=%.4f small_p_ms=%.4f mul_mm_ms=%.4f best_pre=%s ratio_best=%.3f (per-shape, not gated)",
				shape.Name, shape.Format, shape.Out, shape.In, P, pt.ControlMS, pt.SmallPMS, pt.MulMMMS, pt.BestPreArm, pt.RatioBest)

			agg := byFormat[shape.Format][P]
			agg.control += shape.Weight * pt.ControlMS
			agg.mulmm += shape.Weight * pt.MulMMMS
			agg.smallP += shape.Weight * pt.SmallPMS
			if P == kquantMulMMReceiptTFLOPsP {
				macsP += shape.Weight * float64(shape.Out) * float64(shape.In)
			}
			res.Points = append(res.Points, pt)
		}
		receipt.Shapes = append(receipt.Shapes, res)
	}

	// The gate: per format, per-layer weighted time, mul_mm vs the best pre-#13692 arm at that P.
	var mulmmMSP, controlMSP float64
	for _, f := range kquantMulMMReceiptFormats {
		floor := receipt.Floors[f]
		for _, P := range kquantMulMMReceiptPrompts {
			agg := byFormat[f][P]
			fp := kquantMulMMFormatPoint{Format: f, P: P, Gated: P >= floor,
				ControlMS: agg.control, MulMMMS: agg.mulmm, BestPreArm: "control", BestPreMS: agg.control}
			if kquantSmallPApplies(P) {
				fp.SmallPMS = agg.smallP
				if agg.smallP < fp.BestPreMS {
					fp.BestPreArm, fp.BestPreMS = "small_p", agg.smallP
				}
			}
			fp.Ratio = fp.BestPreMS / fp.MulMMMS
			t.Logf("[HW-WITNESSED] %s weighted P=%d control_ms=%.4f small_p_ms=%.4f mul_mm_ms=%.4f best_pre=%s ratio=%.3f gated=%t",
				f, P, fp.ControlMS, fp.SmallPMS, fp.MulMMMS, fp.BestPreArm, fp.Ratio, fp.Gated)
			if fp.Gated && !(fp.Ratio >= kquantMulMMReceiptGate) {
				t.Errorf("%s P=%d: per-layer weighted best-pre/mul_mm=%.3f below the fak#9937 gate %.2f at P >= floor %d",
					f, P, fp.Ratio, kquantMulMMReceiptGate, floor)
			}
			if P == kquantMulMMReceiptTFLOPsP {
				mulmmMSP += agg.mulmm
				controlMSP += agg.control
			}
			receipt.FormatGates = append(receipt.FormatGates, fp)
		}
	}

	tflops := func(ms float64) float64 { return 2 * macsP * kquantMulMMReceiptTFLOPsP / (ms * 1e9) }
	receipt.TFLOPsMulMM, receipt.TFLOPsControl = tflops(mulmmMSP), tflops(controlMSP)
	t.Logf("[HW-WITNESSED] layer-weighted 7B P=%d: mul_mm %.2f TFLOP/s, control %.2f TFLOP/s",
		kquantMulMMReceiptTFLOPsP, receipt.TFLOPsMulMM, receipt.TFLOPsControl)
	if !q4kFinite(receipt.TFLOPsMulMM) || receipt.TFLOPsMulMM <= 0 || math.IsNaN(receipt.TFLOPsControl) {
		t.Errorf("non-finite layer-weighted TFLOP/s mul_mm=%g control=%g", receipt.TFLOPsMulMM, receipt.TFLOPsControl)
	}

	if t.Failed() {
		t.Logf("a gate failed; no receipt written to %s", path)
		return
	}
	receipt.Verdict = "PASS"
	receipt.GeneratedAt = time.Now().UTC().Format(time.RFC3339)
	payload, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(payload, '\n'), 0o644); err != nil {
		t.Fatalf("write receipt %s: %v", path, err)
	}
	t.Logf("[HW-WITNESSED] wrote %s receipt to %s", receipt.Schema, path)
}
