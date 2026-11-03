package compute

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// Untagged required mode prevents a missing CUDA build from becoming a green
// zero-test result. Each mode requires its own matching rebuilt native archive.
var cudaQ56KMMQRequired = flag.String("fak-cuda-q56k-mmq-required", "",
	"require retained Q56 prompt MMQ (1) or a separate default archive (default)")

type cudaQ56KMMQWitness interface {
	cudaQ56KMMVQWitness
	cudaQ56KMMQForTest() (bool, uint64, uint64, uint64)
	cudaQ56KMMQNullInputForTest(Tensor, Tensor, int) int
}

// The small matrix set covers the selector boundary, padded prompt/output tails,
// canonical two-byte Q6 blocks, and a production-sized reduction dimension.
// It is a numerical/ownership witness, not a throughput benchmark.
// fak-test:runtime slow est=30s lane=optin
func TestCUDAQ56KMMQ(t *testing.T) {
	mode := *cudaQ56KMMQRequired
	if mode == "" {
		t.Skip("set required mode 1 or default on a separately admitted CUDA node")
	}
	if mode != "1" && mode != "default" {
		t.Fatal("required mode must be 1 or default")
	}
	raw, ok := Lookup("cuda")
	if !ok {
		t.Fatal("required CUDA backend absent; no CPU or zero-test fallback")
	}
	be, ok := raw.(cudaQ56KMMQWitness)
	if !ok {
		t.Fatalf("required complete native witness interface absent: %T", raw)
	}
	enabled, _, _, _ := be.cudaQ56KMMQForTest()
	if enabled != (mode == "1") {
		t.Fatalf("native typed selection=%v does not match required mode=%s", enabled, mode)
	}
	t.Cleanup(be.Recycle)
	for _, kind := range []Dtype{Q5_K, Q6_K} {
		for _, shape := range [][3]int{{127, 9, 256}, {128, 128, 256}, {129, 129, 256}, {128, 9, 5120}} {
			p, out, in := shape[0], shape[1], shape[2]
			t.Run(fmt.Sprintf("%s-P%d-O%d-K%d", kind, p, out, in), func(t *testing.T) {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				codes, weights := cudaQ56Fixture(kind, out, in)
				original := append([]byte(nil), codes...)
				var host Tensor
				if kind == Q5_K {
					host = NewQ5K(be, []int{out, in}, codes)
				} else {
					host = NewQ6K(be, []int{out, in}, codes)
				}
				w := be.Upload(host, kind)
				defer be.Free(w)
				values, effective := cudaQ56MMQActivations(p, in)
				x := be.Upload(NewF32(be, []int{p, in}, values), F32)
				defer be.Free(x)
				_, q5, q6, _ := be.cudaQ56KMMQForTest()
				y := be.BatchedMatMul(w, x, p)
				defer be.Free(y)
				got := be.Read(y) // real completion, not the submission counter
				_, a5, a6, _ := be.cudaQ56KMMQForTest()
				selected := enabled && p >= 128
				want5, want6 := uint64(0), uint64(0)
				if selected && kind == Q5_K {
					want5 = 1
				}
				if selected && kind == Q6_K {
					want6 = 1
				}
				if a5-q5 != want5 || a6-q6 != want6 {
					t.Fatalf("wrong actual launch deltas %d/%d want %d/%d", a5-q5, a6-q6, want5, want6)
				}
				if len(got) != p*out {
					t.Fatalf("output extent=%d", len(got))
				}
				for r := 0; r < p; r++ {
					activation := values[r*in : (r+1)*in]
					if selected {
						activation = effective[r*in : (r+1)*in]
					}
					for o := 0; o < out; o++ {
						var want, magnitude float64
						for k := 0; k < in; k++ {
							v := float64(weights[o*in+k]) * float64(activation[k])
							want += v
							magnitude += math.Abs(v)
						}
						if selected && kind == Q5_K {
							// DS4 subtracts min * ORIGINAL half-rounded F32 sum,
							// not min * sum(quantized x). Fixture sums are exactly
							// representable half values; assert that premise.
							for k := 0; k < in; k += 32 {
								var originalSum, effectiveSum float32
								for j := 0; j < 32; j++ {
									originalSum += values[r*in+k+j]
									effectiveSum += effective[r*in+k+j]
								}
								if Float16BitsToFloat32(Float32ToFloat16Bits(originalSum)) != originalSum {
									t.Fatal("DS4 fixture sum must be exactly representable as half")
								}
								b := codes[(o*(in/256)+k/256)*176:]
								_, min := kquantbits.ScaleMinK4((k%256)/32, b[4:16])
								dm := Float16BitsToFloat32(binary.LittleEndian.Uint16(b[2:]))
								correction := float64(dm*float32(min)) * float64(effectiveSum-originalSum)
								want += correction
								magnitude += math.Abs(correction)
							}
						}
						tolerance := 1e-4 + 2e-5*magnitude
						if math.IsNaN(float64(got[r*out+o])) || math.IsInf(float64(got[r*out+o]), 0) ||
							math.Abs(float64(got[r*out+o])-want) > tolerance {
							t.Fatalf("row=%d output=%d got=%g want=%g tol=%g selected=%v", r, o, got[r*out+o], want, tolerance, selected)
						}
					}
				}
				readX := be.Read(x)
				if len(readX) != len(values) {
					t.Fatal("activation extent changed")
				}
				for i, v := range readX {
					if math.Float32bits(v) != math.Float32bits(values[i]) {
						t.Fatal("activation input changed")
					}
				}
				if !bytes.Equal(codes, original) {
					t.Fatal("immutable packed source changed")
				}
				if enabled && p == 128 && out == 128 {
					cudaQ56MMQRefusals(t, be, w, x, p)
				}
			})
		}
	}
}

// Balanced quarter-step pairs give a nonzero ORIGINAL sum of d/4 and a zero
// quantized sum. This distinguishes Q5 DS4 minimum correction from an incorrect
// Q6-like scale-only oracle. Every d is an exact power of two, amax=127*d;
// signed half ties test roundf-away-from-zero. Row zero is an all-zero block.
func cudaQ56MMQActivations(p, in int) ([]float32, []float32) {
	x, effective := make([]float32, p*in), make([]float32, p*in)
	for b := in / 32; b < len(x)/32; b++ {
		d := float32(math.Ldexp(1, b%5-5))
		x[b*32], x[b*32+1] = 127*d, -127*d
		x[b*32+2], x[b*32+3] = 0.5*d, -0.5*d
		for j := 4; j < 32; j += 2 {
			v := (float32((j*17+b*7)%123+1) + 0.25) * d
			x[b*32+j], x[b*32+j+1] = v, -v
		}
		x[b*32+31] += 0.25 * d
		for j := 0; j < 32; j++ {
			effective[b*32+j] = float32(math.Round(float64(x[b*32+j]/d))) * d
		}
	}
	return x, effective
}

func cudaQ56MMQRefusals(t *testing.T, be cudaQ56KMMQWitness, w, x Tensor, p int) {
	t.Helper()
	out := w.Shape[0]
	sentinel := make([]float32, p*out)
	for i := range sentinel {
		sentinel[i] = float32(i%13 + 1)
	}
	y := be.Upload(NewF32(be, []int{p, out}, sentinel), F32)
	defer be.Free(y)
	witness := func(label string, call func()) {
		t.Helper()
		t.Run(label, func(t *testing.T) {
			bt, bl := be.cudaAllocationCountsForTest()
			_, q5, q6, fence := be.cudaQ56KMMQForTest()
			be.ResetHostXfer()
			be.ResetH2DXfer()
			call()
			at, al := be.cudaAllocationCountsForTest()
			_, a5, a6, afterFence := be.cudaQ56KMMQForTest()
			if bt != at || bl != al || q5 != a5 || q6 != a6 || fence != afterFence ||
				be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
				t.Fatal("pre-submit refusal changed owners, transfers, submissions or fence")
			}
		})
	}
	witness("native-null-input", func() {
		if rc := be.cudaQ56KMMQNullInputForTest(w, y, p); rc != -75703 {
			t.Fatalf("native null input status=%d want -75703", rc)
		}
	})
	for i, v := range be.Read(y) {
		if v != sentinel[i] {
			t.Fatal("native refusal changed destination")
		}
	}
	refuse := func(label string, weight, input Tensor) {
		witness(label, func() {
			var failure any
			func() { defer func() { failure = recover() }(); be.BatchedMatMul(weight, input, p) }()
			e, ok := failure.(*CUDAOpError)
			if !ok || e.Op != "Q56KMMQ" {
				t.Fatalf("expected typed MMQ preflight refusal, got %T %v", failure, failure)
			}
		})
	}
	bad := w
	q := *w.Quant
	q.Block = 32
	bad.Quant = &q
	refuse("noncanonical-block", bad, x)
	bad = w
	bad.Layout = ColMajor
	refuse("layout", bad, x)
	badX := x
	badX.Shape = []int{p, w.Shape[1] - 1}
	refuse("shape", w, badX)
	refuse("host-input", w, NewF32(be, []int{p, w.Shape[1]}, make([]float32, p*w.Shape[1])))
	if !be.GraphBegin() {
		t.Fatal("required real capture unavailable; require FAK_CUDA_GRAPH=1 and timestamp profiling off")
	}
	defer be.GraphAbort()
	refuse("capture-before-allocation", w, x)
	if !be.IsCapturing() {
		t.Fatal("refusal ended caller-owned capture")
	}
}
