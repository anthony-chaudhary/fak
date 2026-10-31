package compute

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"runtime"
	"testing"
)

// Untagged so a required invocation cannot pass with zero CUDA tests.
// -fak-cuda-q56k-mmvq-required=1 requires the typed qualification archive;
// =default requires a separate default archive and witnesses rollback.
// This is real Upload -> ordinary consumer -> checked Read, not a fake CUDA
// result oracle. Nonfinite activations, asynchronous fault injection, full-model
// quality, full-sequence/lane execution and performance remain separate debt.
type cudaQ56KMMVQWitness interface {
	Backend
	cudaQ56KMMVQForTest() (bool, uint64, uint64, uint64)
	cudaQ56KMMVQNullInputForTest(Tensor, Tensor) int
	cudaAllocationCountsForTest() (int, int)
	GraphBegin() bool
	GraphAbort()
	IsCapturing() bool
	ResetHostXfer()
	ResetH2DXfer()
	HostXferBytes() uint64
	H2DXferBytes() uint64
	Recycle()
}

var cudaQ56KMMVQRequired = flag.String("fak-cuda-q56k-mmvq-required", "", "require the Q56 qualification (1) or separate default archive (default)")

// fak-test:runtime slow est=30s lane=optin
func TestCUDAQ56KMMVQ(t *testing.T) {
	mode := *cudaQ56KMMVQRequired
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
	be, ok := raw.(cudaQ56KMMVQWitness)
	if !ok {
		t.Fatalf("required complete native witness interface absent: %T", raw)
	}
	enabled, _, _, _ := be.cudaQ56KMMVQForTest()
	if enabled != (mode == "1") {
		t.Fatalf("native typed selection=%v does not match required mode=%s", enabled, mode)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	t.Cleanup(be.Recycle)
	for _, kind := range []Dtype{Q5_K, Q6_K} {
		for _, in := range []int{256, 5120} {
			const out = 9 // odd rows: second-row clamp must never read a tenth row
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
			for _, p := range []int{1, 2, 3, 4, 5} {
				t.Run(fmt.Sprintf("%s-K%d-P%d", kind, in, p), func(t *testing.T) {
					values, effective := cudaQ56Activations(p, in)
					x := be.Upload(NewF32(be, []int{p, in}, values), F32)
					defer be.Free(x)
					invoke := func(vector bool) {
						t.Helper()
						_, q5, q6, _ := be.cudaQ56KMMVQForTest()
						var y Tensor
						if vector {
							view := x
							view.Shape = []int{in}
							y = be.MatMul(w, view)
						} else {
							y = be.BatchedMatMul(w, x, p)
						}
						got := be.Read(y)
						defer be.Free(y)
						_, a5, a6, _ := be.cudaQ56KMMVQForTest()
						want5, want6 := uint64(0), uint64(0)
						selected := enabled && p < 4
						if selected {
							if kind == Q5_K {
								want5 = 1
							} else {
								want6 = 1
							}
						}
						if a5-q5 != want5 || a6-q6 != want6 {
							t.Fatalf("wrong real launch deltas %d/%d want %d/%d", a5-q5, a6-q6, want5, want6)
						}
						activation := values
						if selected {
							activation = effective
						}
						if len(got) != p*out {
							t.Fatalf("output extent=%d", len(got))
						}
						for r := 0; r < p; r++ {
							for o := 0; o < out; o++ {
								var want, abs float64
								for k := 0; k < in; k++ {
									v := float64(weights[o*in+k]) * float64(activation[r*in+k])
									want += v
									abs += math.Abs(v)
								}
								tolerance := 1e-4 + 2e-5*abs // reduction error, not permission to use the F32 oracle for q8_1
								if math.IsNaN(float64(got[r*out+o])) || math.IsInf(float64(got[r*out+o]), 0) || math.Abs(float64(got[r*out+o])-want) > tolerance {
									t.Fatalf("row=%d output=%d got=%g want=%g tol=%g selected=%v", r, o, got[r*out+o], want, tolerance, selected)
								}
							}
						}
					}
					invoke(false)
					if p == 1 {
						invoke(true)
					}
					gotX := be.Read(x)
					for i, v := range values {
						if math.Float32bits(gotX[i]) != math.Float32bits(v) {
							t.Fatal("input changed")
						}
					}
				})
			}
			if !bytes.Equal(codes, original) {
				t.Fatal("immutable packed weights changed")
			}
			if enabled && in == 256 {
				t.Run(kind.String()+"-native-null-input", func(t *testing.T) {
					sentinel := make([]float32, out)
					for i := range sentinel {
						sentinel[i] = float32(i + 1)
					}
					y := be.Upload(NewF32(be, []int{out}, sentinel), F32)
					defer be.Free(y)
					beforeT, beforeL := be.cudaAllocationCountsForTest()
					_, q5, q6, fence := be.cudaQ56KMMVQForTest()
					be.ResetHostXfer()
					be.ResetH2DXfer()
					if rc := be.cudaQ56KMMVQNullInputForTest(w, y); rc != -75603 {
						t.Fatalf("checked native null input status=%d, want pre-submit -75603", rc)
					}
					afterT, afterL := be.cudaAllocationCountsForTest()
					_, a5, a6, afterFence := be.cudaQ56KMMVQForTest()
					if beforeT != afterT || beforeL != afterL || q5 != a5 || q6 != a6 || fence != afterFence || be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
						t.Fatal("native null refusal changed owners, transfers, submissions or fence")
					}
					got := be.Read(y)
					for i, v := range sentinel {
						if got[i] != v {
							t.Fatal("native null refusal changed output")
						}
					}
				})
				values, _ := cudaQ56Activations(1, in)
				x := be.Upload(NewF32(be, []int{in}, values), F32)
				defer be.Free(x)
				refusal := func(t *testing.T, name string, call func()) {
					t.Helper()
					t.Run(kind.String()+"-"+name, func(t *testing.T) {
						beforeT, beforeL := be.cudaAllocationCountsForTest()
						_, q5, q6, fence := be.cudaQ56KMMVQForTest()
						be.ResetHostXfer()
						be.ResetH2DXfer()
						var err any
						func() { defer func() { err = recover() }(); call() }()
						if _, ok := err.(*CUDAOpError); !ok {
							t.Fatalf("expected typed preflight refusal, got %T %v", err, err)
						}
						afterT, afterL := be.cudaAllocationCountsForTest()
						_, a5, a6, afterFence := be.cudaQ56KMMVQForTest()
						if beforeT != afterT || beforeL != afterL || q5 != a5 || q6 != a6 || fence != afterFence || be.HostXferBytes() != 0 || be.H2DXferBytes() != 0 {
							t.Fatal("preflight refusal allocated, transferred, submitted or fenced")
						}
					})
				}
				malformed := w
				q := *w.Quant
				q.Block = 32
				malformed.Quant = &q
				refusal(t, "noncanonical-block", func() { be.MatMul(malformed, x) })
				badLayout := w
				badLayout.Layout = ColMajor
				refusal(t, "layout", func() { be.MatMul(badLayout, x) })
				hostX := NewF32(be, []int{in}, values)
				refusal(t, "host-input", func() { be.MatMul(w, hostX) })
				badShape := x
				badShape.Shape = []int{in - 1}
				refusal(t, "shape", func() { be.MatMul(w, badShape) })
				refusal(t, "zero-panel", func() { be.BatchedMatMul(w, x, 0) })
				t.Run(kind.String()+"-capture", func(t *testing.T) {
					if !be.GraphBegin() {
						t.Fatal("required real capture unavailable; require FAK_CUDA_GRAPH=1 and no timestamp profiling")
					}
					defer be.GraphAbort()
					refusal(t, "preallocation", func() { be.MatMul(w, x) })
					if !be.IsCapturing() {
						t.Fatal("refusal ended caller-owned capture")
					}
				})
			}
		}
	}
}

// Independent CPU GGUF decoders predate this CUDA port. Exercise all packed
// bits/scales, including Q6's alternating two-byte-aligned block starts.
func cudaQ56Fixture(kind Dtype, out, in int) ([]byte, []float32) {
	block := 176
	if kind == Q6_K {
		block = 210
	}
	raw := make([]byte, out*(in/256)*block)
	decoded := make([]float32, out*in)
	for b := 0; b < len(raw)/block; b++ {
		row := raw[b*block : (b+1)*block]
		for i := range row {
			row[i] = byte(i*37 + b*19 + 11)
		}
		if kind == Q5_K {
			binary.LittleEndian.PutUint16(row, 0x2800)     // 1/32
			binary.LittleEndian.PutUint16(row[2:], 0x2000) // 1/128
			dequantQ5K(decoded[b*256:], row)
		} else {
			binary.LittleEndian.PutUint16(row[208:], 0x2400) // 1/64
			dequantQ6K(decoded[b*256:], row)
		}
	}
	return raw, decoded
}

// Each block max is exactly127*2^e, so the CUDA F32 division and half-scale
// conversion both produce exact2^e. Quarter steps and signed half ties exercise
// roundf-away-from-zero; the effective oracle is explicitly NOT original F32.
func cudaQ56Activations(p, in int) ([]float32, []float32) {
	x := make([]float32, p*in)
	effective := make([]float32, len(x))
	for b := 0; b < len(x)/32; b++ {
		d := float32(math.Ldexp(1, b%5-5))
		for j := 0; j < 32; j++ {
			v := (float32((j*17+b*7)%247-123) + 0.25) * d
			if j == 0 {
				v = 127 * d
			}
			if j == 1 {
				v = -127 * d
			}
			if j == 2 {
				v = 0.5 * d
			}
			if j == 3 {
				v = -0.5 * d
			}
			x[b*32+j] = v
			effective[b*32+j] = float32(math.Round(float64(v/d))) * d
		}
	}
	return x, effective
}
