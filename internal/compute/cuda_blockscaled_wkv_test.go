//go:build cuda && cgo

package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sync/atomic"
	"testing"
)

// This opt-in component witness calls the new #13987 API, not MatMul or the
// SM100-only exported cast. On a separately admitted SM120 node with a qualified
// native archive, run:
//
// FAK_CUDA_BLOCKSCALED_WKV_REQUIRED=1 go test -tags cuda ./internal/compute -run '^TestCUDABlockScaledWKVSM120Witness$' -count=1 -v
//
// Required mode never skips an absent device, unsupported profile or heuristic.
// Run alone: transfer counters and the producer stream are backend-global. The
// largest fixture needs about 180 MiB host and 112 MiB device operand/scratch
// storage. Prior-case input/output pools retained by Backend.Free bring the
// whole-test device estimate to about 127 MiB, plus CUDA's context/workspace;
// these are estimates, not measured resource admission. No timing, model quality
// or fault-recovery claim follows from this exact-representable fixture.
// Nonfinite activation/result and rounding/saturation cases remain separate.
// Capture, stale-epoch and asynchronous-failure witnesses require separate real
// owner transitions;
// this test does not forge another owner's latch, capture or pending state.
// fak-test:runtime slow est=30s lane=optin
func TestCUDABlockScaledWKVSM120Witness(t *testing.T) {
	if os.Getenv("FAK_CUDA_BLOCKSCALED_WKV_REQUIRED") != "1" {
		t.Skip("set FAK_CUDA_BLOCKSCALED_WKV_REQUIRED=1 on an admitted SM120 acceptance node")
	}
	backend, ok := Lookup("cuda")
	if !ok || backend == nil || backend.Name() != "cuda" {
		t.Fatal("required witness: exact CUDA backend is not registered")
	}
	c, ok := backend.(*cudaBackend)
	if !ok {
		t.Fatalf("registered CUDA backend has type %T", backend)
	}
	api, ok := backend.(BlockScaledWKVBackend)
	if !ok {
		t.Fatal("CUDA backend does not implement BlockScaledWKVBackend")
	}
	setEnabled := func(t *testing.T, enabled bool) {
		t.Helper()
		if err := ConfigureCUDABlockScaledWKV(backend, enabled); err != nil {
			t.Fatalf("explicit blockscaled configuration: %v", err)
		}
	}
	setEnabled(t, true)
	t.Cleanup(func() { setEnabled(t, false) })
	for _, rows := range []int{128, 512, 4096} {
		capability, err := api.BlockScaledWKVSupported(rows)
		if err != nil {
			t.Fatalf("required P=%d support: %v", rows, err)
		}
		if capability.Rows != rows || capability.ComputeMajor != 12 || capability.ComputeMinor != 0 || capability.CUDARuntimeVersion < 12090 || capability.CUBLASVersion < 120901 || capability.Qualified {
			t.Fatalf("unexpected component capability: %+v", capability)
		}
		t.Logf("admitted component (not model-qualified): %+v", capability)
	}

	// Eight nonzeros per output row bound the CPU oracle to O(P*512*8).
	// The first five j values cover all 160 scale columns within every group
	// of 32 weight rows, including all 128x4 vendor scale-tile boundaries.
	type term struct {
		k     int
		value float64
	}
	var terms [512][8]term
	codes := make([]byte, 512*5120)
	scales := make([]byte, 16*160)
	for tile := 0; tile < 16; tile++ {
		for kb := 0; kb < 160; kb++ {
			scales[tile*160+kb] = byte(127 + (tile*11+kb*7)%5 - 2)
		}
	}
	code := [...]byte{0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0}
	value := [...]float64{0.5, 1, 2, -0.5, -1, -2}
	for col := range terms {
		for j := range terms[col] {
			kb := (col*5 + j*23) % 160
			k := kb*32 + (col*7+j*11)%32
			v := (col + j*3) % len(code)
			codes[col*5120+k] = code[v]
			terms[col][j] = term{k, math.Ldexp(value[v], int(scales[(col/32)*160+kb])-127)}
		}
	}
	weights, err := api.UploadBlockScaledWKV(codes, scales)
	if err != nil || weights == nil {
		t.Fatalf("required weight upload: weights=%T error=%v", weights, err)
	}
	t.Cleanup(func() {
		if err := weights.Release(); err != nil {
			t.Errorf("owned weight retirement (never force release on uncertainty): %v", err)
		}
	})

	type observations struct {
		h2d, d2h, fence uint64
		faults          int
	}
	observe := func() observations {
		return observations{c.H2DXferBytes(), c.HostXferBytes(), atomic.LoadUint64(&c.fenceGen), c.faultLatch.faultCount()}
	}
	refuseRun := func(t *testing.T, label string, w BlockScaledWKVWeights, x Tensor, rows int, unsupported bool) {
		t.Helper()
		before := observe()
		out, err := api.BlockScaledWKV(w, x, rows)
		if out.buf != nil {
			backend.Free(out)
			t.Fatalf("%s refusal exposed an output", label)
		}
		if err == nil || (unsupported && !errors.Is(err, ErrBlockScaledGEMMUnsupported)) {
			t.Fatalf("%s: expected refusal, got %v", label, err)
		}
		if after := observe(); after != before {
			t.Fatalf("%s preflight changed transfer/fence/fault state: before=%+v after=%+v", label, before, after)
		}
	}
	refuseUpload := func(t *testing.T, label string, w, s []byte, unsupported bool) {
		t.Helper()
		before := observe()
		got, err := api.UploadBlockScaledWKV(w, s)
		if got != nil {
			_ = got.Release()
			t.Fatalf("%s refusal exposed weights", label)
		}
		if err == nil || (unsupported && !errors.Is(err, ErrBlockScaledGEMMUnsupported)) {
			t.Fatalf("%s: expected refusal, got %v", label, err)
		}
		if after := observe(); after != before {
			t.Fatalf("%s preflight changed transfer/fence/fault state: before=%+v after=%+v", label, before, after)
		}
	}
	checkInput := func(t *testing.T, x Tensor, want []float32) {
		t.Helper()
		got := backend.Read(x)
		if len(got) != len(want) {
			t.Fatalf("input extent changed: got %d want %d", len(got), len(want))
		}
		for i, v := range got {
			if math.Float32bits(v) != math.Float32bits(want[i]) {
				t.Fatalf("input mutated at %d: got %g want %g", i, v, want[i])
			}
		}
	}

	for _, rows := range []int{128, 512, 4096} {
		if !t.Run(fmt.Sprintf("P%d", rows), func(t *testing.T) {
			xHost := make([]float32, rows*5120)
			for row := 0; row < rows; row++ {
				for kb := 0; kb < 160; kb++ {
					scale := float32(math.Ldexp(1, (row*3+kb*5)%7-3))
					for lane := 0; lane < 32; lane++ {
						// Every block contains +/-4. Its cast scale is 2^(e-6),
						// so normalized values are exact E4M3 multiples of 64.
						xHost[row*5120+kb*32+lane] = float32((row*7+kb*3+lane*5)%9-4) * scale
					}
				}
			}
			x := c.UploadClass(NewF32(Default(), []int{rows, 5120}, xHost), F32, MemoryActivation, "blockscaled-wkv-witness")
			t.Cleanup(func() { backend.Free(x) })
			xb, ok := x.buf.(*cudaBuf)
			if !ok || xb == nil || xb.ptr == nil || xb.managed || xb.device != 0 || x.Backend() != backend {
				t.Fatal("fixture did not obtain an owned device-0 F32 input")
			}
			if rows == 128 {
				if _, err := api.BlockScaledWKVSupported(127); !errors.Is(err, ErrBlockScaledGEMMUnsupported) {
					t.Fatalf("unsupported P accepted: %v", err)
				}
				refuseRun(t, "unsupported P", weights, x, 127, true)
				refuseRun(t, "nil weights", nil, x, rows, false)
				refuseRun(t, "host input", weights, NewF32(Default(), []int{rows, 5120}, xHost), rows, false)
				wrongShape := x
				wrongShape.Shape = []int{rows, 5119} // metadata view only; no foreign buffer/state edit
				refuseRun(t, "wrong input shape", weights, wrongShape, rows, false)
				refuseUpload(t, "short codes", codes[:len(codes)-1], scales, false)
				if !t.Run("off", func(t *testing.T) {
					setEnabled(t, false)
					t.Cleanup(func() { setEnabled(t, true) })
					if _, err := api.BlockScaledWKVSupported(rows); !errors.Is(err, ErrBlockScaledGEMMUnsupported) {
						t.Fatalf("off support accepted: %v", err)
					}
					refuseUpload(t, "off upload", codes, scales, true)
					refuseRun(t, "off run", weights, x, rows, true)
				}) {
					return
				}
				// Mutate only our host upload sources after success. A live weight
				// owner must retain its immutable device copy, not alias these bytes.
				savedCode := codes[0]
				codes[0] = 0x7f
				refuseUpload(t, "NaN weight", codes, scales, false)
				codes[0] = savedCode
				scales[0] = 255
				refuseUpload(t, "NaN scale", codes, scales, false)
				codes[0] = 0x7f // keep both host sources invalid for all positive runs
				checkInput(t, x, xHost)
			}

			before := observe()
			out, err := api.BlockScaledWKV(weights, x, rows)
			if err != nil {
				t.Fatalf("required native run: %v", err)
			}
			t.Cleanup(func() { backend.Free(out) })
			after := observe()
			if after.h2d != before.h2d || after.d2h-before.d2h != 8 || after.fence != before.fence+1 || after.faults != before.faults {
				t.Fatalf("run must publish after its fence and two 4-byte status reads: before=%+v after=%+v", before, after)
			}
			ob, ok := out.buf.(*cudaBuf)
			if !ok || ob == nil || ob.ptr == nil || ob.managed || ob.device != 0 || out.Backend() != backend || out.Dtype != F32 || out.Layout != RowMajor || out.Quant != nil || len(out.Shape) != 2 || out.Shape[0] != rows || out.Shape[1] != 512 {
				t.Fatalf("output is not owned device F32[%d,512]: shape=%v", rows, out.Shape)
			}
			got := backend.Read(out)
			if len(got) != rows*512 {
				t.Fatalf("output length %d, want %d", len(got), rows*512)
			}
			nonzero := false
			for row := 0; row < rows; row++ {
				for col := range terms {
					var want float64
					for _, term := range terms[col] {
						want += float64(xHost[row*5120+term.k]) * term.value
					}
					// Dyadic products and eight-term sums fit exactly in F32;
					// no quantization tolerance can hide layout/transpose errors.
					if v := got[row*512+col]; float64(v) != want {
						t.Fatalf("row=%d col=%d got=%g want=%g", row, col, v, want)
					}
					nonzero = nonzero || want != 0
				}
			}
			if !nonzero {
				t.Fatal("reference fixture accidentally became all zero")
			}
			checkInput(t, x, xHost)
			if rows == 4096 {
				if !t.Run("release-while-off", func(t *testing.T) {
					setEnabled(t, false)
					t.Cleanup(func() { setEnabled(t, true) })
					if err := weights.Release(); err != nil {
						t.Fatalf("valid owner release while off: %v", err)
					}
					if err := weights.Release(); err != nil {
						t.Fatalf("idempotent owner release: %v", err)
					}
				}) {
					return
				}
				refuseRun(t, "released weights", weights, x, rows, false)
				checkInput(t, x, xHost)
			}
		}) {
			return // Do not submit later cases after an actual device failure.
		}
	}
}
