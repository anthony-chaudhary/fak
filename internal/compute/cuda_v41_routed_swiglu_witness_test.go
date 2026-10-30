//go:build cuda && cgo && fak_cuda_v41_routed_swiglu

// SPDX-License-Identifier: Apache-2.0
package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// Oracle/fixtures copied from vulkan_v41_routed_swiglu_test.go at be41c0a4.
// Vulkan-specific Bounds deliberately NOT copied. Host comparison is diagnostic.
func cudaRoutedWitnessBF16(v float32) float32 {
	bits := math.Float32bits(v)
	return math.Float32frombits((bits + 0x7fff + (bits>>16)&1) &^ 0xffff)
}

// cudaRoutedWitnessHostLane is the host graph through the route-weight product:
// widened BF16 gate/up, the asymmetric clamp, F32 silu(g)*u with
// float32(math.Exp), then the F32 weight product. It returns the clamped gate
// as well, which scales the exp error bound.
func cudaRoutedWitnessHostLane(gate, up, limit, weight float32) (pre, g float32) {
	g, u := cudaRoutedWitnessBF16(gate), cudaRoutedWitnessBF16(up)
	if limit > 0 {
		g = min(g, limit)
		u = max(-limit, min(u, limit))
	}
	f := (g / (1 + float32(math.Exp(float64(-g))))) * u
	return f * weight, g
}

func cudaRoutedWitnessLanes(rng *rand.Rand, n int, limit float32) (gate, up []float32) {
	gate, up = make([]float32, n), make([]float32, n)
	scales := []float64{1e-3, 0.05, 1, 4, 12, 40}
	for i := range gate {
		s := scales[i%len(scales)]
		gate[i] = float32(rng.NormFloat64() * s)
		up[i] = float32(rng.NormFloat64() * s)
	}
	special := []struct{ g, u float32 }{
		{0, 1}, {float32(math.Copysign(0, -1)), 1}, {1, 0}, {1, float32(math.Copysign(0, -1))},
		{limit, limit}, {-limit, -limit}, {limit * 4, limit * 4}, {-limit * 4, -limit * 4},
		{-88, 3}, {-90, 3}, {-200, -3}, {88, 3}, {1e-40, 1}, {-1e-40, 2},
		{3.0e38, 1e-30}, {1.00390625, -0.51}, {0.501953125, 0.501953125},
	}
	for i, s := range special {
		if i < n {
			gate[i], up[i] = s.g, s.u
		}
	}
	return gate, up
}

// fak-test:runtime integration est=5s lane=optin
// Structural/fault witness only; ending SKIP prevents full numerical acceptance.
func TestCUDAV41RoutedSwiGLUDiagnosticWitness(t *testing.T) {
	if os.Getenv("FAK_CUDA_ROUTED_SWIGLU_WITNESS") != "1" {
		t.Skip("explicit device opt-in required; skip is not acceptance")
	}
	for _, key := range []string{"FAK_WITNESS_SOURCE_SHA", "FAK_WITNESS_LIB_SHA256", "FAK_WITNESS_DEVICE_UUID", "FAK_WITNESS_BUILD_RECEIPT"} {
		if os.Getenv(key) == "" {
			t.Fatalf("missing evidence pin %s", key)
		}
		t.Logf("operator evidence %s=%s; verify against receipts", key, os.Getenv(key))
	}
	c, ok := Pick("cuda").(*cudaBackend)
	if !ok || c == nil {
		t.Fatal("CUDA backend absent")
	}
	if c.V41RoutedSwiGLUAvailable() {
		t.Fatal("unexpected public qualification")
	}
	exact := func(t *testing.T, label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length", label)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]: %08x != %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	upload := func(t *testing.T, x []float32) Tensor {
		t.Helper()
		d := c.UploadClass(NewF32(Default(), []int{len(x)}, x), F32, MemoryActivation, "routed-swiglu-witness")
		t.Cleanup(func() { c.Free(d) })
		exact(t, "checked upload", c.Read(d), x)
		return d
	}
	invoke := func(t *testing.T, g, u Tensor, limit, weight float32) (Tensor, error) {
		t.Helper()
		out, err := c.cudaV41RoutedSwiGLU(g, u, limit, weight, true)
		if err != nil {
			if out.buf != nil {
				t.Fatal("error published output")
			}
			return out, err
		}
		t.Cleanup(func() { c.Free(out) })
		ob, ok := out.buf.(*cudaBuf)
		if !ok || ob == nil || ob.ptr == nil {
			t.Fatal("absent output")
		}
		for _, x := range []Tensor{g, u} {
			if ob.ptr == x.buf.(*cudaBuf).ptr {
				t.Fatal("output aliases input")
			}
		}
		return out, nil
	}
	rng := rand.New(rand.NewSource(13815))
	for _, tc := range []struct {
		n             int
		limit, weight float32
	}{{2048, 10, .501953125}, {2048, 0, 1}, {2304, .75, 2.5}, {1 << 20, 10, .125}, {1, 10, -.75}, {65, -1, 0}, {65, 10, -32}} {
		t.Run(fmt.Sprintf("diagnostic_n%d_limit%g_weight%g", tc.n, tc.limit, tc.weight), func(t *testing.T) {
			g, u := cudaRoutedWitnessLanes(rng, tc.n, tc.limit)
			gd, ud := upload(t, g), upload(t, u)
			out, err := invoke(t, gd, ud, tc.limit, tc.weight)
			if err != nil {
				t.Fatal(err)
			}
			got := c.Read(out)
			if len(got) != tc.n {
				t.Fatal("output length")
			}
			mismatch := 0
			for i, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || math.Float32bits(v)&0xffff != 0 {
					t.Fatalf("lane %d not finite widened BF16", i)
				}
				pre, _ := cudaRoutedWitnessHostLane(g[i], u[i], tc.limit, tc.weight)
				want := cudaRoutedWitnessBF16(pre)
				if math.Float32bits(v) != math.Float32bits(want) {
					mismatch++
					if mismatch <= 16 {
						t.Logf("DIAGNOSTIC lane=%d gate=%08x up=%08x device=%08x host=%08x", i, math.Float32bits(g[i]), math.Float32bits(u[i]), math.Float32bits(v), math.Float32bits(want))
					}
				}
			}
			t.Logf("DIAGNOSTIC mismatches=%d/%d; no CUDA numerical threshold or acceptance", mismatch, tc.n)
			exact(t, "gate immutable", c.Read(gd), g)
			exact(t, "up immutable", c.Read(ud), u)
		})
	}
	ones := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = 1
		}
		return x
	}
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	run := func(t *testing.T, g, u []float32, limit, weight float32) error {
		gd, ud := upload(t, g), upload(t, u)
		_, err := invoke(t, gd, ud, limit, weight)
		exact(t, "fault gate immutable", c.Read(gd), g)
		exact(t, "fault up immutable", c.Read(ud), u)
		return err
	}
	for name, tc := range map[string]struct {
		gate, up      []float32
		limit, weight float32
		stage         V41RoutedSwiGLUStage
		index         int
	}{
		"nonfinite gate":        {[]float32{1, 1, 1, inf, 1, nan, 1, 1}, ones(8), 10, 1, V41RoutedSwiGLUStageActivation, 3},
		"up rounds to BF16 inf": {ones(8), []float32{1, 1, 1, 1, 1, 1, 1, math.MaxFloat32}, 10, 1, V41RoutedSwiGLUStageActivation, 7},
		"activation overflow":   {[]float32{1, 1e20, 1}, []float32{1, 1e20, 1}, 0, 1, V41RoutedSwiGLUStageActivation, 1},
		"weighted overflow":     {[]float32{1, 1, 1e19, 1e19}, []float32{1, 1, 1e19, 1e19}, 0, 10, V41RoutedSwiGLUStageDownOperand, 2},
		// silu(2^64) is 2^64 and 2^64*0x5f7f0000 is the BF16 maximum exactly; the
		// finite weighted product 3.398e38 then rounds past it to BF16 infinity.
		"down operand BF16 inf":  {[]float32{1, 0x1p64}, []float32{1, math.Float32frombits(0x5f7f0000)}, 0, 1.0025, V41RoutedSwiGLUStageDownOperand, 1},
		"activation before down": {[]float32{1e19, 1, 1, nan}, []float32{1e19, 1, 1, 1}, 0, 10, V41RoutedSwiGLUStageActivation, 3},
	} {
		t.Run(name, func(t *testing.T) {
			var fault *V41RoutedSwiGLUFault
			if err := run(t, tc.gate, tc.up, tc.limit, tc.weight); !errors.As(err, &fault) || fault.Stage != tc.stage || fault.Index != tc.index {
				t.Fatalf("err = %v, want %v fault at lane %d", err, tc.stage, tc.index)
			}
		})
	}

	t.Run("queued_producers", func(t *testing.T) {
		// No read/sync between producers and candidate. This exercises submission
		// order, not proof that producers were still pending at kernel submission.
		g, u := ones(129), ones(129)
		gateDelta, upDelta := make([]float32, 129), make([]float32, 129)
		for i := range g {
			if i%2 == 0 {
				gateDelta[i] = -1
			} else {
				upDelta[i] = -1
			}
		}
		gd, ud := upload(t, g), upload(t, u)
		gchange, uchange := upload(t, gateDelta), upload(t, upDelta)
		// Even lanes require gate 1->0, odd lanes require up 1->0. A stale
		// relevant producer gives nonzero; both correctly ordered give zero.
		c.AddInPlace(gd, gchange)
		c.AddInPlace(ud, uchange)
		out, err := invoke(t, gd, ud, 10, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i := range g {
			g[i] += gateDelta[i]
			u[i] += upDelta[i]
		}
		exact(t, "zero activation", c.Read(out), make([]float32, 129))
		exact(t, "queued gate immutable", c.Read(gd), g)
		exact(t, "queued up immutable", c.Read(ud), u)
	})
	t.Run("admission", func(t *testing.T) {
		gd, ud := upload(t, ones(4)), upload(t, ones(4))
		short := upload(t, ones(3))
		for _, tc := range []struct {
			name          string
			g, u          Tensor
			limit, weight float32
			witness       bool
		}{
			{"host", NewF32(Default(), []int{4}, ones(4)), ud, 10, 1, true},
			{"shape", gd, short, 10, 1, true}, {"nan_limit", gd, ud, nan, 1, true},
			{"inf_weight", gd, ud, 10, inf, true}, {"public", gd, ud, 10, 1, false},
		} {
			t.Run(tc.name, func(t *testing.T) {
				out, err := c.cudaV41RoutedSwiGLU(tc.g, tc.u, tc.limit, tc.weight, tc.witness)
				var backend *BackendError
				if !errors.As(err, &backend) || out.buf != nil {
					t.Fatalf("not fail closed: %v", err)
				}
			})
		}
	})
	t.Run("status_decoder", func(t *testing.T) {
		if v41RoutedSwiGLUStatusBytes != 16 {
			t.Fatal("status size")
		}
		good := [4]uint32{v41RoutedSwiGLUStatusMagic, 8, math.MaxUint32, math.MaxUint32}
		if err := decodeV41RoutedSwiGLUStatus(good, 8); err != nil {
			t.Fatal(err)
		}
		for _, words := range [][4]uint32{{0, 8, math.MaxUint32, math.MaxUint32}, {v41RoutedSwiGLUStatusMagic, 7, math.MaxUint32, math.MaxUint32}, {v41RoutedSwiGLUStatusMagic, 8, 8, math.MaxUint32}, {math.MaxUint32, math.MaxUint32, math.MaxUint32, math.MaxUint32}} {
			if decodeV41RoutedSwiGLUStatus(words, 8) == nil {
				t.Fatal("malformed status accepted")
			}
		}
		both := [4]uint32{v41RoutedSwiGLUStatusMagic, 8, 7, 0}
		var f *V41RoutedSwiGLUFault
		if e := decodeV41RoutedSwiGLUStatus(both, 8); !errors.As(e, &f) || f.Stage != V41RoutedSwiGLUStageActivation || f.Index != 7 {
			t.Fatalf("status ordering: %v", e)
		}
	})
	t.Run("healthy_after_faults", func(t *testing.T) {
		if err := run(t, ones(64), ones(64), 10, 1); err != nil {
			t.Fatal(err)
		}
	})
	t.Skip("DIAGNOSTIC ONLY: CUDA numerical qualification, native injected-failure/lifetime and full-model acceptance remain pending; subtest success is not qualification")
}
