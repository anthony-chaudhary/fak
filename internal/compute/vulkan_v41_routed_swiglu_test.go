//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"testing"
)

func v41RoutedTestBF16(v float32) float32 {
	bits := math.Float32bits(v)
	return math.Float32frombits((bits + 0x7fff + (bits>>16)&1) &^ 0xffff)
}

// v41RoutedSwiGLUHostLane is the host graph through the route-weight product:
// widened BF16 gate/up, the asymmetric clamp, F32 silu(g)*u with
// float32(math.Exp), then the F32 weight product. It returns the clamped gate
// as well, which scales the exp error bound.
func v41RoutedSwiGLUHostLane(gate, up, limit, weight float32) (pre, g float32) {
	g, u := v41RoutedTestBF16(gate), v41RoutedTestBF16(up)
	if limit > 0 {
		g = min(g, limit)
		u = max(-limit, min(u, limit))
	}
	f := (g / (1 + float32(math.Exp(float64(-g))))) * u
	return f * weight, g
}

// v41RoutedSwiGLUBounds is the admitted interval for the device BF16 output.
// Vulkan bounds exp at 3+2|x| ULP and division at 2.5 ULP; add, multiply and
// the bit rounding are exact under RoundingModeRTE. Composed with the host's
// own roundings this is a relative error of at most (9.5+2|g|)*2^-23. The
// 2^-100 floor admits only the exp-overflow tail (|g| > 88.7), where the host
// flushes silu to a signed zero and the device may return a value below 1e-30.
func v41RoutedSwiGLUBounds(pre, g float32) (lo, hi float32) {
	tol := (9.5+2*math.Abs(float64(g)))*0x1p-23*math.Abs(float64(pre)) + 0x1p-100
	return v41RoutedTestBF16(float32(float64(pre) - tol)), v41RoutedTestBF16(float32(float64(pre) + tol))
}

func v41RoutedSwiGLUWitnessLanes(rng *rand.Rand, n int, limit float32) (gate, up []float32) {
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

// fak-test:runtime integration est=3s lane=optin
// Ref #13815. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// physical witness. Every device lane must be the BF16 rounding of a value
// inside the documented Vulkan precision bound around the host graph.
func TestVulkanV41RoutedSwiGLUParity(t *testing.T) {
	v := vk(t)
	if !v.V41RoutedSwiGLUAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 routed SwiGLU module or binary32 float controls")
		}
		t.Skip("V4.1 routed SwiGLU module unavailable")
	}
	rng := rand.New(rand.NewSource(13815))
	for _, c := range []struct {
		n             int
		limit, weight float32
	}{
		{2048, 10, .501953125}, {2048, 0, 1}, {2304, .75, 2.5}, {1 << 20, 10, .125}, {1, 10, -.75},
	} {
		gate, up := v41RoutedSwiGLUWitnessLanes(rng, c.n, c.limit)
		gd := v.Upload(NewF32(Default(), []int{c.n}, gate), F32)
		ud := v.Upload(NewF32(Default(), []int{c.n}, up), F32)
		out, err := v.V41RoutedSwiGLU(gd, ud, c.limit, c.weight)
		if err != nil {
			v.Free(gd)
			v.Free(ud)
			t.Fatalf("n=%d limit=%g: %v", c.n, c.limit, err)
		}
		got := v.Read(out)
		v.Free(out)
		v.Free(gd)
		v.Free(ud)
		if len(got) != c.n {
			t.Fatalf("n=%d: read %d lanes", c.n, len(got))
		}
		exact, tail, maxULP := 0, 0, uint32(0)
		for i, value := range got {
			pre, g := v41RoutedSwiGLUHostLane(gate[i], up[i], c.limit, c.weight)
			want := v41RoutedTestBF16(pre)
			if math.Float32bits(value)&0xffff != 0 {
				t.Fatalf("n=%d lane %d = %#08x is not a widened BF16 value", c.n, i, math.Float32bits(value))
			}
			if math.Float32bits(value) == math.Float32bits(want) {
				exact++
				continue
			}
			lo, hi := v41RoutedSwiGLUBounds(pre, g)
			if !(value >= lo && value <= hi) {
				t.Fatalf("n=%d limit=%g weight=%g lane %d gate=%g up=%g: device %g outside [%g,%g] (host %g)", c.n, c.limit, c.weight, i, gate[i], up[i], value, lo, hi, want)
			}
			// Lanes the relative bound alone would admit are counted in BF16 ULPs;
			// the rest are the exp-overflow tail the absolute floor admits.
			if math.Abs(float64(value-want)) > 0x1p-100 && math.Signbit(float64(value)) == math.Signbit(float64(want)) {
				a, b := math.Float32bits(value)>>16, math.Float32bits(want)>>16
				maxULP = max(maxULP, max(a, b)-min(a, b))
			} else {
				tail++
			}
		}
		t.Logf("routed_swiglu n=%d limit=%g weight=%g exact_lanes=%d/%d max_bf16_ulp=%d tail_lanes=%d", c.n, c.limit, c.weight, exact, c.n, maxULP, tail)
	}
}

// fak-test:runtime integration est=2s lane=optin
// Ref #13815. Numeric faults are typed verdicts naming the host stage and the
// lowest lane; malformed operands are backend errors. Neither publishes output.
func TestVulkanV41RoutedSwiGLUFaults(t *testing.T) {
	v := vk(t)
	if !v.V41RoutedSwiGLUAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 routed SwiGLU module or binary32 float controls")
		}
		t.Skip("V4.1 routed SwiGLU module unavailable")
	}
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	run := func(gate, up []float32, limit, weight float32) error {
		t.Helper()
		gd := v.Upload(NewF32(Default(), []int{len(gate)}, gate), F32)
		defer v.Free(gd)
		ud := v.Upload(NewF32(Default(), []int{len(up)}, up), F32)
		defer v.Free(ud)
		out, err := v.V41RoutedSwiGLU(gd, ud, limit, weight)
		if err == nil {
			v.Free(out)
		} else if out.Buf() != nil {
			t.Fatal("failed routed SwiGLU published an output")
		}
		return err
	}
	ones := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = 1
		}
		return s
	}
	for name, c := range map[string]struct {
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
			if err := run(c.gate, c.up, c.limit, c.weight); !errors.As(err, &fault) || fault.Stage != c.stage || fault.Index != c.index {
				t.Fatalf("err = %v, want %v fault at lane %d", err, c.stage, c.index)
			}
		})
	}
	t.Run("width mismatch", func(t *testing.T) {
		var backend *BackendError
		if err := run(ones(4), ones(5), 10, 1); !errors.As(err, &backend) || !errors.Is(err, ErrVulkanInvalidGeometry) {
			t.Fatalf("err = %v, want invalid geometry", err)
		}
	})
	t.Run("host tensor", func(t *testing.T) {
		host := NewF32(Default(), []int{4}, ones(4))
		ud := v.Upload(NewF32(Default(), []int{4}, ones(4)), F32)
		defer v.Free(ud)
		var backend *BackendError
		if _, err := v.V41RoutedSwiGLU(host, ud, 10, 1); !errors.As(err, &backend) || !errors.Is(err, ErrVulkanInvalidGeometry) {
			t.Fatalf("err = %v, want invalid geometry", err)
		}
	})
	t.Run("healthy after faults", func(t *testing.T) {
		if err := run(ones(64), ones(64), 10, 1); err != nil {
			t.Fatalf("routed SwiGLU after numeric faults: %v", err)
		}
	})
}
