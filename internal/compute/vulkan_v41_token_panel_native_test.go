//go:build vulkan && (windows || linux) && cgo

// Run on a Vulkan host: go test -tags=vulkan ./internal/compute -run TestVulkanV41TokenPanelNative
package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"
)

// Ref #12640. The existing subprocess harness sets the panel policy before
// init() builds optional pipelines. Setting it with t.Setenv after vk(t) is too
// late. An empty value is not "0", so it exercises the default-on startup.
func v41TokenPanelNativeStartup(t *testing.T, q2, q3 string) bool {
	t.Helper()
	if _, ok := Lookup("vulkan"); !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("UNQUALIFIED: required Vulkan device is not registered; native token-panel parity did not execute")
		}
		t.Skip("UNQUALIFIED: no Vulkan device; native token-panel parity did not execute")
	}
	return runVulkanStartupEnvFixture(t, 2*time.Minute, map[string]string{
		"FAK_V41_Q2K_TOKEN_PANEL":      q2,
		"FAK_V41_Q3K_TOKEN_PANEL":      q3,
		"FAK_VULKAN_Q2K_MATVEC":        "1",
		"FAK_STRIX_WAVE32_GEMV_DECODE": "0",
	})
}

func v41TokenPanelNativePhysicalDevice(t *testing.T) *vulkanBackend {
	t.Helper()
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("UNQUALIFIED: child must start with required-device mode and dispatch profiling")
	}
	v := vk(t)
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		t.Fatalf("UNQUALIFIED: native parity needs a physical GPU, deviceType=%d tier=%s", kind, v.Tier())
	}
	identity := v41TokenPanelNativeSnapshot(t, v).Identity
	t.Logf("engine=fak-native backend=%s device=%q driver=%q runtime=%q", identity.Backend, identity.Device, identity.Driver, identity.Runtime)
	return v
}

func v41TokenPanelNativeDevice(t *testing.T) *vulkanBackend {
	t.Helper()
	v := v41TokenPanelNativePhysicalDevice(t)
	if !v41Q2KTokenPanelEnabled() || !v41Q3KTokenPanelEnabled() || !v.SupportsQ3KMatMul() {
		t.Fatal("UNQUALIFIED: default-on token panels disabled or native Q3_K scalar pipeline unavailable")
	}
	previous := vulkanDebugSelectQ2KMatvec(true)
	t.Cleanup(func() { vulkanDebugSelectQ2KMatvec(previous) })
	// The selector returns its previous state. This second call observes whether
	// the first one actually selected a built q2k_matvec pipeline.
	if !vulkanDebugSelectQ2KMatvec(true) {
		t.Fatal("UNQUALIFIED: native Q2_K mode-0 matvec pipeline unavailable")
	}
	return v
}

func v41TokenPanelNativeSupports(v *vulkanBackend, w Tensor, rows int) bool {
	if w.Dtype == Q2_K {
		return v.SupportsV41Q2KTokenPanel(w, rows)
	}
	return v.SupportsV41Q3KTokenPanel(w, rows)
}

func v41TokenPanelNativeCall(v *vulkanBackend, dtype Dtype, w, x Tensor, rows int) (Tensor, error) {
	if dtype == Q2_K {
		return v.V41Q2KTokenPanel(w, x, rows)
	}
	return v.V41Q3KTokenPanel(w, x, rows)
}

func v41TokenPanelNativeSnapshot(t *testing.T, v *vulkanBackend) BackendExecutionSnapshot {
	t.Helper()
	s, err := v.BackendExecutionSnapshot()
	if err != nil || !s.TransferCountersObserved || !s.DeviceAllocationObserved {
		t.Fatalf("UNQUALIFIED: native counters/allocation observation unavailable: %+v %v", s, err)
	}
	return s
}

func v41TokenPanelNativeFixture(dtype Dtype, out, in, rows int, exact bool) (Tensor, []float32) {
	blockBytes := 84
	if dtype == Q3_K {
		blockBytes = 110
	}
	raw := make([]byte, out*(in/256)*blockBytes)
	rng := rand.New(rand.NewSource(12640 + int64(out*in)))
	_, _ = rng.Read(raw)
	for off := 0; off < len(raw); off += blockBytes {
		if dtype == Q2_K {
			binary.LittleEndian.PutUint16(raw[off+80:], 0x2000) // 1/128
			binary.LittleEndian.PutUint16(raw[off+82:], 0x1c00) // 1/256
			if !exact {
				binaryPutFloat16(raw[off+80:], float32(0.01+0.05*rng.Float64()))
				binaryPutFloat16(raw[off+82:], float32(0.005+0.02*rng.Float64()))
			}
		} else {
			binary.LittleEndian.PutUint16(raw[off+108:], 0x1c00) // 1/256
			if !exact {
				binaryPutFloat16(raw[off+108:], float32(0.01+0.05*rng.Float64()))
			}
		}
	}
	x := make([]float32, rows*in)
	for token := 0; token < rows; token++ {
		for col := 0; col < in; col++ {
			value := float32((col*3+token*11)%31-15) / 32
			if !exact {
				value = float32(math.Ldexp(rng.Float64()*2-1, (col+token)%7-3))
				value = math.Float32frombits(math.Float32bits(value) & 0xffff0000)
			}
			x[token*in+col] = value
		}
	}
	if dtype == Q2_K {
		return NewQ2K(Default(), []int{out, in}, raw), x
	}
	return NewQ3K(Default(), []int{out, in}, raw), x
}

// These native cases independently require scalar/tile bit equality; compiled
// arithmetic-order evidence does not replace physical parity. There is no
// declared tile-specific ULP allowance. Exact fixtures have |integer numerator| <= 5120*105*15 < 2^24
// (Q2) or 2304*128*15 < 2^24 (Q3), on the common 1/8192 grid. Every
// intermediate is representable, including either contraction choice.
// BF16-input fixtures additionally require observed scalar/tile bit equality;
// a mismatch is an unqualified model route, not an Approx-tolerance pass.
// Passing these finite fixtures does not prove universal kernel equivalence.
// The ordinary MatMul Approx contract does not qualify this model route.
// The V4.1 attn.wkv shape pins the dense KV panel route to per-token MatMul
// bits (Ref #12640).
// fak-test:runtime integration est=30s lane=optin
func TestVulkanV41TokenPanelNativeParity(t *testing.T) {
	if v41TokenPanelNativeStartup(t, "", "") {
		return
	}
	v := v41TokenPanelNativeDevice(t)
	for _, dtype := range []Dtype{Q2_K, Q3_K} {
		fullOut, fullIn := 2304, 5120
		if dtype == Q3_K {
			fullOut, fullIn = 5120, 2304
		}
		shapes := [][2]int{{1, 256}, {65, 1280}, {fullOut, fullIn}}
		kvShape := [2]int{512, 4096} // V4.1 attn.wkv cold-prefill panel
		if dtype == Q2_K {
			shapes = append(shapes, kvShape)
		}
		for _, shape := range shapes {
			for _, rows := range []int{2, 4, 8, 9, 64} {
				// Keep the grid additions bounded while covering an odd output
				// tail, a masked token tail and the full admitted token extent.
				if rows > 8 && shape != [2]int{65, 1280} && shape != kvShape {
					continue
				}
				for _, exact := range []bool{true, false} {
					t.Run(fmt.Sprintf("%s/out%d_in%d_rows%d/exact_%t", dtype, shape[0], shape[1], rows, exact), func(t *testing.T) {
						out, in := shape[0], shape[1]
						hw, values := v41TokenPanelNativeFixture(dtype, out, in, rows, exact)
						w := v.Upload(hw, dtype)
						defer v.Free(w)
						if !v41TokenPanelNativeSupports(v, w, rows) {
							if rows > 8 {
								t.Skip("UNQUALIFIED: loaded module or device does not admit this grid extent; grid parity did not execute")
							}
							t.Fatal("UNQUALIFIED: tile module, float controls, or actual device geometry unavailable")
						}
						wb, ok := w.buf.(*vulkanBuf)
						blockBytes := 84
						if dtype == Q3_K {
							blockBytes = 110
						}
						if !ok || wb.ptr == nil || w.Dtype != dtype || wb.n != (out*(in/256)*blockBytes+3)&^3 {
							t.Fatal("packed resident weight extent/padding was not preserved")
						}
						x := v.UploadClass(NewF32(Default(), []int{len(values)}, values), F32, MemoryActivation, "token-panel native parity")
						defer v.Free(x)
						want := make([]float32, 0, rows*out)
						beforeScalar := v.VulkanDebugDispatchProfileSnapshot()
						for row := 0; row < rows; row++ {
							func() {
								dx := v.UploadClass(NewF32(Default(), []int{in}, values[row*in:(row+1)*in]), F32, MemoryActivation, "token-panel scalar row")
								defer v.Free(dx)
								dy := v.MatMul(w, dx)
								defer v.Free(dy)
								values := v.Read(dy)
								if len(values) != out {
									t.Fatalf("native scalar row=%d length=%d, want %d", row, len(values), out)
								}
								want = append(want, values...)
							}()
						}
						v41TokenPanelNativeDispatches(t, dtype, beforeScalar, v.VulkanDebugDispatchProfileSnapshot(), uint64(rows))
						got := v41TokenPanelNativeRead(t, v, dtype, w, x, rows, rows >= 8)
						v41TokenPanelNativeEqual(t, got, want)
						assertBitIdentical(t, "input remains unchanged", values, v.Read(x))
						t.Logf("NATIVE_TOKEN_PANEL_PARITY dtype=%s out=%d in=%d rows=%d exact_fixture=%t step_batch=%t scalar_dispatches=%d tile_dispatches=1 bit_equal=%d", dtype, out, in, rows, exact, rows >= 8, rows, len(got))
					})
				}
			}
		}
	}
}

func v41TokenPanelNativeDispatches(t *testing.T, dtype Dtype, before, after VulkanDispatchProfile, want uint64) {
	t.Helper()
	compute := after.ComputeDispatches - before.ComputeDispatches
	q2 := after.Q2KMatmulDispatches - before.Q2KMatmulDispatches
	q4 := after.Q4KMatmulDispatches - before.Q4KMatmulDispatches
	other := after.OtherComputeDispatches - before.OtherComputeDispatches
	matmul := after.OtherMatmulDispatches - before.OtherMatmulDispatches
	if compute != want || q4 != 0 ||
		(dtype == Q2_K && (q2 != want || other != 0 || matmul != 0)) ||
		(dtype == Q3_K && (q2 != 0 || other != want || matmul != want)) {
		t.Fatalf("UNQUALIFIED: dtype=%s want %d native matmul dispatches, got compute=%d q2=%d q4=%d other=%d other_matmul=%d", dtype, want, compute, q2, q4, other, matmul)
	}
}

func v41TokenPanelNativeRead(t *testing.T, v *vulkanBackend, dtype Dtype, w, x Tensor, rows int, step bool) []float32 {
	t.Helper()
	var output Tensor
	defer func() {
		v.Free(output)
		if b, ok := output.buf.(*vulkanBuf); ok && b.ptr != nil {
			t.Error("caller-owned panel storage remained live after Free")
		}
	}()
	if step {
		if !v.BeginV41StepBatch() {
			t.Fatal("UNQUALIFIED: could not open native V4.1 step batch")
		}
		defer func() {
			if err := v.EndV41StepBatch(); err != nil {
				t.Errorf("closing native step batch: %v", err)
			}
		}()
	}
	before := v41TokenPanelNativeSnapshot(t, v)
	beforeDispatch := v.VulkanDebugDispatchProfileSnapshot()
	var err error
	output, err = v41TokenPanelNativeCall(v, dtype, w, x, rows)
	if err != nil {
		t.Fatalf("selected native tile failed, without scalar replay: %v", err)
	}
	if output.Backend() != Backend(v) || output.Dtype != F32 || output.Layout != RowMajor || output.Quant != nil || len(output.Shape) != 1 || output.Shape[0] != rows*w.Shape[0] || output.Buf() == x.Buf() || output.Buf() == w.Buf() {
		t.Fatal("tile result is not fresh caller-owned flat token-major F32 storage")
	}
	if b, ok := output.buf.(*vulkanBuf); !ok || b.ptr == nil || !b.v41CheckedRead {
		t.Fatal("tile result lacks live checked-readback storage")
	}
	issued := v41TokenPanelNativeSnapshot(t, v)
	delta, err := BackendExecutionDelta(before, issued)
	if err != nil || delta.Counters.ComputeDispatches != 1 || delta.Counters.H2DCount != 0 || delta.Counters.H2DBytes != 0 || delta.Counters.D2HCount != 0 || delta.Counters.D2HBytes != 0 || delta.Counters.D2DCopies != 0 || delta.Counters.D2DBytes != 0 || delta.Counters.Fallbacks != 0 {
		t.Fatalf("tile did not issue exactly one resident native dispatch: %+v %v", delta.Counters, err)
	}
	v41TokenPanelNativeDispatches(t, dtype, beforeDispatch, v.VulkanDebugDispatchProfileSnapshot(), 1)
	return append([]float32(nil), v.Read(output)...)
}

func v41TokenPanelNativeEqual(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("native panel length=%d scalar length=%d", len(got), len(want))
	}
	var mismatches int
	var maxULP uint64
	first := -1
	ordered := func(x float32) uint64 {
		bits := math.Float32bits(x)
		if bits&0x80000000 != 0 {
			return uint64(^bits)
		}
		return uint64(bits | 0x80000000)
	}
	for i := range want {
		if math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) || math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("UNQUALIFIED: nonfinite native output at index=%d scalar=%g tile=%g", i, want[i], got[i])
		}
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			mismatches++
			if first < 0 {
				first = i
			}
			a, b := ordered(got[i]), ordered(want[i])
			if a < b {
				a, b = b, a
			}
			maxULP = max(maxULP, a-b)
		}
	}
	if mismatches != 0 {
		t.Fatalf("UNQUALIFIED: native tile/scalar bit mismatches=%d/%d max_ordered_float32_ulp=%d first_index=%d scalar=%g(0x%08x) tile=%g(0x%08x); no model-route ULP allowance is declared", mismatches, len(got), maxULP, first, want[first], math.Float32bits(want[first]), got[first], math.Float32bits(got[first]))
	}
}

// Native pre-dispatch refusals must leave no output, dispatch, or allocation;
// failed checked readback must publish no host slice and keep cleanup usable.
// This does not inject an allocation/dispatch/device-loss failure or prove the
// separate model cohort's status-ordering and session-retirement contract.
// fak-test:runtime integration est=3s lane=optin
func TestVulkanV41TokenPanelNativeRefusalCleanup(t *testing.T) {
	if v41TokenPanelNativeStartup(t, "", "") {
		return
	}
	v := v41TokenPanelNativeDevice(t)
	for _, dtype := range []Dtype{Q2_K, Q3_K} {
		t.Run(dtype.String(), func(t *testing.T) {
			const out, in, rows = 65, 256, 2
			hw, values := v41TokenPanelNativeFixture(dtype, out, in, rows, true)
			w := v.Upload(hw, dtype)
			defer v.Free(w)
			x := v.UploadClass(NewF32(Default(), []int{len(values)}, values), F32, MemoryActivation, "token-panel refusal input")
			defer v.Free(x)
			if !v41TokenPanelNativeSupports(v, w, rows) {
				t.Fatal("UNQUALIFIED: required native tile capability unavailable")
			}
			refuse := func(w, x Tensor, rows int, sentinel error) {
				t.Helper()
				before := v41TokenPanelNativeSnapshot(t, v)
				y, err := v41TokenPanelNativeCall(v, dtype, w, x, rows)
				defer v.Free(y)
				var backend *BackendError
				if y.Buf() != nil || !errors.As(err, &backend) || !errors.Is(err, sentinel) {
					t.Fatalf("invalid call published output or lost typed error: output=%v error=%v", y.Buf(), err)
				}
				after := v41TokenPanelNativeSnapshot(t, v)
				if after.Counters != before.Counters || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
					t.Fatal("pre-dispatch refusal changed native counters or live allocation bytes")
				}
			}
			for _, n := range []int{0, 1, 65} {
				if v41TokenPanelNativeSupports(v, w, n) {
					t.Fatalf("unsupported row count %d admitted", n)
				}
				refuse(w, x, n, ErrVulkanInvalidGeometry)
			}
			// Legacy modules decline these widths; grid modules admit them.
			// A capability refusal must retain the same no-output/no-work path.
			for _, n := range []int{3, 9} {
				if !v41TokenPanelNativeSupports(v, w, n) {
					refuse(w, x, n, ErrVulkanInvalidGeometry)
				}
			}
			badW, badX := w, x
			badW.Layout, badX.Layout = ColMajor, ColMajor
			refuse(badW, x, rows, ErrVulkanInvalidGeometry)
			refuse(w, badX, rows, ErrVulkanInvalidGeometry)
			badX = x
			badX.Shape = []int{rows, in}
			refuse(w, badX, rows, ErrVulkanInvalidGeometry)
			refuse(hw, x, rows, ErrVulkanInvalidGeometry)
			refuse(w, NewF32(Default(), []int{len(values)}, values), rows, ErrVulkanInvalidGeometry)
			short := v.UploadClass(NewF32(Default(), []int{1}, []float32{1}), F32, MemoryActivation, "token-panel short storage")
			defer v.Free(short)
			badX = short
			badX.Shape = x.Shape
			refuse(w, badX, rows, ErrVulkanInvalidGeometry)
			v.Free(short)
			refuse(w, badX, rows, ErrVulkanInvalidGeometry)
			func() {
				v.BeginBatch()
				defer v.FlushBatch()
				if v41TokenPanelNativeSupports(v, w, rows) {
					t.Fatal("foreign recording window admitted")
				}
				refuse(w, x, rows, ErrVulkanInvalidGeometry)
			}()
			t.Run("disabled-opt-in", func(t *testing.T) {
				t.Setenv("FAK_V41_Q2K_TOKEN_PANEL", "0")
				t.Setenv("FAK_V41_Q3K_TOKEN_PANEL", "0")
				if v41TokenPanelNativeSupports(v, w, rows) {
					t.Fatal("disabled opt-in admitted")
				}
				refuse(w, x, rows, ErrVulkanExecutionFailed)
			})
			want := v41TokenPanelNativeRead(t, v, dtype, w, x, rows, false)
			func() {
				y, err := v41TokenPanelNativeCall(v, dtype, w, x, rows)
				defer v.Free(y)
				if err != nil {
					t.Fatal(err)
				}
				v.VulkanDebugSetD2HStagingFailureOnce(true)
				defer v.VulkanDebugSetD2HStagingFailureOnce(false)
				be := requireVulkanReadPanic(t, func() { _ = v.Read(y) })
				requireVulkanReadError(t, be, VulkanClassAllocationFailed, ErrVulkanAllocationFailed)
			}()
			got := v41TokenPanelNativeRead(t, v, dtype, w, x, rows, false)
			v41TokenPanelNativeEqual(t, got, want)
			assertBitIdentical(t, "input after refusals and read failure", values, v.Read(x))
		})
	}
}

// Default-on Q3 must reach a real native dispatch while Q2=0 keeps Q2
// unavailable. A late re-enable must not invent an unbuilt pipeline.
// fak-test:runtime integration est=3s lane=optin
func TestVulkanV41TokenPanelNativeQ3Only(t *testing.T) {
	for _, startupQ3 := range []string{"1", "0"} {
		t.Run("startup-q3-"+startupQ3, func(t *testing.T) {
			if v41TokenPanelNativeStartup(t, "0", startupQ3) {
				return
			}
			v := v41TokenPanelNativePhysicalDevice(t)
			const out, in, rows = 65, 256, 2
			hw, values := v41TokenPanelNativeFixture(Q3_K, out, in, rows, true)
			w := v.Upload(hw, Q3_K)
			defer v.Free(w)
			hq2, _ := v41TokenPanelNativeFixture(Q2_K, out, in, rows, true)
			q2 := v.Upload(hq2, Q2_K)
			defer v.Free(q2)
			x := v.UploadClass(NewF32(Default(), []int{len(values)}, values), F32, MemoryActivation, "Q3-only token-panel input")
			defer v.Free(x)
			refuse := func(t *testing.T, weight Tensor) {
				t.Helper()
				before := v41TokenPanelNativeSnapshot(t, v)
				if v41TokenPanelNativeSupports(v, weight, rows) {
					t.Fatal("disabled or unbuilt token-panel pipeline admitted")
				}
				y, err := v41TokenPanelNativeCall(v, weight.Dtype, weight, x, rows)
				defer v.Free(y)
				var backend *BackendError
				if y.Buf() != nil || !errors.As(err, &backend) || !errors.Is(err, ErrVulkanExecutionFailed) {
					t.Fatalf("unavailable tile published output or lost typed error: output=%v error=%v", y.Buf(), err)
				}
				after := v41TokenPanelNativeSnapshot(t, v)
				if after.Counters != before.Counters || after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
					t.Fatal("unavailable tile changed native counters or live allocation bytes")
				}
			}
			refuse(t, q2)
			t.Run("late-q2", func(t *testing.T) {
				t.Setenv("FAK_V41_Q2K_TOKEN_PANEL", "1")
				refuse(t, q2)
			})
			if startupQ3 == "0" {
				t.Setenv("FAK_V41_Q3K_TOKEN_PANEL", "1")
				refuse(t, w)
				return
			}
			if !v.SupportsQ3KMatMul() || !v.SupportsV41Q3KTokenPanel(w, rows) {
				t.Fatal("UNQUALIFIED: Q3-only scalar/tile module, float controls, or device geometry unavailable")
			}
			want := make([]float32, 0, rows*out)
			beforeScalar := v.VulkanDebugDispatchProfileSnapshot()
			for row := 0; row < rows; row++ {
				func() {
					dx := v.Upload(NewF32(Default(), []int{in}, values[row*in:(row+1)*in]), F32)
					defer v.Free(dx)
					dy := v.MatMul(w, dx)
					defer v.Free(dy)
					want = append(want, v.Read(dy)...)
				}()
			}
			v41TokenPanelNativeDispatches(t, Q3_K, beforeScalar, v.VulkanDebugDispatchProfileSnapshot(), rows)
			got := v41TokenPanelNativeRead(t, v, Q3_K, w, x, rows, false)
			v41TokenPanelNativeEqual(t, got, want)
			t.Run("disabled-q3", func(t *testing.T) {
				t.Setenv("FAK_V41_Q3K_TOKEN_PANEL", "0")
				refuse(t, w)
			})
			t.Logf("NATIVE_TOKEN_PANEL_Q3_ONLY out=%d in=%d rows=%d scalar_dispatches=%d tile_dispatches=1 bit_equal=%d", out, in, rows, rows, len(got))
		})
	}
}
