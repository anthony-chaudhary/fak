//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=5s lane=optin
// Qualification needs a matching rebuilt native library and a physical GPU with
// eligible device-local pool capacity. A skip supplies no native witness.
func TestVulkanV41FFNNormPublicationNative(t *testing.T) {
	v := vk(t)
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("physical GPU required")
		}
		t.Skip("physical GPU absent")
	}
	owner := v.ForwardExecutionOwner()
	if owner == nil {
		t.Fatal("canonical backend has no execution owner")
	}
	owner.Lock()
	defer owner.Unlock()
	const width = 65
	const eps = float32(1e-5)
	t.Run("capacity_miss", func(t *testing.T) {
		gain := make([]float32, width)
		for i := range gain {
			gain[i] = 1
		}
		g := v.Upload(NewF32(Default(), []int{width}, gain), F32)
		defer func() { v.Free(g); v.Trim() }()
		v.Recycle()
		v.Trim() // Empty idle pools; the live gain remains owned by this test.
		const rows = 2
		input := make([]float32, rows*width)
		for i := range input {
			input[i] = float32(i%17-8) * .125
		}
		want := append([]float32(nil), input...)
		before := v41FFNWitnessSnapshot(t, v)
		p, selected, err := TryV41FFNNormPublication(v, g, input, rows, width, eps)
		after := v41FFNWitnessSnapshot(t, v)
		if p != nil || selected || err != nil {
			t.Fatalf("cold capacity miss: selected=%v publication=%v err=%v", selected, p != nil, err)
		}
		v41FFNWitnessBits(t, input, want)
		if before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
			t.Fatalf("miss changed observed counters/allocation: before=%+v after=%+v", before, after)
		}
		v41FFNWitnessBits(t, v.Read(g), gain)
	})
	for _, rows := range []int{2, 32, 64} {
		t.Run(fmt.Sprintf("positive_rows_%d", rows), func(t *testing.T) {
			gain := make([]float32, width)
			for i := range gain {
				gain[i] = .5 + float32(i%7)*.125
			}
			g := v.Upload(NewF32(Default(), []int{width}, gain), F32)
			defer func() { v.Free(g); v.Trim() }()
			input := make([]float32, rows*width)
			for i := range input {
				input[i] = float32(i%29-14) * .125
			}
			x := v.UploadClass(NewF32(Default(), []int{rows, width}, input), F32, MemoryActivation, "FFN witness ordinary norm input")
			raw := v.RMSNorm(x, g, eps)
			wantRaw := append([]float32(nil), v.Read(raw)...)
			v.Free(raw)
			v.Free(x)
			v.Recycle()
			v.Trim()
			v41FFNWitnessWarm(t, v, len(input))
			before := v41FFNWitnessSnapshot(t, v)
			p, selected, err := TryV41FFNNormPublication(v, g, input, rows, width, eps)
			v41FFNWitnessSelected(t, selected, err)
			if p == nil {
				t.Fatal("selected route returned no publication")
			}
			after := v41FFNWitnessSnapshot(t, v)
			v41FFNWitnessDispatch(t, before, after, len(input))
			v41FFNWitnessBits(t, input, wantRaw)
			saved := make([][]float32, rows)
			views := make([][]float32, rows)
			for row := range saved {
				got, err := p.Row(row)
				if err != nil || len(got) != width || cap(got) != width {
					t.Fatalf("row %d: len=%d cap=%d err=%v", row, len(got), cap(got), err)
				}
				views[row] = got
				saved[row] = append([]float32(nil), got...)
				for i, value := range got {
					bits := math.Float32bits(wantRaw[row*width+i])
					if bits&0x7f800000 == 0x7f800000 {
						t.Fatal("ordinary raw norm is nonfinite")
					}
					want := (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
					if math.Float32bits(value) != want {
						t.Fatalf("row=%d lane=%d publication=%08x want=%08x", row, i, math.Float32bits(value), want)
					}
				}
			}
			// The publication is independent host storage, not an alias of caller input
			// or a native loan whose contents change on subsequent checkout/retirement.
			for i := range input {
				input[i] = -123
			}
			v41FFNWitnessWarm(t, v, len(input))
			v.Trim()
			for row := range views {
				v41FFNWitnessBits(t, views[row], saved[row])
			}
			v41FFNWitnessBits(t, v.Read(g), gain)
		})
	}
	t.Run("numeric_row_verdict", func(t *testing.T) {
		const rows = 3
		gain := make([]float32, width)
		for i := range gain {
			gain[i] = math.MaxFloat32
		}
		g := v.Upload(NewF32(Default(), []int{width}, gain), F32)
		defer func() { v.Free(g); v.Trim() }()
		v.Recycle()
		v.Trim()
		input := make([]float32, rows*width)
		for i := width; i < 2*width; i++ {
			input[i] = 1
		}
		v41FFNWitnessWarm(t, v, len(input))
		before := v41FFNWitnessSnapshot(t, v)
		p, selected, err := TryV41FFNNormPublication(v, g, input, rows, width, 1e-8)
		v41FFNWitnessSelected(t, selected, err)
		if p == nil {
			t.Fatal("selected numeric route returned no publication")
		}
		after := v41FFNWitnessSnapshot(t, v)
		v41FFNWitnessDispatch(t, before, after, len(input))
		// Establish the actual raw arithmetic precondition before consuming Row.
		// The fault must be checked BF16 rounding, not NaN/Inf from RMSNorm itself.
		for i, value := range input {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("raw norm nonfinite at %d", i)
			}
		}
		for i := width; i < 2*width; i++ {
			bits := math.Float32bits(input[i])
			rounded := (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
			if rounded&0x7f800000 != 0x7f800000 {
				t.Fatalf("raw fault row does not overflow BF16 at lane=%d value=%g", i-width, input[i])
			}
		}
		good, err := p.Row(0)
		if err != nil || len(good) != width {
			t.Fatalf("prior row: %v", err)
		}
		for _, value := range good {
			if value != 0 {
				t.Fatalf("prior row value=%g", value)
			}
		}
		failed, err := p.Row(1)
		var arithmetic *V41BF16CheckedArithmeticError
		if failed != nil || !errors.As(err, &arithmetic) || arithmetic.Site != V41BF16PublishFFNNormOutput || arithmetic.Index != 0 {
			t.Fatalf("fault row=%v error=%v", failed, err)
		}
		later, err := p.Row(2)
		var protocol *V41BF16CheckedProtocolError
		if later != nil || !errors.As(err, &protocol) {
			t.Fatalf("later row=%v err=%v", later, err)
		}
		end := v41FFNWitnessSnapshot(t, v)
		if end.Counters != after.Counters {
			t.Fatal("host verdict consumption submitted extra device work")
		}
		v41FFNWitnessBits(t, v.Read(g), gain)
	})
}

func v41FFNWitnessWarm(t *testing.T, v *vulkanBackend, count int) {
	t.Helper()
	// Allocate both data owners simultaneously so the pool cannot return the
	// same allocation twice. The larger upload establishes bytes+16 staging.
	var held []Tensor
	defer func() {
		for _, x := range held {
			v.Free(x)
		}
	}()
	for _, n := range []int{count, count, 4, count + 4} {
		values := make([]float32, n)
		for i := range values {
			values[i] = 17
		}
		held = append(held, v.UploadClass(NewF32(Default(), []int{n}, values), F32, MemoryActivation, "FFN witness pool preparation"))
	}
}
func v41FFNWitnessSelected(t *testing.T, selected bool, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	if !selected {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required fused FFN route did not select after bounded pool preparation")
		}
		t.Skip("eligible native FFN pool/module capacity unavailable")
	}
}
func v41FFNWitnessSnapshot(t *testing.T, v *vulkanBackend) BackendExecutionSnapshot {
	t.Helper()
	s, err := v.BackendExecutionSnapshot()
	if err != nil || !s.TransferCountersObserved || !s.DeviceAllocationObserved {
		t.Fatalf("native observation unavailable: %v", err)
	}
	return s
}
func v41FFNWitnessDispatch(t *testing.T, before, after BackendExecutionSnapshot, count int) {
	t.Helper()
	d, err := BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if d.Counters.ComputeDispatches != 2 || d.Counters.H2DCount != 1 || d.Counters.H2DBytes != uint64(4*count) || d.Counters.D2HCount != 3 || d.Counters.D2HBytes != uint64(8*count+16) || d.Counters.Fallbacks != 0 {
		t.Fatalf("fused FFN execution delta: %+v", d.Counters)
	}
}
func v41FFNWitnessBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("lane=%d got=%08x want=%08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}
