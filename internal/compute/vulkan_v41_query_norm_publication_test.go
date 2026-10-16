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
// Requires a matching rebuilt native library and a physical GPU. A skipped
// prepared route supplies no qualification. This tests the canonical API, not
// model callback replay, transport fault injection or physical retirement.
func TestVulkanV41QueryNormPublicationNative(t *testing.T) {
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
	const width = 1280
	const eps = float32(1e-5)
	gainFor := func(t *testing.T, large bool) (Tensor, []float32) {
		t.Helper()
		gain := make([]float32, width)
		for i := range gain {
			gain[i] = .5 + float32(i%7)*.125
			if large {
				gain[i] = math.MaxFloat32
			}
		}
		g := v.Upload(NewF32(Default(), []int{width}, gain), F32)
		t.Cleanup(func() { v.Free(g); v.Recycle(); v.Trim() })
		return g, gain
	}
	t.Run("capacity_miss", func(t *testing.T) {
		g, gain := gainFor(t, false)
		v.Recycle()
		v.Trim()
		const rows = 2
		input := make([]float32, rows*width)
		for i := range input {
			input[i] = float32(i%17-8) * .031
		}
		want := append([]float32(nil), input...)
		end := v41QueryWitnessStep(t, v)
		before := v41QueryWitnessSnapshot(t, v)
		result, selected, err := TryV41QueryNormPublication(v, g, input, rows, width, 3, eps)
		after := v41QueryWitnessSnapshot(t, v)
		if result != nil || selected || err != nil {
			t.Fatalf("capacity miss selected=%v result=%v error=%v", selected, result != nil, err)
		}
		v41QueryWitnessBits(t, input, want)
		if before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
			t.Fatal("capacity miss changed counters or live allocation")
		}
		if !v.V41StepBatchRecording() {
			t.Fatal("capacity miss ended recording")
		}
		end()
		v41QueryWitnessBits(t, v.Read(g), gain)
	})
	for _, geometry := range [][2]int{{2, 3}, {3, 64}, {64, 64}} {
		rows, capacity := geometry[0], geometry[1]
		t.Run(fmt.Sprintf("positive_rows_%d_capacity_%d", rows, capacity), func(t *testing.T) {
			g, gain := gainFor(t, false)
			input := make([]float32, rows*width)
			roundedInput := make([]float32, len(input))
			changed := false
			for i := range input {
				input[i] = float32(i%29-14) * .031
				roundedInput[i] = math.Float32frombits(v41QueryWitnessRNE(input[i]))
				changed = changed || math.Float32bits(input[i]) != math.Float32bits(roundedInput[i])
			}
			if !changed {
				t.Fatal("input publication oracle is vacuous")
			}
			// Same native RMSNorm geometry, with independently integer-rounded input.
			x := v.UploadClass(NewF32(Default(), []int{rows, width}, roundedInput), F32, MemoryActivation, "query reference input")
			norm := v.RMSNorm(x, g, eps)
			wantRaw := append([]float32(nil), v.Read(norm)...)
			v.Free(norm)
			v.Free(x)
			v.Recycle()
			v.Trim()
			v41QueryWitnessWarm(v, capacity, width, len(input))
			end := v41QueryWitnessStep(t, v)
			before := v41QueryWitnessSnapshot(t, v)
			result, selected, err := TryV41QueryNormPublication(v, g, input, rows, width, capacity, eps)
			v41QueryWitnessSelected(t, selected, err)
			if result == nil {
				t.Fatal("selected route returned nil result")
			}
			after := v41QueryWitnessSnapshot(t, v)
			v41QueryWitnessTraffic(t, before, after, len(input))
			if !v.V41StepBatchRecording() {
				t.Fatal("selected route did not resume recording")
			}
			v41QueryWitnessBits(t, input, wantRaw)
			views := make([][]float32, rows)
			saved := make([][]float32, rows)
			for row := range views {
				if err := result.InputRowError(row); err != nil {
					t.Fatalf("input row %d: %v", row, err)
				}
				for _, value := range input[row*width : (row+1)*width] {
					v41QueryWitnessFinite(t, value)
				}
				values, rowErr := result.Row(row)
				if rowErr != nil || len(values) != width || cap(values) != width {
					t.Fatalf("row %d len=%d cap=%d error=%v", row, len(values), cap(values), rowErr)
				}
				views[row], saved[row] = values, append([]float32(nil), values...)
				for i, value := range values {
					if math.Float32bits(value) != v41QueryWitnessRNE(wantRaw[row*width+i]) {
						t.Fatalf("output row=%d lane=%d differs from RNE", row, i)
					}
				}
			}
			if v41QueryWitnessSnapshot(t, v).Counters != after.Counters {
				t.Fatal("host result inspection submitted work")
			}
			end()
			for i := range input {
				input[i] = -123
			}
			v41QueryWitnessWarm(v, capacity, width, len(input))
			v.Recycle()
			v.Trim()
			for row := range views {
				v41QueryWitnessBits(t, views[row], saved[row])
			}
			v41QueryWitnessBits(t, v.Read(g), gain)
		})
	}
	for _, site := range []string{"input", "output"} {
		t.Run("numeric_"+site+"_row_verdict", func(t *testing.T) {
			const rows, capacity = 3, 4
			g, gain := gainFor(t, site == "output")
			v.Recycle()
			v.Trim()
			input := make([]float32, rows*width)
			for i := width; i < 2*width; i++ {
				input[i] = 1
			}
			if site == "input" {
				input[width] = math.MaxFloat32
			}
			v41QueryWitnessWarm(v, capacity, width, len(input))
			end := v41QueryWitnessStep(t, v)
			before := v41QueryWitnessSnapshot(t, v)
			result, selected, err := TryV41QueryNormPublication(v, g, input, rows, width, capacity, 1e-8)
			v41QueryWitnessSelected(t, selected, err)
			if result == nil {
				t.Fatal("selected numeric route returned nil result")
			}
			after := v41QueryWitnessSnapshot(t, v)
			v41QueryWitnessTraffic(t, before, after, len(input))
			if !v.V41StepBatchRecording() {
				t.Fatal("numeric result did not resume recording")
			}
			if err := result.InputRowError(0); err != nil {
				t.Fatal(err)
			}
			for _, value := range input[:width] {
				v41QueryWitnessFinite(t, value)
			}
			good, err := result.Row(0)
			if err != nil || len(good) != width || cap(good) != width {
				t.Fatalf("prior row: %v", err)
			}
			for _, value := range good {
				if value != 0 {
					t.Fatalf("prior row value=%g", value)
				}
			}
			var arithmetic *V41BF16CheckedArithmeticError
			var protocol *V41BF16CheckedProtocolError
			if site == "input" {
				err = result.InputRowError(1)
				if !errors.As(err, &arithmetic) || arithmetic.Site != V41BF16PublishQueryNormInput || arithmetic.Index != 0 {
					t.Fatalf("input fault: %v", err)
				}
				if err := result.InputRowError(2); !errors.As(err, &protocol) {
					t.Fatalf("later input: %v", err)
				}
				// Do not consume output from the faulting/later input rows. Their raw norm
				// arithmetic is deliberately not the valid-output contract.
			} else {
				for row := 0; row < rows; row++ {
					if err := result.InputRowError(row); err != nil {
						t.Fatal(err)
					}
				}
				for _, value := range input {
					v41QueryWitnessFinite(t, value)
				}
				for _, value := range input[width : 2*width] {
					if v41QueryWitnessRNE(value)&0x7f800000 != 0x7f800000 {
						t.Fatalf("raw output does not establish finite BF16 overflow: %g", value)
					}
				}
				bad, rowErr := result.Row(1)
				if bad != nil || !errors.As(rowErr, &arithmetic) || arithmetic.Site != V41BF16PublishQueryNormOutput || arithmetic.Index != 0 {
					t.Fatalf("output fault row=%v error=%v", bad, rowErr)
				}
				later, rowErr := result.Row(2)
				if later != nil || !errors.As(rowErr, &protocol) {
					t.Fatalf("later output row=%v error=%v", later, rowErr)
				}
			}
			if v41QueryWitnessSnapshot(t, v).Counters != after.Counters {
				t.Fatal("numeric verdict consumption submitted work")
			}
			end()
			v41QueryWitnessBits(t, v.Read(g), gain)
		})
	}
}

func v41QueryWitnessStep(t *testing.T, v *vulkanBackend) func() {
	t.Helper()
	if !v.BeginV41StepBatch() {
		t.Fatal("cannot begin genuine query step recording")
	}
	open, live := true, false
	var setup Tensor
	end := func() {
		if open {
			open = false
			if err := v.EndV41StepBatch(); err != nil {
				t.Errorf("ending query recording: %v", err)
			}
		}
		if live {
			live = false
			v.Free(setup)
		}
	}
	t.Cleanup(end)
	// This distinct live allocation establishes the native upload arena without
	// taking the 16-byte verdict bucket. Counters are sampled after record-time
	// upload accounting; setup work may share the query's first checked flush.
	setup = v.UploadClass(NewF32(Default(), []int{8}, make([]float32, 8)), F32, MemoryActivation, "query upload arena setup")
	live = true
	if !v.V41StepBatchRecording() {
		t.Fatal("setup did not leave recording live")
	}
	return end
}
func v41QueryWitnessWarm(v *vulkanBackend, capacity, width, count int) {
	var held []Tensor
	defer func() {
		for _, x := range held {
			v.Free(x)
		}
	}()
	for _, n := range []int{capacity * width, capacity * 5120, 4, count + 4} {
		held = append(held, v.UploadClass(NewF32(Default(), []int{n}, make([]float32, n)), F32, MemoryActivation, "query heterogeneous pool preparation"))
	}
}
func v41QueryWitnessSelected(t *testing.T, selected bool, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	if !selected {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required query route missed after bounded pool/step preparation")
		}
		t.Skip("eligible query pool/module capacity unavailable")
	}
}
func v41QueryWitnessSnapshot(t *testing.T, v *vulkanBackend) BackendExecutionSnapshot {
	t.Helper()
	s, err := v.BackendExecutionSnapshot()
	if err != nil || !s.TransferCountersObserved || !s.DeviceAllocationObserved {
		t.Fatalf("native observation unavailable: %v", err)
	}
	return s
}
func v41QueryWitnessTraffic(t *testing.T, before, after BackendExecutionSnapshot, count int) {
	t.Helper()
	d, err := BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if d.Counters.ComputeDispatches != 3 || d.Counters.H2DCount != 1 || d.Counters.H2DBytes != uint64(4*count) || d.Counters.D2HCount != 4 || d.Counters.D2HBytes != uint64(8*count+32) || d.Counters.Fallbacks != 0 {
		t.Fatalf("query execution delta: %+v", d.Counters)
	}
}
func v41QueryWitnessRNE(v float32) uint32 {
	b := math.Float32bits(v)
	return (b + 0x7fff + ((b >> 16) & 1)) & 0xffff0000
}
func v41QueryWitnessFinite(t *testing.T, v float32) {
	t.Helper()
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		t.Fatalf("raw norm nonfinite: %g", v)
	}
}
func v41QueryWitnessBits(t *testing.T, got, want []float32) {
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
