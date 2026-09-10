//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"testing"
)

// TestVulkanKVCopyOnWriteFork pins the device-resident fork contract: cloning a
// prefix shares its three Vulkan allocations, while the first write privately
// detaches a branch and leaves every other owner usable.
func TestVulkanKVCopyOnWriteFork(t *testing.T) {
	v := vk(t)
	c := cpu()
	cfg := KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 8, RopeTheta: 10000}
	const cloneCount = 8
	v.Trim()
	baseline, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("allocation baseline: %v", err)
	}

	appendRow := func(kv KVStore, raw, value []float32, pos int) {
		kr := append([]float32(nil), raw...)
		cos, sin := ropeRow(cfg.RopeTheta, cfg.HeadDim, pos)
		applyRope(kr, cos, sin)
		rawT := v.Upload(NewF32(c, []int{cfg.HeadDim}, raw), F32)
		ropeT := v.Upload(NewF32(c, []int{cfg.HeadDim}, kr), F32)
		valueT := v.Upload(NewF32(c, []int{cfg.HeadDim}, value), F32)
		kv.AppendKV(0, rawT, ropeT, valueT, pos)
		v.Free(rawT)
		v.Free(ropeT)
		v.Free(valueT)
	}

	raw0 := []float32{.1, .2, .3, .4, .5, .6, .7, .8}
	val0 := []float32{.8, .7, .6, .5, .4, .3, .2, .1}
	rawBase1 := []float32{.2, -.1, .4, -.3, .6, -.5, .8, -.7}
	valBase1 := []float32{-.7, .8, -.5, .6, -.3, .4, -.1, .2}
	source := v.NewKV(cfg)
	appendRow(source, raw0, val0, 0)
	appendRow(source, rawBase1, valBase1, 1) // leaves one row of spare geometric capacity
	sourceVK := source.(*vulkanKV)

	beforeFork, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("fork allocation baseline: %v", err)
	}
	window, err := v.BeginBackendExecutionWindow()
	if err != nil {
		source.Free()
		t.Fatalf("begin clone observation: %v", err)
	}
	clones := make([]KVStore, cloneCount)
	allShared := true
	for i := range clones {
		clones[i] = source.Clone()
		clone := clones[i].(*vulkanKV)
		if clone.K[0].ptr != sourceVK.K[0].ptr || clone.Kraw[0].ptr != sourceVK.Kraw[0].ptr || clone.V[0].ptr != sourceVK.V[0].ptr {
			allShared = false
		}
	}
	observation, err := window.End()
	if err != nil {
		t.Fatalf("end clone observation: %v", err)
	}
	t.Logf("forks=%d allocation_live_delta=%d allocation_peak_delta=%d d2d_copies=%d d2d_bytes=%d",
		cloneCount, int64(observation.DeviceAllocationLiveBytes)-int64(beforeFork.DeviceAllocationLiveBytes), int64(observation.DeviceAllocationPeakBytes)-int64(beforeFork.DeviceAllocationLiveBytes),
		observation.Counters.D2DCopies, observation.Counters.D2DBytes)
	if observation.DeviceAllocationPeakBytes != beforeFork.DeviceAllocationLiveBytes || observation.DeviceAllocationLiveBytes != beforeFork.DeviceAllocationLiveBytes || observation.Counters.D2DCopies != 0 || observation.Counters.D2DBytes != 0 {
		t.Fatalf("fork copied or allocated device storage: live/peak=%d/%d d2d=%d/%d", observation.DeviceAllocationLiveBytes, observation.DeviceAllocationPeakBytes, observation.Counters.D2DCopies, observation.Counters.D2DBytes)
	}
	if !allShared {
		t.Fatal("one or more clones did not share all prefix allocations")
	}

	// The empty-cache case must remain allocation-free as well.
	empty := v.NewKV(cfg)
	emptyClone := empty.Clone().(*vulkanKV)
	if emptyClone.K[0].ptr != nil || emptyClone.Kraw[0].ptr != nil || emptyClone.V[0].ptr != nil {
		t.Fatal("empty fork acquired device storage")
	}
	empty.Free()
	emptyClone.Free()

	mutated := clones[0]
	raw1 := []float32{-.2, .3, -.4, .5, -.6, .7, -.8, .9}
	val1 := []float32{.9, -.8, .7, -.6, .5, -.4, .3, -.2}
	appendRow(mutated, raw1, val1, 2)
	if mutatedVK := mutated.(*vulkanKV); mutatedVK.K[0].ptr != sourceVK.K[0].ptr || mutatedVK.Kraw[0].ptr != sourceVK.Kraw[0].ptr || mutatedVK.V[0].ptr != sourceVK.V[0].ptr {
		t.Fatal("first high-water append copied the visible prefix despite spare capacity")
	}
	// A second fork at the old logical end would overwrite the first branch's
	// tail, so it must detach before writing.
	appendRow(clones[1], []float32{1, 2, 3, 4, 5, 6, 7, 8}, []float32{8, 7, 6, 5, 4, 3, 2, 1}, 2)
	secondWriter := clones[1].(*vulkanKV)
	if secondWriter.K[0].ptr == sourceVK.K[0].ptr || secondWriter.Kraw[0].ptr == sourceVK.Kraw[0].ptr || secondWriter.V[0].ptr == sourceVK.V[0].ptr {
		t.Fatal("divergent writer overwrote a tail already claimed by a sibling")
	}
	if got := mutated.Evict(0, 1); got != 1 {
		t.Fatalf("Evict removed %d positions, want 1", got)
	}
	mutatedVK := mutated.(*vulkanKV)
	if mutatedVK.K[0].ptr == sourceVK.K[0].ptr || mutatedVK.Kraw[0].ptr == sourceVK.Kraw[0].ptr || mutatedVK.V[0].ptr == sourceVK.V[0].ptr {
		t.Fatal("first mutation did not detach all affected segments")
	}

	cold := v.NewKV(cfg)
	appendRow(cold, rawBase1, valBase1, 0)
	appendRow(cold, raw1, val1, 1)
	if got, want := v.Read(mutated.KeysView(0)), v.Read(cold.KeysView(0)); maxAbs(got, want) > 1e-6 {
		t.Fatalf("detached keys differ from cold cache: got=%v want=%v", got, want)
	}
	if got, want := v.Read(mutated.ValuesView(0)), v.Read(cold.ValuesView(0)); maxAbs(got, want) > 1e-6 {
		t.Fatalf("detached values differ from cold cache: got=%v want=%v", got, want)
	}

	// Drop the original owner before using a still-shared sibling. This catches
	// premature frees independently of the mutated branch comparison above.
	source.Free()
	prefixCold := v.NewKV(cfg)
	appendRow(prefixCold, raw0, val0, 0)
	appendRow(prefixCold, rawBase1, valBase1, 1)
	query := []float32{.3, -.2, .1, .4, -.5, .6, -.7, .8}
	q := v.Upload(NewF32(c, []int{cfg.HeadDim}, query), F32)
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	gotAttention := v.Attention(q, clones[2], 0, true, 1, scale)
	wantAttention := v.Attention(q, prefixCold, 0, true, 1, scale)
	if got, want := v.Read(gotAttention), v.Read(wantAttention); maxAbs(got, want) > 1e-5 {
		t.Fatalf("sibling attention after original free differs: got=%v want=%v", got, want)
	}
	v.Free(gotAttention)
	v.Free(wantAttention)
	v.Free(q)

	// Exercise non-LIFO release order and repeated Free. Shared allocations must
	// be released exactly once when their final owner closes.
	for _, i := range []int{4, 2, 7, 1, 6, 3, 5, 0} {
		clones[i].Free()
		clones[i].Free()
	}
	cold.Free()
	prefixCold.Free()
	v.Trim()
	after, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("allocation final snapshot: %v", err)
	}
	if after.DeviceAllocationLiveBytes != baseline.DeviceAllocationLiveBytes {
		t.Fatalf("KV fork leaked device storage after trim: before=%d after=%d", baseline.DeviceAllocationLiveBytes, after.DeviceAllocationLiveBytes)
	}
}
