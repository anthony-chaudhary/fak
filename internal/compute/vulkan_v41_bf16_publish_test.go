//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"math"
	"os"
	"testing"
)

func v41BF16PublishOracle(bits uint32) uint32 {
	return (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
}

// v41BF16PublishWitnessWords covers RNE ties both ways, the words either side of
// each tie, subnormals, both signed zeros, the largest non-overflowing word and
// a deterministic walk across every finite exponent.
func v41BF16PublishWitnessWords() []uint32 {
	words := []uint32{
		0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x00007fff, 0x00008000, 0x00008001,
		0x00018000, 0x80018000, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000,
		0x3f807fff, 0x3f808000, 0x3f808001, 0x3f817fff, 0x3f818000, 0x3f818001,
		0x7f7f0000, 0x7f7f7fff, 0xff7f7fff, 0x7f7effff, 0x7f7e8000,
	}
	state := uint32(0x9e3779b9)
	for len(words) < 1<<16 {
		state = state*1664525 + 1013904223
		word := state
		if word&0x7f800000 == 0x7f800000 {
			word ^= 0x00800000
		}
		if v41BF16PublishOracle(word)&0x7f800000 == 0x7f800000 {
			word &^= 0x40000000
		}
		words = append(words, word)
	}
	return words
}

// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no physical
// witness. Every ordinary site is bit-exact against the integer RNE oracle.
// fak-test:runtime integration est=5s lane=optin
func TestVulkanV41BF16Publish(t *testing.T) {
	v := vk(t)
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	if !v.SupportsV41BF16Publish(1) || !v.SupportsV41BF16Publish(1<<16) {
		t.Fatal("registered ordinary Vulkan build lacks the V4.1 BF16 publication")
	}
	words := v41BF16PublishWitnessWords()
	values := make([]float32, len(words))
	for i, w := range words {
		values[i] = math.Float32frombits(w)
	}
	upload := func(values []float32) Tensor {
		return v.UploadClass(NewF32(Default(), []int{len(values)}, values), F32, MemoryActivation, "V4.1 BF16 publication witness input")
	}
	snapshot := func() BackendExecutionSnapshot {
		v.Trim()
		s, err := v.BackendExecutionSnapshot()
		if err != nil || !s.TransferCountersObserved || !s.DeviceAllocationObserved {
			t.Fatalf("counters unavailable: %v %+v", err, s)
		}
		return s
	}
	var sites []uint32
	for site := uint32(0); site <= 64; site++ {
		if V41BF16PublishSite(site) {
			sites = append(sites, site)
		}
	}
	if len(sites) != 25 {
		t.Fatalf("ordinary site set has %d sites, want 25", len(sites))
	}
	dx := upload(values)
	defer v.Free(dx)
	before := snapshot()
	for _, site := range sites {
		start, err := v.BackendExecutionSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		out, err := v.V41BF16Publish(dx, site)
		if err != nil {
			t.Fatalf("site %d: %v", site, err)
		}
		if out.Buf() == dx.Buf() || out.Dtype != F32 || len(out.Shape) != 1 || out.Shape[0] != len(values) {
			t.Fatalf("site %d returned a non-fresh or malformed tensor", site)
		}
		end, err := v.BackendExecutionSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		d, err := BackendExecutionDelta(start, end)
		if err != nil || d.Counters.ComputeDispatches != 1 || d.Counters.H2DCount != 0 || d.Counters.D2HCount != 1 || d.Counters.D2HBytes != 16 || d.Counters.Fallbacks != 0 {
			t.Fatalf("site %d dispatch/transfer contract: %+v %v", site, d, err)
		}
		got := v.Read(out)
		for i, w := range words {
			if math.Float32bits(got[i]) != v41BF16PublishOracle(w) {
				t.Fatalf("site %d lane %d word %08x got %08x want %08x", site, i, w, math.Float32bits(got[i]), v41BF16PublishOracle(w))
			}
		}
		v.Free(out)
	}
	for i, value := range v.Read(dx) {
		if math.Float32bits(value) != words[i] {
			t.Fatalf("input mutated at lane %d", i)
		}
	}
	if after := snapshot(); after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
		t.Fatalf("publication leaked device bytes: before=%d after=%d", before.DeviceAllocationLiveBytes, after.DeviceAllocationLiveBytes)
	}

	t.Run("closed-site-policy", func(t *testing.T) {
		for _, site := range []uint32{0, 3, 4, 28, math.MaxUint32} {
			start, _ := v.BackendExecutionSnapshot()
			out, err := v.V41BF16Publish(dx, site)
			end, _ := v.BackendExecutionSnapshot()
			d, _ := BackendExecutionDelta(start, end)
			var backend *BackendError
			if out.Buf() != nil || !errors.As(err, &backend) || backend.Class != VulkanClassInvalidGeometry || d.Counters.ComputeDispatches != 0 {
				t.Fatalf("site %d not refused before dispatch: %v %+v", site, err, d.Counters)
			}
		}
	})
	t.Run("lowest-index-fault", func(t *testing.T) {
		for _, tc := range []struct {
			bits  []uint32
			index int
		}{
			{[]uint32{0x3f800000, 0x7f7f8000, 0x7fc00000, 0x7f800000}, 1},
			{[]uint32{0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0xffc00001, 0, 0, 0, 0xff7f8000}, 5},
			{[]uint32{0xff800000}, 0},
		} {
			row := make([]float32, len(tc.bits))
			for i, b := range tc.bits {
				row[i] = math.Float32frombits(b)
			}
			in := upload(row)
			start, _ := v.BackendExecutionSnapshot()
			out, err := v.V41BF16Publish(in, V41BF16PublishRoutedDown)
			end, _ := v.BackendExecutionSnapshot()
			v.Free(in)
			d, _ := BackendExecutionDelta(start, end)
			var numeric *V41BF16CheckedArithmeticError
			if out.Buf() != nil || !errors.As(err, &numeric) || numeric.Site != V41BF16PublishRoutedDown || numeric.Index != tc.index {
				t.Fatalf("fault %x: want lowest index %d, got %v", tc.bits, tc.index, err)
			}
			if d.Counters.ComputeDispatches != 1 || d.Counters.D2HCount != 1 || d.Counters.D2HBytes != 16 {
				t.Fatalf("failed output was read or status absent: %+v", d.Counters)
			}
		}
	})
	t.Run("batch-refusal", func(t *testing.T) {
		v.BeginBatch()
		out, err := v.V41BF16Publish(dx, V41BF16PublishKVNormInput)
		v.FlushBatch()
		var backend *BackendError
		if out.Buf() != nil || !errors.As(err, &backend) {
			t.Fatalf("active batch did not refuse the synchronous publication: %v", err)
		}
		out, err = v.V41BF16Publish(dx, V41BF16PublishKVNormInput)
		if err != nil {
			t.Fatalf("publication after the batch flushed: %v", err)
		}
		v.Free(out)
	})
	if after := snapshot(); after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
		t.Fatalf("refusals leaked device bytes: before=%d after=%d", before.DeviceAllocationLiveBytes, after.DeviceAllocationLiveBytes)
	}
	t.Logf("V41_BF16_PUBLISH_PHYSICAL_COMPLETE sites=%d words=%d", len(sites), len(words))
}
