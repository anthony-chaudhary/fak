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
	policy := map[uint32]bool{}
	for _, site := range []uint32{
		V41BF16PublishFinalNormInput, V41BF16PublishFinalNormOutput,
		V41BF16PublishQueryNormInput, V41BF16PublishQueryNormOutput,
		V41BF16PublishKVNormInput, V41BF16PublishKVNormOutput,
		V41BF16PublishFFNNormInput, V41BF16PublishFFNNormOutput,
		V41BF16PublishTailRoPEInput, V41BF16PublishTailRoPEOutput,
		V41BF16PublishInverseOutputInput, V41BF16PublishInverseOutputOutput,
		V41BF16PublishGroupedOutputInput, V41BF16PublishGroupedOutputProjection,
		V41BF16PublishSharedExpertInput, V41BF16PublishSharedExpertProjection,
		V41BF16PublishSharedExpertDownInput,
		V41BF16PublishRoutedGateUp, V41BF16PublishRoutedDown,
		V41BF16PublishIndexerProjection, V41BF16PublishIndexerKeyNorm,
		V41BF16PublishIndexerRotary, V41BF16PublishIndexerScaledWeight,
		V41BF16PublishCompressorNormInput, V41BF16PublishCompressorNormOutput,
	} {
		policy[site] = true
	}
	if len(sites) != len(policy) {
		t.Fatalf("ordinary site set has %d sites, want the %d named policy sites", len(sites), len(policy))
	}
	for _, site := range sites {
		if !policy[site] {
			t.Fatalf("site %d admitted outside the named policy", site)
		}
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
	// The deferred form issues the same kernel and leaves its status on the
	// device: the status and row then arrive behind one fence, bit-exact, and
	// a fault decodes to the synchronous verdict with the row withheld.
	t.Run("deferred", func(t *testing.T) {
		if !v.SupportsV41DeferredStatus(len(values)) {
			t.Fatal("registered ordinary Vulkan build lacks the deferred status record")
		}
		for _, site := range sites {
			start, _ := v.BackendExecutionSnapshot()
			out, status, err := v.V41BF16PublishDeferred(dx, site)
			if err != nil {
				t.Fatalf("site %d: %v", site, err)
			}
			issued, _ := v.BackendExecutionSnapshot()
			verdicts, rows, err := v.V41ReadDeferred([]V41DeferredStatus{status}, []Tensor{out})
			end, _ := v.BackendExecutionSnapshot()
			v.Free(out)
			if err != nil || len(verdicts) != 1 || verdicts[0] != nil || len(rows) != 1 || len(rows[0]) != len(words) {
				t.Fatalf("site %d deferred read: verdicts=%v err=%v", site, verdicts, err)
			}
			publish, _ := BackendExecutionDelta(start, issued)
			read, _ := BackendExecutionDelta(issued, end)
			if publish.Counters.ComputeDispatches != 1 || publish.Counters.D2HCount != 0 ||
				read.Counters.ComputeDispatches != 0 || read.Counters.D2HCount != 2 || read.Counters.D2HBytes != uint64(16+4*len(words)) {
				t.Fatalf("site %d deferred contract: publish %+v read %+v", site, publish.Counters, read.Counters)
			}
			for i, w := range words {
				if math.Float32bits(rows[0][i]) != v41BF16PublishOracle(w) {
					t.Fatalf("site %d deferred lane %d word %08x got %08x want %08x", site, i, w, math.Float32bits(rows[0][i]), v41BF16PublishOracle(w))
				}
			}
		}
		row := make([]float32, 10)
		for i, b := range []uint32{0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000, 0xffc00001, 0, 0, 0, 0xff7f8000} {
			row[i] = math.Float32frombits(b)
		}
		in := upload(row)
		defer v.Free(in)
		_, syncErr := v.V41BF16Publish(in, V41BF16PublishRoutedDown)
		out, status, err := v.V41BF16PublishDeferred(in, V41BF16PublishRoutedDown)
		if err != nil {
			t.Fatal(err)
		}
		verdicts, _, err := v.V41ReadDeferred([]V41DeferredStatus{status}, []Tensor{out})
		v.Free(out)
		var syncBackend, deferredBackend *BackendError
		var syncNumeric, deferredNumeric *V41BF16CheckedArithmeticError
		if err != nil || len(verdicts) != 1 || !errors.As(syncErr, &syncBackend) || !errors.As(verdicts[0], &deferredBackend) ||
			syncBackend.Site != deferredBackend.Site || syncBackend.Class != deferredBackend.Class ||
			!errors.As(syncErr, &syncNumeric) || !errors.As(verdicts[0], &deferredNumeric) || *syncNumeric != *deferredNumeric || deferredNumeric.Index != 5 {
			t.Fatalf("deferred verdict %v (err %v) differs from the synchronous %v", verdicts, err, syncErr)
		}
	})
	if after := snapshot(); after.DeviceAllocationLiveBytes != before.DeviceAllocationLiveBytes {
		t.Fatalf("refusals leaked device bytes: before=%d after=%d", before.DeviceAllocationLiveBytes, after.DeviceAllocationLiveBytes)
	}
	t.Logf("V41_BF16_PUBLISH_PHYSICAL_COMPLETE sites=%d words=%d", len(sites), len(words))
}
