//go:build vulkan && (windows || linux) && cgo

// Run from the repository root with the current shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41CompressorRingVulkan$' -count=1
// The synthetic payload witnesses the session seam on a physical GPU, not a checkpoint.

package model

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41RingDispatchProfiler interface {
	VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	VulkanDebugResetDispatchProfile()
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
// The ring session and a control session on the same backend run the same
// prefill and decode tokens; the control keeps the host-staged group with the
// device gated pool, which is the path the ring replaces. Logits must be
// bit-identical, and with the dispatch profiler on every closed group must save
// exactly 2*ratio projection readbacks plus the pool's own flush.
func TestV41CompressorRingVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical compressor ring seam witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	t.Logf("Vulkan device: %s", be.Tier())
	ring, okRing := be.(compute.V41CompressorRingBackend)
	pool, okPool := be.(compute.V41CompressorPoolBackend)
	if !okRing || !ring.SupportsV41CompressorRing() || !okPool || !pool.SupportsV41CompressorPool() {
		t.Fatal("registered Vulkan build lacks the V4.1 compressor ring or pool module (rebuild the current shader bundle)")
	}
	// Once-per-device admission probes of sibling modules (a lazy self-check)
	// land on this throwaway session, so the measured delta is the ring's alone.
	warmModel, _ := v41IncrementalEngramFixture(t, true)
	warm, err := warmModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	warmPrefix, warmNext := v41IncrementalPrefix(warmModel.Cfg, 4)
	warm.Prefill(warmPrefix)
	warm.Step(warmNext)
	warm.Close()
	for _, prefixLen := range []int{4, 3} {
		m, _ := v41IncrementalEngramFixture(t, true)
		controlModel, _ := v41IncrementalEngramFixture(t, true)
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		control, err := controlModel.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		if s.v41State().gatherOps.compressorRing == nil || control.v41State().gatherOps.compressorPool == nil {
			t.Fatal("Vulkan session did not bind the compressor ring and pool seams")
		}
		control.v41State().gatherOps.compressorRing = nil

		prefix, next := v41IncrementalPrefix(m.Cfg, prefixLen)
		v41RingBitsEqualRows(t, "prefill", s.Prefill(prefix), control.Prefill(prefix))
		profiler, profiled := be.(v41RingDispatchProfiler)
		profiled = profiled && compute.VulkanDispatchProfileEnabled()
		submits := func(run func() []float32) ([]float32, uint64) {
			if !profiled {
				return run(), 0
			}
			profiler.VulkanDebugResetDispatchProfile()
			out := run()
			p := profiler.VulkanDebugDispatchProfileSnapshot()
			return out, p.BatchSubmits + p.OneShotSubmits
		}
		const steps, ratio = 6, 2
		var ringSubmits, controlSubmits uint64
		for i := 0; i < steps; i++ {
			got, ringN := submits(func() []float32 { return s.Step(next) })
			want, controlN := submits(func() []float32 { return control.Step(next) })
			v41RingBitsEqualRows(t, "decode step", got, want)
			ringSubmits += ringN
			controlSubmits += controlN
			next = (next*7 + 3) % m.Cfg.VocabSize
		}
		layers := m.Cfg.NumLayers
		groups := (prefixLen+steps)/ratio - prefixLen/ratio
		stats := s.v41State().gatherOps.compressorRing.stats
		want := v41CompressorRingStats{Stores: (steps + prefixLen%ratio) * layers, Emits: groups * layers, Reseeds: prefixLen % ratio * layers}
		if stats != want {
			t.Fatalf("prefix %d ring stats %+v, want %+v", prefixLen, stats, want)
		}
		t.Logf("prefix %d: %d decode steps, %d closed groups x %d layers bit-identical; queue submits ring=%d control=%d (profiled=%v)",
			prefixLen, steps, groups, layers, ringSubmits, controlSubmits, profiled)
		if profiled {
			if saved, wantSaved := int64(controlSubmits)-int64(ringSubmits), int64((2*ratio+1)*groups*layers); saved != wantSaved {
				t.Fatalf("prefix %d submits saved %d, want %d", prefixLen, saved, wantSaved)
			}
		}
		s.Close()
		control.Close()
	}
}
