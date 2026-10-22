//go:build vulkan && (windows || linux) && cgo

// Run from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41ColdStageAttributionVulkan$' -count=1 -v
// Without FAK_VULKAN_DISPATCH_PROFILE=1 the rows still close and the always-on
// device_submits still name each stage's submits; the profiler's dispatch and
// submit counts are then flagged not live instead of reading as zero work.

package model

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41ColdStageAttributionVulkan is the device half of the stage witness:
// on a real Vulkan backend the stage rows of a cold prefill and of each decode
// step close exactly on the backend's own BackendExecutionDelta, always-on
// submit and DeviceWork totals, every row's always-on submits equal the
// profiler's when it is on, the routed and attention stages own device work,
// and attribution leaves every logit bit-identical.
// fak-test:justify why=integration when=changed:internal/model/v41_cold_stage_attribution*.go,internal/compute/vulkan_backend_counters.go
// fak-test:runtime slow est=20s lane=optin
func TestV41ColdStageAttributionVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("device stage-attribution witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	for _, fx := range append(v41DecodeSubmitsFixtures(), v41DecodeSubmitsPinnedFixtures()...) {
		t.Run(fx.name, func(t *testing.T) {
			plainModel, m := fx.build(t), fx.build(t)
			defer func() {
				for _, mm := range []*Model{plainModel, m} {
					if err := mm.CloseWeights(); err != nil {
						t.Error(err)
					}
				}
			}()
			plain, err := plainModel.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer plain.Close()
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			if bound, _ := v41DeviceParitySeams(s.v41State()); len(bound) == 0 {
				t.Fatal("Vulkan session bound no V4.1 device seam")
			}
			s.EnableV41ColdStageAttribution()

			prompt := v41DecodeSubmitsPrompt(16, m.Cfg.VocabSize)
			want := append([]float32(nil), plain.Prefill(prompt)...)
			window, got := v41ColdStageVulkanWindow(t, be, s, func() []float32 { return s.Prefill(prompt) })
			v41ColdStageRequireBitIdentical(t, "prefill", got, want)
			if c := window.Total().Counters; c.D2HCount == 0 && c.H2DCount == 0 {
				t.Fatal("cold prefill moved no bytes through the observed backend")
			}
			if window.Stages[V41ColdStageQKV].Entries != uint64(m.Cfg.NumLayers) {
				t.Fatalf("qkv entries=%d want %d", window.Stages[V41ColdStageQKV].Entries, m.Cfg.NumLayers)
			}
			for _, stage := range []V41ColdStage{V41ColdStageRouted, V41ColdStageAttention} {
				if row := window.Stages[stage]; row.DeviceSubmits == 0 || row.DeviceDispatches == 0 {
					t.Fatalf("cold prefill %s row owns no device work: %+v", stage, row)
				}
			}
			v41ColdStageLog(t, fx.name+" backend=vulkan phase=prefill", window)

			next := argmaxF32(got)
			for step := 0; step < 2; step++ {
				want = append([]float32(nil), plain.Step(next)...)
				window, got = v41ColdStageVulkanWindow(t, be, s, func() []float32 { return s.Step(next) })
				v41ColdStageRequireBitIdentical(t, "decode", got, want)
				v41ColdStageLog(t, fx.name+" backend=vulkan phase=decode", window)
				next = argmaxF32(got)
			}
		})
	}
}

func v41ColdStageVulkanWindow(t *testing.T, be compute.Backend, s *Session, run func() []float32) (V41ColdStageAttribution, []float32) {
	t.Helper()
	snapBefore, ok, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !ok {
		t.Fatalf("snapshot before: %t %v", ok, err)
	}
	workBefore, _ := compute.ReadDeviceWork(be.Name())
	submitsBefore, _, _ := compute.ReadBackendCounters(be)
	before, _ := s.V41ColdStageAttribution()
	logits := append([]float32(nil), run()...)
	after, _ := s.V41ColdStageAttribution()
	submitsAfter, _, _ := compute.ReadBackendCounters(be)
	workAfter, _ := compute.ReadDeviceWork(be.Name())
	snapAfter, ok, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !ok {
		t.Fatalf("snapshot after: %t %v", ok, err)
	}
	window, err := after.Sub(before)
	if err != nil {
		t.Fatal(err)
	}
	delta, err := compute.BackendExecutionDelta(snapBefore, snapAfter)
	if err != nil {
		t.Fatal(err)
	}
	if !window.CountersObserved || !window.DeviceSubmitsObserved || !window.DeviceWorkObserved ||
		window.DispatchCountsLive != compute.VulkanDispatchProfileEnabled() || !submitsBefore.SubmitsObserved || !submitsAfter.SubmitsObserved {
		t.Fatalf("observability: %+v", window)
	}
	total := window.Total()
	if total.Counters != delta.Counters {
		t.Fatalf("stage rows do not close on the backend delta:\n rows=%+v\n backend=%+v", total.Counters, delta.Counters)
	}
	if total.DeviceSubmits != submitsAfter.Submits-submitsBefore.Submits || total.DeviceSubmits == 0 {
		t.Fatalf("stage rows do not close on the always-on submit count: rows=%d backend=%d", total.DeviceSubmits, submitsAfter.Submits-submitsBefore.Submits)
	}
	if window.DispatchCountsLive {
		for stage := V41ColdStage(0); stage <= V41ColdStageCount; stage++ {
			row := window.Unassigned
			if stage < V41ColdStageCount {
				row = window.Stages[stage]
			}
			if row.DeviceSubmits != row.Counters.DispatchSubmits || row.DeviceDispatches != row.Counters.ComputeDispatches {
				t.Fatalf("%s always-on submits/dispatches=%d/%d profiler=%d/%d", stage, row.DeviceSubmits, row.DeviceDispatches,
					row.Counters.DispatchSubmits, row.Counters.ComputeDispatches)
			}
		}
	}
	if total.DeviceDispatches != workAfter.Dispatches-workBefore.Dispatches || total.SubmitWaitNanos != workAfter.SubmitWaitNS-workBefore.SubmitWaitNS {
		t.Fatalf("stage rows do not close on DeviceWork: rows=%d/%d backend=%d/%d", total.DeviceDispatches, total.SubmitWaitNanos,
			workAfter.Dispatches-workBefore.Dispatches, workAfter.SubmitWaitNS-workBefore.SubmitWaitNS)
	}
	return window, logits
}
