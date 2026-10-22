//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"testing"
)

// TestVulkanDeviceSubmitsAlwaysOn: the always-on submit count moves with
// one-shot and batch submits whatever the dispatch profile, and with the
// profile on it moves exactly with the profiler's batch plus one-shot submits.
// The parent run covers the process's own profile setting; a profile-off
// parent also re-runs the case in a profiled child.
// fak-test:runtime fast est=2s lane=optin
func TestVulkanDeviceSubmitsAlwaysOn(t *testing.T) {
	v := vk(t)
	c := cpu()
	w := v.Upload(NewF32(c, []int{4, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}), F32)
	x := v.Upload(NewF32(c, []int{4}, []float32{1, 2, 3, 4}), F32)
	before, ok, err := ReadBackendCounters(v)
	if !ok || err != nil || !before.SubmitsObserved {
		t.Fatalf("counter read before: %+v %t %v", before, ok, err)
	}
	_ = v.Read(v.MatMul(w, x))
	v.BeginBatch()
	_ = v.MatMul(w, x)
	v.FlushBatch()
	after, ok, err := ReadBackendCounters(v)
	if !ok || err != nil || !after.SubmitsObserved || after.DispatchCountsLive != before.DispatchCountsLive {
		t.Fatalf("counter read after: %+v %t %v", after, ok, err)
	}
	submits := after.Submits - before.Submits
	profiled := after.Counters.DispatchSubmits - before.Counters.DispatchSubmits
	if submits < 2 {
		t.Fatalf("always-on submits moved %d for one one-shot readback and one batch", submits)
	}
	if after.DispatchCountsLive && profiled != submits || !after.DispatchCountsLive && profiled != 0 {
		t.Fatalf("live=%t always-on submits=%d profiler submits=%d", after.DispatchCountsLive, submits, profiled)
	}
	if !after.DispatchCountsLive && os.Getenv(vulkanProfileFixtureChildEnv) != t.Name() {
		runVulkanProfileFixture(t)
	}
}
