package compute

import (
	"strings"
	"sync/atomic"
	"testing"
)

func resetHostFallbacksForTest(t *testing.T) {
	t.Helper()
	hostFallbacks.mu.Lock()
	saved := hostFallbacks.byStage
	hostFallbacks.byStage = nil
	hostFallbacks.mu.Unlock()
	t.Cleanup(func() {
		hostFallbacks.mu.Lock()
		hostFallbacks.byStage = saved
		hostFallbacks.mu.Unlock()
	})
}

func TestDeviceDispatchCountersAdvanceForFakeBackend(t *testing.T) {
	var dispatches, readback atomic.Uint64
	t.Cleanup(RegisterDeviceWorkReader("fake-device-work", func() DeviceWork {
		return DeviceWork{Dispatches: dispatches.Load(), ReadbackBytes: readback.Load(), SubmitWaitNS: 7}
	}))
	find := func() DeviceWork {
		for _, w := range DeviceWorkSnapshot() {
			if w.Backend == "fake-device-work" {
				return w
			}
		}
		t.Fatal("registered backend missing from snapshot")
		return DeviceWork{}
	}
	before := find()
	dispatches.Add(26)
	readback.Add(1 << 20)
	after := find()
	if got := after.Dispatches - before.Dispatches; got != 26 {
		t.Fatalf("dispatch delta = %d, want 26", got)
	}
	if got := after.ReadbackBytes - before.ReadbackBytes; got != 1<<20 {
		t.Fatalf("readback delta = %d, want %d", got, 1<<20)
	}
	if after.SubmitWaitNS != 7 {
		t.Fatalf("submit wait = %d, want 7", after.SubmitWaitNS)
	}
}

func TestDeviceDispatchUnregisteredBackendIsAbsent(t *testing.T) {
	for _, w := range DeviceWorkSnapshot() {
		if w.Backend == "never-registered" {
			t.Fatal("an unregistered backend must be absent, not a zero series")
		}
	}
}

func TestHostFallbackCountsByStage(t *testing.T) {
	resetHostFallbacksForTest(t)
	DeclareHostFallbackStage("logits_readback")
	CountHostFallback("host_sampling")
	CountHostFallback("host_sampling")
	got := map[string]uint64{}
	for _, c := range HostFallbackCounts() {
		got[c.Stage] = c.Count
	}
	if got["host_sampling"] != 2 {
		t.Fatalf("host_sampling = %d, want 2", got["host_sampling"])
	}
	if v, ok := got["logits_readback"]; !ok || v != 0 {
		t.Fatalf("declared stage = (%d, %v), want (0, true)", v, ok)
	}
}

func TestHostFallbackLabelsStayBounded(t *testing.T) {
	resetHostFallbacksForTest(t)
	CountHostFallback("Not A Label!")
	for i := 0; i < 2*maxHostFallbackStages; i++ {
		CountHostFallback("stage_" + strings.Repeat("x", i%40) + string(rune('a'+i%26)))
	}
	counts := HostFallbackCounts()
	if len(counts) > maxHostFallbackStages {
		t.Fatalf("stages = %d, want <= %d", len(counts), maxHostFallbackStages)
	}
	var other uint64
	for _, c := range counts {
		if c.Stage == HostFallbackOtherStage {
			other = c.Count
		}
	}
	if other == 0 {
		t.Fatal("invalid and overflow stages must fold into the other stage")
	}
}
