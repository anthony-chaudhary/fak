package compute

import (
	"errors"
	"os"
	"strings"
	"testing"
)

type observationTestBackend struct {
	Backend
	name      string
	snapshots []BackendExecutionSnapshot
	err       error
	calls     int
	window    BackendExecutionWindow
	windowErr error
}

func TestVulkanObservationWindowNativeAccountingStructure(t *testing.T) {
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	for _, required := range []string{
		"g_submissionStatus != VK_SUCCESS ||",
		"!backendObservationQuiescent() || !g_deviceAllocationAccountingValid",
		"!g_allocationWindowActive || token == 0 || token != g_allocationWindowToken",
		"!h2d_count || !h2d_bytes || !d2h_count || !d2h_bytes ||",
		"!g_deviceAllocationAccountingValid || !live_bytes",
		"untrackDeviceAllocation(block.capacity, block.allocationDeviceLocal);",
		"untrackDeviceAllocation(b->allocationBytes, b->allocationDeviceLocal);",
	} {
		if !strings.Contains(source, required) {
			t.Fatalf("native observation guard/accounting seam missing %q", required)
		}
	}
	quiescence := sourceSection(t, source, "static bool backendObservationQuiescent()", "int fvk_device_allocation_window_begin")
	for _, pending := range []string{
		"!g_batching", "g_batchCmd == VK_NULL_HANDLE", "g_batchOps == 0",
		"g_batchD2DCount == 0", "g_batchD2DBytes == 0", "!g_restore.stage",
		"g_restore.cmd == VK_NULL_HANDLE", "g_restore.entries == 0",
		"g_restore.payloadBytes == 0",
	} {
		if !strings.Contains(quiescence, pending) {
			t.Fatalf("native observation quiescence guard missing %q", pending)
		}
	}
	transferSnapshot := sourceSection(t, source, "int fvk_transfer_counters", "int fvk_device_allocation_snapshot")
	if !strings.Contains(transferSnapshot, "g_submissionStatus != VK_SUCCESS") ||
		strings.Index(transferSnapshot, "!h2d_count") > strings.Index(transferSnapshot, "*h2d_count =") {
		t.Fatal("transfer snapshot must reject a poisoned context and incomplete output tuple before writing")
	}
	allocationSnapshot := sourceSection(t, source, "int fvk_device_allocation_snapshot", "static bool backendObservationQuiescent()")
	if !strings.Contains(allocationSnapshot, "g_submissionStatus != VK_SUCCESS") ||
		strings.Index(allocationSnapshot, "!live_bytes") > strings.Index(allocationSnapshot, "*live_bytes =") {
		t.Fatal("allocation snapshot must reject a poisoned context and missing output before writing")
	}
	begin := sourceSection(t, source, "int fvk_device_allocation_window_begin", "int fvk_device_allocation_window_end")
	if !strings.Contains(begin, "g_submissionStatus != VK_SUCCESS") || !strings.Contains(begin, "!backendObservationQuiescent()") {
		t.Fatal("observation begin must reject a poisoned or non-quiescent context")
	}
	end := sourceSection(t, source, "int fvk_device_allocation_window_end", "void fvk_sync")
	closed := strings.Index(end, "g_allocationWindowActive = false")
	failureCheck := strings.Index(end, "g_submissionStatus != VK_SUCCESS")
	if closed < 0 || failureCheck < 0 || closed > failureCheck || !strings.Contains(end, "!backendObservationQuiescent()") {
		t.Fatal("valid-token observation end must close before rejecting a poisoned or non-quiescent context")
	}
	restore := sourceSection(t, source, "int fvk_restore_submit", "void fvk_restore_finish")
	forcedFailure := strings.Index(restore, "g_submissionStatus = VK_ERROR_DEVICE_LOST")
	forcedReturn := strings.Index(restore, "return (int)VK_ERROR_DEVICE_LOST")
	if forcedFailure < 0 || forcedReturn < 0 || forcedFailure > forcedReturn {
		t.Fatal("forced restore submission failure must poison the context before returning")
	}
	if strings.Count(source, "trackDeviceAllocation(blockBytes, memoryType);") != 1 {
		t.Fatal("weight-arena VkDeviceMemory block must be counted exactly once")
	}
	reuse := sourceSection(t, source, "void* fvk_malloc(size_t bytes)", "void* fvk_malloc_weight")
	if strings.Contains(reuse, "trackDeviceAllocation") || strings.Contains(reuse, "untrackDeviceAllocation") {
		t.Fatal("reusing a pooled buffer must not change live VkDeviceMemory accounting")
	}
	h2d := sourceSection(t, source, "void copyHostToDevice", "void copyDeviceToHost")
	if strings.Index(h2d, "endSubmitWait(cmd);") > strings.Index(h2d, "checkedCounterAdd(g_h2dCount") {
		t.Fatal("H2D counters must advance only after successful submission completion")
	}
	d2h := sourceSection(t, source, "void copyDeviceToHost", "// ---- SPIR-V")
	if strings.Index(d2h, "memcpy(host") > strings.Index(d2h, "checkedCounterAdd(g_d2hCount") {
		t.Fatal("D2H counters must advance only after the host copy completes")
	}
	batch := sourceSection(t, source, "void batchFlush()", "// staging copy")
	if strings.Index(batch, "endSubmitWait(g_batchCmd)") > strings.Index(batch, "checkedCounterAdd(g_d2dCount") {
		t.Fatal("batched D2D counters must advance only after successful flush completion")
	}
	if strings.Index(restore, "if (r != VK_SUCCESS)") > strings.Index(restore, "checkedCounterAdd(g_h2dCount") ||
		!strings.Contains(restore, "size_t submittedEntries = g_restore.entries;") {
		t.Fatal("restore H2D counts and bytes must cover completed copy entries only")
	}
}

func sourceSection(t *testing.T, source, begin, end string) string {
	t.Helper()
	start := strings.Index(source, begin)
	if start < 0 {
		t.Fatalf("source section start %q missing", begin)
	}
	stop := strings.Index(source[start:], end)
	if stop < 0 {
		t.Fatalf("source section end %q missing", end)
	}
	return source[start : start+stop]
}

func (b *observationTestBackend) Name() string            { return b.name }
func (b *observationTestBackend) Tier() string            { return "test" }
func (b *observationTestBackend) Class() CorrectnessClass { return Approx }
func (b *observationTestBackend) Caps() Caps              { return Caps{DeviceMemory: true} }
func (b *observationTestBackend) BackendExecutionSnapshot() (BackendExecutionSnapshot, error) {
	b.calls++
	if b.err != nil {
		return BackendExecutionSnapshot{}, b.err
	}
	return b.snapshots[b.calls-1], nil
}
func (b *observationTestBackend) BeginBackendExecutionWindow() (BackendExecutionWindow, error) {
	return b.window, b.windowErr
}

type observationTestWindow struct {
	observation BackendExecutionObservation
	err         error
	calls       int
}

func (w *observationTestWindow) End() (BackendExecutionObservation, error) {
	w.calls++
	if w.calls > 1 {
		return BackendExecutionObservation{}, errors.New("stale observation window")
	}
	return w.observation, w.err
}

type unsupportedObservationBackend struct {
	Backend
	name string
}

func (b unsupportedObservationBackend) Name() string            { return b.name }
func (b unsupportedObservationBackend) Tier() string            { return "test" }
func (b unsupportedObservationBackend) Class() CorrectnessClass { return Approx }
func (b unsupportedObservationBackend) Caps() Caps              { return Caps{} }

func TestPhysicalObservationDeltaExcludesPriorCumulativeCounters(t *testing.T) {
	identity := BackendRuntimeIdentity{Backend: "vulkan", Device: "observed-device", Driver: "observed-driver", Runtime: "vulkan-1.3.0"}
	before := BackendExecutionSnapshot{
		Identity: identity,
		Counters: BackendCounterSnapshot{
			ComputeDispatches: 100, Q4KMatmulDispatches: 80, OtherDispatches: 20, DispatchSubmits: 50,
			H2DBytes: 1000, H2DCount: 10, D2HBytes: 100, D2HCount: 5, D2DCopies: 7, D2DBytes: 700,
			Q4KStageCalls: 4, Q4KStageBytes: 400,
			Fallbacks: 2, TensorHomeHits: 10, TensorHomeAdmissions: 3, TensorHomeBypasses: 1, TensorHomeCopiedBytes: 300,
		},
		TensorHomeEntries: 3, TensorHomeResidentBytes: 300,
		DeviceMemoryTotalBytes: 10000, DeviceMemoryFreeBytes: 7000, DeviceMemoryObserved: true,
		TransferCountersObserved: true, DeviceAllocationLiveBytes: 2048, DeviceAllocationObserved: true,
	}
	after := before
	after.Counters.ComputeDispatches += 9
	after.Counters.Q4KMatmulDispatches += 7
	after.Counters.OtherDispatches += 2
	after.Counters.DispatchSubmits += 3
	after.Counters.H2DBytes += 64
	after.Counters.H2DCount += 2
	after.Counters.D2HBytes += 8
	after.Counters.D2HCount++
	after.Counters.D2DCopies += 1
	after.Counters.D2DBytes += 32
	after.Counters.Q4KStageCalls += 2
	after.Counters.Q4KStageBytes += 32
	after.Counters.TensorHomeHits += 4
	after.Counters.TensorHomeAdmissions += 1
	after.Counters.TensorHomeCopiedBytes += 16
	after.TensorHomeEntries = 4
	after.TensorHomeResidentBytes = 316
	after.DeviceMemoryFreeBytes = 6800

	got, err := BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatal(err)
	}
	if got.Counters.ComputeDispatches != 9 || got.Counters.Q4KMatmulDispatches != 7 ||
		got.Counters.H2DBytes != 64 || got.Counters.H2DCount != 2 || got.Counters.D2HCount != 1 ||
		got.Counters.D2DCopies != 1 || got.Counters.D2DBytes != 32 || got.Counters.TensorHomeHits != 4 {
		t.Fatalf("delta includes cumulative history or lost execution values: %+v", got.Counters)
	}
	if got.Counters.Fallbacks != 0 || got.TensorHomeEntries != 4 || got.TensorHomeResidentBytes != 316 ||
		got.DeviceMemoryFreeBytes != 6800 || !got.TransferCountersObserved || got.DeviceAllocationObserved ||
		got.DeviceAllocationLiveBytes != 0 || got.DeviceAllocationPeakBytes != 0 {
		t.Fatalf("closing gauges/fallback delta mismatch: %+v", got)
	}
}

func TestPhysicalObservationRejectsResetIdentityDriftAndMissingIdentity(t *testing.T) {
	identity := BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "vulkan-1.3.0"}
	before := BackendExecutionSnapshot{Identity: identity, Counters: BackendCounterSnapshot{ComputeDispatches: 2}}

	t.Run("counter reset", func(t *testing.T) {
		after := before
		after.Counters.ComputeDispatches = 1
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "reset") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	t.Run("identity drift", func(t *testing.T) {
		after := before
		after.Identity.Device = "other"
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "changed") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	t.Run("missing identity", func(t *testing.T) {
		backend := &observationTestBackend{name: "vulkan", snapshots: []BackendExecutionSnapshot{{Identity: BackendRuntimeIdentity{Backend: "vulkan"}}}}
		if got, available, err := CaptureBackendExecutionSnapshot(backend); err == nil || available || got != (BackendExecutionSnapshot{}) {
			t.Fatalf("got=%+v available=%v err=%v", got, available, err)
		}
	})
	t.Run("transfer availability drift", func(t *testing.T) {
		after := before
		after.TransferCountersObserved = true
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "transfer-counter") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	t.Run("device allocation availability drift", func(t *testing.T) {
		after := before
		after.DeviceAllocationObserved = true
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "device-allocation") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
	t.Run("new transfer counter reset", func(t *testing.T) {
		before := before
		before.Counters.D2DBytes = 2
		after := before
		after.Counters.D2DBytes = 1
		if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) || !strings.Contains(err.Error(), "reset") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
}

func TestPhysicalObservationUnsupportedAndObserverErrorStayUnavailable(t *testing.T) {
	unsupported := unsupportedObservationBackend{name: "cpu-ref"}
	if got, available, err := CaptureBackendExecutionSnapshot(unsupported); err != nil || available || got != (BackendExecutionSnapshot{}) {
		t.Fatalf("unsupported got=%+v available=%v err=%v", got, available, err)
	}

	backend := &observationTestBackend{name: "vulkan", err: errors.New("snapshot failed")}
	if got, available, err := CaptureBackendExecutionSnapshot(backend); err == nil || available || got != (BackendExecutionSnapshot{}) {
		t.Fatalf("failed observer got=%+v available=%v err=%v", got, available, err)
	}
}

func TestBackendExecutionObservationWindowIsOptionalAndSingleUse(t *testing.T) {
	if window, available, err := BeginBackendExecutionObservation(unsupportedObservationBackend{name: "cpu-ref"}); err != nil || available || window != nil {
		t.Fatalf("unsupported window=%v available=%v err=%v", window, available, err)
	}

	want := BackendExecutionObservation{
		Identity:                 BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "vulkan-1.3.0"},
		Counters:                 BackendCounterSnapshot{H2DCount: 1, H2DBytes: 64},
		TransferCountersObserved: true, DeviceAllocationLiveBytes: 1024,
		DeviceAllocationPeakBytes: 2048, DeviceAllocationObserved: true,
	}
	fake := &observationTestWindow{observation: want}
	backend := &observationTestBackend{name: "vulkan", window: fake}
	window, available, err := BeginBackendExecutionObservation(backend)
	if err != nil || !available || window == nil {
		t.Fatalf("window=%v available=%v err=%v", window, available, err)
	}
	if got, err := window.End(); err != nil || got != want {
		t.Fatalf("first end got=%+v err=%v", got, err)
	}
	if got, err := window.End(); err == nil || got != (BackendExecutionObservation{}) {
		t.Fatalf("second end got=%+v err=%v", got, err)
	}

	backend.window = nil
	if got, available, err := BeginBackendExecutionObservation(backend); err == nil || available || got != nil {
		t.Fatalf("nil provider window=%v available=%v err=%v", got, available, err)
	}
	backend.windowErr = errors.New("begin failed")
	if got, available, err := BeginBackendExecutionObservation(backend); err == nil || available || got != nil {
		t.Fatalf("failed provider window=%v available=%v err=%v", got, available, err)
	}
}

func TestBackendExecutionWindowDeltaRequiresCompleteConsistentAllocationPeak(t *testing.T) {
	identity := BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "vulkan-1.3.0"}
	before := BackendExecutionSnapshot{
		Identity: identity, Counters: BackendCounterSnapshot{H2DCount: 10, H2DBytes: 100},
		TransferCountersObserved: true, DeviceAllocationLiveBytes: 1000, DeviceAllocationObserved: true,
	}
	after := before
	after.Counters.H2DCount++
	after.Counters.H2DBytes += 64
	after.DeviceAllocationLiveBytes = 1200

	got, err := BackendExecutionWindowDelta(before, after, 1200, 1600)
	if err != nil {
		t.Fatal(err)
	}
	if !got.TransferCountersObserved || got.Counters.H2DCount != 1 || got.Counters.H2DBytes != 64 ||
		!got.DeviceAllocationObserved || got.DeviceAllocationLiveBytes != 1200 || got.DeviceAllocationPeakBytes != 1600 {
		t.Fatalf("window delta lost complete observed values: %+v", got)
	}

	for name, mutate := range map[string]func(*BackendExecutionSnapshot, *BackendExecutionSnapshot) (uint64, uint64){
		"missing transfers": func(before, after *BackendExecutionSnapshot) (uint64, uint64) {
			before.TransferCountersObserved = false
			return 1200, 1600
		},
		"missing allocation": func(before, after *BackendExecutionSnapshot) (uint64, uint64) {
			after.DeviceAllocationObserved = false
			return 1200, 1600
		},
		"closing mismatch": func(before, after *BackendExecutionSnapshot) (uint64, uint64) {
			return 1199, 1600
		},
		"peak below endpoint": func(before, after *BackendExecutionSnapshot) (uint64, uint64) {
			return 1200, 1199
		},
		"counter reset": func(before, after *BackendExecutionSnapshot) (uint64, uint64) {
			after.Counters.H2DCount = before.Counters.H2DCount - 1
			return 1200, 1600
		},
	} {
		t.Run(name, func(t *testing.T) {
			badBefore, badAfter := before, after
			live, peak := mutate(&badBefore, &badAfter)
			if got, err := BackendExecutionWindowDelta(badBefore, badAfter, live, peak); err == nil || got != (BackendExecutionObservation{}) {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
	}
}

func TestBackendExecutionDeltaRejectsIncompleteTransferPairs(t *testing.T) {
	identity := BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "vulkan-1.3.0"}
	base := BackendExecutionSnapshot{Identity: identity, TransferCountersObserved: true}
	for _, direction := range []struct {
		name     string
		setCount func(*BackendCounterSnapshot, uint64)
		setBytes func(*BackendCounterSnapshot, uint64)
	}{
		{name: "H2D", setCount: func(c *BackendCounterSnapshot, v uint64) { c.H2DCount = v }, setBytes: func(c *BackendCounterSnapshot, v uint64) { c.H2DBytes = v }},
		{name: "D2H", setCount: func(c *BackendCounterSnapshot, v uint64) { c.D2HCount = v }, setBytes: func(c *BackendCounterSnapshot, v uint64) { c.D2HBytes = v }},
		{name: "D2D", setCount: func(c *BackendCounterSnapshot, v uint64) { c.D2DCopies = v }, setBytes: func(c *BackendCounterSnapshot, v uint64) { c.D2DBytes = v }},
	} {
		t.Run(direction.name+" closing count only", func(t *testing.T) {
			after := base
			direction.setCount(&after.Counters, 1)
			if got, err := BackendExecutionDelta(base, after); err == nil || got != (BackendExecutionObservation{}) {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
		t.Run(direction.name+" closing bytes only", func(t *testing.T) {
			after := base
			direction.setBytes(&after.Counters, 1)
			if got, err := BackendExecutionDelta(base, after); err == nil || got != (BackendExecutionObservation{}) {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
		t.Run(direction.name+" delta mismatch", func(t *testing.T) {
			before, after := base, base
			direction.setCount(&before.Counters, 1)
			direction.setBytes(&before.Counters, 1)
			direction.setCount(&after.Counters, 2)
			direction.setBytes(&after.Counters, 1)
			if got, err := BackendExecutionDelta(before, after); err == nil || got != (BackendExecutionObservation{}) {
				t.Fatalf("got=%+v err=%v", got, err)
			}
		})
	}
}
