package nativeperf

import (
	"strings"
	"testing"
	"time"
)

func deterministicDelayReceipt(t *testing.T) PhaseReceipt {
	t.Helper()
	r := NewPhaseRecorder("inkernel", "cuda", "qwen3.8_cuda", 30*time.Millisecond)
	adds := []struct {
		phase, parent Phase
		start, end    time.Duration
		kind          WorkKind
	}{
		{PhaseQueueAdmission, "", 0, 3 * time.Millisecond, WorkWait},
		{PhaseModelLoad, "", 3 * time.Millisecond, 5 * time.Millisecond, WorkActive},
		{PhaseTokenization, "", 5 * time.Millisecond, 7 * time.Millisecond, WorkActive},
		{PhaseKVLookup, "", 7 * time.Millisecond, 8 * time.Millisecond, WorkActive},
		{PhaseKVRestore, "", 8 * time.Millisecond, 10 * time.Millisecond, WorkWait},
		{PhaseHostUpload, PhasePrefill, 10 * time.Millisecond, 14 * time.Millisecond, WorkWait},
		{PhasePrefill, "", 10 * time.Millisecond, 18 * time.Millisecond, WorkActive},
		{PhaseKernel, PhasePrefill, 14 * time.Millisecond, 17 * time.Millisecond, WorkActive},
		{PhaseSynchronization, PhasePrefill, 17 * time.Millisecond, 18 * time.Millisecond, WorkWait},
		{PhaseDecode, "", 18 * time.Millisecond, 25 * time.Millisecond, WorkActive},
		{PhaseKernel, PhaseDecode, 19 * time.Millisecond, 23 * time.Millisecond, WorkActive},
		{PhaseSampling, PhaseDecode, 23 * time.Millisecond, 25 * time.Millisecond, WorkActive},
		{PhaseHostDownload, "", 25 * time.Millisecond, 27 * time.Millisecond, WorkWait},
		{PhaseKVEvict, "", 27 * time.Millisecond, 28 * time.Millisecond, WorkActive},
		{PhaseOutput, "", 28 * time.Millisecond, 30 * time.Millisecond, WorkActive},
	}
	for _, add := range adds {
		if err := r.Add(add.phase, add.parent, add.start, add.end, add.kind); err != nil {
			t.Fatalf("Add(%s): %v", add.phase, err)
		}
	}
	receipt, err := r.Finalize(0)
	if err != nil {
		t.Fatalf("Finalize: %v", err)
	}
	return receipt
}

func TestPhaseRecorderDeterministicDelaysReconcileWithoutNestedDoubleCounting(t *testing.T) {
	receipt := deterministicDelayReceipt(t)
	if receipt.Engine != "inkernel" || receipt.FallbackActive {
		t.Fatalf("receipt does not identify non-fallback fak-native: %+v", receipt)
	}
	if err := receipt.Validate(0); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	byPhase := map[Phase]PhaseAccounting{}
	var exclusive time.Duration
	for _, phase := range receipt.Phases {
		byPhase[phase.Phase] = phase
		exclusive += phase.Exclusive.Wall
	}
	if exclusive != receipt.Wall {
		t.Fatalf("exclusive wall=%s want request wall=%s", exclusive, receipt.Wall)
	}
	checks := map[Phase]PhaseTiming{
		PhaseQueueAdmission: {Wall: 3 * time.Millisecond, Wait: 3 * time.Millisecond},
		PhaseHostUpload:     {Wall: 4 * time.Millisecond, Wait: 4 * time.Millisecond},
		PhaseKernel:         {Wall: 7 * time.Millisecond, Active: 7 * time.Millisecond},
		PhasePrefill:        {},
		PhaseDecode:         {Wall: time.Millisecond, Active: time.Millisecond},
	}
	for phase, want := range checks {
		if got := byPhase[phase].Exclusive; got != want {
			t.Errorf("%s exclusive=%+v want %+v", phase, got, want)
		}
	}
	if got := byPhase[PhasePrefill].Inclusive.Wall; got != 8*time.Millisecond {
		t.Fatalf("prefill inclusive wall=%s want 8ms", got)
	}
	if got := byPhase[PhaseDecode].Inclusive.Wall; got != 7*time.Millisecond {
		t.Fatalf("decode inclusive wall=%s want 7ms", got)
	}
}

func TestReceiptMetricsObservePhasesExportsBoundedExclusiveWallActiveWait(t *testing.T) {
	receipt := deterministicDelayReceipt(t)
	metrics := NewReceiptMetrics(time.Minute)
	if !metrics.ObservePhases(receipt, time.Unix(1700000000, 0)) {
		t.Fatal("ObservePhases rejected valid fak-native receipt")
	}
	text := metrics.Prometheus(time.Unix(1700000001, 0))
	for _, want := range []string{
		`phase="queue_admission",kind="wait"} 0.003`,
		`phase="host_device_upload",kind="wait"} 0.004`,
		`phase="kernel",kind="active"} 0.007`,
		`phase="prefill",kind="wall"} 0`,
		`phase="decode",kind="wall"} 0.001`,
		`engine="inkernel",backend="cuda",forward_path="qwen_cuda"`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("metrics missing %q\n%s", want, text)
		}
	}
	if strings.Contains(text, "qwen3.8_cuda") || strings.Contains(text, "llama") {
		t.Fatalf("unbounded or fallback identity leaked into metrics:\n%s", text)
	}
}

func TestPhaseReceiptRejectsFallbackAndUnreconciledAccounting(t *testing.T) {
	receipt := deterministicDelayReceipt(t)
	receipt.FallbackActive = true
	if err := receipt.Validate(0); err == nil {
		t.Fatal("fallback-active receipt accepted")
	}
	receipt.FallbackActive = false
	receipt.Phases[0].Exclusive.Active++
	if err := receipt.Validate(0); err == nil {
		t.Fatal("unreconciled active/wait timing accepted")
	}
}

func TestReceiptMetricsObservePhasesConcurrent(t *testing.T) {
	receipt := deterministicDelayReceipt(t)
	metrics := NewReceiptMetrics(time.Minute)
	const workers = 16
	done := make(chan struct{}, workers)
	for i := 0; i < workers; i++ {
		go func() {
			metrics.ObservePhases(receipt, time.Unix(1700000000, 0))
			_ = metrics.Prometheus(time.Unix(1700000001, 0))
			done <- struct{}{}
		}()
	}
	for i := 0; i < workers; i++ {
		<-done
	}
	text := metrics.Prometheus(time.Unix(1700000001, 0))
	if !strings.Contains(text, `phase="kernel",kind="active"} 0.112`) {
		t.Fatalf("concurrent kernel total did not reconcile:\n%s", text)
	}
}
