package dispatchtick

import "testing"

func TestThreadPressureAbsoluteInventoryAbstains(t *testing.T) {
	got := EvaluateThreadPressure(ThreadPressureInput{
		Cores: 32, TotalThreads: 13500, ThreadsPerCore: 400, ThreadsPerWorker: 200,
	})
	if got.Signal != ThreadSignalInventory || got.HardCap != nil {
		t.Fatalf("inventory result = %+v, want visible inventory with no hard cap", got)
	}
	cap, limiter := ApplyThreadPressure(4, got)
	if cap != 4 || limiter != "" {
		t.Fatalf("inventory fold = %d/%q, want identity 4/empty", cap, limiter)
	}
}

func TestThreadPressureBaselineChargesOnlyMarginalThreads(t *testing.T) {
	got := EvaluateThreadPressure(ThreadPressureInput{
		Cores: 32, TotalThreads: 13500, BaselineThreads: 13000,
		ThreadsPerCore: 400, ThreadsPerWorker: 200,
	})
	if got.Signal != ThreadSignalBaselineDelta || got.HardCap == nil {
		t.Fatalf("baseline result = %+v", got)
	}
	if got.ChargedThreads != 500 || *got.HardCap != 61 {
		t.Fatalf("baseline charge/cap = %d/%d, want 500/61", got.ChargedThreads, *got.HardCap)
	}
	cap, limiter := ApplyThreadPressure(16, got)
	if cap != 16 || limiter != "" {
		t.Fatalf("non-binding fold = %d/%q, want 16/empty", cap, limiter)
	}
}

func TestThreadPressureWorkerAttributedContracts(t *testing.T) {
	got := EvaluateThreadPressure(ThreadPressureInput{
		Cores: 4, TotalThreads: 5000, BaselineThreads: 4900,
		WorkerAttributedThreads: 1400, ThreadsPerCore: 400, ThreadsPerWorker: 200,
	})
	if got.Signal != ThreadSignalWorkerAttributed || got.HardCap == nil || *got.HardCap != 1 {
		t.Fatalf("attributed result = %+v, want hard cap 1", got)
	}
	cap, limiter := ApplyThreadPressure(4, got)
	if cap != 1 || limiter != ThreadSignalWorkerAttributed {
		t.Fatalf("binding fold = %d/%q, want 1/%q", cap, limiter, ThreadSignalWorkerAttributed)
	}
}

func TestThreadPressureInvalidBudgetFailsOpenWithoutInventingCapacity(t *testing.T) {
	got := EvaluateThreadPressure(ThreadPressureInput{Cores: 32, TotalThreads: 13500})
	if got.HardCap != nil || got.TotalThreads != 13500 {
		t.Fatalf("invalid-budget result = %+v", got)
	}
}
