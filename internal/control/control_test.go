package control

import (
	"sync"
	"testing"
	"time"
)

func TestDiffAndResourceImpact(t *testing.T) {
	current := DefaultConfig()
	proposed := current

	// Case 1: Pure scalar modification -> LOW risk, 0 VRAM delta, no drain
	logLvl := "debug"
	deadline := uint32(10000)
	proposed = proposed.Apply(ConfigPatch{
		LogLevel:             &logLvl,
		CompletionDeadlineMs: &deadline,
	})

	diff := ComputeDiff(current, proposed)
	if len(diff) != 2 {
		t.Fatalf("expected 2 diff entries, got %d: %v", len(diff), diff)
	}
	if diff["log_level"].From != "info" || diff["log_level"].To != "debug" {
		t.Errorf("log_level diff mismatch: %+v", diff["log_level"])
	}

	impact := ComputeImpact(current, proposed, 10)
	if impact.RiskLevel != RiskLow {
		t.Errorf("expected RiskLow, got %s", impact.RiskLevel)
	}
	if impact.DrainRequired {
		t.Errorf("expected DrainRequired=false for pure scalar change")
	}
	if impact.VRAMDeltaBytes != 0 {
		t.Errorf("expected 0 VRAM delta, got %d", impact.VRAMDeltaBytes)
	}

	// Case 2: Memory expansion -> MEDIUM risk, positive VRAM delta, no drain
	expandBlocks := uint32(40000)
	proposedExpand := current
	proposedExpand.TargetKVBlocks = expandBlocks
	impactExpand := ComputeImpact(current, proposedExpand, 5)
	if impactExpand.RiskLevel != RiskMedium {
		t.Errorf("expected RiskMedium for expansion, got %s", impactExpand.RiskLevel)
	}
	if impactExpand.VRAMDeltaBytes <= 0 {
		t.Errorf("expected positive VRAM delta, got %d", impactExpand.VRAMDeltaBytes)
	}
	if impactExpand.DrainRequired {
		t.Errorf("expected DrainRequired=false for expansion")
	}

	// Case 3: Memory contraction -> HIGH_DRAIN_REQUIRED risk, negative VRAM delta, drain required
	contractBlocks := uint32(16384)
	proposedContract := current
	proposedContract.TargetKVBlocks = contractBlocks
	impactContract := ComputeImpact(current, proposedContract, 8)
	if impactContract.RiskLevel != RiskHighDrainRequired {
		t.Errorf("expected RiskHighDrainRequired for contraction, got %s", impactContract.RiskLevel)
	}
	if impactContract.VRAMDeltaBytes >= 0 {
		t.Errorf("expected negative VRAM delta, got %d", impactContract.VRAMDeltaBytes)
	}
	if !impactContract.DrainRequired {
		t.Errorf("expected DrainRequired=true for contraction")
	}
	if impactContract.EstimatedDrainMS <= 0 {
		t.Errorf("expected positive EstimatedDrainMS, got %d", impactContract.EstimatedDrainMS)
	}
}

func TestManager_DryRunPreservesLiveState(t *testing.T) {
	initial := DefaultConfig()
	mgr, err := NewManager(initial, DefaultWatchdogConfig(), func() int { return 4 })
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	cur := mgr.Active()
	if cur.Epoch != 1 {
		t.Fatalf("expected initial epoch 1, got %d", cur.Epoch)
	}

	newDepth := uint32(5)
	newTokens := uint32(32768)
	patch := ConfigPatch{
		SpeculativeDraftDepth: &newDepth,
		MaxBatchTokens:        &newTokens,
	}

	// Dry run evaluation
	res, err := mgr.DryRun(patch)
	if err != nil {
		t.Fatalf("DryRun failed: %v", err)
	}
	if res.Status != "dry_run" {
		t.Errorf("expected status 'dry_run', got %s", res.Status)
	}
	if !res.Valid {
		t.Errorf("expected res.Valid=true")
	}
	if res.ConfigEpoch != 1 {
		t.Errorf("expected ConfigEpoch=1 in dry_run, got %d", res.ConfigEpoch)
	}

	// Verify live state remains unmutated
	afterDryRun := mgr.Active()
	if afterDryRun.Epoch != 1 {
		t.Errorf("active epoch changed after dry run: %d", afterDryRun.Epoch)
	}
	if afterDryRun.Config.SpeculativeDraftDepth != initial.SpeculativeDraftDepth {
		t.Errorf("active config mutated after dry run: draft depth = %d", afterDryRun.Config.SpeculativeDraftDepth)
	}
}

func TestManager_ApplyAndDoubleBufferedLKG(t *testing.T) {
	initial := DefaultConfig()
	mgr, err := NewManager(initial, DefaultWatchdogConfig(), nil)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	if mgr.Active().Epoch != 1 {
		t.Fatalf("expected active epoch 1, got %d", mgr.Active().Epoch)
	}
	if mgr.LKG().Epoch != 1 {
		t.Fatalf("expected LKG epoch 1, got %d", mgr.LKG().Epoch)
	}

	// Apply candidate configuration
	newDepth := uint32(5)
	res, err := mgr.Apply(ConfigPatch{
		SpeculativeDraftDepth: &newDepth,
	}, false)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	if res.Status != "applied" {
		t.Errorf("expected status 'applied', got %s", res.Status)
	}
	if res.ConfigEpoch != 2 {
		t.Errorf("expected epoch 2, got %d", res.ConfigEpoch)
	}

	// Verify Active is Epoch 2 while LKG remains Epoch 1 (double-buffering)
	active := mgr.Active()
	if active.Epoch != 2 || active.Config.SpeculativeDraftDepth != 5 {
		t.Errorf("active config not updated to epoch 2 with depth 5: %+v", active)
	}

	lkg := mgr.LKG()
	if lkg.Epoch != 1 || lkg.Config.SpeculativeDraftDepth != initial.SpeculativeDraftDepth {
		t.Errorf("LKG config prematurely updated before stabilization: %+v", lkg)
	}

	// Verify audit event in event stream
	events := mgr.EventStream().Snapshot()
	if len(events) != 1 {
		t.Fatalf("expected 1 audit event, got %d", len(events))
	}
	if events[0].Event != EventSystemConfigApplied || events[0].ToEpoch != 2 {
		t.Errorf("unexpected event: %+v", events[0])
	}
}

func TestWatchdog_AutomaticRollback_SpeculativeCollapse(t *testing.T) {
	initial := DefaultConfig()
	initial.SpeculativeDraftDepth = 3
	initial.SpeculativeAcceptanceThreshold = 0.80

	wcfg := DefaultWatchdogConfig()
	wcfg.StabilizationWindow = 10 * time.Second

	mgr, err := NewManager(initial, wcfg, nil)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	// Apply candidate with depth 6
	newDepth := uint32(6)
	_, err = mgr.Apply(ConfigPatch{
		SpeculativeDraftDepth: &newDepth,
	}, false)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	if mgr.Active().Epoch != 2 {
		t.Fatalf("expected active epoch 2, got %d", mgr.Active().Epoch)
	}

	// Feed healthy baseline telemetry sample
	triggered, _, _ := mgr.IngestTelemetry(TelemetrySample{
		SpeculativeAcceptanceRate: 0.82,
		TTFTp99MS:                 120.0,
		Error5xxRate:              0.0001,
	})
	if triggered {
		t.Fatalf("unexpected trigger on healthy sample")
	}

	// Feed collapsed speculative acceptance sample (0.82 -> 0.35, collapse > 50%)
	triggered, triggerName, detail := mgr.IngestTelemetry(TelemetrySample{
		SpeculativeAcceptanceRate: 0.35,
		TTFTp99MS:                 125.0,
		Error5xxRate:              0.0001,
	})
	if !triggered {
		t.Fatalf("expected rollback trigger for speculative collapse, got none")
	}
	if triggerName != TriggerSpeculativeCollapse {
		t.Errorf("expected trigger %s, got %s (detail: %s)", TriggerSpeculativeCollapse, triggerName, detail)
	}

	// Verify microsecond automatic rollback to LKG
	active := mgr.Active()
	if active.Epoch != 3 {
		t.Errorf("expected monotonic epoch 3 after rollback, got %d", active.Epoch)
	}
	if active.Config.SpeculativeDraftDepth != initial.SpeculativeDraftDepth {
		t.Errorf("active config did not roll back to LKG: depth = %d, want %d",
			active.Config.SpeculativeDraftDepth, initial.SpeculativeDraftDepth)
	}

	// Verify SYSTEM_CONFIG_AUTOMATIC_ROLLBACK in event stream
	events := mgr.EventStream().Snapshot()
	var rollbackFound bool
	for _, ev := range events {
		if ev.Event == EventSystemConfigAutomaticRollback {
			rollbackFound = true
			if ev.Trigger != TriggerSpeculativeCollapse {
				t.Errorf("expected event trigger %s, got %s", TriggerSpeculativeCollapse, ev.Trigger)
			}
			if ev.FromEpoch != 2 || ev.ToEpoch != 3 {
				t.Errorf("expected from 2 to 3, got from %d to %d", ev.FromEpoch, ev.ToEpoch)
			}
		}
	}
	if !rollbackFound {
		t.Errorf("SYSTEM_CONFIG_AUTOMATIC_ROLLBACK not found in event stream: %+v", events)
	}
}

func TestWatchdog_AutomaticRollback_LatencySLABreach(t *testing.T) {
	initial := DefaultConfig()
	initial.DeclaredLatencySLAMS = 150.0

	wcfg := DefaultWatchdogConfig()
	mgr, _ := NewManager(initial, wcfg, nil)

	newTokens := uint32(32768)
	_, _ = mgr.Apply(ConfigPatch{MaxBatchTokens: &newTokens}, false)

	// Feed latency spike: 220ms > 150ms SLA
	triggered, triggerName, _ := mgr.IngestTelemetry(TelemetrySample{
		TTFTp99MS:    220.0,
		Error5xxRate: 0.0,
	})
	if !triggered {
		t.Fatalf("expected rollback trigger on latency SLA breach")
	}
	if triggerName != TriggerLatencySLABreach {
		t.Errorf("expected trigger %s, got %s", TriggerLatencySLABreach, triggerName)
	}

	active := mgr.Active()
	if active.Epoch != 3 {
		t.Errorf("expected epoch 3 after rollback, got %d", active.Epoch)
	}
	if active.Config.MaxBatchTokens != initial.MaxBatchTokens {
		t.Errorf("config did not roll back to LKG tokens %d, got %d",
			initial.MaxBatchTokens, active.Config.MaxBatchTokens)
	}
}

func TestWatchdog_AutomaticRollback_5xxErrorRate(t *testing.T) {
	initial := DefaultConfig()
	wcfg := DefaultWatchdogConfig()
	mgr, _ := NewManager(initial, wcfg, nil)

	newDepth := uint32(4)
	_, _ = mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &newDepth}, false)

	// Feed 5xx error rate: 0.005 (0.5% > 0.1% ceiling)
	triggered, triggerName, _ := mgr.IngestTelemetry(TelemetrySample{
		TotalRequests: 1000,
		Errors5xx:     5, // 0.5%
	})
	if !triggered {
		t.Fatalf("expected rollback trigger on 5xx error rate exceeded")
	}
	if triggerName != Trigger5xxErrorRateExceeded {
		t.Errorf("expected trigger %s, got %s", Trigger5xxErrorRateExceeded, triggerName)
	}

	active := mgr.Active()
	if active.Epoch != 3 {
		t.Errorf("expected epoch 3 after rollback, got %d", active.Epoch)
	}
	if active.Config.SpeculativeDraftDepth != initial.SpeculativeDraftDepth {
		t.Errorf("config did not roll back to LKG draft depth %d, got %d",
			initial.SpeculativeDraftDepth, active.Config.SpeculativeDraftDepth)
	}
}

func TestManager_WatchdogStabilizationPromotesToLKG(t *testing.T) {
	initial := DefaultConfig()
	wcfg := DefaultWatchdogConfig()
	wcfg.StabilizationWindow = 50 * time.Millisecond

	mgr, _ := NewManager(initial, wcfg, nil)

	newDepth := uint32(5)
	_, _ = mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &newDepth}, false)

	if mgr.LKG().Epoch != 1 {
		t.Fatalf("LKG should be epoch 1 prior to stabilization")
	}

	// Sleep past stabilization window and check
	time.Sleep(70 * time.Millisecond)
	mgr.Watchdog().CheckStabilization(time.Now().UTC())

	if mgr.LKG().Epoch != 2 {
		t.Errorf("expected LKG to be promoted to epoch 2, got %d", mgr.LKG().Epoch)
	}
	if mgr.LKG().Config.SpeculativeDraftDepth != 5 {
		t.Errorf("expected LKG depth 5, got %d", mgr.LKG().Config.SpeculativeDraftDepth)
	}

	// Verify SYSTEM_CONFIG_COMMITTED_LKG event in stream
	var commitFound bool
	for _, ev := range mgr.EventStream().Snapshot() {
		if ev.Event == EventSystemConfigCommittedLKG {
			commitFound = true
			break
		}
	}
	if !commitFound {
		t.Errorf("SYSTEM_CONFIG_COMMITTED_LKG not found in event stream")
	}
}

func TestManager_ConcurrentReadAndMutations(t *testing.T) {
	initial := DefaultConfig()
	mgr, _ := NewManager(initial, DefaultWatchdogConfig(), nil)

	var wg sync.WaitGroup
	workers := 10
	iterations := 50

	// Concurrent readers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				vc := mgr.Active()
				if vc.Epoch == 0 {
					t.Errorf("zero epoch read")
				}
				_ = mgr.LKG()
			}
		}()
	}

	// Concurrent dry-run and apply operations
	for i := 0; i < workers; i++ {
		wg.Add(1)
		workerID := i
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				depth := uint32(j % 8)
				if workerID%2 == 0 {
					_, _ = mgr.DryRun(ConfigPatch{SpeculativeDraftDepth: &depth})
				} else {
					_, _ = mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &depth}, false)
				}
			}
		}()
	}

	wg.Wait()

	finalActive := mgr.Active()
	if finalActive.Epoch <= 1 {
		t.Errorf("expected epoch to advance under concurrent apply, got %d", finalActive.Epoch)
	}
}

func TestManager_ConcurrentApplyAndTelemetryRollback_NoDeadlock(t *testing.T) {
	initial := DefaultConfig()
	wcfg := DefaultWatchdogConfig()
	wcfg.StabilizationWindow = 1 * time.Millisecond

	mgr, err := NewManager(initial, wcfg, nil)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	const iterations = 1000
	var wg sync.WaitGroup
	wg.Add(3)

	// Goroutine 1: Repeatedly applies configuration patches (mgr.Apply(..., false))
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			depth := uint32((i % 8) + 1)
			_, _ = mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &depth}, false)
		}
	}()

	// Goroutine 2: Repeatedly feeds telemetry samples triggering rollbacks (mgr.IngestTelemetry(...))
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			mgr.IngestTelemetry(TelemetrySample{
				Timestamp:    time.Now().UTC(),
				Error5xxRate: 0.05,
			})
		}
	}()

	// Goroutine 3: Repeatedly checks stabilization (mgr.Watchdog().CheckStabilization(...))
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			mgr.Watchdog().CheckStabilization(time.Now().UTC().Add(time.Hour))
		}
	}()

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("deadlock detected: timed out waiting for concurrent apply, telemetry rollback, and stabilization checks")
	}
}

func TestManager_ObserverReentrancyNoDeadlock(t *testing.T) {
	initial := DefaultConfig()
	mgr, err := NewManager(initial, DefaultWatchdogConfig(), nil)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	var reentrantApplyDone bool
	var reentrantRollbackDone bool
	var reentrantRegisterDone bool

	mgr.RegisterObserver(func(vc VersionedConfig) {
		if vc.Epoch == 2 && !reentrantApplyDone {
			reentrantApplyDone = true
			newDepth := uint32(7)
			// Reentrant Apply from within Apply observer callback
			_, err := mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &newDepth}, false)
			if err != nil {
				t.Errorf("reentrant Apply failed: %v", err)
			}
		} else if vc.Epoch == 3 && !reentrantRollbackDone {
			reentrantRollbackDone = true
			// Reentrant Rollback from within Apply observer callback
			_, err := mgr.Rollback("manual", "reentrant rollback")
			if err != nil {
				t.Errorf("reentrant Rollback failed: %v", err)
			}
		} else if vc.Epoch == 4 && !reentrantRegisterDone {
			reentrantRegisterDone = true
			// Reentrant RegisterObserver from within Rollback observer callback
			mgr.RegisterObserver(func(VersionedConfig) {})
		}
	})

	done := make(chan struct{})
	go func() {
		newDepth := uint32(4)
		_, err := mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &newDepth}, false)
		if err != nil {
			t.Errorf("initial Apply failed: %v", err)
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("deadlock detected: timed out waiting for reentrant observer operations")
	}

	if !reentrantApplyDone {
		t.Error("reentrant Apply was not executed")
	}
	if !reentrantRollbackDone {
		t.Error("reentrant Rollback was not executed")
	}
	if !reentrantRegisterDone {
		t.Error("reentrant RegisterObserver was not executed")
	}
}
