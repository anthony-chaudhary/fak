package control

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestManager_RegisterObserver verifies that registered EpochObservers receive
// callbacks synchronously whenever the active configuration epoch advances,
// both during standard candidate application and during rollback transitions.
func TestManager_RegisterObserver(t *testing.T) {
	initial := DefaultConfig()
	mgr, err := NewManager(initial, DefaultWatchdogConfig(), nil)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	var mu sync.Mutex
	var observedEpochs []uint64

	// Register observer to capture active versioned config transitions.
	mgr.RegisterObserver(func(vc VersionedConfig) {
		mu.Lock()
		defer mu.Unlock()
		observedEpochs = append(observedEpochs, vc.Epoch)
	})

	// Apply a configuration update advancing Epoch from 1 to 2.
	newDepth := uint32(5)
	_, err = mgr.Apply(ConfigPatch{
		SpeculativeDraftDepth: &newDepth,
	}, false)
	if err != nil {
		t.Fatalf("Apply failed: %v", err)
	}

	// Rollback to LKG advancing Epoch from 2 to 3.
	_, err = mgr.Rollback("manual", "operator requested rollback")
	if err != nil {
		t.Fatalf("Rollback failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(observedEpochs) != 2 {
		t.Fatalf("expected 2 observer notifications, got %d: %v", len(observedEpochs), observedEpochs)
	}
	if observedEpochs[0] != 2 {
		t.Errorf("expected first notification for Epoch 2, got %d", observedEpochs[0])
	}
	if observedEpochs[1] != 3 {
		t.Errorf("expected second notification for Epoch 3, got %d", observedEpochs[1])
	}
}

// TestWatchdog_Status verifies that the Watchdog Status query correctly reports
// the lifecycle phase (CanaryState), evaluated epoch, baseline acceptance rate,
// and positive elapsed duration across all lifecycle transitions.
func TestWatchdog_Status(t *testing.T) {
	wcfg := DefaultWatchdogConfig()
	wd := NewWatchdog(wcfg, nil, nil)

	// Invariant 1: Freshly initialized watchdog must be in CanaryStateIdle.
	state, epoch, baseline, elapsed := wd.Status()
	if state != CanaryStateIdle {
		t.Errorf("expected CanaryStateIdle, got %v", state)
	}
	if epoch != 0 {
		t.Errorf("expected initial epoch 0, got %d", epoch)
	}
	if baseline != 0 {
		t.Errorf("expected baseline 0, got %f", baseline)
	}
	if elapsed != 0 {
		t.Errorf("expected elapsed 0 in idle state, got %v", elapsed)
	}

	// Invariant 2: Once evaluation begins, state transitions to CanaryStateEvaluating
	// with valid epoch, baseline acceptance rate, and non-negative elapsed duration.
	wd.StartEvaluation(2, 200.0, 0.85)
	state, epoch, baseline, elapsed = wd.Status()
	if state != CanaryStateEvaluating {
		t.Errorf("expected CanaryStateEvaluating, got %v", state)
	}
	if epoch != 2 {
		t.Errorf("expected epoch 2, got %d", epoch)
	}
	if baseline != 0.85 {
		t.Errorf("expected baseline 0.85, got %f", baseline)
	}
	if elapsed < 0 {
		t.Errorf("expected non-negative elapsed duration, got %v", elapsed)
	}

	// Invariant 3: Triggering rollback updates state to CanaryStateRolledBack.
	triggered, triggerName, _ := wd.IngestTelemetry(TelemetrySample{
		Error5xxRate: 0.05, // Exceeds ceiling
	})
	if !triggered || triggerName != Trigger5xxErrorRateExceeded {
		t.Fatalf("expected rollback trigger, got %v, %s", triggered, triggerName)
	}
	state, _, _, _ = wd.Status()
	if state != CanaryStateRolledBack {
		t.Errorf("expected CanaryStateRolledBack, got %v", state)
	}

	// Invariant 4: CanaryStateStabilized state reporting.
	wd2 := NewWatchdog(WatchdogConfig{StabilizationWindow: 10 * time.Millisecond}, nil, nil)
	wd2.StartEvaluation(3, 250.0, 0.90)
	time.Sleep(15 * time.Millisecond)
	wd2.CheckStabilization(time.Now().UTC())
	state2, epoch2, _, _ := wd2.Status()
	if state2 != CanaryStateStabilized {
		t.Errorf("expected CanaryStateStabilized, got %v", state2)
	}
	if epoch2 != 3 {
		t.Errorf("expected epoch 3, got %d", epoch2)
	}
}

// TestEventStream_Subscribe_And_Latest verifies subscription registration,
// synchronous delivery to active listeners, unsubscribe mechanics, and
// latest event snapshot retrieval on both populated and empty streams.
func TestEventStream_Subscribe_And_Latest(t *testing.T) {
	stream := NewEventStream(16)

	// Invariant 1: Latest on an empty stream must return nil without panic.
	if latest := stream.Latest(); latest != nil {
		t.Fatalf("expected nil latest on empty stream, got %+v", latest)
	}

	var received []AuditEvent
	var mu sync.Mutex

	// Invariant 2: Subscribed listeners receive every newly appended event.
	unsub := stream.Subscribe(func(ev AuditEvent) {
		mu.Lock()
		defer mu.Unlock()
		received = append(received, ev)
	})

	first := stream.Append(AuditEvent{
		Event:     EventSystemConfigApplied,
		FromEpoch: 1,
		ToEpoch:   2,
		Detail:    "first update",
	})

	mu.Lock()
	if len(received) != 1 || received[0].Seq != first.Seq {
		t.Errorf("expected subscriber to receive first event, got %v", received)
	}
	mu.Unlock()

	// Invariant 3: Latest returns the most recently appended audit event.
	latest := stream.Latest()
	if latest == nil || latest.Seq != first.Seq || latest.Detail != "first update" {
		t.Errorf("Latest mismatch: got %+v, expected seq %d", latest, first.Seq)
	}

	// Invariant 4: Calling unsubscribe removes listener so subsequent events are dropped.
	unsub()

	_ = stream.Append(AuditEvent{
		Event:     EventSystemConfigCommittedLKG,
		FromEpoch: 2,
		ToEpoch:   2,
		Detail:    "second update after unsubscribe",
	})

	mu.Lock()
	if len(received) != 1 {
		t.Errorf("unsubscribed listener should not receive second event, got %d events", len(received))
	}
	mu.Unlock()

	// Latest still points to the newest appended event.
	latest2 := stream.Latest()
	if latest2 == nil || latest2.Detail != "second update after unsubscribe" {
		t.Errorf("Latest mismatch after second append: %+v", latest2)
	}

	// Invariant 5: Ring buffer wrapping when capacity is exceeded.
	smallStream := NewEventStream(2)
	e1 := smallStream.Append(AuditEvent{Detail: "item1"})
	e2 := smallStream.Append(AuditEvent{Detail: "item2"})
	e3 := smallStream.Append(AuditEvent{Detail: "item3"})

	snap := smallStream.Snapshot()
	if len(snap) != 2 {
		t.Fatalf("expected bounded snapshot length 2, got %d", len(snap))
	}
	if snap[0].Seq != e2.Seq || snap[1].Seq != e3.Seq {
		t.Errorf("expected items 2 and 3 in ring buffer, got seqs %d and %d (e1=%d)", snap[0].Seq, snap[1].Seq, e1.Seq)
	}

	// Invariant 6: Default capacity when <= 0 is passed.
	defaultCapStream := NewEventStream(0)
	if defaultCapStream.capacity != 1024 {
		t.Errorf("expected default capacity 1024, got %d", defaultCapStream.capacity)
	}
}

// TestServingConfig_Clone_And_JSON verifies ServingConfig deep-copy isolation,
// JSON marshaling fidelity, and comprehensive patch application for all fields.
func TestServingConfig_Clone_And_JSON(t *testing.T) {
	orig := DefaultConfig()

	// Invariant 1: Clone produces an identical value copy.
	cloned := orig.Clone()
	if cloned != orig {
		t.Errorf("cloned config does not equal original: %+v vs %+v", cloned, orig)
	}

	// Invariant 2: Mutations to the clone do not corrupt the original.
	cloned.MaxBatchTokens = 99999
	if orig.MaxBatchTokens == 99999 {
		t.Errorf("mutating clone modified original: orig.MaxBatchTokens = %d", orig.MaxBatchTokens)
	}

	// Invariant 3: MarshalJSON produces valid JSON bytes.
	bytes, err := orig.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON failed: %v", err)
	}
	var unmarshaled ServingConfig
	if err := json.Unmarshal(bytes, &unmarshaled); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}
	if unmarshaled.MaxBatchTokens != orig.MaxBatchTokens {
		t.Errorf("roundtrip JSON mismatch: got %d, expected %d", unmarshaled.MaxBatchTokens, orig.MaxBatchTokens)
	}

	// Invariant 4: Comprehensive Apply covering every field in ConfigPatch.
	deadline := uint32(15000)
	streamTimeout := uint32(20000)
	maxWaiting := uint32(512)
	compactBudget := 16000
	compactHead := 0
	logLevel := "WARN "
	draftDepth := uint32(4)
	threshold := 0.75
	maxBatch := uint32(32768)
	maxModel := uint32(16384)
	maxSeqs := uint32(128)
	priority := " DEADLINE_FIRST "
	preempt := " SWAP "
	kvBlocks := uint32(20000)
	blockSize := uint32(4096)
	draftLimit := uint32(16)
	vram := uint64(32 * 1024 * 1024 * 1024)
	weights := uint64(10 * 1024 * 1024 * 1024)
	headroom := uint64(4 * 1024 * 1024 * 1024)
	sla := 180.0

	fullPatch := ConfigPatch{
		CompletionDeadlineMs:           &deadline,
		StreamProgressTimeoutMs:        &streamTimeout,
		MaxWaitingSeqs:                 &maxWaiting,
		CompactHistoryBudget:           &compactBudget,
		CompactAnchorHead:              &compactHead,
		LogLevel:                       &logLevel,
		SpeculativeDraftDepth:          &draftDepth,
		SpeculativeAcceptanceThreshold: &threshold,
		MaxBatchTokens:                 &maxBatch,
		MaxModelLen:                    &maxModel,
		MaxNumSeqs:                     &maxSeqs,
		PriorityStrategy:               &priority,
		PreemptionStrategy:             &preempt,
		TargetKVBlocks:                 &kvBlocks,
		BlockSizeBytes:                 &blockSize,
		MaxPreallocatedDraftLimit:      &draftLimit,
		AvailableVRAMBytes:             &vram,
		ModelWeightsBytes:              &weights,
		ActivationHeadroomBytes:        &headroom,
		DeclaredLatencySLAMS:           &sla,
	}

	patched := orig.Apply(fullPatch)
	if patched.CompletionDeadlineMs != deadline ||
		patched.StreamProgressTimeoutMs != streamTimeout ||
		patched.MaxWaitingSeqs != maxWaiting ||
		patched.CompactHistoryBudget != compactBudget ||
		patched.CompactAnchorHead != compactHead ||
		patched.LogLevel != "warn" ||
		patched.SpeculativeDraftDepth != draftDepth ||
		patched.SpeculativeAcceptanceThreshold != threshold ||
		patched.MaxBatchTokens != maxBatch ||
		patched.MaxModelLen != maxModel ||
		patched.MaxNumSeqs != maxSeqs ||
		patched.PriorityStrategy != "deadline_first" ||
		patched.PreemptionStrategy != "swap" ||
		patched.TargetKVBlocks != kvBlocks ||
		patched.BlockSizeBytes != blockSize ||
		patched.MaxPreallocatedDraftLimit != draftLimit ||
		patched.AvailableVRAMBytes != vram ||
		patched.ModelWeightsBytes != weights ||
		patched.ActivationHeadroomBytes != headroom ||
		patched.DeclaredLatencySLAMS != sla {
		t.Errorf("comprehensive apply mismatch on patched config: %+v", patched)
	}
}

// TestValidationError_And_ValidationErrors_Error verifies formatting invariants
// for individual ValidationError items and aggregated ValidationErrors lists.
func TestValidationError_And_ValidationErrors_Error(t *testing.T) {
	// Invariant 1: ValidationError with Field populated formatted as "[CODE] field: message".
	errWithField := ValidationError{
		Code:    "ERR_TEST_CODE",
		Field:   "test_field",
		Message: "something is wrong",
	}
	expectedWithField := "[ERR_TEST_CODE] test_field: something is wrong"
	if errWithField.Error() != expectedWithField {
		t.Errorf("expected %q, got %q", expectedWithField, errWithField.Error())
	}

	// Invariant 2: ValidationError without Field formatted as "[CODE] message".
	errWithoutField := ValidationError{
		Code:    "ERR_NO_FIELD",
		Message: "general error",
	}
	expectedWithoutField := "[ERR_NO_FIELD] general error"
	if errWithoutField.Error() != expectedWithoutField {
		t.Errorf("expected %q, got %q", expectedWithoutField, errWithoutField.Error())
	}

	// Invariant 3: ValidationErrors on empty slice returns empty string.
	var emptyErrs ValidationErrors
	if emptyErrs.Error() != "" {
		t.Errorf("expected empty string for empty ValidationErrors, got %q", emptyErrs.Error())
	}
	if emptyErrs.HasErrors() {
		t.Errorf("expected HasErrors()=false for empty ValidationErrors")
	}

	// Invariant 4: ValidationErrors aggregates multiple errors joined with semicolons.
	multiErrs := ValidationErrors{errWithField, errWithoutField}
	if !multiErrs.HasErrors() {
		t.Errorf("expected HasErrors()=true for populated ValidationErrors")
	}
	expectedAgg := expectedWithField + "; " + expectedWithoutField
	if multiErrs.Error() != expectedAgg {
		t.Errorf("expected aggregated %q, got %q", expectedAgg, multiErrs.Error())
	}
}

// TestDiff_AllFields verifies that ComputeDiff captures modifications across
// every single configurable field in ServingConfig.
func TestDiff_AllFields(t *testing.T) {
	c1 := DefaultConfig()
	c2 := c1
	c2.CompletionDeadlineMs += 1
	c2.StreamProgressTimeoutMs = 10000
	c2.MaxWaitingSeqs += 1
	c2.CompactHistoryBudget += 1
	c2.CompactAnchorHead = 0
	c2.LogLevel = "debug"
	c2.SpeculativeDraftDepth += 1
	c2.SpeculativeAcceptanceThreshold += 0.05
	c2.MaxBatchTokens += 1
	c2.MaxModelLen += 1
	c2.MaxNumSeqs += 1
	c2.PriorityStrategy = "cost_fairness"
	c2.PreemptionStrategy = "swap"
	c2.TargetKVBlocks += 1
	c2.BlockSizeBytes += 1
	c2.MaxPreallocatedDraftLimit += 1
	c2.AvailableVRAMBytes += 1
	c2.ModelWeightsBytes += 1
	c2.ActivationHeadroomBytes += 1
	c2.DeclaredLatencySLAMS += 1.0

	diff := ComputeDiff(c1, c2)
	expectedFields := []string{
		"completion_deadline_ms",
		"stream_progress_timeout_ms",
		"max_waiting_seqs",
		"compact_history_budget",
		"compact_anchor_head",
		"log_level",
		"speculative_draft_depth",
		"speculative_acceptance_threshold",
		"max_batch_tokens",
		"max_model_len",
		"max_num_seqs",
		"priority_strategy",
		"preemption_strategy",
		"target_kv_blocks",
		"block_size_bytes",
		"max_preallocated_draft_slots",
		"available_vram_bytes",
		"model_weights_bytes",
		"activation_headroom_bytes",
		"declared_latency_sla_ms",
	}

	for _, field := range expectedFields {
		if _, ok := diff[field]; !ok {
			t.Errorf("missing diff field: %s", field)
		}
	}
}

// TestComputeImpact_Contractions_And_Drains verifies risk assessment and drain
// time estimations for each contraction vector and algorithmic update.
func TestComputeImpact_Contractions_And_Drains(t *testing.T) {
	base := DefaultConfig()

	// Invariant 1: Wait queue contraction triggers drain.
	qContract := base
	qContract.MaxWaitingSeqs = base.MaxWaitingSeqs - 100
	impactQ := ComputeImpact(base, qContract, 10)
	if !impactQ.DrainRequired || impactQ.RiskLevel != RiskHighDrainRequired {
		t.Errorf("expected HighDrainRequired on queue contraction, got %+v", impactQ)
	}

	// Invariant 2: Concurrency contraction triggers active sequence drain.
	cContract := base
	cContract.MaxNumSeqs = base.MaxNumSeqs - 10
	impactC := ComputeImpact(base, cContract, 10)
	if !impactC.DrainRequired || impactC.RiskLevel != RiskHighDrainRequired {
		t.Errorf("expected HighDrainRequired on concurrency contraction, got %+v", impactC)
	}

	// Invariant 3: Batch token contraction triggers in-flight iteration drain.
	bContract := base
	bContract.MaxBatchTokens = base.MaxBatchTokens - 1000
	impactB := ComputeImpact(base, bContract, 10)
	if !impactB.DrainRequired || impactB.RiskLevel != RiskHighDrainRequired {
		t.Errorf("expected HighDrainRequired on batch token contraction, got %+v", impactB)
	}

	// Invariant 4: Drain time capped by CompletionDeadlineMs.
	heavyActive := base
	heavyActive.TargetKVBlocks = base.TargetKVBlocks / 2
	heavyActive.CompletionDeadlineMs = 200                // 200 ms ceiling
	impactHeavy := ComputeImpact(base, heavyActive, 1000) // 1000 * 25 = 25000ms > 200ms
	if impactHeavy.EstimatedDrainMS != 200 {
		t.Errorf("expected drain time capped at 200ms, got %d", impactHeavy.EstimatedDrainMS)
	}

	// Invariant 5: Drain time floor of 50ms when activeSeqs is small.
	lightActive := base
	lightActive.TargetKVBlocks = base.TargetKVBlocks / 2
	impactLight := ComputeImpact(base, lightActive, 1) // 1 * 25 = 25ms < 50ms
	if impactLight.EstimatedDrainMS != 50 {
		t.Errorf("expected drain time floor of 50ms, got %d", impactLight.EstimatedDrainMS)
	}

	// Invariant 6: Algorithmic policy updates trigger Medium risk without drain.
	algoCfg := base
	algoCfg.PreemptionStrategy = "swap"
	algoCfg.PriorityStrategy = "deadline_first"
	algoCfg.SpeculativeDraftDepth = base.SpeculativeDraftDepth + 1
	algoCfg.MaxBatchTokens = base.MaxBatchTokens + 1024
	impactAlgo := ComputeImpact(base, algoCfg, 0)
	if impactAlgo.RiskLevel != RiskMedium || impactAlgo.DrainRequired {
		t.Errorf("expected RiskMedium without drain for algorithmic changes, got %+v", impactAlgo)
	}

	// Invariant 7: Pure algorithmic change with zero VRAM delta verifies detail strings.
	algoOnly := base
	algoOnly.PreemptionStrategy = "swap"
	impactAlgoOnly := ComputeImpact(base, algoOnly, 0)
	if impactAlgoOnly.RiskLevel != RiskMedium || impactAlgoOnly.VRAMDeltaBytes != 0 {
		t.Errorf("expected RiskMedium with 0 VRAM delta for preemption change, got %+v", impactAlgoOnly)
	}
}

// TestManager_NilReceivers_And_EdgeCases verifies defensive nil receiver handling
// and boundary conditions across Manager methods.
func TestManager_NilReceivers_And_EdgeCases(t *testing.T) {
	var nilMgr *Manager

	// Invariant 1: Active() on nil Manager returns safe default config.
	act := nilMgr.Active()
	if act.Epoch != 0 || act.Config.MaxBatchTokens != DefaultConfig().MaxBatchTokens {
		t.Errorf("unexpected Active on nil Manager: %+v", act)
	}

	// Invariant 2: LKG() on nil Manager returns safe default config.
	lkg := nilMgr.LKG()
	if lkg.Epoch != 0 || lkg.Config.MaxBatchTokens != DefaultConfig().MaxBatchTokens {
		t.Errorf("unexpected LKG on nil Manager: %+v", lkg)
	}

	// Invariant 3: EventStream() and Watchdog() on nil Manager return nil.
	if nilMgr.EventStream() != nil {
		t.Errorf("expected nil EventStream on nil Manager")
	}
	if nilMgr.Watchdog() != nil {
		t.Errorf("expected nil Watchdog on nil Manager")
	}

	// Invariant 4: Apply() on nil Manager returns typed error.
	if _, err := nilMgr.Apply(ConfigPatch{}, false); err == nil {
		t.Errorf("expected error applying on nil Manager")
	}

	// Invariant 5: Rollback() on nil Manager returns typed error.
	if _, err := nilMgr.Rollback("manual", "test"); err == nil {
		t.Errorf("expected error rolling back on nil Manager")
	}

	// Invariant 6: IngestTelemetry() on nil Manager returns false without panic.
	if triggered, _, _ := nilMgr.IngestTelemetry(TelemetrySample{}); triggered {
		t.Errorf("expected false telemetry on nil Manager")
	}

	// Invariant 7: activeQueueDepth returns 0 when queueDepthFn is nil.
	validMgr, err := NewManager(DefaultConfig(), DefaultWatchdogConfig(), nil)
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}
	if depth := validMgr.activeQueueDepth(); depth != 0 {
		t.Errorf("expected 0 queue depth when queueDepthFn is nil, got %d", depth)
	}

	// Invariant 8: NewManager fails when initial configuration violates invariants.
	badCfg := DefaultConfig()
	badCfg.MaxBatchTokens = 100 // Violates max_batch_tokens >= max_model_len (8192)
	if _, err := NewManager(badCfg, DefaultWatchdogConfig(), nil); err == nil {
		t.Fatalf("expected NewManager to reject invalid initial configuration")
	}

	// Invariant 9: Apply fails when patch produces an invalid configuration.
	badBatch := uint32(50)
	if _, err := validMgr.Apply(ConfigPatch{MaxBatchTokens: &badBatch}, false); err == nil {
		t.Fatalf("expected Apply to reject invalid patch")
	}

	// Invariant 10: IngestTelemetry with nil watchdog.
	mgrNoWatchdog := &Manager{}
	if triggered, _, _ := mgrNoWatchdog.IngestTelemetry(TelemetrySample{}); triggered {
		t.Errorf("expected false telemetry on Manager with nil watchdog")
	}

	// Invariant 11: Uninitialized Manager with nil atomic pointers falls back to DefaultConfig.
	if uninitAct := mgrNoWatchdog.Active(); uninitAct.Epoch != 0 || uninitAct.Config.MaxBatchTokens != DefaultConfig().MaxBatchTokens {
		t.Errorf("unexpected Active on uninitialized Manager: %+v", uninitAct)
	}
	if uninitLKG := mgrNoWatchdog.LKG(); uninitLKG.Epoch != 0 || uninitLKG.Config.MaxBatchTokens != DefaultConfig().MaxBatchTokens {
		t.Errorf("unexpected LKG on uninitialized Manager: %+v", uninitLKG)
	}
}

// TestWatchdog_EdgeCases verifies Watchdog boundary handling for zero-config defaults,
// telemetry calculation with count fields, and stabilization status queries.
func TestWatchdog_EdgeCases(t *testing.T) {
	// Invariant 1: Zero-value WatchdogConfig receives sensible production defaults.
	wd := NewWatchdog(WatchdogConfig{}, nil, nil)
	if wd.cfg.StabilizationWindow != 30*time.Second ||
		wd.cfg.Max5xxErrorRate != 0.001 ||
		wd.cfg.MaxAcceptanceDropRatio != 0.50 ||
		wd.cfg.DefaultDeclaredLatencyMS != 250.0 {
		t.Errorf("unexpected zero-config defaults: %+v", wd.cfg)
	}

	// Invariant 2: StartEvaluation with zero SLA and zero initial acceptance
	// uses configured defaults.
	wd.StartEvaluation(1, 0, 0)
	if wd.slaMS != 250.0 {
		t.Errorf("expected default SLA 250.0, got %f", wd.slaMS)
	}

	// Invariant 3: IngestTelemetry computes Error5xxRate from request counts.
	triggered, trigger, detail := wd.IngestTelemetry(TelemetrySample{
		TotalRequests: 1000,
		Errors5xx:     10, // 1% error rate > 0.1% ceiling
		Error5xxRate:  0,  // zero triggers count-based calculation
	})
	if !triggered || trigger != Trigger5xxErrorRateExceeded {
		t.Fatalf("expected 5xx trigger via counts, got %v (%s: %s)", triggered, trigger, detail)
	}

	// Invariant 4: IngestTelemetry ignored when not in CanaryStateEvaluating.
	wdIdle := NewWatchdog(WatchdogConfig{}, nil, nil)
	if trig, _, _ := wdIdle.IngestTelemetry(TelemetrySample{Error5xxRate: 0.99}); trig {
		t.Errorf("telemetry should be ignored when watchdog is in idle state")
	}

	// Invariant 5: CheckStabilization returns true when already stabilized.
	wdStabilized := NewWatchdog(WatchdogConfig{StabilizationWindow: time.Millisecond}, nil, nil)
	wdStabilized.StartEvaluation(1, 100.0, 0.8)
	time.Sleep(5 * time.Millisecond)
	if !wdStabilized.CheckStabilization(time.Now().UTC()) {
		t.Errorf("expected CheckStabilization=true")
	}
	// Calling again when already in CanaryStateStabilized returns true.
	if !wdStabilized.CheckStabilization(time.Now().UTC()) {
		t.Errorf("expected CheckStabilization=true when already stabilized")
	}

	// Invariant 6: CheckStabilization returns false when stabilization window has not elapsed.
	wdLong := NewWatchdog(WatchdogConfig{StabilizationWindow: 1 * time.Hour}, nil, nil)
	wdLong.StartEvaluation(5, 200.0, 0.8)
	if wdLong.CheckStabilization(time.Now().UTC()) {
		t.Errorf("expected CheckStabilization=false before window elapsed")
	}

	// Invariant 7: IngestTelemetry automatically promotes when window has elapsed without error.
	var promotedEpoch uint64
	wdAutoPromote := NewWatchdog(
		WatchdogConfig{StabilizationWindow: 5 * time.Millisecond},
		nil,
		func(epoch uint64) error {
			promotedEpoch = epoch
			return nil
		},
	)
	wdAutoPromote.StartEvaluation(7, 250.0, 0.85)
	time.Sleep(10 * time.Millisecond)
	trig, _, _ := wdAutoPromote.IngestTelemetry(TelemetrySample{
		SpeculativeAcceptanceRate: 0.85,
		TTFTp99MS:                 100.0,
		Error5xxRate:              0.0001,
	})
	if trig {
		t.Errorf("unexpected rollback trigger during auto-promotion sample")
	}
	state, epoch, _, _ := wdAutoPromote.Status()
	if state != CanaryStateStabilized || epoch != 7 || promotedEpoch != 7 {
		t.Errorf("expected CanaryStateStabilized with promotedEpoch=7, got state=%s epoch=%d promoted=%d", state, epoch, promotedEpoch)
	}
}

// TestValidator_AdditionalBounds verifies edge cases in syntactic checks
// and relational VRAM fixed-overhead validations.
func TestValidator_AdditionalBounds(t *testing.T) {
	// Invariant 1: Fixed overhead (weights + headroom) exceeding total available VRAM.
	cfg := DefaultConfig()
	cfg.AvailableVRAMBytes = 16 * 1024 * 1024 * 1024
	cfg.ModelWeightsBytes = 14 * 1024 * 1024 * 1024
	cfg.ActivationHeadroomBytes = 4 * 1024 * 1024 * 1024 // 14 + 4 = 18GB > 16GB VRAM
	errs := Validate(cfg)
	if !errs.HasErrors() {
		t.Fatalf("expected VRAM overcommit error for fixed overhead exceeding available VRAM")
	}
	var foundOvercommit bool
	for _, e := range errs {
		if e.Code == ErrRelationalInvariantVRAMOvercommit && strings.Contains(e.Message, "fixed overhead") {
			foundOvercommit = true
			break
		}
	}
	if !foundOvercommit {
		t.Errorf("expected fixed overhead VRAM overcommit error in %v", errs)
	}

	// Invariant 2: StreamProgressTimeoutMs exceeding 600000 ceiling.
	cfgTimeout := DefaultConfig()
	cfgTimeout.StreamProgressTimeoutMs = 700000
	errsTimeout := Validate(cfgTimeout)
	if !errsTimeout.HasErrors() {
		t.Fatalf("expected error for StreamProgressTimeoutMs > 600000")
	}

	// Invariant 3: CompactHistoryBudget out of bounds.
	cfgBudgetNeg := DefaultConfig()
	cfgBudgetNeg.CompactHistoryBudget = -1
	if errs := Validate(cfgBudgetNeg); !errs.HasErrors() {
		t.Fatalf("expected error for CompactHistoryBudget < 0")
	}

	cfgBudgetHigh := DefaultConfig()
	cfgBudgetHigh.CompactHistoryBudget = 20000000
	if errs := Validate(cfgBudgetHigh); !errs.HasErrors() {
		t.Fatalf("expected error for CompactHistoryBudget > 10000000")
	}
}
