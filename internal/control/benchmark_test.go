package control

import (
	"sync"
	"testing"
	"time"
)

var (
	sinkVersionedConfig  VersionedConfig
	sinkValidationErrors ValidationErrors
	sinkDiff             map[string]FieldDiff
	sinkImpact           ResourceImpact
	sinkConfig           ServingConfig
	sinkApplyResult      *ApplyResult
	sinkBool             bool
	sinkString           string
	sinkAuditEvent       AuditEvent
	sinkAuditEvents      []AuditEvent
	sinkBytes            []byte
)

// BenchmarkManager_Active measures atomic lock-free retrieval of the active configuration
// on the request serving hot-path.
func BenchmarkManager_Active(b *testing.B) {
	mgr, err := NewManager(DefaultConfig(), DefaultWatchdogConfig(), nil)
	if err != nil {
		b.Fatalf("NewManager failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkVersionedConfig = mgr.Active()
	}
}

// BenchmarkManager_Active_Parallel measures concurrent atomic read throughput across goroutines.
func BenchmarkManager_Active_Parallel(b *testing.B) {
	mgr, err := NewManager(DefaultConfig(), DefaultWatchdogConfig(), nil)
	if err != nil {
		b.Fatalf("NewManager failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	var mu sync.Mutex
	b.RunParallel(func(pb *testing.PB) {
		var local VersionedConfig
		for pb.Next() {
			local = mgr.Active()
		}
		mu.Lock()
		sinkVersionedConfig = local
		mu.Unlock()
	})
}

// BenchmarkServingConfig_Apply measures patch application overhead across multi-tier parameters.
func BenchmarkServingConfig_Apply(b *testing.B) {
	cfg := DefaultConfig()
	logLvl := "debug"
	deadline := uint32(15000)
	depth := uint32(4)
	tokens := uint32(32768)
	patch := ConfigPatch{
		LogLevel:              &logLvl,
		CompletionDeadlineMs:  &deadline,
		SpeculativeDraftDepth: &depth,
		MaxBatchTokens:        &tokens,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkConfig = cfg.Apply(patch)
	}
}

// BenchmarkValidate_Default measures shift-left validation on baseline production config.
func BenchmarkValidate_Default(b *testing.B) {
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkValidationErrors = Validate(cfg)
	}
}

// BenchmarkValidate_Relational measures validation verifying syntactic bounds and relational invariants.
func BenchmarkValidate_Relational(b *testing.B) {
	cfg := DefaultConfig()
	cfg.MaxBatchTokens = 32768
	cfg.MaxModelLen = 16384
	cfg.SpeculativeDraftDepth = 6
	cfg.MaxPreallocatedDraftLimit = 8
	cfg.AvailableVRAMBytes = 48 * 1024 * 1024 * 1024
	cfg.ModelWeightsBytes = 28 * 1024 * 1024 * 1024
	cfg.ActivationHeadroomBytes = 4 * 1024 * 1024 * 1024
	cfg.TargetKVBlocks = 65536
	cfg.BlockSizeBytes = 2048

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkValidationErrors = Validate(cfg)
	}
}

// BenchmarkComputeDiff measures structural diff computation comparing configurations across 20+ fields.
func BenchmarkComputeDiff(b *testing.B) {
	current := DefaultConfig()
	proposed := current
	logLvl := "debug"
	deadline := uint32(15000)
	depth := uint32(5)
	tokens := uint32(32768)
	proposed = proposed.Apply(ConfigPatch{
		LogLevel:              &logLvl,
		CompletionDeadlineMs:  &deadline,
		SpeculativeDraftDepth: &depth,
		MaxBatchTokens:        &tokens,
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDiff = ComputeDiff(current, proposed)
	}
}

// BenchmarkComputeImpact_Scalar measures resource impact calculation for scalar updates (Low risk).
func BenchmarkComputeImpact_Scalar(b *testing.B) {
	current := DefaultConfig()
	proposed := current
	logLvl := "debug"
	deadline := uint32(15000)
	proposed = proposed.Apply(ConfigPatch{
		LogLevel:             &logLvl,
		CompletionDeadlineMs: &deadline,
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkImpact = ComputeImpact(current, proposed, 10)
	}
}

// BenchmarkComputeImpact_Contraction measures impact calculation for contraction (HighDrainRequired risk).
func BenchmarkComputeImpact_Contraction(b *testing.B) {
	current := DefaultConfig()
	proposed := current
	proposed.TargetKVBlocks = 16384 // contraction
	proposed.MaxWaitingSeqs = 512   // queue contraction

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkImpact = ComputeImpact(current, proposed, 32)
	}
}

// BenchmarkManager_DryRun measures end-to-end dry-run candidate evaluation without state mutation.
func BenchmarkManager_DryRun(b *testing.B) {
	mgr, err := NewManager(DefaultConfig(), DefaultWatchdogConfig(), func() int { return 16 })
	if err != nil {
		b.Fatalf("NewManager failed: %v", err)
	}

	depth := uint32(5)
	tokens := uint32(32768)
	patch := ConfigPatch{
		SpeculativeDraftDepth: &depth,
		MaxBatchTokens:        &tokens,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := mgr.DryRun(patch)
		if err != nil {
			b.Fatalf("DryRun failed: %v", err)
		}
		sinkApplyResult = res
	}
}

// BenchmarkManager_Apply measures dynamic configuration hot-swapping under lock with epoch advance.
func BenchmarkManager_Apply(b *testing.B) {
	mgr, err := NewManager(DefaultConfig(), DefaultWatchdogConfig(), nil)
	if err != nil {
		b.Fatalf("NewManager failed: %v", err)
	}

	depth4 := uint32(4)
	depth5 := uint32(5)
	patch4 := ConfigPatch{SpeculativeDraftDepth: &depth4}
	patch5 := ConfigPatch{SpeculativeDraftDepth: &depth5}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := patch4
		if i%2 == 1 {
			p = patch5
		}
		res, err := mgr.Apply(p, false)
		if err != nil {
			b.Fatalf("Apply failed: %v", err)
		}
		sinkApplyResult = res
	}
}

// BenchmarkManager_Rollback measures restoring the Last-Known-Good configuration snapshot.
func BenchmarkManager_Rollback(b *testing.B) {
	mgr, err := NewManager(DefaultConfig(), DefaultWatchdogConfig(), nil)
	if err != nil {
		b.Fatalf("NewManager failed: %v", err)
	}

	depth := uint32(5)
	_, _ = mgr.Apply(ConfigPatch{SpeculativeDraftDepth: &depth}, false)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vc, err := mgr.Rollback("benchmark", "benchmark-rollback")
		if err != nil {
			b.Fatalf("Rollback failed: %v", err)
		}
		sinkVersionedConfig = *vc
	}
}

// BenchmarkWatchdog_IngestTelemetry_Healthy measures telemetry ingestion during normal serving.
func BenchmarkWatchdog_IngestTelemetry_Healthy(b *testing.B) {
	wcfg := DefaultWatchdogConfig()
	wcfg.StabilizationWindow = 24 * time.Hour // ensure evaluation state persists
	wd := NewWatchdog(wcfg, nil, nil)
	wd.StartEvaluation(2, 250.0, 0.80)

	sample := TelemetrySample{
		SpeculativeAcceptanceRate: 0.80,
		TTFTp99MS:                 120.0,
		Error5xxRate:              0.0001,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		triggered, trig, detail := wd.IngestTelemetry(sample)
		sinkBool = triggered
		sinkString = trig + detail
	}
}

// BenchmarkWatchdog_IngestTelemetry_AnomalyRollback measures telemetry evaluation detecting an SLA breach.
func BenchmarkWatchdog_IngestTelemetry_AnomalyRollback(b *testing.B) {
	wcfg := DefaultWatchdogConfig()
	wcfg.StabilizationWindow = 24 * time.Hour
	wd := NewWatchdog(wcfg, nil, nil)

	// Anomaly sample: 5xx error rate 0.5% > 0.1% threshold
	sample := TelemetrySample{
		SpeculativeAcceptanceRate: 0.80,
		TTFTp99MS:                 120.0,
		Error5xxRate:              0.005,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wd.StartEvaluation(2, 250.0, 0.80)
		triggered, trig, detail := wd.IngestTelemetry(sample)
		if !triggered {
			b.Fatal("expected anomaly trigger")
		}
		sinkBool = triggered
		sinkString = trig + detail
	}
}

// BenchmarkEventStream_Append measures appending audit events into the bounded in-memory journal.
func BenchmarkEventStream_Append(b *testing.B) {
	stream := NewEventStream(1024)
	ev := AuditEvent{
		Event:     EventSystemConfigApplied,
		FromEpoch: 1,
		ToEpoch:   2,
		Detail:    "benchmark audit event",
		Config:    DefaultConfig(),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkAuditEvent = stream.Append(ev)
	}
}

// BenchmarkEventStream_Snapshot measures copying the chronological audit event stream.
func BenchmarkEventStream_Snapshot(b *testing.B) {
	stream := NewEventStream(1024)
	ev := AuditEvent{
		Event:     EventSystemConfigApplied,
		FromEpoch: 1,
		ToEpoch:   2,
		Detail:    "benchmark audit event",
		Config:    DefaultConfig(),
	}
	for i := 0; i < 100; i++ {
		stream.Append(ev)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkAuditEvents = stream.Snapshot()
	}
}

// BenchmarkServingConfig_MarshalJSON measures JSON serialization performance of ServingConfig.
func BenchmarkServingConfig_MarshalJSON(b *testing.B) {
	cfg := DefaultConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data, err := cfg.MarshalJSON()
		if err != nil {
			b.Fatalf("MarshalJSON failed: %v", err)
		}
		sinkBytes = data
	}
}
