package agent

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func requireV41PhaseSnapshot(t *testing.T, got, want V41PhaseSnapshot) {
	t.Helper()
	gv, wv := reflect.ValueOf(got), reflect.ValueOf(want)
	typ := gv.Type()
	for i := 0; i < gv.NumField(); i++ {
		if !reflect.DeepEqual(gv.Field(i).Interface(), wv.Field(i).Interface()) {
			t.Fatalf("V41 phase field %s = %v, want %v", typ.Field(i).Name, gv.Field(i).Interface(), wv.Field(i).Interface())
		}
	}
}

func v41AttributionFixture(base int) model.V41ExpertFaultAttributionPhase {
	return model.V41ExpertFaultAttributionPhase{
		IncrementalDeviceDispatchNanos: int64(base + 32), IncrementalDeviceGateUpCalls: base + 33, IncrementalDeviceDownCalls: base + 34,
		Tokens: base + 1, AttentionContractionCalls: base + 2, AttentionContractionNanos: int64(base + 3),
		Faults: base + 4, FaultedBytes: int64(base + 5), FaultDoorNanos: int64(base + 6),
		DequantBytes: int64(base + 7), DequantNanos: int64(base + 8),
		Contractions: base + 9, ContractionNanos: int64(base + 10),
		MHCProjectionDeviceCalls: base + 27, MHCProjectionHostCalls: base + 28,
		MHCProjectionNanos: int64(base + 29), MHCProjectionActivationUploadBytes: int64(base + 30), MHCProjectionReadbackBytes: int64(base + 31),
		DenseProjectionDeviceCalls: base + 11, DenseProjectionHostCalls: base + 12,
		DenseProjectionNanos: int64(base + 13), DenseProjectionActivationUploadBytes: int64(base + 14), DenseProjectionReadbackBytes: int64(base + 15),
		GroupedOutputDeviceCalls: base + 16, GroupedOutputHostCalls: base + 17,
		GroupedOutputNanos: int64(base + 18), GroupedOutputActivationUploadBytes: int64(base + 19), GroupedOutputReadbackBytes: int64(base + 20),
		ExpertActivationDeviceCalls: base + 21, ExpertActivationHostCalls: base + 22,
		ExpertActivationNanos: int64(base + 23), ExpertActivationReadbackBytes: int64(base + 24),
		IncrementalEngramInjections: base + 25, IncrementalEngramNanos: int64(base + 26),
	}
}

func v41SnapshotFixture(base int) V41PhaseSnapshot {
	return V41PhaseSnapshot{
		Tokens: base + 1, AttentionContractionCalls: base + 2, AttentionContractionNanos: int64(base + 3),
		Faults: base + 4, FaultedBytes: int64(base + 5), FaultNanos: int64(base + 6),
		DequantBytes: int64(base + 7), DequantNanos: int64(base + 8),
		Contractions: base + 9, ContractionNanos: int64(base + 10),
		MHCProjectionDeviceCalls: base + 27, MHCProjectionHostCalls: base + 28,
		MHCProjectionNanos: int64(base + 29), MHCProjectionActivationUploadBytes: int64(base + 30), MHCProjectionReadbackBytes: int64(base + 31),
		DenseProjectionDeviceCalls: base + 11, DenseProjectionHostCalls: base + 12,
		DenseProjectionNanos: int64(base + 13), DenseProjectionActivationUploadBytes: int64(base + 14), DenseProjectionReadbackBytes: int64(base + 15),
		GroupedOutputDeviceCalls: base + 16, GroupedOutputHostCalls: base + 17,
		GroupedOutputNanos: int64(base + 18), GroupedOutputActivationUploadBytes: int64(base + 19), GroupedOutputReadbackBytes: int64(base + 20),
		ExpertActivationDeviceCalls: base + 21, ExpertActivationHostCalls: base + 22,
		ExpertActivationNanos: int64(base + 23), ExpertActivationReadbackBytes: int64(base + 24),
		IncrementalEngramInjections: base + 25, IncrementalEngramNanos: int64(base + 26),
	}
}

// TestV41PhaseAttributionLatestRinglessSnapshot proves the V4.1 source is a
// model-lifetime gauge: it survives without a ring, replaces rather than sums,
// preserves all 31 numeric fields per phase, and cannot be mutated by a reader.
// fak-test:runtime fast est=1ms lane=default
func TestV41PhaseAttributionLatestRinglessSnapshot(t *testing.T) {
	p := &InKernelPlanner{}
	p.foldV41PhasesLocked(model.V41ExpertFaultAttribution{
		Prefill: v41AttributionFixture(10), Decode: v41AttributionFixture(100),
	})

	first := p.MoEResidencyStats()
	if first.Requests != 0 || first.Tokens != 0 || first.Last.Ring.Enabled {
		t.Fatalf("ringless V4.1 snapshot changed ring framing: requests=%d tokens=%d last_ring=%v", first.Requests, first.Tokens, first.Last.Ring.Enabled)
	}
	if first.V41Phases == nil {
		t.Fatal("V41 phases absent, want model_lifetime snapshot")
	}
	if first.V41Phases.Scope != "model_lifetime" {
		t.Fatalf("V41 phase scope = %q, want model_lifetime", first.V41Phases.Scope)
	}
	requireV41PhaseSnapshot(t, first.V41Phases.Prefill, v41SnapshotFixture(10))
	requireV41PhaseSnapshot(t, first.V41Phases.Decode, v41SnapshotFixture(100))

	first.V41Phases.Prefill.Tokens = 9999
	if got := p.MoEResidencyStats().V41Phases.Prefill.Tokens; got != 11 {
		t.Fatalf("reader mutation changed stored prefill tokens to %d, want 11", got)
	}

	// A later cumulative source snapshot replaces the prior one. If the planner
	// summed snapshots, the small values below would be inflated by the first fold.
	p.foldV41PhasesLocked(model.V41ExpertFaultAttribution{
		Prefill: v41AttributionFixture(20), Decode: v41AttributionFixture(200),
	})
	latest := p.MoEResidencyStats()
	if latest.V41Phases == nil {
		t.Fatal("latest V41 source was dropped")
	}
	requireV41PhaseSnapshot(t, latest.V41Phases.Prefill, v41SnapshotFixture(20))
	requireV41PhaseSnapshot(t, latest.V41Phases.Decode, v41SnapshotFixture(200))

	// Checkpoint-only folding is independent and must retain the V4.1 snapshot.
	p.foldMoEResidency(checkpointResidencyReport(3, 2, 4096), 17)
	both := p.MoEResidencyStats()
	if both.Checkpoint == nil {
		t.Fatal("checkpoint snapshot absent after checkpoint fold")
	}
	if both.V41Phases == nil {
		t.Fatal("checkpoint fold dropped the retained V41 snapshot")
	}
	requireV41PhaseSnapshot(t, both.V41Phases.Prefill, v41SnapshotFixture(20))
	if both.Requests != 0 || both.Tokens != 0 {
		t.Fatalf("model-lifetime sources diluted ring framing: requests=%d tokens=%d", both.Requests, both.Tokens)
	}
}

// TestV41PhaseAttributionZeroSourceIsInert ensures a non-V4.1 model does not
// fabricate a debug-visible phase snapshot.
// fak-test:runtime fast est=1ms lane=default
func TestV41PhaseAttributionZeroSourceIsInert(t *testing.T) {
	p := &InKernelPlanner{}
	p.foldV41PhasesLocked(model.V41ExpertFaultAttribution{})
	if got := p.MoEResidencyStats().V41Phases; got != nil {
		t.Fatalf("zero V4.1 source produced scope %q, want nil", got.Scope)
	}
}

// TestV41PhaseCoverageDebugExport preserves timed zero versus unavailable at
// the existing reader/JSON boundary, without summing lifetime observations.
// fak-test:runtime fast est=1ms lane=default
func TestV41PhaseCoverageDebugExport(t *testing.T) {
	coverage := model.V41ExpertPhaseCoverage{
		Scope: "model_lifetime", RequestIsolation: "unsupported",
		RequestIsolationReason: "shared_model_phase", Dependency: "request_scoped_phase_ledger",
		RingPageInSemantics:  "h2d_upload_bytes_not_os_faults_disk_reads_or_host_f32_dequant",
		HostMaterialization:  model.V41PhaseFieldCoverage{Status: "not_applicable", Reason: "device_handled"},
		ContractionTime:      model.V41PhaseFieldCoverage{Status: "measured", Boundary: "host_contraction", MeasuredSamples: 1},
		DeviceDispatchTime:   model.V41PhaseFieldCoverage{Status: "unavailable", Reason: "no_completion_timer", Boundary: "submission", UnavailableSamples: 2},
		DeviceActivationTime: model.V41PhaseFieldCoverage{Status: "measured", Boundary: "completion", Overlap: "included_in_dispatch", MeasuredSamples: 3},
	}
	prefill := v41AttributionFixture(10)
	prefill.ContractionNanos = 0 // An observed timer sample may have zero resolution.
	prefill.Coverage = coverage
	decode := model.V41ExpertFaultAttributionPhase{Contractions: 2, Coverage: coverage}
	decode.Coverage.ContractionTime = model.V41PhaseFieldCoverage{
		Status: "unavailable", Reason: "device_contraction_not_timed", UnavailableSamples: 2,
	}
	p := &InKernelPlanner{}
	// Populate the actual fixed workload vocabulary at its maximum counter
	// width, so the existing export ceiling also covers the new default fields.
	maxRow := model.V41WorkloadRows{ObservedOperations: ^uint64(0), ExecutedRows: ^uint64(0), AppendedRows: ^uint64(0)}
	source := model.V41ExpertFaultAttribution{
		Prefill: prefill, Decode: decode,
		Workload: model.V41WorkloadLifetime{
			ColdFull: maxRow, SuffixSerial: maxRow, SuffixBatch: maxRow,
			SuffixRefold: maxRow, SerialDecode: maxRow, DecodeRefold: maxRow, CohortDecode: maxRow,
		},
		WorkloadCoverage: model.V41WorkloadCoverageDefault(),
	}
	p.foldV41PhasesLocked(source)
	p.foldV41PhasesLocked(source)
	got := p.MoEResidencyStats()
	if got.V41Phases.Workload != source.Workload || got.V41Phases.WorkloadCoverage != source.WorkloadCoverage {
		t.Fatal("default export lost workload rows or coverage")
	}
	wantPrefill := v41SnapshotFixture(10)
	wantPrefill.ContractionNanos = 0
	requireV41PhaseSnapshot(t, got.V41Phases.Prefill, wantPrefill)
	if got.V41Phases.ExpertCoverage.Prefill != (V41ExpertCoverageSnapshot{Coverage: coverage, IncrementalDeviceDispatchNanos: 42, IncrementalDeviceGateUpCalls: 43, IncrementalDeviceDownCalls: 44}) {
		t.Fatalf("prefill expert coverage lost timing or dispatch totals: %+v", got.V41Phases.ExpertCoverage.Prefill)
	}
	if got.V41Phases.ExpertCoverage.Decode.Coverage != decode.Coverage || got.V41Phases.Decode.ContractionNanos != 0 || got.V41Phases.Decode.Contractions != 2 {
		t.Fatalf("decode lost unavailable timing or replaced numeric counters: %+v", got.V41Phases.Decode)
	}
	encoded, err := json.Marshal(got.V41Phases)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded) > 16*1024 {
		t.Fatalf("fixed phase export exceeded 16KiB: %d bytes", len(encoded))
	}
	var roundtrip V41PhaseLifetimeSnapshot
	if err := json.Unmarshal(encoded, &roundtrip); err != nil {
		t.Fatal(err)
	}
	if roundtrip != *got.V41Phases {
		t.Fatalf("JSON lost phase counters or coverage: %s", encoded)
	}
	var wire struct {
		Prefill        map[string]json.RawMessage `json:"prefill"`
		Decode         map[string]json.RawMessage `json:"decode"`
		ExpertCoverage struct {
			Prefill map[string]json.RawMessage `json:"prefill"`
			Decode  map[string]json.RawMessage `json:"decode"`
		} `json:"expert_coverage"`
	}
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	for _, phase := range []map[string]json.RawMessage{wire.Prefill, wire.Decode} {
		if string(phase["contraction_nanos"]) != "0" || len(phase) != 31 {
			t.Fatalf("zero elapsed requires exported coverage: %s", encoded)
		}
	}
	for _, phase := range []map[string]json.RawMessage{wire.ExpertCoverage.Prefill, wire.ExpertCoverage.Decode} {
		if len(phase["coverage"]) == 0 {
			t.Fatalf("zero elapsed requires separate exported coverage: %s", encoded)
		}
	}
	got.V41Phases.ExpertCoverage.Prefill.Coverage.ContractionTime.Status = "reader_mutation"
	if p.MoEResidencyStats().V41Phases.ExpertCoverage.Prefill.Coverage != coverage {
		t.Fatal("reader mutation altered retained coverage")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41PhaseAttributionBusyReadRetainsLastCounters(t *testing.T) {
	p := &InKernelPlanner{}
	source := model.V41ExpertFaultAttribution{Prefill: v41AttributionFixture(10), Decode: v41AttributionFixture(100)}
	source.Prefill.HeadProjectionDeviceCalls = 7
	source.Decode.RoutedLayerDispatches = 9
	source.Workload.SerialDecode = model.V41WorkloadRows{ObservedOperations: 3, ExecutedRows: 3, AppendedRows: 3}
	source.WorkloadCoverage = model.V41WorkloadCoverageDefault()
	p.foldV41PhasesLocked(source)
	previous := *p.MoEResidencyStats().V41Phases
	coverage := model.V41ExpertPhaseCoverage{Scope: "unavailable", RequestIsolationReason: "ledger_busy"}
	workloadCoverage := model.V41WorkloadCoverageDefault()
	workloadCoverage.Scope, workloadCoverage.Basis = "unavailable", "ledger_busy"
	busy := model.V41ExpertFaultAttribution{
		WorkloadCoverage: workloadCoverage,
		Prefill:          model.V41ExpertFaultAttributionPhase{Coverage: coverage},
		Decode:           model.V41ExpertFaultAttributionPhase{Coverage: coverage},
	}
	p.foldV41PhasesLocked(busy)
	stale := p.MoEResidencyStats().V41Phases
	if stale.Availability != "stale" || stale.AvailabilityReason != "ledger_busy" || stale.ExpertCoverage.Prefill.Coverage != coverage || stale.ExpertCoverage.Decode.Coverage != coverage {
		t.Fatalf("busy availability absent: %+v", stale)
	}
	// Normalize only metadata: every numeric subsystem must still match.
	retained := *stale
	retained.Availability, retained.AvailabilityReason = "", ""
	retained.ExpertCoverage.Prefill.Coverage, retained.ExpertCoverage.Decode.Coverage = previous.ExpertCoverage.Prefill.Coverage, previous.ExpertCoverage.Decode.Coverage
	if retained != previous {
		t.Fatalf("busy read replaced prior cumulative counters: got %+v want %+v", retained, previous)
	}
	// A valid smaller observation can indicate a genuine source reset. It must
	// replace the cache and clear stale metadata rather than being clamped.
	p.foldV41PhasesLocked(model.V41ExpertFaultAttribution{Prefill: v41AttributionFixture(1)})
	latest := p.MoEResidencyStats().V41Phases
	if latest.Availability != "" || latest.AvailabilityReason != "" || latest.Prefill.Tokens != 2 || latest.Decode.Tokens != 0 || latest.Projections.Prefill.HeadProjectionDeviceCalls != 0 {
		t.Fatalf("successful source reset did not replace stale snapshot: %+v", latest)
	}
	fresh := &InKernelPlanner{}
	fresh.foldV41PhasesLocked(busy)
	if unavailable := fresh.MoEResidencyStats().V41Phases; unavailable.Availability != "unavailable" || unavailable.AvailabilityReason != "ledger_busy" || unavailable.WorkloadCoverage != workloadCoverage {
		t.Fatalf("first busy read claimed a measured snapshot: %+v", unavailable)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41PhaseCoverageUnobservedOppositeDebugOnly(t *testing.T) {
	for _, prefillObserved := range []bool{true, false} {
		source := model.V41ExpertFaultAttribution{}
		observed := v41AttributionFixture(10)
		observed.Coverage = model.V41ExpertPhaseCoverage{
			Scope: "model_lifetime", RequestIsolation: "unsupported",
			RequestIsolationReason: "shared_model_phase", Dependency: "request_scoped_phase_ledger",
			RingPageInSemantics:      "h2d_upload_bytes_not_os_faults_disk_reads_or_host_f32_dequant",
			ContractionTime:          model.V41PhaseFieldCoverage{Status: "measured", MeasuredSamples: 7},
			DeviceBypassContractions: 9,
		}
		if prefillObserved {
			source.Prefill = observed
		} else {
			source.Decode = observed
		}
		p := &InKernelPlanner{}
		p.foldV41PhasesLocked(source)
		phases := p.MoEResidencyStats().V41Phases
		unobserved, raw := phases.ExpertCoverage.Decode, source.Decode
		if !prefillObserved {
			unobserved, raw = phases.ExpertCoverage.Prefill, source.Prefill
		}
		if raw != (model.V41ExpertFaultAttributionPhase{}) {
			t.Fatal("debug projection changed untouched raw model phase")
		}
		coverage := unobserved.Coverage
		if coverage.Scope != observed.Coverage.Scope || coverage.RequestIsolation != "unsupported" || coverage.RequestIsolationReason != observed.Coverage.RequestIsolationReason || coverage.Dependency != observed.Coverage.Dependency || coverage.RingPageInSemantics != observed.Coverage.RingPageInSemantics || coverage.DeviceBypassContractions != 0 {
			t.Fatalf("unobserved scope semantics lost or copied work: %+v", coverage)
		}
		for _, field := range []model.V41PhaseFieldCoverage{coverage.HostMaterialization, coverage.FaultTime, coverage.DequantTime, coverage.ContractionTime, coverage.DeviceDispatchTime, coverage.DeviceActivationTime} {
			if field != (model.V41PhaseFieldCoverage{Status: "unavailable", Reason: "phase_not_observed"}) {
				t.Fatalf("unobserved phase claimed a sample: %+v", field)
			}
		}
		unobserved.Coverage = model.V41ExpertPhaseCoverage{}
		if unobserved != (V41ExpertCoverageSnapshot{}) {
			t.Fatalf("debug qualification copied numeric work: %+v", unobserved)
		}
	}
}
