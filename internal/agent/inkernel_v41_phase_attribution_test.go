package agent

import (
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
