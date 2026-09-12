package gateway

import (
	"strings"
	"testing"
)

func TestFeatureTrackerMetricsOncePerRequest(t *testing.T) {
	var metrics featureActivationMetrics
	tracker := NewFeatureActivationTracker(FeatureCatalog{})
	tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed)
	tracker.RecordActivation(FeatureVDSO, FeatureOutcomeUsed)
	for i := 0; i < 2; i++ {
		if snapshot, first := tracker.Finalize(); first {
			metrics.ObserveFinal(snapshot)
		}
	}
	counts := metrics.Snapshot()
	if len(counts) != 1 || counts[FeatureVDSO] != 1 {
		t.Fatalf("request counts=%v", counts)
	}
	counts[FeatureVDSO] = 99
	var rendered strings.Builder
	metrics.Render(&rendered)
	if !strings.Contains(rendered.String(), `fak_gateway_feature_used_requests_total{feature="vdso"} 1`) {
		t.Fatalf("metric lost request count or aliased snapshot: %s", rendered.String())
	}
}
