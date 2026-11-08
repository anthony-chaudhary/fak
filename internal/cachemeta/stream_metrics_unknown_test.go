package cachemeta

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestStreamMetricsUnknownLabelIdentity(t *testing.T) {
	metrics := NewStreamMetrics()
	for _, plane := range []Plane{"", "unknown"} {
		for _, tier := range []ResidencyTier{"", "unknown"} {
			entry := Entry{Plane: plane, Residency: Residency{Tier: tier}}
			entry.Metrics.BytesTransferred = 7
			metrics.Observe("hit", entry)
		}
	}
	snapshot := metrics.Snapshot()
	if snapshot.Events != 4 || snapshot.BytesMoved != 28 || len(snapshot.Rows) != 1 {
		t.Fatalf("render-equivalent labels did not merge: %+v", snapshot)
	}
	want := StreamRow{Plane: "unknown", Tier: "unknown", Kind: "hit", Count: 4, Bytes: 28}
	if snapshot.Rows[0] != want {
		t.Fatalf("row=%+v want=%+v", snapshot.Rows[0], want)
	}
	exposition := snapshot.Prometheus()
	const labelSet = `fak_cache_event_breakdown_total{plane="unknown",tier="unknown",kind="hit"}`
	if strings.Count(exposition, labelSet) != 1 || !strings.Contains(exposition, labelSet+" 4\n") {
		t.Fatalf("duplicate or incorrect sample: %s", exposition)
	}
	// A real tier remains a distinct series; normalization is not broad merging.
	metrics.Observe("hit", Entry{Residency: Residency{Tier: TierHBM}})
	if got := metrics.Snapshot(); len(got.Rows) != 2 || got.Events != 5 || got.BytesMoved != 28 {
		t.Fatalf("distinct tier was merged: %+v", got)
	}
}
