package procguard

import "testing"

func TestEvaluateSystemCommitHeadroomBoundary(t *testing.T) {
	for _, tt := range []struct {
		name     string
		snapshot MemorySnapshot
		required uint64
		refuse   bool
		observed uint64
	}{
		{"below", MemorySnapshot{Metric: MemoryMetricCommit, SystemBytes: 91, SystemLimit: 100}, 10, true, 9},
		{"exact", MemorySnapshot{Metric: MemoryMetricCommit, SystemBytes: 90, SystemLimit: 100}, 10, true, 10},
		{"above", MemorySnapshot{Metric: MemoryMetricCommit, SystemBytes: 89, SystemLimit: 100}, 10, false, 11},
		{"underflow", MemorySnapshot{Metric: MemoryMetricCommit, SystemBytes: 101, SystemLimit: 100}, 1, true, 0},
		{"rss abstains", MemorySnapshot{Metric: MemoryMetricRSS, SystemBytes: 99, SystemLimit: 100}, 10, false, 1},
		{"missing limit abstains", MemorySnapshot{Metric: MemoryMetricCommit}, 10, false, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := EvaluateSystemCommitHeadroom(tt.snapshot, tt.required)
			if got.Refuse != tt.refuse || got.ObservedBytes != tt.observed {
				t.Fatalf("result=%+v", got)
			}
			if got.Refuse && got.Reason != SystemCommitHeadroomReason {
				t.Fatalf("reason=%q", got.Reason)
			}
		})
	}
}

func TestRequiredSystemCommitHeadroom(t *testing.T) {
	for _, raw := range []string{"", "0", "-1", "+1", "1GB", "17592186044416"} {
		if got := RequiredSystemCommitHeadroom(func(string) string { return raw }); got != DefaultSystemCommitHeadroomBytes {
			t.Fatalf("raw=%q got=%d want default=%d", raw, got, DefaultSystemCommitHeadroomBytes)
		}
	}
	if got := RequiredSystemCommitHeadroom(func(string) string { return " 456 " }); got != 456<<20 {
		t.Fatalf("override=%d", got)
	}
}

func TestEvaluateSystemCommitHeadroomPreservesPhysicalFields(t *testing.T) {
	snapshot := MemorySnapshot{
		Metric:                     MemoryMetricCommit,
		SystemBytes:                50,
		SystemLimit:                100,
		HostPhysicalBytes:          16 << 30,
		HostPhysicalAvailableBytes: 8 << 30,
	}
	headroom := EvaluateSystemCommitHeadroom(snapshot, 10)
	if headroom.PhysicalTotalBytes != 16<<30 {
		t.Fatalf("PhysicalTotalBytes got %d, want %d", headroom.PhysicalTotalBytes, uint64(16<<30))
	}
	if headroom.PhysicalAvailableBytes != 8<<30 {
		t.Fatalf("PhysicalAvailableBytes got %d, want %d", headroom.PhysicalAvailableBytes, uint64(8<<30))
	}
}
