package accountobs

import (
	"fmt"
	"net/http"
	"testing"
	"time"
)

// BenchmarkAdmissionRecord measures quota-window admission evaluation across varied account records.
func BenchmarkAdmissionRecord(b *testing.B) {
	now := time.Date(2026, time.August, 8, 12, 0, 0, 0, time.UTC)
	future := now.Add(15 * time.Minute)

	records := []Record{
		{
			Key:        "seat-exhausted",
			UpdatedAt:  now,
			LastStatus: http.StatusTooManyRequests,
			Headers: map[string]string{
				"anthropic-ratelimit-requests-limit":         "5000",
				"anthropic-ratelimit-requests-remaining":     "0",
				"anthropic-ratelimit-requests-reset":         future.Format(time.RFC3339),
				"anthropic-ratelimit-input-tokens-limit":     "400000",
				"anthropic-ratelimit-input-tokens-remaining": "15000",
				"anthropic-ratelimit-input-tokens-reset":     future.Format(time.RFC3339),
			},
		},
		{
			Key:        "seat-available",
			UpdatedAt:  now,
			LastStatus: http.StatusOK,
			Headers: map[string]string{
				"anthropic-ratelimit-requests-limit":         "5000",
				"anthropic-ratelimit-requests-remaining":     "4200",
				"anthropic-ratelimit-requests-reset":         future.Format(time.RFC3339),
				"anthropic-ratelimit-input-tokens-limit":     "400000",
				"anthropic-ratelimit-input-tokens-remaining": "350000",
				"anthropic-ratelimit-input-tokens-reset":     future.Format(time.RFC3339),
			},
		},
		{
			Key:        "seat-mixed",
			UpdatedAt:  now,
			LastStatus: http.StatusOK,
			Headers: map[string]string{
				"x-ratelimit-requests-remaining": "100",
				"x-ratelimit-requests-reset":     future.Format(time.RFC3339),
				"x-ratelimit-tokens-remaining":   "0",
				"x-ratelimit-tokens-reset":       future.Add(5 * time.Minute).Format(time.RFC3339),
			},
		},
	}

	snapshots := make([]Snapshot, len(records))
	for i, r := range records {
		snapshots[i] = Snapshot{
			LastStatus: r.LastStatus,
			LastAt:     r.UpdatedAt,
			Headers:    r.Headers,
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap := snapshots[i%len(snapshots)]
		skip, until := Admit(snap, now)
		_ = skip
		_ = until
	}
}

// BenchmarkStoreHarvest measures end-to-end quota observation harvesting and durable state coalescing.
func BenchmarkStoreHarvest(b *testing.B) {
	dir := b.TempDir()
	store := Store{Dir: dir}
	harvester := NewHarvester(store, "seat-bench")
	baseTime := time.Date(2026, time.August, 8, 12, 0, 0, 0, time.UTC)

	headers := []http.Header{
		{
			"Anthropic-Ratelimit-Requests-Limit":         {"5000"},
			"Anthropic-Ratelimit-Requests-Remaining":     {"4990"},
			"Anthropic-Ratelimit-Requests-Reset":         {"2026-08-08T12:15:00Z"},
			"Anthropic-Ratelimit-Input-Tokens-Limit":     {"400000"},
			"Anthropic-Ratelimit-Input-Tokens-Remaining": {"385000"},
			"Anthropic-Ratelimit-Input-Tokens-Reset":     {"2026-08-08T12:15:00Z"},
			"Anthropic-Ratelimit-Unified-5h-Utilization": {"0.35"},
			"Anthropic-Ratelimit-Unified-5h-Status":      {"allowed"},
			"Authorization":                              {"Bearer sk-secret"},
			"Content-Type":                               {"application/json"},
		},
		{
			"Anthropic-Ratelimit-Requests-Remaining":     {"4980"},
			"Anthropic-Ratelimit-Input-Tokens-Remaining": {"380000"},
			"Anthropic-Ratelimit-Unified-5h-Utilization": {"0.38"},
		},
	}

	var step time.Duration
	harvester.Now = func() time.Time {
		step += time.Millisecond
		return baseTime.Add(step)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h := headers[i%len(headers)]
		if err := harvester.Observe(http.StatusOK, h); err != nil {
			b.Fatalf("harvester observe failed: %v", err)
		}
	}
}

// BenchmarkTrackerObserve measures in-memory filtering and folding of provider response headers.
func BenchmarkTrackerObserve(b *testing.B) {
	tr := New()
	headers := []http.Header{
		{
			"Anthropic-Ratelimit-Requests-Limit":         {"5000"},
			"Anthropic-Ratelimit-Requests-Remaining":     {"4990"},
			"Anthropic-Ratelimit-Requests-Reset":         {"2026-08-08T12:15:00Z"},
			"Anthropic-Ratelimit-Input-Tokens-Limit":     {"400000"},
			"Anthropic-Ratelimit-Input-Tokens-Remaining": {"385000"},
			"Anthropic-Ratelimit-Unified-5h-Utilization": {"0.35"},
			"Anthropic-Ratelimit-Unified-5h-Status":      {"allowed"},
			"Authorization":                              {"Bearer sk-secret-token"},
			"Content-Type":                               {"application/json"},
			"Date":                                       {"Fri, 08 Aug 2026 12:00:00 GMT"},
		},
		{
			"Anthropic-Ratelimit-Requests-Remaining":     {"4985"},
			"Anthropic-Ratelimit-Input-Tokens-Remaining": {"382000"},
			"Anthropic-Ratelimit-Unified-5h-Utilization": {"0.36"},
			"Anthropic-Ratelimit-Unified-7d-Utilization": {"0.15"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tr.Observe(http.StatusOK, headers[i%len(headers)])
	}
}

// BenchmarkSnapshotUnified measures extraction and sorting of unified subscription windows.
func BenchmarkSnapshotUnified(b *testing.B) {
	reset := time.Date(2026, 8, 8, 17, 0, 0, 0, time.UTC)
	snap := Snapshot{
		Responses: 10,
		Headers: map[string]string{
			"anthropic-ratelimit-unified-status":         "allowed_warning",
			"anthropic-ratelimit-unified-reset":          fmt.Sprint(reset.Unix()),
			"anthropic-ratelimit-unified-5h-utilization": "42",
			"anthropic-ratelimit-unified-5h-status":      "allowed",
			"anthropic-ratelimit-unified-5h-reset":       fmt.Sprint(reset.Unix()),
			"anthropic-ratelimit-unified-7d-utilization": "0.18",
			"anthropic-ratelimit-unified-7d-status":      "allowed",
			"anthropic-ratelimit-unified-7d-reset":       fmt.Sprint(reset.Add(24 * time.Hour).Unix()),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		windows := snap.Unified()
		if len(windows) != 3 {
			b.Fatalf("unexpected window count: %d", len(windows))
		}
	}
}

// BenchmarkSnapshotFamilies measures parsing and sorting of API-key rate-limit families.
func BenchmarkSnapshotFamilies(b *testing.B) {
	snap := Snapshot{
		Responses: 10,
		Headers: map[string]string{
			"anthropic-ratelimit-requests-limit":          "5000",
			"anthropic-ratelimit-requests-remaining":      "4800",
			"anthropic-ratelimit-requests-reset":          "2026-08-08T13:00:00Z",
			"anthropic-ratelimit-input-tokens-limit":      "400000",
			"anthropic-ratelimit-input-tokens-remaining":  "375000",
			"anthropic-ratelimit-input-tokens-reset":      "2026-08-08T13:00:00Z",
			"anthropic-ratelimit-output-tokens-limit":     "80000",
			"anthropic-ratelimit-output-tokens-remaining": "79000",
			"anthropic-ratelimit-output-tokens-reset":     "2026-08-08T13:00:00Z",
			"anthropic-ratelimit-unified-5h-utilization":  "0.25",
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		families := snap.Families()
		if len(families) != 3 {
			b.Fatalf("unexpected family count: %d", len(families))
		}
	}
}

// BenchmarkUsageOverageRejection measures classification of usage and overage rejection signals.
func BenchmarkUsageOverageRejection(b *testing.B) {
	headers := []http.Header{
		{
			"Anthropic-Ratelimit-Unified-Status":         {"allowed"},
			"Anthropic-Ratelimit-Unified-Overage-Status": {"rejected"},
			"Anthropic-Ratelimit-Unified-Reset":          {"1786190000"},
		},
		{
			"Anthropic-Ratelimit-Unified-5h-Status": {"rejected"},
			"Anthropic-Ratelimit-Unified-5h-Reset":  {"1786190000"},
		},
		{
			"Anthropic-Ratelimit-Unified-Status":         {"allowed"},
			"Anthropic-Ratelimit-Unified-5h-Status":      {"allowed"},
			"Anthropic-Ratelimit-Unified-7d-Utilization": {"0.54"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rej := UsageOverageRejection(headers[i%len(headers)])
		_ = rej.Rejected
	}
}

// BenchmarkSnapshotReport measures formatting of the human-readable account observation summary.
func BenchmarkSnapshotReport(b *testing.B) {
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	reset := now.Add(45 * time.Minute)
	snap := Snapshot{
		Responses:   15,
		RateLimited: 1,
		Headers: map[string]string{
			"anthropic-ratelimit-unified-5h-utilization": "68",
			"anthropic-ratelimit-unified-5h-status":      "allowed_warning",
			"anthropic-ratelimit-unified-5h-reset":       fmt.Sprint(reset.Unix()),
			"anthropic-ratelimit-requests-limit":         "5000",
			"anthropic-ratelimit-requests-remaining":     "4200",
			"anthropic-ratelimit-requests-reset":         reset.Format(time.RFC3339),
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := snap.Report(now)
		if report == "" {
			b.Fatal("unexpected empty report")
		}
	}
}
