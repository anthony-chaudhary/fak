package accountobs

import (
	"net/http"
	"testing"
	"time"
)

func TestAdmitUsesOnlyExhaustedFutureQuotaWindows(t *testing.T) {
	now := time.Date(2026, time.August, 8, 12, 0, 0, 0, time.UTC)
	future := now.Add(20 * time.Minute)

	tests := []struct {
		name      string
		snapshot  Snapshot
		wantSkip  bool
		wantUntil time.Time
	}{
		{
			name: "zero remaining with future reset blocks",
			snapshot: Snapshot{Headers: map[string]string{
				"x-ratelimit-requests-remaining": "0",
				"x-ratelimit-requests-reset":     future.Format(time.RFC3339),
			}},
			wantSkip:  true,
			wantUntil: future,
		},
		{
			name: "retry after alone never blocks",
			snapshot: Snapshot{LastStatus: http.StatusTooManyRequests, Headers: map[string]string{
				"retry-after": "120",
			}},
		},
		{
			name: "healthy 200 with spurious retry after never blocks",
			snapshot: Snapshot{LastStatus: http.StatusOK, Headers: map[string]string{
				"x-ratelimit-requests-remaining": "7",
				"x-ratelimit-requests-reset":     future.Format(time.RFC3339),
				"retry-after":                    "120",
			}},
		},
		{
			name: "expired zero window no longer blocks",
			snapshot: Snapshot{Headers: map[string]string{
				"x-ratelimit-tokens-remaining": "0",
				"x-ratelimit-tokens-reset":     now.Add(-time.Second).Format(time.RFC3339),
			}},
		},
		{
			name: "earliest exhausted window is reevaluation time",
			snapshot: Snapshot{Headers: map[string]string{
				"x-ratelimit-requests-remaining": "0",
				"x-ratelimit-requests-reset":     future.Format(time.RFC3339),
				"x-ratelimit-tokens-remaining":   "0",
				"x-ratelimit-tokens-reset":       now.Add(10 * time.Minute).Format(time.RFC3339),
			}},
			wantSkip:  true,
			wantUntil: now.Add(10 * time.Minute),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotSkip, gotUntil := Admit(tt.snapshot, now)
			if gotSkip != tt.wantSkip || !gotUntil.Equal(tt.wantUntil) {
				t.Fatalf("Admit() = (%v, %s), want (%v, %s)", gotSkip, gotUntil, tt.wantSkip, tt.wantUntil)
			}
		})
	}
}
