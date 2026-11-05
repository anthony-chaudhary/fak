package fleetbottleneck

import "testing"

func TestRankHeadlines(t *testing.T) {
	tests := []struct {
		name string
		s    Snapshot
		want Class
	}{
		{"seats", Snapshot{Machines: 1, Sessions: 10, SeatCapacity: 10}, ClassSeats},
		{"throttle", Snapshot{Machines: 1, ThrottledSeats: 4}, ClassThrottle},
		{"resume", Snapshot{Machines: 1, ResumeBacklog: 20}, ClassResume},
		{"host", Snapshot{Machines: 1, HostLoad: .95}, ClassHost},
		{"auth", Snapshot{Machines: 1, AuthBlocked: 3}, ClassAuth},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Rank(tt.s)
			if len(got) == 0 || got[0].Class != tt.want {
				t.Fatalf("rank=%+v want %s", got, tt.want)
			}
		})
	}
}
func TestRankIdleIsSilent(t *testing.T) {
	if got := Rank(Snapshot{}); got != nil {
		t.Fatalf("got=%v", got)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestRankThrottleCountsDoNotOverflow(t *testing.T) {
	maxInt := int64(^uint(0) >> 1)
	for _, tc := range []struct {
		name      string
		throttled int64
		healthy   int64
		wantScore float64
	}{
		{"ordinary", 1, 1, .875},
		{"overflow", maxInt, maxInt, .875},
		{"preserve rounded tie", 3160077690863166028, 790019422715791087, .95},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.throttled > maxInt || tc.healthy > maxInt {
				t.Skip("fixture requires 64-bit int")
			}
			got := Rank(Snapshot{
				Machines:       1,
				ThrottledSeats: int(tc.throttled),
				HealthySeats:   int(tc.healthy),
				HostLoad:       1,
			})
			if len(got) != 2 {
				t.Fatalf("rank=%+v, want host and throttle", got)
			}
			if got[0].Class != ClassHost || got[1].Class != ClassThrottle || got[1].Score != tc.wantScore {
				t.Fatalf("rank=%+v, want host before throttle at score %v", got, tc.wantScore)
			}
		})
	}
}
