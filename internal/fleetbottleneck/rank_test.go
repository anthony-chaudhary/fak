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
