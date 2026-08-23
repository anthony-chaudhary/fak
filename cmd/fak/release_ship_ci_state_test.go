package main

import "testing"

func TestReleaseShipCIState(t *testing.T) {
	tests := []struct{ status, conclusion, want string }{
		{"queued", "", "pending"},
		{"in_progress", "", "pending"},
		{"completed", "success", "green"},
		{"completed", "failure", "failed"},
		{"completed", "timed_out", "failed"},
		{"completed", "cancelled", "cancelled_or_starved"},
		{"completed", "stale", "cancelled_or_starved"},
		{"completed", "", "missing"},
	}
	for _, tt := range tests {
		t.Run(tt.status+"/"+tt.conclusion, func(t *testing.T) {
			if got := releaseShipCIState(tt.status, tt.conclusion); got != tt.want {
				t.Fatalf("got %q, want %q", got, tt.want)
			}
		})
	}
}
