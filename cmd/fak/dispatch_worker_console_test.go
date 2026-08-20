package main

import "testing"

func TestDispatchWorkerNeedsHiddenConsole(t *testing.T) {
	for _, tc := range []struct {
		backend string
		want    bool
	}{
		{backend: "codex", want: true},
		{backend: " CODEX ", want: true},
		{backend: "claude", want: false},
		{backend: "opencode", want: false},
		{backend: "", want: false},
	} {
		if got := dispatchWorkerNeedsHiddenConsole(tc.backend); got != tc.want {
			t.Errorf("dispatchWorkerNeedsHiddenConsole(%q) = %v, want %v", tc.backend, got, tc.want)
		}
	}
}
