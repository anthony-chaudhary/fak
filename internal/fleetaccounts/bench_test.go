package fleetaccounts

import (
	"testing"
)

// BenchmarkFleetAccounts measures the performance of account routing and status folding across candidate seats.
func BenchmarkFleetAccounts(b *testing.B) {
	rows := []Account{
		{
			Dir:          "C:/u/.claude-weighted",
			Product:      "claude",
			Account:      ".claude-weighted",
			Tag:          "weighted",
			Kind:         KindWorker,
			ModelTier:    intp(1),
			Available:    boolp(true),
			RouteWeight:  intp(10),
			LiveSessions: intp(0),
		},
		{
			Dir:          "C:/u/.claude-idle",
			Product:      "claude",
			Account:      ".claude-idle",
			Tag:          "idle",
			Kind:         KindWorker,
			ModelTier:    intp(1),
			Available:    boolp(true),
			RouteWeight:  intp(5),
			LiveSessions: intp(0),
		},
		{
			Dir:          "C:/u/.claude-backup",
			Product:      "claude",
			Account:      ".claude-backup",
			Tag:          "backup",
			Kind:         KindWorker,
			ModelTier:    intp(2),
			Available:    boolp(true),
			RouteWeight:  intp(1),
			LiveSessions: intp(0),
		},
	}
	pol := DefaultPolicy()
	reg := Registry{
		GeneratedUTC: "2026-09-04T12:00:00Z",
		Throttle:     map[string]any{"claude": false},
		Auth:         map[string]any{"claude": true},
		Sessions: []Session{
			{Account: ".claude-weighted", Disp: "running", AgeMin: 5},
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		annotated := Annotate(rows, reg)
		res := RouteAccount(annotated, "implement the feature and optimize kernel", "engineering", false, false, "claude", pol)
		if !res.OK || res.Account == nil {
			b.Fatal("failed to route account")
		}
	}
}
