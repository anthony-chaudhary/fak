package fleetmemory

import "testing"

// BenchmarkFleetMemory exercises lesson matching and session injection
// in a loop to measure in-memory query throughput.
func BenchmarkFleetMemory(b *testing.B) {
	ledger := New([]Lesson{
		{
			ID:      "bash-git-hang",
			Fact:    "Bash git hangs on this host — use PowerShell",
			Trigger: Trigger{Host: "fleet-win", Tool: "Bash"},
			Witness: "memory:bash_git_gh_hang_use_powershell",
		},
		{
			ID:      "wsl-go-test",
			Fact:    "native go test is OS-blocked — route through WSL",
			Trigger: Trigger{PathGlobs: []string{"internal/*"}},
			Witness: "memory:wsl_go_test_capture_technique",
		},
		{
			ID:      "off-trunk",
			Fact:    "never open a feature branch — the guard refuses off-trunk commits",
			Trigger: Trigger{RefusalToken: "OFF_TRUNK"},
		},
	})

	ctx := SessionContext{
		Host:         "fleet-win",
		Paths:        []string{"internal/fleetmemory/ledger.go"},
		Tool:         "Bash",
		RefusalToken: "OFF_TRUNK",
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ledger.Inject(ctx)
		_, _ = ledger.Match("Bash git hangs on this host, use PowerShell")
	}
}
