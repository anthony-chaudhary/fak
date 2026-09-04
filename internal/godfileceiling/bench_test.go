package godfileceiling_test

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/godfileceiling"
)

func BenchmarkLineCount(b *testing.B) {
	sample := bytes.Repeat([]byte("package main\n\nfunc hello() {\n\tprintln(\"hello world\")\n}\n"), 250)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = godfileceiling.LineCount(sample)
	}
}

func BenchmarkExcluded(b *testing.B) {
	paths := []string{
		"internal/godfileceiling/godfileceiling.go",
		".claude/worktrees/worker-1/internal/agent/chat.go",
		"vendor/github.com/stretchr/testify/assert.go",
		"internal/gateway/metrics.go",
		"cmd/fak/main.go",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		for _, p := range paths {
			_ = godfileceiling.Excluded(p)
		}
	}
}

func BenchmarkEvaluate(b *testing.B) {
	measured := map[string]int{
		"cmd/fak/cachevalue_status.go":            3014,
		"cmd/fak/dispatch_tick.go":                1720,
		"cmd/fak/loop.go":                         1540,
		"cmd/fak/release_ship.go":                 1700,
		"internal/agent/chat.go":                  1650,
		"internal/compute/cuda.go":                1560,
		"internal/dispatchtick/router.go":         1760,
		"internal/fleetpane/fleetpane.go":         2080,
		"internal/gateway/gateway.go":             3130,
		"internal/gateway/http.go":                1810,
		"internal/gateway/messages.go":            1730,
		"internal/gateway/metrics.go":             3350,
		"internal/operatorbrief/operatorbrief.go": 1570,
		"internal/sessionaudit/sessionaudit.go":   1730,
		"internal/small/foo.go":                   350,
		"internal/small/bar.go":                   800,
	}
	caps := godfileceiling.Baseline
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = godfileceiling.Evaluate(measured, caps)
	}
}

func BenchmarkProposeBaseline(b *testing.B) {
	measured := map[string]int{
		"cmd/fak/cachevalue_status.go": 3014,
		"cmd/fak/dispatch_tick.go":     1720,
		"cmd/fak/loop.go":              1540,
		"internal/small/foo.go":        350,
		"internal/small/bar.go":        800,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = godfileceiling.ProposeBaseline(measured)
	}
}

func BenchmarkRepin(b *testing.B) {
	measured := map[string]int{
		"cmd/fak/cachevalue_status.go": 2900,
		"cmd/fak/dispatch_tick.go":     1700,
		"cmd/fak/loop.go":              1500,
	}
	oldCaps := map[string]int{
		"cmd/fak/cachevalue_status.go": 3000,
		"cmd/fak/dispatch_tick.go":     1730,
		"cmd/fak/loop.go":              1544,
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, _ = godfileceiling.Repin(measured, oldCaps)
	}
}
