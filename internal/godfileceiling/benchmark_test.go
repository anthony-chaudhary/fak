package godfileceiling_test

import (
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/godfileceiling"
)

// BenchmarkGodFileCeilingScan benchmarks evaluating measured file trees against pinned ceiling caps.
func BenchmarkGodFileCeilingScan(b *testing.B) {
	caps := make(map[string]int, 20)
	caps["cmd/fak/cachevalue_status.go"] = 3014
	caps["cmd/fak/dispatch_tick.go"] = 1730
	caps["cmd/fak/loop.go"] = 1544
	caps["cmd/fak/release_ship.go"] = 1708
	caps["internal/agent/chat.go"] = 1664
	caps["internal/compute/cuda.go"] = 1562
	caps["internal/dispatchtick/router.go"] = 1768
	caps["internal/fleetpane/fleetpane.go"] = 2091
	caps["internal/gateway/gateway.go"] = 3135
	caps["internal/gateway/http.go"] = 1819
	caps["internal/gateway/messages.go"] = 1739
	caps["internal/gateway/metrics.go"] = 3354
	caps["internal/operatorbrief/operatorbrief.go"] = 1576
	caps["internal/sessionaudit/sessionaudit.go"] = 1737

	measured := make(map[string]int, 250)
	for k, v := range caps {
		measured[k] = v
	}
	for i := 0; i < 200; i++ {
		path := "internal/pkg" + strconv.Itoa(i) + "/file.go"
		measured[path] = 100 + (i % 1200)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := godfileceiling.Evaluate(measured, caps)
		if !v.OK {
			b.Fatalf("unexpected failure during scan benchmark: %+v", v.Violations)
		}
	}
}

// BenchmarkGodFileCeilingLineCount benchmarks physical line counting over source file buffers.
func BenchmarkGodFileCeilingLineCount(b *testing.B) {
	buf := []byte(
		"package example\n\nimport \"fmt\"\n\n// Sample comment line.\nfunc Hello() {\n\tfmt.Println(\"hello\")\n}\n",
	)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n := godfileceiling.LineCount(buf)
		if n != 8 {
			b.Fatalf("unexpected line count: %d", n)
		}
	}
}

// TestGodFileCeilingBenchmarkExecution ensures that benchmark evaluation passes basic sanity assertions.
func TestGodFileCeilingBenchmarkExecution(t *testing.T) {
	caps := map[string]int{"cmd/fak/loop.go": 1600}
	measured := map[string]int{"cmd/fak/loop.go": 1544, "internal/small/file.go": 200}
	v := godfileceiling.Evaluate(measured, caps)
	if !v.OK {
		t.Fatalf("expected valid evaluation, got violations: %+v", v.Violations)
	}
}
