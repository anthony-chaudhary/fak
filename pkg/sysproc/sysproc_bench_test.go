package sysproc

import (
	"os/exec"
	"testing"
)

// benchName names an executable that does not exist. exec.Command defers
// resolution to Start, so these loops measure only the configuration seam and
// never spawn a process.
const benchName = "fak-sysproc-bench"

// BenchmarkCommand measures the spawn-time hot path shared by eight dependent
// packages: one background-configured *exec.Cmd per spawned child.
func BenchmarkCommand(b *testing.B) {
	var sink *exec.Cmd
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = Command(benchName, "bench")
	}
	_ = sink
}

// BenchmarkConfigureBackground measures console-window suppression applied to
// a fresh command, the per-spawn cost of every hidden Windows child.
func BenchmarkConfigureBackground(b *testing.B) {
	var sink *exec.Cmd
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = exec.Command(benchName)
		ConfigureBackground(sink)
	}
	_ = sink
}

// BenchmarkConfigureDetached measures detaching a child from its launcher's
// session or console, applied once per detached spawn.
func BenchmarkConfigureDetached(b *testing.B) {
	var sink *exec.Cmd
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = exec.Command(benchName)
		ConfigureDetached(sink)
	}
	_ = sink
}

// BenchmarkConfigureDurableChild measures the survive-the-launcher flag
// combination, the per-spawn cost of every long-lived managed child.
func BenchmarkConfigureDurableChild(b *testing.B) {
	var sink *exec.Cmd
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = exec.Command(benchName)
		ConfigureDurableChild(sink)
	}
	_ = sink
}

// BenchmarkConfigureProcessGroup measures starting a child as the root of a
// new process group, the per-spawn cost behind group-scoped signal handling.
func BenchmarkConfigureProcessGroup(b *testing.B) {
	var sink *exec.Cmd
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sink = exec.Command(benchName)
		ConfigureProcessGroup(sink)
	}
	_ = sink
}
