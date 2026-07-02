package compute

import (
	"runtime"
	"testing"
)

// TestFarMemoryInfoContract checks the far-memory probes' fail-open contract on
// whatever host runs the tests: an unconfirmed tier reports exactly (0, FreeUnknown,
// false) — never a fabricated number (#1470's fence) — and only the linux sysfs probe
// may confirm one, with free never exceeding total when it does.
func TestFarMemoryInfoContract(t *testing.T) {
	probes := map[string]func() (int64, int64, bool){
		"numa_far": NUMAFarMemoryInfo,
		"cxl":      CXLMemoryInfo,
	}
	for name, probe := range probes {
		total, free, known := probe()
		if !known {
			if total != 0 || free != FreeUnknown {
				t.Errorf("%s: unknown must fail open to (0, FreeUnknown), got (%d, %d)", name, total, free)
			}
			continue
		}
		if runtime.GOOS != "linux" {
			t.Errorf("%s: only the linux probe may report known=true, got known on %s", name, runtime.GOOS)
		}
		if total <= 0 {
			t.Errorf("%s: known=true with non-positive total %d", name, total)
		}
		if free < 0 || free > total {
			t.Errorf("%s: free %d outside [0, total=%d]", name, free, total)
		}
	}
}
