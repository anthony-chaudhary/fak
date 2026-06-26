//go:build linux

package compute

import "testing"

func TestParseMeminfoKB(t *testing.T) {
	cases := []struct {
		line string
		want int64
		ok   bool
	}{
		{"MemAvailable:   1976543210 kB", 1976543210 * 1024, true},
		{"MemTotal:       2113929216 kB", 2113929216 * 1024, true},
		{"MemFree: 715827882 kB", 715827882 * 1024, true},
		{"MemAvailable:", 0, false},
		{"garbage", 0, false},
		{"MemAvailable: notanumber kB", 0, false},
	}
	for _, c := range cases {
		got, ok := parseMeminfoKB(c.line)
		if ok != c.ok || got != c.want {
			t.Errorf("parseMeminfoKB(%q) = (%d, %v), want (%d, %v)", c.line, got, ok, c.want, c.ok)
		}
	}
}

// TestProcMeminfoAvailableLive reads the real /proc/meminfo on the test host and asserts the
// available figure is positive and not larger than total — the invariant the fit check relies on.
func TestProcMeminfoAvailableLive(t *testing.T) {
	total, avail, ok := procMeminfoAvailable()
	if !ok {
		t.Skip("/proc/meminfo unavailable on this host")
	}
	if total <= 0 || avail <= 0 {
		t.Fatalf("non-positive memory: total=%d avail=%d", total, avail)
	}
	if avail > total {
		t.Fatalf("available %d exceeds total %d", avail, total)
	}
}
