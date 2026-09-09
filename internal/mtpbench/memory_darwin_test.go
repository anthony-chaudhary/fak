//go:build darwin && cgo

package mtpbench

import "testing"

func TestParseSwapUsedFailClosed(t *testing.T) {
	got, ok := parseSwapUsed("total = 4096.00M  used = 12.25M  free = 4083.75M")
	if !ok || got != uint64(12.25*(1<<20)) {
		t.Fatalf("got=%d ok=%v", got, ok)
	}
	for _, raw := range []string{"", "used = nope", "used = 1.0", "used = 1.0G", "used = NaNM", "used = +InfM", "used = -1M", "used = 999999999999999999999999999999999M", "free = 1.0M"} {
		if _, ok := parseSwapUsed(raw); ok {
			t.Fatalf("accepted %q", raw)
		}
	}
}
