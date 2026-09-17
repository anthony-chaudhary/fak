package agent

import (
	"encoding/json"
	"fmt"
	"testing"
)

func TestHashEventsFixedWidth(t *testing.T) {
	// Zero value must zero-pad, not shrink.
	if got := fmt.Sprintf("%016x", uint64(0)); got != "0000000000000000" {
		t.Fatalf("zero-pad broken: got %q", got)
	}
	// Nil and empty event slices must yield exactly 16 hex chars (no slice panic).
	for _, evs := range [][]traceEvent{nil, {}, {{Turn: 1, Arm: "fak", Tool: "t"}}} {
		h := hashEvents(evs)
		if len(h) != 16 {
			t.Fatalf("hashEvents len=%d, want 16 (events=%v, hash=%q)", len(h), evs, h)
		}
	}
	// Width property: brute-force a suffix whose fnv1a hex would be <16 chars
	// unpadded (leading zero nibble), and assert the padded form stays 16.
	found := false
	for i := 0; i < 100000; i++ {
		b, _ := json.Marshal([]traceEvent{{Note: fmt.Sprintf("padprobe-%d", i)}})
		raw := fmt.Sprintf("%x", fnv1a(b))
		if len(raw) < 16 {
			h := hashEvents([]traceEvent{{Note: fmt.Sprintf("padprobe-%d", i)}})
			if len(h) != 16 {
				t.Fatalf("leading-zero case len=%d, want 16 (raw=%q padded=%q)", len(h), raw, h)
			}
			if h != fmt.Sprintf("%016x", fnv1a(b)) {
				t.Fatalf("hashEvents mismatch: got %q want %q", h, fmt.Sprintf("%016x", fnv1a(b)))
			}
			found = true
			break
		}
	}
	if !found {
		t.Log("no leading-zero-nibble payload found in 100k probes; nil/empty width checks still hold")
	}
}
