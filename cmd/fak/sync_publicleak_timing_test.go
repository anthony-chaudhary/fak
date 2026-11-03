package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// fak-test:runtime fast est=10ms lane=default
func TestCleanSyncEmitsLeakTimingWithoutNoisyEnvelope(t *testing.T) {
	// Injected values test JSON wiring, not measured performance.
	p := &syncPublicLeakPreflight{OK: true, Timing: hooks.PublicLeakTreeTiming{EnumerationNS: 1, ScanNS: 2, TrackedPaths: 3}}
	var out, stderr bytes.Buffer
	info := safesync.Assessment{OK: true, State: safesync.StateInSync}
	if code := outputSyncResult(&out, &stderr, "check", true, info, p, nil); code != syncExitOK {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(out.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if _, exists := raw["public_leak"]; exists {
		t.Fatal("clean check emitted findings envelope")
	}
	var got hooks.PublicLeakTreeTiming
	if err := json.Unmarshal(raw["public_leak_timing"], &got); err != nil {
		t.Fatal(err)
	}
	if got != p.Timing {
		t.Fatalf("timing changed: %+v", got)
	}
	out.Reset()
	if code := outputSyncResult(&out, &stderr, "check", false, info, p, nil); code != syncExitOK {
		t.Fatal(code)
	}
	if out.String() != "in sync: local branch already matches the remote; nothing to do\n" {
		t.Fatalf("text output changed: %q", out.String())
	}
}
