package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// fak-test:runtime fast est=50ms
func TestPerfVerbCompactNamesSlowestTurnIDs(t *testing.T) {
	slow := perfledger.NewRecord(time.UnixMilli(4), "stop", perfledger.LocalitySelfHosted, 400, 40, 0, 9*time.Second, time.Second)
	slow.RequestID, slow.InboundRequestID = "fak-slow", "rt-slow"
	busy := 88
	slow.GPUBusyPercent = &busy
	path := writePerfLedger(t, append(perfFixture(), slow)...)
	var out, errOut bytes.Buffer
	if rc := runPerf(&out, &errOut, []string{"--ledger", path}); rc != 0 {
		t.Fatalf("rc = %d, want 0", rc)
	}
	for _, want := range []string{"request_id=fak-slow", "inbound_request_id=rt-slow", "gpu busy p50=88% (measured 1/4)"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("compact missing %q: %s", want, out.String())
		}
	}
}
