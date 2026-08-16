package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestRenderLoopHealthCarriesUtilityFailureEvidence(t *testing.T) {
	rep := loopmgr.HealthReport{Rollup: loopmgr.HealthRollup{Runs: 4, Effects: 1, Failed: 3, FailureAlert: 1, NeverSucceeded: 1}}
	rep.Rows = []loopmgr.HealthRow{{LoopID: "scout", Runs: 4, Failed: 3, FailureAlert: true, NeverSucceeded: true, ConsecutiveFailures: 3}}
	var out bytes.Buffer
	renderLoopHealth(&out, rep, "", "")
	for _, want := range []string{"utility: runs=4 effects=1", "FAILED", "ALERT", "never-succeeded(3)"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("render missing %q:\n%s", want, out.String())
		}
	}
}
