package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestRenderLoopHealthCarriesUtilityFailureEvidence(t *testing.T) {
	rep := loopmgr.HealthReport{
		Rollup: loopmgr.HealthRollup{
			Loops:          3,
			Live:           3,
			Runs:           10,
			Effects:        3,
			NoFuel:         1,
			Failed:         6,
			FailureAlert:   2,
			NeverSucceeded: 1,
			CostedRuns:     2,
		},
		Rows: []loopmgr.HealthRow{
			{
				LoopID:              "never",
				State:               loopmgr.HealthLive,
				CadenceSeconds:      60,
				Runs:                3,
				Failed:              3,
				ConsecutiveFailures: 3,
				FailureAlert:        true,
				NeverSucceeded:      true,
				KeepRate:            -1,
			},
			{
				LoopID:              "regressed",
				State:               loopmgr.HealthLive,
				CadenceSeconds:      60,
				Runs:                4,
				Failed:              2,
				ConsecutiveFailures: 2,
				FailureAlert:        true,
				Witnessed:           2,
				KeepRate:            0.5,
			},
			{
				LoopID:         "recovered",
				State:          loopmgr.HealthLive,
				CadenceSeconds: 60,
				Runs:           3,
				Failed:         1,
				Witnessed:      2,
				KeepRate:       2.0 / 3.0,
			},
		},
	}

	rendered := captureLoopHealth(t, rep)
	for _, want := range []string{
		"utility: runs=10 effects=3 no-fuel=1 unattributed=0 failed=6 alerting=2 never-succeeded=1 cost=$0.000/2run",
		"LOOP STATE LAST AGE CADENCE RUNS FAILED ALERT",
		"never live - - 1.0m 3 3 never-succeeded(3)",
		"regressed live - - 1.0m 4 2 repeated-failure(2)",
		"recovered live - - 1.0m 3 1 -",
	} {
		if !strings.Contains(compactLoopHealth(rendered), want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func TestRenderLoopHealthUtilityCostSeparatesMeasuredZeroFromUnmeasured(t *testing.T) {
	rep := loopmgr.HealthReport{
		Rollup: loopmgr.HealthRollup{Loops: 1, Live: 1, Runs: 1, CostedRuns: 1},
		Rows:   []loopmgr.HealthRow{{LoopID: "measured", State: loopmgr.HealthLive, Runs: 1, KeepRate: -1}},
	}
	measured := captureLoopHealth(t, rep)
	if !strings.Contains(measured, "cost=$0.000/1run") {
		t.Fatalf("measured zero cost rendered as unmeasured:\n%s", measured)
	}

	rep.Rollup.CostedRuns = 0
	unmeasured := captureLoopHealth(t, rep)
	if !strings.Contains(unmeasured, "cost=unmeasured") {
		t.Fatalf("missing unmeasured cost:\n%s", unmeasured)
	}
}

func captureLoopHealth(t *testing.T, rep loopmgr.HealthReport) string {
	t.Helper()
	var out bytes.Buffer
	renderLoopHealth(&out, rep, "", "")
	return out.String()
}

func compactLoopHealth(rendered string) string {
	lines := strings.Split(rendered, "\n")
	for i, line := range lines {
		lines[i] = strings.Join(strings.Fields(line), " ")
	}
	return strings.Join(lines, "\n")
}
