package main

// footprint_heldaccuracy.go — the #3533 held-accuracy fault-in-recall eval, a
// `--held-accuracy` mode of `fak footprint` (epic #3229, QA gate #2 of 3 for the #3537
// default-on flip). It runs the gateway's fixed eval set (one representative cold tool per
// category) through the PRODUCTION deferral transform ARMED vs ABLATED and reports the
// held-accuracy PAIR (armed_pass/total vs ablated_pass/total) plus the gate verdict.
//
// HONESTY: Mode is deterministic-faultin-sim — it WITNESSES that a deferred cold tool is
// never silently lost (it stays present, searchable, and schema-intact, so it is faultable
// in), NOT that a live model actually searches for and completes the task. That live number
// is the dogfood run (#3536); the JSON carries live_accuracy_claim_allowed:false so nobody
// re-labels this a task-completion accuracy claim.

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func runFootprintHeldAccuracy(out, errw io.Writer, asJSON bool) int {
	rep := gateway.DeferHeldAccuracyReport()

	if asJSON {
		_ = writeIndentedJSONNoEscape(out, map[string]any{
			"schema":                      "fak-footprint-held-accuracy/1",
			"mode":                        rep.Mode,
			"provenance":                  rep.Provenance,
			"live_accuracy_claim_allowed": rep.LiveAccuracyClaimAllowed,
			"armed_pass":                  rep.ArmedPass,
			"ablated_pass":                rep.AblatedPass,
			"total":                       rep.Total,
			"gate_holds":                  rep.GateHolds,
			"offenders":                   rep.Offenders,
			"results":                     rep.Results,
		})
		// A regression is a non-zero exit so CI/`make` can gate on it.
		if !rep.GateHolds {
			return 1
		}
		return 0
	}

	fmt.Fprintf(out, "held-accuracy (#3533, epic #3229) — cold-tool-deferral fault-in recall · %s\n", rep.Mode)
	fmt.Fprintf(out, "  held-accuracy: armed %d/%d vs ablated %d/%d  →  gate %s (armed >= ablated)\n",
		rep.ArmedPass, rep.Total, rep.AblatedPass, rep.Total, gateVerdict(rep.GateHolds))
	for _, r := range rep.Results {
		mark := "pass"
		note := ""
		if !r.ArmedPass {
			mark = "FAIL"
			note = " — " + r.Reason
		}
		fmt.Fprintf(out, "  [%s] %-14s %s (%s)%s\n", mark, r.Task.Category, r.Task.Tool, r.Task.Name, note)
	}
	if !rep.GateHolds {
		fmt.Fprintf(out, "  REGRESSION: deferral lost capability for: %s\n", strings.Join(rep.Offenders, ", "))
	}
	fmt.Fprintln(out, "  note: mechanical recall (a deferred tool stays faultable-in), not a live-model")
	fmt.Fprintln(out, "  task-completion accuracy — that is the dogfood run (#3536).")
	if !rep.GateHolds {
		return 1
	}
	return 0
}

func gateVerdict(holds bool) string {
	if holds {
		return "HOLDS"
	}
	return "RED"
}
