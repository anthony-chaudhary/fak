package main

// dosobserve.go wires #588: when -dos-observe is set, each rsiloop keep/revert verdict
// is mirrored to the DOS audit journal as a `dos improve --observe` receipt — a record-
// only telemetry rung that journals what the loop decided WITHOUT letting the external
// command re-gate the decision.
//
// WHY OBSERVE-ONLY (the non-forgeability fence). The loop's keep-bit is already
// computed by shipgate.Evaluate over float witnesses Run measured itself — that is the
// authority, and it is never read back from here. `dos improve --observe` recomputes a
// verdict from the SAME witnesses purely to record it; in --observe mode it journals,
// it does not gate. So even if the float-to-int scaling below blurred a microscopic gain
// (round(metric*1e9) keeps 1e-9 resolution), it could only change the RECORDED receipt,
// never the loop's verdict — which is also carried verbatim in --narrated. Re-gating the
// loop through an integer `--work` would erode the keep-bit's non-forgeability; emitting
// a receipt of it does not.
//
// DEGRADES SILENTLY. dos is absent on CI runners and bench nodes; -dos-observe there is
// a no-op (one stderr note, the loop unaffected). A verdict exit code from dos improve
// (3 REVERT / 4 ESCALATE) is the tool reporting — NOT an error — so it is never surfaced;
// only a genuine spawn failure is.

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// roundUnit maps a float KPI to dos's non-negative integer --work unit "for the
// recorded number only": round(metric*1e9). It is the int the receipt carries; the
// loop's own decision stays on the float.
func roundUnit(x float64) int64 { return int64(math.Round(x * 1e9)) }

// scaleWorkPair scales a candidate metric and its baseline to the (work, baselineWork)
// integers dos improve records, preserving the loop's own ordering so the receipt's
// numbers agree with the loop's verdict. For a higher-better metric that is a direct
// round; for a lower-better one it subtracts each from the pair's max (a constant-minus,
// NOT a negation) so the recorded values stay non-negative while work > baselineWork iff
// the loop measured an improvement. Both shipped harnesses are higher-better today; the
// lower-better branch keeps the receipt correct if one is ever wired.
func scaleWorkPair(candidate, baseline float64, lowerBetter bool) (work, baselineWork int64) {
	if lowerBetter {
		c := math.Max(candidate, baseline)
		return roundUnit(c - candidate), roundUnit(c - baseline)
	}
	return roundUnit(candidate), roundUnit(baseline)
}

// dosImproveArgs builds the argv for one `dos improve --observe` receipt from a journal
// row. Every field is environment-authored (measured by the loop, not narrated by it);
// --narrated carries the loop's own verdict verbatim for the operator and is parsed for
// nothing. Pure and deterministic, so it is unit-tested without spawning dos.
func dosImproveArgs(workspace string, maxReverts int, r rsiloop.Row) []string {
	work, baselineWork := scaleWorkPair(r.Candidate_, r.Baseline, r.LowerBetter)
	args := []string{
		"improve", "--observe",
		"--work", strconv.FormatInt(work, 10),
		"--baseline-work", strconv.FormatInt(baselineWork, 10),
		"--consecutive-reverts", strconv.Itoa(r.BreakerCount),
		"--max-reverts", strconv.Itoa(maxReverts),
		"--lane", "rsiloop",
		"--subject", fmt.Sprintf("rsiloop::%s::cycle%d", r.MetricName, r.Cycle),
		"--narrated", fmt.Sprintf("rsiloop verdict=%s candidate=%q improved=%v measured=%v",
			r.Decision, r.Candidate, r.Improved, r.Measured),
	}
	if r.SuiteGreen {
		args = append(args, "--suite-passed")
	}
	if r.TruthClean {
		args = append(args, "--truth-clean")
	}
	if workspace != "" {
		args = append(args, "--workspace", workspace)
	}
	return args
}

// dosObserveReceipt returns an rsiloop.Observer that emits a `dos improve --observe`
// receipt per verdict, or nil (a no-op) when dos is not on PATH — the silent-degrade
// path for CI/bench nodes. maxReverts mirrors the loop's breaker threshold so the
// receipt's escalation context matches the loop's. The receipt's value is the journal
// row dos writes, so the command's own stdout/stderr is discarded; a verdict exit code
// is expected and ignored, and only a real spawn failure is surfaced (never silently
// swallowed).
func dosObserveReceipt(workspace string, maxReverts int) rsiloop.Observer {
	dosPath, err := exec.LookPath("dos")
	if err != nil {
		fmt.Fprintln(os.Stderr, "rsiloop -dos-observe: 'dos' not on PATH — emitting no receipts (loop unaffected)")
		return nil
	}
	return func(r rsiloop.Row) {
		cmd := exec.Command(dosPath, dosImproveArgs(workspace, maxReverts, r)...)
		cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
		if runErr := cmd.Run(); runErr != nil {
			var exitErr *exec.ExitError
			if !errors.As(runErr, &exitErr) {
				// Not a verdict exit code (3/4) — a genuine failure to run the tool.
				fmt.Fprintf(os.Stderr, "rsiloop -dos-observe: dos improve did not run for cycle %d: %v\n", r.Cycle, runErr)
			}
		}
	}
}
