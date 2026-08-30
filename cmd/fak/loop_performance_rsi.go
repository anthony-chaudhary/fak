package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/perfrsiscore"
)

// reserveLoopPerformanceRSIOutput allocates a unique name beside the selected loop ledger,
// then removes the reservation before the child starts. The exact target is therefore absent
// at admission and any regular file found there after Wait was created during this run.
func reserveLoopPerformanceRSIOutput(ledger string) (string, error) {
	f, err := os.CreateTemp(filepath.Dir(ledger), ".fak-performance-rsi-*.json")
	if err != nil {
		return "", err
	}
	name := f.Name()
	if err := f.Close(); err != nil {
		_ = os.Remove(name)
		return "", err
	}
	if err := os.Remove(name); err != nil {
		return "", err
	}
	return name, nil
}

func appendLoopEnvAllow(current string, names ...string) string {
	seen := map[string]bool{}
	var out []string
	for _, name := range append(strings.Split(current, ","), names...) {
		name = strings.TrimSpace(name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return strings.Join(out, ",")
}

func scoreAutomaticLoopPerformanceRSI(input, runID string, prepErr error) perfrsiscore.LoopTurnReceipt {
	unavailable := func(diagnostic string) perfrsiscore.LoopTurnReceipt {
		return perfrsiscore.LoopTurnReceipt{
			Schema:                perfrsiscore.LoopTurnSchema,
			Status:                perfrsiscore.LoopTurnUnavailable,
			Reason:                "SCORE_INPUT_UNAVAILABLE",
			Input:                 input,
			UnavailableDiagnostic: diagnostic,
			InvocationOutcomes:    perfrsiscore.OutcomeCounts{Refusal: 1},
		}
	}
	if prepErr != nil {
		return unavailable("prepare run-scoped performance-RSI output: " + prepErr.Error())
	}
	info, err := os.Lstat(input)
	if err != nil {
		if os.IsNotExist(err) {
			return unavailable(loopPerformanceRSIOutputEnv + " was not produced for this run")
		}
		return unavailable("inspect run-scoped performance-RSI output: " + err.Error())
	}
	if !info.Mode().IsRegular() {
		return unavailable(loopPerformanceRSIOutputEnv + " is not a regular file")
	}
	if info.Size() == 0 {
		return unavailable(loopPerformanceRSIOutputEnv + " was unchanged for this run")
	}
	if info.Size() > loopPerformanceRSIMaxBytes {
		return unavailable(fmt.Sprintf("%s exceeds %d bytes", loopPerformanceRSIOutputEnv, loopPerformanceRSIMaxBytes))
	}

	receipt := perfrsiscore.ScoreLoopTurn(input)
	if receipt.Status == perfrsiscore.LoopTurnScored && receipt.Snapshot != runID {
		return unavailable(fmt.Sprintf("performance-RSI snapshot %q does not match loop run %q", receipt.Snapshot, runID))
	}
	return receipt
}

// writeLoopRunReport emits one `fak loop run --json` report. Every report -- the containment
// refusal's and the completed run's alike -- names the same four things: the report schema,
// the ledger it was recorded in, and the loop/run it describes. Those live here so a refusal
// can never publish a differently identified report than a run that finished; outcome carries
// only the fields that differ (status/reason, or exit code and duration). It returns false
// when the encode failed and the caller must exit 1 instead of reporting success.
func writeLoopRunReport(stdout, stderr io.Writer, ledger, loopID, runID string, outcome map[string]any) bool {
	rep := map[string]any{
		"schema":      "fak.loop-run-report.v1",
		"ledger_path": ledger,
		"loop_id":     loopID,
		"run_id":      runID,
	}
	for k, v := range outcome {
		rep[k] = v
	}
	if err := writeIndentedJSON(stdout, rep); err != nil {
		fmt.Fprintf(stderr, "fak loop run: encode json: %v\n", err)
		return false
	}
	return true
}
