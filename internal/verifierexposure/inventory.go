package verifierexposure

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DeclaredInventory is the reviewed seed set. Sources are checked on every run:
// the declaration cannot silently outlive the gate it purports to measure.
func DeclaredInventory() []Gate {
	return []Gate{
		{Name: "dos-commit-audit", Kind: Deterministic, Sources: []string{"dos.toml"}, CheckerBytesPinned: true, SchemaPinned: true, IndependentlyRemeasured: true},
		{Name: "dos-verify", Kind: Deterministic, Sources: []string{"dos.toml"}, CheckerBytesPinned: true, SchemaPinned: true, IndependentlyRemeasured: true},
		{Name: "witness-rungs-w1-w3", Kind: Deterministic, Sources: []string{"internal/witness/witness.go"}, SignalProbes: []SignalProbe{{Path: "internal/witness/witness.go", Contains: "WitnessConfirmed"}}, CheckerBytesPinned: true, SchemaPinned: true, IndependentlyRemeasured: true},
		{Name: "trajctl-judge", Kind: LLMJudge, Sources: []string{"internal/trajctl/judgeclient.go", "internal/trajctl/rubric.go"}, SignalProbes: []SignalProbe{{Path: "internal/trajctl/judgeclient.go", Contains: "Temperature: 0"}}, CheckerBytesPinned: true, SchemaPinned: true, TemperatureZero: true},
		{Name: "policy-smart-approval", Kind: Deterministic, Sources: []string{"internal/egressfloor/approval.go"}, SignalProbes: []SignalProbe{{Path: "internal/egressfloor/approval.go", Contains: "AdjudicateApproval"}}, SchemaPinned: true, IndependentlyRemeasured: true},
		{Name: "safecommit", Kind: Deterministic, Sources: []string{"internal/safecommit/safecommit.go"}, SignalProbes: []SignalProbe{{Path: "internal/safecommit/safecommit.go", Contains: "PATHSPEC_RACE"}}, SchemaPinned: true},
		{Name: "antipattern", Kind: Deterministic, Sources: []string{"internal/antipattern/antipattern.go"}, SignalProbes: []SignalProbe{{Path: "internal/antipattern/antipattern.go", Contains: "fak-antipattern-scorecard/1"}}, SchemaPinned: true},
		{Name: "ship-integrity", Kind: Deterministic, Sources: []string{"internal/shipgate/shipgate.go"}, SchemaPinned: true, IndependentlyRemeasured: true},
		{Name: "kpi-tests", Kind: SelfReport, Sources: []string{"internal/antipattern/checker_games_test.go"}, CheckerBytesPinned: false},
	}
}

// Gather verifies that each declared gate still has a tracked source-shaped file.
// Missing declarations fail closed through Report.InventoryErrors.
func Gather(root string) Report {
	gates := DeclaredInventory()
	var errs []string
	for _, g := range gates {
		for _, source := range g.Sources {
			info, err := os.Stat(filepath.Join(root, filepath.FromSlash(source)))
			if err != nil || info.IsDir() {
				errs = append(errs, fmt.Sprintf("%s: source %s unavailable", g.Name, source))
			}
		}
		for _, probe := range g.SignalProbes {
			data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(probe.Path)))
			if err != nil || !strings.Contains(string(data), probe.Contains) {
				errs = append(errs, fmt.Sprintf("%s: signal %q absent from %s", g.Name, probe.Contains, probe.Path))
			}
		}
	}
	return Fold(gates, errs)
}

func Markdown(r Report) string {
	var b strings.Builder
	fmt.Fprintln(&b, "# Verifier exposure scorecard")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Schema: `%s`  \n", r.Schema)
	fmt.Fprintf(&b, "Baseline: `verifier_exposure_debt = %d` of %d gates at threshold `%.2f` (grade `%s`).\n\n", r.VerifierExposureDebt, r.GateCount, r.DebtThreshold, r.Grade)
	fmt.Fprintln(&b, "| Rank | Gate | Kind | Exposure | Above debt floor |")
	fmt.Fprintln(&b, "|---:|---|---|---:|:---:|")
	for i, g := range r.Worklist {
		fmt.Fprintf(&b, "| %d | `%s` | `%s` | %.2f | %v |\n", i+1, g.Name, g.Kind, g.Exposure, g.Exposure >= r.DebtThreshold)
	}
	fmt.Fprintln(&b, "\nExposure is a declared-signal heuristic, not an empirical exploit probability. Higher is hardened first; missing inventory sources fail the grade closed.")
	return b.String()
}
