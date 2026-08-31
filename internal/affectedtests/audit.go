package affectedtests

import "sort"

// PackageObservation records whether a package's test run observed a failure.
type PackageObservation struct {
	Package string `json:"package"`
	Failed  bool   `json:"failed"`
}

// TestObservation is one selected or full-truth test-run observation.
type TestObservation struct {
	Complete bool                 `json:"complete"`
	Packages []PackageObservation `json:"packages"`
}

// SelectionAudit compares the failures seen by selected tests with full-suite truth.
type SelectionAudit struct {
	Complete         bool     `json:"complete"`
	Sound            bool     `json:"sound"`
	SelectedFailures []string `json:"selected_failures"`
	FullFailures     []string `json:"full_failures"`
	SelectorMisses   []string `json:"selector_misses"`
}

// AuditSelection reports full-truth failures that the selected run did not observe.
// An incomplete selected or truth observation fails closed, even when no miss is visible.
func AuditSelection(selected, full TestObservation) SelectionAudit {
	selectedFailures := failedPackages(selected.Packages)
	fullFailures := failedPackages(full.Packages)
	selectedSet := make(map[string]struct{}, len(selectedFailures))
	for _, pkg := range selectedFailures {
		selectedSet[pkg] = struct{}{}
	}

	misses := make([]string, 0)
	for _, pkg := range fullFailures {
		if _, ok := selectedSet[pkg]; !ok {
			misses = append(misses, pkg)
		}
	}

	complete := selected.Complete && full.Complete
	return SelectionAudit{
		Complete:         complete,
		Sound:            complete && len(misses) == 0,
		SelectedFailures: selectedFailures,
		FullFailures:     fullFailures,
		SelectorMisses:   misses,
	}
}

func failedPackages(observations []PackageObservation) []string {
	failures := make(map[string]struct{})
	for _, observation := range observations {
		if observation.Failed {
			failures[observation.Package] = struct{}{}
		}
	}

	packages := make([]string, 0, len(failures))
	for pkg := range failures {
		packages = append(packages, pkg)
	}
	sort.Strings(packages)
	return packages
}
