package hooks

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/launchvisual"
)

const (
	launchNumbersDocPrefix = "docs/launch/"
	launchNumbersData      = launchvisual.DefaultData
	launchNumbersIndex     = launchvisual.DefaultIndex
	launchNumbersCure      = "go run ./cmd/fak launch-visual"
)

// launchNumbersTriggered reports whether a staged path can change a launch number: anything
// under docs/launch/, the figure data file, or the published headline index.
func launchNumbersTriggered(staged string) bool {
	p := strings.TrimPrefix(staged, "./")
	return strings.HasPrefix(p, launchNumbersDocPrefix) || p == launchNumbersData || p == launchNumbersIndex
}

func checkLaunchNumbers(d *StagedDiff) ([]Finding, error) {
	relevant := 0
	for _, p := range d.StagedPaths {
		if launchNumbersTriggered(p) {
			relevant++
		}
	}
	d.NoteCandidates("LAUNCH_NUMBERS_FRESHNESS", relevant, "staged launch-number path(s)")
	if relevant == 0 {
		return nil, nil
	}
	read, tree, err := launchvisual.GitTreeReader(d.Root, "")
	if err != nil {
		return nil, fmt.Errorf("launch numbers freshness: %w", err)
	}
	return launchNumbersFindings(read, tree), nil
}

// launchNumbersFindings turns a verification of the tree read through read into findings.
// The tree is named in each detail so a refusal can be reproduced against the same tree.
func launchNumbersFindings(read launchvisual.ReadFunc, tree string) []Finding {
	rep, err := launchvisual.Verify(read, launchNumbersData, launchNumbersIndex)
	if err != nil {
		return []Finding{{Gate: "LAUNCH_NUMBERS_FRESHNESS", File: launchNumbersData,
			Detail: fmt.Sprintf("launch numbers cannot be verified against tree %s: %v", tree, err)}}
	}
	var out []Finding
	for _, rel := range rep.Drift {
		out = append(out, Finding{Gate: "LAUNCH_NUMBERS_FRESHNESS", File: rel,
			Detail: fmt.Sprintf("launch SVG is stale against the index scored in tree %s; run `%s` and stage the result -- this gate scores the STAGED tree, not the worktree", tree, launchNumbersCure)})
	}
	for _, v := range rep.Violations {
		out = append(out, Finding{Gate: "LAUNCH_NUMBERS_FRESHNESS", File: v.File, Line: v.Line,
			Detail: fmt.Sprintf("%s does not trace to a headline row (%s); cite a row's display value, a non-null ratio as N.NNx, or add the token to allowed_numbers in %s", v.Token, v.Reason, launchNumbersData)})
	}
	return out
}
