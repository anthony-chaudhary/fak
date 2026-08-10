package devcmd

import (
	"errors"
	"os"
	"path/filepath"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

const mainGoFile = "cmd/fak/main.go"
const reasonVerbUntiered = "VERB_UNTIERED"

func verbTierFindings(root string) ([]indexPolicyFinding, error) {
	body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(mainGoFile)))
	if err != nil {
		return nil, err
	}
	verbs := devindex.DispatchVerbs(body)
	if len(verbs) == 0 {
		return nil, errors.New("runtime dispatch parser returned no verbs")
	}
	var findings []indexPolicyFinding
	for _, verb := range verbs {
		if _, ok := devindex.TierOf(verb); ok {
			continue
		}
		findings = append(findings, indexPolicyFinding{
			Reason: reasonVerbUntiered,
			File:   mainGoFile,
			Detail: "dispatched verb " + verb + " has no tier - classify it in internal/devindex/tiers.go",
		})
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].Detail < findings[j].Detail })
	return findings, nil
}
