package hooks

import (
	"regexp"
	"strings"
)

// gate_docplacement.go — the DOC_PLACEMENT gate, a port of tools/check_doc_placement.py. Two
// rules: (1) a root-level *.md must be in the front-door allowlist; (2) a newly-added dated
// note under docs/notes/ must be listed in INDEX.md.

// allowedRootMD is the front-door allowlist (check_doc_placement.py), kept in sync with the
// Python fallback.
var allowedRootMD = map[string]bool{
	"README.md": true, "START-HERE.md": true, "INSTALL.md": true, "INDEX.md": true, "LEARNING-PATH.md": true,
	"CONTRIBUTING.md": true, "CLA.md": true, "AGENTS.md": true, "AGENT.md": true, "CLAUDE.md": true, "GEMINI.md": true, "GOAL.md": true,
	"SECURITY.md": true, "PUBLIC-SCRUB-POLICY.md": true,
	"ARCHITECTURE.md": true, "EXTENDING.md": true, "GETTING-STARTED.md": true, "GPU.md": true,
	"POLICY.md": true, "PARTITION.md": true, "STATUS.md": true, "CLAIMS.md": true,
	"SOTA-COMPARISON.md": true, "DOGFOOD-CLAUDE.md": true, "SUBSYSTEM-CHECKS.md": true,
	"BENCHMARK-AUTHORITY.md": true, "BENCHMARK-GALLERY.md": true,
	"BENCHMARK-GOVERNANCE.md": true, "BENCHMARK-TEMPLATE.md": true,
	"HERO-BENCHMARK-2026-06-21.md": true,
	"CODE_OF_CONDUCT.md":           true, "CHANGELOG.md": true, "GOVERNANCE.md": true, "MAINTAINERS.md": true,
	"ROADMAP.md": true, "AUTHORS.md": true, "NOTICE.md": true, "SUPPORT.md": true, "HISTORY.md": true,
	"TRADEMARK.md": true, "LICENSING.md": true,
}

// isAllowedRootMD returns true if the root markdown file is allowlisted or is a unique goal file (GOAL-*.md).
func isAllowedRootMD(name string) bool {
	return allowedRootMD[name] || (strings.HasPrefix(name, "GOAL-") && strings.HasSuffix(name, ".md"))
}

// datedRE matches a YYYY-MM-DD date anywhere in a note basename (check_doc_placement.py L106).
var datedRE = regexp.MustCompile(`20\d\d-\d\d-\d\d`)

func gateDocPlacement(d *StagedDiff) ([]Finding, error) {
	// Rule 1: root *.md not in the allowlist (over the touched ACMR set). Shared with the tree twin.
	findings := rootMDPlacementFindings(d.StagedPaths)

	// This gate judges MARKDOWN, not the whole staged set: root-level .md under rule 1 plus
	// newly-added docs/notes/ notes under rule 2. Counting that union is what distinguishes a
	// commit of forty docs from a commit of forty .go files, which today report identically (#5602).
	judged := 0
	for _, p := range d.StagedPaths {
		norm := strings.ReplaceAll(p, "\\", "/")
		if strings.HasSuffix(norm, ".md") && !strings.Contains(norm, "/") {
			judged++
		}
	}

	// Rule 2: a newly-added dated docs/notes/ note must be listed in INDEX.md.
	index, _ := d.IndexMD() // missing INDEX.md => "" => everything reads as unlisted, matching the Python ([] on OSError) only when index exists
	if index != "" {
		for _, p := range d.AddedPaths {
			if !strings.HasPrefix(p, "docs/notes/") || !strings.HasSuffix(p, ".md") {
				continue
			}
			base := baseName(p)
			if base == "README.md" {
				continue
			}
			judged++
			isDated := datedRE.MatchString(base) || strings.HasPrefix(base, "PLAN-")
			if isDated && !strings.Contains(index, base) {
				findings = append(findings, Finding{
					Gate: "DOC_PLACEMENT", File: p,
					Detail: "new docs/notes/ note not listed in INDEX.md: " + p + "  —  add a one-line entry to INDEX.md",
				})
			}
		}
	}
	d.NoteCandidates("DOC_PLACEMENT", judged, "staged markdown doc(s)")
	return findings, nil
}

func baseName(p string) string {
	if i := strings.LastIndex(p, "/"); i >= 0 {
		return p[i+1:]
	}
	return p
}
