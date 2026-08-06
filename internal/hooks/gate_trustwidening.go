package hooks

import (
	"path/filepath"
	"strings"
)

// gateTrustWidening is the polarity-aware, self-referential review seam for the
// permission surfaces that govern fak and its host agent. Path-only self-modify
// gates intentionally treat tightening and widening alike; this gate adds the
// missing semantic half by warning only when the staged diff adds a grant.
//
// This is advisory by default. A human can confirm that the broader capability
// is intentional without turning harmless deny/removal edits into review noise.
func gateTrustWidening(d *StagedDiff) ([]Finding, error) {
	var findings []Finding
	// The filter here is TRUST-CONFIG files, a narrow slice of any staged set. A commit that
	// touches none admits zero — the state this gate could not previously report (#5602).
	judged := 0
	for file, lines := range d.AddedByFile {
		if !trustConfigPath(file) {
			continue
		}
		judged += len(lines)
		content, exists := d.FileBytes(file)
		for _, line := range lines {
			if !addedTrustGrant(line.Text, content, exists, line.New) {
				continue
			}
			findings = append(findings, Finding{
				Gate:   "TRUST_WIDENING",
				File:   file,
				Line:   line.New,
				Detail: "ESCALATE: staged trust configuration adds a permission grant; confirm the wider capability is intentional: " + strings.TrimSpace(line.Text),
			})
		}
	}
	d.NoteCandidates("TRUST_WIDENING", judged, "added line(s) in trust-config file(s)")
	return findings, nil
}

func trustConfigPath(path string) bool {
	path = filepath.ToSlash(strings.TrimSpace(path))
	if path == ".claude/settings.json" || path == "cmd/fak/guard-default-policy.json" {
		return true
	}
	base := strings.ToLower(filepath.Base(path))
	return base == "policy.json" || strings.HasSuffix(base, "-policy.json")
}

func addedTrustGrant(line string, content []byte, exists bool, lineNumber int) bool {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "//") {
		return false
	}
	lower := strings.ToLower(line)

	// Compact one-line additions carry both the key and one or more newly added
	// values, so they are grants rather than section markers.
	for _, marker := range []string{`"allow": [`, `"allowedtools": [`, `"self_modify_globs": [`} {
		if strings.Contains(lower, marker) && !strings.HasSuffix(lower, "[") {
			return true
		}
	}

	// A bare JSON array member is a widening only when its enclosing array is a
	// known grant field. Looking at the resulting file prevents a newly added deny
	// regex or unrelated policy string from being mislabeled as a grant.
	if !(strings.HasPrefix(line, `"`) && (strings.HasSuffix(line, `",`) || strings.HasSuffix(line, `"`))) {
		return false
	}
	if !exists || lineNumber <= 0 {
		return false
	}
	return trustGrantSectionAtLine(content, lineNumber)
}

func trustGrantSectionAtLine(content []byte, lineNumber int) bool {
	lines := strings.Split(strings.ReplaceAll(string(content), "\r\n", "\n"), "\n")
	if lineNumber > len(lines) {
		return false
	}
	for i := lineNumber - 2; i >= 0; i-- {
		line := strings.TrimSpace(strings.ToLower(lines[i]))
		if strings.HasPrefix(line, "]") {
			return false
		}
		for _, key := range []string{`"allow"`, `"allowedtools"`, `"self_modify_globs"`} {
			if strings.HasPrefix(line, key) && strings.Contains(line, "[") {
				return true
			}
		}
	}
	return false
}
