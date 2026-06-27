package hooks

import (
	"regexp"
	"strings"
)

// gate_provenance.go — the PROVENANCE_LABEL gate, a port of tools/check_provenance_labels.py.
// It refuses an added line that labels a known-MODELED number "measured": the line must mention
// "measured", NOT match any ALLOW_PATTERNS carve-out, and match a MODELED family's context AND
// number. It scans added lines of staged *.md/*.html/*.txt/*.json (minus skip prefixes/basenames).

// modeledClaim is the single MODELED family (check_provenance_labels.py L48-59).
type modeledClaim struct {
	context []*regexp.Regexp
	numbers []*regexp.Regexp
	fix     string
}

var modeledClaims = []modeledClaim{
	{
		context: mustAll(`webvoyager`, `webbench`, `643[\s-]*task`, `643\s*\)`),
		numbers: mustAll(`9\.7\s*(?:[xX]|\x{00D7})`, `8\.8\s*(?:[xX]|\x{00D7})`, `8\.8\s*(?:\x{2013}|-)\s*9\.7`),
		fix:     `call it "modeled" (closed-form geometry, no wall-clock); the 9.7x is the A/C ratio vs the naive re-prefill floor`,
	},
}

// allowPatterns — the carve-outs (check_provenance_labels.py L66-86), all case-insensitive.
var allowPatterns = mustAllI(
	`\bmodeled\b`,
	`measured\s*\*{0,2}\s*4\.1`,
	`/\s*measured\s+4\.1`,
	`not\s+(?:a\s+)?(?:wall-clock\s+)?measured`,
	`not\s+a\s+wall-clock\s+measurement`,
	`not\s+['"]?measured`,
	`mislabel`,
	`false\s+['"]?measured`,
	`fuses\s+two\s+unrelated`,
	`naive\s+arm\s+is\s+modeled\s+from\s+the\s+measured`,
	`from\s+['"]?measured['"]?\s+to\s+modeled`,
	`"measured"\s*->\s*`,
	`end-to-end\s+measured`,
)

var measuredRE = regexp.MustCompile(`(?i)\bmeasured\b`)

var provenanceSkipPrefixes = []string{"docs/releases/", "vendor/", "node_modules/"}
var provenanceSkipBasenames = map[string]bool{"llms-full.txt": true, "check_provenance_labels.py": true}
var provenanceScanExts = map[string]bool{".md": true, ".html": true, ".txt": true, ".json": true}

func gateProvenanceLabel(d *StagedDiff) ([]Finding, error) {
	var findings []Finding
	for _, f := range d.sortedFiles() {
		if startsWithAny(f, provenanceSkipPrefixes) || provenanceSkipBasenames[baseName(f)] {
			continue
		}
		if !provenanceScanExts[lowerExt(f)] {
			continue
		}
		for _, al := range d.AddedByFile[f] {
			if fix, bad := lineViolates(al.Text); bad {
				findings = append(findings, Finding{
					Gate: "PROVENANCE_LABEL", File: f, Line: al.New,
					Detail: trim160(al.Text) + " — fix: " + fix,
				})
			}
		}
	}
	return findings, nil
}

// lineViolates ports _line_violates (L137-148): measured present, no allow-pattern, and a family
// whose context AND number both match the lowercased line.
func lineViolates(line string) (string, bool) {
	if !measuredRE.MatchString(line) {
		return "", false
	}
	for _, ap := range allowPatterns {
		if ap.MatchString(line) {
			return "", false
		}
	}
	low := strings.ToLower(line)
	for _, fam := range modeledClaims {
		if anyMatch(fam.context, low) && anyMatch(fam.numbers, low) {
			return fam.fix, true
		}
	}
	return "", false
}

func anyMatch(res []*regexp.Regexp, s string) bool {
	for _, re := range res {
		if re.MatchString(s) {
			return true
		}
	}
	return false
}

func mustAll(pats ...string) []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(pats))
	for i, p := range pats {
		out[i] = regexp.MustCompile(p)
	}
	return out
}

func mustAllI(pats ...string) []*regexp.Regexp {
	ci := make([]string, len(pats))
	for i, p := range pats {
		ci[i] = `(?i)` + p
	}
	return mustAll(ci...)
}

func trim160(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 160 {
		return s[:160]
	}
	return s
}
