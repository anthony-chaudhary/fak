package hooks

import (
	"regexp"
	"strings"
)

// gate_brandconsistency.go — the BRAND_CONSISTENCY gate, and since #6265 the ONLY
// implementation of the check (it began as a byte-faithful port of the now-retired
// the retired Python checker, whose golden vectors live on in the test beside it).
// It guards fak's PRIMARY product descriptor against re-drift: the durable brand keeps ONE
// primary noun ("the Fused Agent Kernel" / "agent kernel") and retires "agent tool firewall"
// / "tool-call policy gateway" as PRIMARY descriptors. A retired phrase is still ALLOWED as a
// synonym-list / "also described as" / named-asset reference — so a line is flagged ONLY when
// it uses a retired phrase as the primary noun for fak (a copula "fak is an agent tool
// firewall" or a "fak — X" banner) AND carries no legitimate-use marker. See issue #591 (this
// guard) / #589 (the brand epic).
//
// This is a TREE-mode-ONLY gate: the retired Python checker exposed only `--audit-tree` (it
// had no `--audit-staged` branch and was never wired into the pre-commit hook), so there is no
// staged Gate twin — only the HygieneGate below, run by `fak hygiene` / `make hygiene`, which
// is what ci.yml and scripts/ci.ps1 now invoke in the Python checker's place.

// brandPrimaryRE: fak declared TO BE a retired descriptor — "fak is a/an/the X", or a
// title/banner "fak — X" / "fak - X" / "fak : X" (article optional there). Inherited from
// the retired Python checker PRIMARY_RE (RETIRED inlined), case-insensitive. The retired
// alternation matches both "tool-call" and "tool call". The char class is {em-dash, ':',
// '-'} (trailing '-' is literal); '\-' is the unambiguous spelling of the same set in RE2.
var brandPrimaryRE = regexp.MustCompile(
	`(?i)\bfak\b[^.\n]{0,40}?(?:\bis\s+(?:an?|the)\s+|\s+[—:\-]\s+(?:the\s+)?)` +
		`(?:agent tool firewall|tool[- ]call policy gateway)`,
)

// brandAllowMarkersRE: markers that make a retired descriptor a LEGIT secondary use, not a
// primary claim (synonym list, "also described as", the named video/poster asset). Inherited
// from the retired Python checker ALLOW_MARKERS.
var brandAllowMarkersRE = regexp.MustCompile(
	`(?i)also described as|alternatename|keywords?|topics?|category|aria-label|` +
		`\balt\b|explainer|reveal|\bcard\b|poster|\.mp4|\.gif|\.svg|agent-firewall|firewall card`,
)

// brandExemptPrefixes / brandExemptFiles / brandScanExts inherit EXEMPT_PREFIXES,
// EXEMPT_FILES and SCAN_EXT: generated corpus + visual assets + the metadata generators are
// skipped wholesale, and only reader-facing text surfaces are scanned. The extension test is
// a case-sensitive suffix match, exactly like the Python `rel.endswith(SCAN_EXT)`.
var brandExemptPrefixes = []string{"visuals/"}

var brandExemptFiles = map[string]bool{
	"llms-full.txt":                true, // generated; mirrors source on regen
	"tools/gen_structured_data.py": true, // emits alternateName/keywords lists
	// This gate and its test carry synthetic retired-descriptor samples (the regex source +
	// the golden vectors inherited from the retired Python oracle), so they are exempt — else
	// the gate would flag its own fixtures on the tree.
	"internal/hooks/gate_brandconsistency.go":      true, // this gate's own regex source
	"internal/hooks/gate_brandconsistency_test.go": true, // the golden vectors
}

var brandScanExts = []string{".md", ".txt", ".go", ".html", ".cff"}

// brandLineViolates ports the per-line decision (audit L84): a primary-descriptor match with
// no legitimate-use marker on the same line.
func brandLineViolates(line string) bool {
	return brandPrimaryRE.MatchString(line) && !brandAllowMarkersRE.MatchString(line)
}

// brandScanned reports whether a tracked path is in scope: not exempt, and a scanned text
// extension. Ports the audit() file filter (L74-76).
func brandScanned(rel string) bool {
	if brandExemptFiles[rel] || startsWithAny(rel, brandExemptPrefixes) {
		return false
	}
	for _, ext := range brandScanExts {
		if strings.HasSuffix(rel, ext) {
			return true
		}
	}
	return false
}

// gateBrandConsistencyTree is the --audit-tree gate: scan every in-scope tracked file
// line-by-line and flag each primary-descriptor violation. Inherits the retired Python checker
// audit(): readlines() enumerated from 1, the same per-line decision. t.Paths is sorted, so
// per-file order is stable.
func gateBrandConsistencyTree(t *TrackedTree) ([]Finding, error) {
	return scanTreeFileLines(t, t.Paths, "BRAND_CONSISTENCY", brandScanned, func(line string) (string, bool) {
		if brandLineViolates(line) {
			return trim160(line), true
		}
		return "", false
	}), nil
}
