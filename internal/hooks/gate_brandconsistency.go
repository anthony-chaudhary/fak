package hooks

import (
	"regexp"
	"strings"
)

// gate_brandconsistency.go — the BRAND_CONSISTENCY gate, a byte-faithful port of
// tools/check_brand_consistency.py. It guards fak's PRIMARY product descriptor against
// re-drift: the durable brand keeps ONE primary noun ("the Fused Agent Kernel" / "agent
// kernel") and retires "agent tool firewall" / "tool-call policy gateway" as PRIMARY
// descriptors. A retired phrase is still ALLOWED as a synonym-list / "also described as"
// / named-asset reference — so a line is flagged ONLY when it uses a retired phrase as the
// primary noun for fak (a copula "fak is an agent tool firewall" or a "fak — X" banner)
// AND carries no legitimate-use marker. See issue #591 (this guard) / #589 (the brand epic).
//
// This is a TREE-mode-ONLY gate: the Python checker exposes only `--audit-tree` (it has no
// `--audit-staged` branch and is not wired into the pre-commit hook), so there is no staged
// Gate twin — only the HygieneGate below, run by `fak hygiene` / `make hygiene`.

// brandPrimaryRE: fak declared TO BE a retired descriptor — "fak is a/an/the X", or a
// title/banner "fak — X" / "fak - X" / "fak: X" (article optional there). Mirrors
// check_brand_consistency.py PRIMARY_RE (RETIRED inlined), case-insensitive. The retired
// alternation matches both "tool-call" and "tool call". The char class is {em-dash, ':',
// '-'} (trailing '-' is literal); '\-' is the unambiguous spelling of the same set in RE2.
var brandPrimaryRE = regexp.MustCompile(
	`(?i)\bfak\b[^.\n]{0,40}?(?:\bis\s+(?:an?|the)\s+|\s+[—:\-]\s+(?:the\s+)?)` +
		`(?:agent tool firewall|tool[- ]call policy gateway)`,
)

// brandAllowMarkersRE: markers that make a retired descriptor a LEGIT secondary use, not a
// primary claim (synonym list, "also described as", the named video/poster asset). Mirrors
// check_brand_consistency.py ALLOW_MARKERS.
var brandAllowMarkersRE = regexp.MustCompile(
	`(?i)also described as|alternatename|keywords?|topics?|category|aria-label|` +
		`\balt\b|explainer|reveal|\bcard\b|poster|\.mp4|\.gif|\.svg|agent-firewall|firewall card`,
)

// brandExemptPrefixes / brandExemptFiles / brandScanExts mirror EXEMPT_PREFIXES,
// EXEMPT_FILES and SCAN_EXT: generated corpus + visual assets + the metadata generators are
// skipped wholesale, and only reader-facing text surfaces are scanned. The extension test is
// a case-sensitive suffix match, exactly like the Python `rel.endswith(SCAN_EXT)`.
var brandExemptPrefixes = []string{"visuals/"}

var brandExemptFiles = map[string]bool{
	"llms-full.txt":                         true, // generated; mirrors source on regen
	"tools/check_brand_consistency.py":      true, // the Python oracle's own docstring examples
	"tools/check_brand_consistency_test.py": true, // the Python oracle's synthetic samples
	"tools/gen_structured_data.py":          true, // emits alternateName/keywords lists
	// The Go twin and its parity test carry the SAME synthetic retired-descriptor samples
	// (the regex source + the replayed golden vectors), so they are exempt for the same reason
	// the Python checker + test are — else the gate would flag its own fixtures on the tree.
	"internal/hooks/gate_brandconsistency.go":      true, // this gate's own regex source
	"internal/hooks/gate_brandconsistency_test.go": true, // the Go parity test's golden vectors
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
// line-by-line and flag each primary-descriptor violation. Mirrors check_brand_consistency.py
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
