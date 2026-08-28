package claimcheck

// ledger.go — the CLAIMS.md exposure lint (#6218).
//
// CLAIMS.md is the honesty ledger, and its three tags ([SHIPPED]/[SIMULATED]/
// [STUB]) answer exactly one question: "is it real". They do not answer the
// orthogonal one — "is it ON for an operator who set no env var". Q6 of the
// net-true-value standard is that second axis, and this package has modeled it as
// a typed value since #1171 (`Realized`: on by default, or honestly gated WITH a
// stated reason). Until this file the two halves were never connected: the lint
// was a one-line awk tag counter, so a [SHIPPED] line reading "reproducible now"
// could mean "reproducible now, once you know which flag to set" and nothing
// mechanical could tell the difference.
//
// The exposure state is declared inline, at the END of a capability line:
//
//	… Witness: `go test ./internal/headroom`. [exposure: gated — DEFAULT-OFF; the build compresses nothing until FAK_COMPRESSOR=native selects it]
//	… Witness: `go test ./internal/policy`.   [exposure: default-on]
//
// The mapping to Q6 is direct and deliberately NOT a parallel vocabulary:
// `default-on` is `Realized{OnByDefault: true}`, `gated — R` is
// `Realized{GateReason: R}`, and the pass/fail rule is `gradeRealized` verbatim —
// a gated line with no stated reason is the seam that fails Q6.
//
// Exposure is total over the ledger, not a per-line opt-in: a [SIMULATED] or
// [STUB] line is PARKED by its tag (the same routing `tools/claims_salience_
// register.py` already performs), and a [SHIPPED] line with no marker asserts
// default-on. The absence of a marker is therefore a claim, not silence — which
// is why a [SHIPPED] line whose own prose discloses gating ("opt-in",
// "off by default", …) must declare its exposure explicitly instead of resting on
// that default. That rule is what keeps the ledger honest as new default-off work
// lands, and what makes "how many claimed capabilities are on for a default
// operator" one command instead of 174 prose judgments.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// The three honesty tags, in the order the ledger header defines them.
var ledgerTags = []string{"[SHIPPED]", "[SIMULATED]", "[STUB]"}

// exposureMarker opens the inline exposure declaration. It must be the last thing
// on the line, closed by the line's final "]".
const exposureMarker = "[exposure:"

// exposureDefaultOn is the marker value for a capability that is on for an
// operator who configured nothing.
const exposureDefaultOn = "default-on"

// exposureGated is the marker value for a capability that ships off; everything
// after it (past a — / - / : separator) is the stated gate reason.
const exposureGated = "gated"

// documentSetMarker identifies a bounded index whose local Markdown links are
// navigation and whose linked pages carry the authoritative claim records.
const documentSetMarker = "<!-- fak:document-set -->"

// gateProse is the narrow, false-positive-free set of phrases a capability line
// uses when it discloses gating in prose. Narrow by design (the same discipline as
// cache-headline-lint): every phrase here means "off unless you opt in" in every
// CLAIMS.md line that carries it — "by default" alone is excluded because it reads
// both ways ("owner/mechanism split by default", "PromotionWarn (the default)").
var gateProse = []string{
	"opt-in",
	"opt in",
	"off by default",
	"default off",
	"default-off",
	"disabled by default",
	"default-inert",
}

// Exposure is a capability line's answer to Q6, as declared in the ledger.
type Exposure string

const (
	// ExposureDefaultOn: on for an operator who configured nothing.
	ExposureDefaultOn Exposure = "default-on"
	// ExposureGated: ships off, with a stated reason (an honest gate).
	ExposureGated Exposure = "gated"
	// ExposureParked: [SIMULATED]/[STUB] — off the critical path by tag, so the
	// tag already carries the exposure and no marker is required.
	ExposureParked Exposure = "parked"
)

// LedgerLine is one parsed CLAIMS.md capability line (a line beginning "- [").
type LedgerLine struct {
	Path     string   `json:"path"`     // repository-relative source page
	N        int      `json:"n"`        // 1-based line number in CLAIMS.md
	Tag      string   `json:"tag"`      // "[SHIPPED]" / "[SIMULATED]" / "[STUB]"; "" when the tag rule failed
	Exposure Exposure `json:"exposure"` // the state every capability line carries
	// Realized is the Q6 value a [SHIPPED] line declares (the zero value, which
	// fails Q6, only ever reaches a violation).
	Realized Realized `json:"realized"`
	// Declared reports whether the state came from an explicit marker rather than
	// from the default-on / parked contract.
	Declared bool `json:"declared"`
	// Text is the line, trimmed of trailing space.
	Text string `json:"text"`
}

// Violation is one ledger defect: the rule it broke, on which line, and why.
type LedgerViolation struct {
	Path   string `json:"path"`
	N      int    `json:"n"`
	Rule   string `json:"rule"`
	Detail string `json:"detail"`
	Text   string `json:"text"`
}

// The closed rule vocabulary. TAG is the awk one-liner this lint replaced; the
// EXPOSURE_* rules are what it could never check.
const (
	// RuleTag: a capability line carries exactly one of the three honesty tags.
	RuleTag = "TAG"
	// RuleExposureNoReason: `[exposure: gated]` with no stated reason — Q6's seam.
	RuleExposureNoReason = "EXPOSURE_NO_REASON"
	// RuleExposureUnknown: a marker value that is neither default-on nor gated.
	RuleExposureUnknown = "EXPOSURE_UNKNOWN"
	// RuleExposureDuplicate: more than one marker on one line.
	RuleExposureDuplicate = "EXPOSURE_DUPLICATE"
	// RuleExposurePlacement: a marker that does not close at the end of the line.
	RuleExposurePlacement = "EXPOSURE_PLACEMENT"
	// RuleExposureUndeclared: a [SHIPPED] line whose prose discloses gating but
	// which declares no exposure — the unstructured disclosure this lint exists
	// to end.
	RuleExposureUndeclared = "EXPOSURE_UNDECLARED"
	// RuleExposureOnParked: an exposure marker on a [SIMULATED]/[STUB] line, whose
	// tag already parks it.
	RuleExposureOnParked = "EXPOSURE_ON_PARKED"
	// RuleEmpty: the ledger parsed to zero capability lines (a broken read is not
	// a pass — the awk lint's `c==0` exit, kept).
	RuleEmpty = "EMPTY"
	// RuleDocumentSetLinkMalformed: an index row is not one local Markdown link.
	RuleDocumentSetLinkMalformed = "DOCUMENT_SET_LINK_MALFORMED"
	// RuleDocumentSetLinkMissing: an indexed claim page cannot be read.
	RuleDocumentSetLinkMissing = "DOCUMENT_SET_LINK_MISSING"
	// RuleDocumentSetLinkDuplicate: two index rows resolve to the same page.
	RuleDocumentSetLinkDuplicate = "DOCUMENT_SET_LINK_DUPLICATE"
	// RuleDocumentSetLinkEscape: an index target resolves outside the document-set root.
	RuleDocumentSetLinkEscape = "DOCUMENT_SET_LINK_ESCAPE"
)

// LedgerReport is the whole-ledger result: the per-state counts (the numbers that
// used to require grepping prose) and every violation.
type LedgerReport struct {
	Lines      []LedgerLine      `json:"-"`
	Capability int               `json:"capability_lines"`
	Shipped    int               `json:"shipped"`
	Simulated  int               `json:"simulated"`
	Stub       int               `json:"stub"`
	DefaultOn  int               `json:"default_on"`
	Gated      int               `json:"gated"`
	Parked     int               `json:"parked"`
	Declared   int               `json:"declared"`
	Violations []LedgerViolation `json:"violations,omitempty"`
}

// OK reports whether the ledger passes: no violations, over a non-empty ledger.
func (r LedgerReport) OK() bool { return len(r.Violations) == 0 }

// LintLedger parses a CLAIMS.md body and grades every capability line's tag and
// exposure state. It reads nothing and writes nothing — pure over src.
func LintLedger(src string) LedgerReport {
	return lintLedger("CLAIMS.md", src)
}

func lintLedger(path, src string) LedgerReport {
	var rep LedgerReport
	for i, raw := range strings.Split(src, "\n") {
		line := strings.TrimRight(raw, " \t\r")
		if !strings.HasPrefix(line, "- [") {
			continue
		}
		rep.Capability++
		l := LedgerLine{Path: path, N: i + 1, Text: line}

		// Rule TAG — the awk lint, ported verbatim: exactly one of the three.
		var tags []string
		for _, t := range ledgerTags {
			if strings.Contains(line, t) {
				tags = append(tags, t)
			}
		}
		if len(tags) != 1 {
			rep.violate(l, RuleTag, fmt.Sprintf("carries %d of the three honesty tags, want exactly 1", len(tags)))
		} else {
			l.Tag = tags[0]
		}
		switch l.Tag {
		case "[SHIPPED]":
			rep.Shipped++
		case "[SIMULATED]":
			rep.Simulated++
		case "[STUB]":
			rep.Stub++
		}

		body, marker, mrep := splitExposure(line)
		if mrep.rule != "" {
			rep.violate(l, mrep.rule, mrep.detail)
		}

		// A [SIMULATED]/[STUB] line is parked by its tag; a marker there would be a
		// second, competing authority.
		if l.Tag == "[SIMULATED]" || l.Tag == "[STUB]" {
			l.Exposure = ExposureParked
			rep.Parked++
			if marker != "" {
				rep.violate(l, RuleExposureOnParked, "the "+l.Tag+" tag already parks this claim off the critical path")
			}
			rep.Lines = append(rep.Lines, l)
			continue
		}

		// [SHIPPED] (or an untagged line, graded as one so its exposure is still
		// checked): the marker decides, and its absence asserts default-on.
		switch {
		case marker == "":
			l.Exposure = ExposureDefaultOn
			l.Realized = Realized{OnByDefault: true}
			if p := proseGate(body); p != "" && l.Tag == "[SHIPPED]" {
				rep.violate(l, RuleExposureUndeclared,
					"prose discloses gating ("+p+") but the line declares no exposure — add `[exposure: gated — <reason>]` (or `[exposure: default-on]` if the claimed capability is on)")
			}
		case marker == exposureDefaultOn:
			l.Exposure = ExposureDefaultOn
			l.Realized = Realized{OnByDefault: true}
			l.Declared = true
		case marker == exposureGated || strings.HasPrefix(marker, exposureGated+" ") ||
			strings.HasPrefix(marker, exposureGated+":") || strings.HasPrefix(marker, exposureGated+"—") ||
			strings.HasPrefix(marker, exposureGated+"-"):
			l.Exposure = ExposureGated
			l.Declared = true
			l.Realized = Realized{GateReason: strings.TrimSpace(strings.TrimLeft(strings.TrimPrefix(marker, exposureGated), " —-:"))}
			// The Q6 rule, reused rather than restated: an off-by-default value with
			// no stated reason is a seam, not a realized gain.
			if q := gradeRealized(l.Realized); !q.Pass {
				rep.violate(l, RuleExposureNoReason, q.Detail)
			}
		default:
			l.Exposure = ExposureDefaultOn
			rep.violate(l, RuleExposureUnknown,
				fmt.Sprintf("exposure %q is neither %q nor %q", marker, exposureDefaultOn, exposureGated))
		}

		switch l.Exposure {
		case ExposureGated:
			rep.Gated++
		default:
			rep.DefaultOn++
		}
		if l.Declared {
			rep.Declared++
		}
		rep.Lines = append(rep.Lines, l)
	}

	if rep.Capability == 0 {
		rep.Violations = append(rep.Violations, LedgerViolation{Path: path, N: 1, Rule: RuleEmpty, Detail: "no capability lines found — the ledger did not parse"})
	}
	sortLedgerViolations(rep.Violations)
	return rep
}

// LintLedgerFile grades a ledger on disk. A document-set marker makes the root
// file an index: each local linked page is resolved once and becomes the sole
// source of capability records. The same report feeds claims-lint and readiness.
func LintLedgerFile(path string) (LedgerReport, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return LedgerReport{}, err
	}
	if !strings.Contains(string(b), documentSetMarker) {
		return lintLedger(filepath.Base(path), string(b)), nil
	}
	return lintDocumentSet(path, string(b))
}

func (r *LedgerReport) violate(l LedgerLine, rule, detail string) {
	r.Violations = append(r.Violations, LedgerViolation{Path: l.Path, N: l.N, Rule: rule, Detail: detail, Text: l.Text})
}

func lintDocumentSet(indexPath, index string) (LedgerReport, error) {
	root, err := filepath.Abs(filepath.Dir(indexPath))
	if err != nil {
		return LedgerReport{}, err
	}
	indexAbs, err := filepath.Abs(indexPath)
	if err != nil {
		return LedgerReport{}, err
	}
	indexName := filepath.ToSlash(filepath.Base(indexPath))
	seen := map[string]int{}
	var rep LedgerReport
	linked := 0

	for i, raw := range strings.Split(index, "\n") {
		line := strings.TrimRight(raw, " \t\r")
		if !strings.HasPrefix(line, "- [") {
			continue
		}
		n := i + 1
		target, ok := documentSetLinkTarget(line)
		if !ok {
			rep.violateAt(indexName, n, RuleDocumentSetLinkMalformed, "document-set row must be one local Markdown link", line)
			continue
		}
		clean, abs, ok := containedDocumentSetTarget(root, target)
		if !ok || samePath(abs, indexAbs) {
			rep.violateAt(indexName, n, RuleDocumentSetLinkEscape, fmt.Sprintf("linked claim page %q must stay below the document-set root", target), line)
			continue
		}
		key := strings.ToLower(filepath.ToSlash(clean))
		if first, duplicate := seen[key]; duplicate {
			rep.violateAt(indexName, n, RuleDocumentSetLinkDuplicate, fmt.Sprintf("linked claim page %q already appears at %s:%d", filepath.ToSlash(clean), indexName, first), line)
			continue
		}
		seen[key] = n
		linked++

		resolved, err := filepath.EvalSymlinks(abs)
		if err != nil {
			rep.violateAt(indexName, n, RuleDocumentSetLinkMissing, fmt.Sprintf("cannot resolve linked claim page %q: %v", filepath.ToSlash(clean), err), line)
			continue
		}
		if !pathWithin(root, resolved) {
			rep.violateAt(indexName, n, RuleDocumentSetLinkEscape, fmt.Sprintf("linked claim page %q resolves outside the document-set root", filepath.ToSlash(clean)), line)
			continue
		}
		body, err := os.ReadFile(resolved)
		if err != nil {
			rep.violateAt(indexName, n, RuleDocumentSetLinkMissing, fmt.Sprintf("cannot read linked claim page %q: %v", filepath.ToSlash(clean), err), line)
			continue
		}
		rep.merge(lintLedger(filepath.ToSlash(clean), string(body)))
	}

	if linked == 0 && len(rep.Violations) == 0 {
		rep.violateAt(indexName, 1, RuleEmpty, "document-set index contains no linked claim pages", "")
	}
	sortLedgerViolations(rep.Violations)
	return rep, nil
}

func documentSetLinkTarget(line string) (string, bool) {
	open := strings.LastIndex(line, "](")
	if open < 3 {
		return "", false
	}
	rest := line[open+2:]
	close := strings.LastIndex(rest, ")")
	if close < 1 || strings.TrimSpace(rest[close+1:]) != "" {
		return "", false
	}
	target := strings.TrimSpace(rest[:close])
	if target == "" || strings.HasPrefix(target, "#") || strings.Contains(target, "://") || strings.HasPrefix(target, "//") {
		return "", false
	}
	return target, true
}

func containedDocumentSetTarget(root, target string) (clean, abs string, ok bool) {
	fromSlash := filepath.FromSlash(target)
	if filepath.IsAbs(fromSlash) || filepath.VolumeName(fromSlash) != "" {
		return "", "", false
	}
	clean = filepath.Clean(fromSlash)
	abs = filepath.Join(root, clean)
	return clean, abs, pathWithin(root, abs)
}

func pathWithin(root, path string) bool {
	rel, err := filepath.Rel(root, path)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func samePath(a, b string) bool {
	return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
}

func (r *LedgerReport) merge(child LedgerReport) {
	r.Lines = append(r.Lines, child.Lines...)
	r.Capability += child.Capability
	r.Shipped += child.Shipped
	r.Simulated += child.Simulated
	r.Stub += child.Stub
	r.DefaultOn += child.DefaultOn
	r.Gated += child.Gated
	r.Parked += child.Parked
	r.Declared += child.Declared
	r.Violations = append(r.Violations, child.Violations...)
}

func (r *LedgerReport) violateAt(path string, n int, rule, detail, text string) {
	r.Violations = append(r.Violations, LedgerViolation{Path: path, N: n, Rule: rule, Detail: detail, Text: text})
}

func sortLedgerViolations(violations []LedgerViolation) {
	sort.SliceStable(violations, func(i, j int) bool {
		if violations[i].Path != violations[j].Path {
			return violations[i].Path < violations[j].Path
		}
		return violations[i].N < violations[j].N
	})
}

// markerReport carries a marker-shape defect out of splitExposure.
type markerReport struct{ rule, detail string }

// splitExposure separates a capability line's prose from its exposure marker. It
// returns the prose body, the marker's inner value (lower-cased key, original-case
// reason), and any marker-shape defect. A line with no marker returns ("", body).
func splitExposure(line string) (body, marker string, rep markerReport) {
	i := strings.Index(line, exposureMarker)
	if i < 0 {
		return line, "", rep
	}
	if strings.Contains(line[i+len(exposureMarker):], exposureMarker) {
		return line[:i], "", markerReport{RuleExposureDuplicate, "more than one `" + exposureMarker + "` marker on the line"}
	}
	if !strings.HasSuffix(line, "]") {
		return line[:i], "", markerReport{RuleExposurePlacement, "the `" + exposureMarker + "` marker must close at the end of the line"}
	}
	return line[:i], strings.TrimSpace(line[i+len(exposureMarker) : len(line)-1]), rep
}

// proseGate returns the first gate-disclosing phrase in body, or "" if none.
func proseGate(body string) string {
	low := strings.ToLower(body)
	for _, p := range gateProse {
		if strings.Contains(low, p) {
			return p
		}
	}
	return ""
}

// String renders the report as the one operator-readable line `make claims-lint`
// prints — the default-off count emitted, never grepped.
func (r LedgerReport) String() string {
	var b strings.Builder
	for _, v := range r.Violations {
		path := v.Path
		if path == "" {
			path = "CLAIMS.md"
		}
		fmt.Fprintf(&b, "VIOLATION %s (%s:%d): %s\n", v.Rule, path, v.N, v.Detail)
		if v.Text != "" {
			fmt.Fprintf(&b, "  %s\n", truncate(v.Text, 160))
		}
	}
	fmt.Fprintf(&b, "claims-lint: %d capability lines (%d shipped, %d simulated, %d stub); exposure: %d default-on, %d gated (%d declared), %d parked; %d violations\n",
		r.Capability, r.Shipped, r.Simulated, r.Stub, r.DefaultOn, r.Gated, r.Declared, r.Parked, len(r.Violations))
	return b.String()
}

func truncate(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n]) + "…"
}
