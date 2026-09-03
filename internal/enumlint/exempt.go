package enumlint

import (
	"fmt"
	"go/ast"
	"sort"
	"strings"
)

// exempt.go — the escape hatch, in two forms, both of which DEMAND A WRITTEN
// REASON.
//
// A hatch nobody must justify is a deny-list nobody enforces. So neither form
// opens without prose: the in-place directive is ignored when its reason is
// blank, and LookupExemption refuses a table entry whose value is "".
//
// The table also fails in the direction it can actually go wrong. It cannot go
// wrong by being SHORT — a site with no entry is a finding, so the default is
// refusal. It goes wrong by being LONG: a site is fixed or deleted, its
// exemption stays, and the next literal that lands under that name is silently
// unchecked forever. TestNoStaleExemptions re-scans with exemptions disabled and
// fails on any key that no longer fires.

// ExemptDirective is the in-place form:
//
//	//enumlint:exempt <reason>
//
// written on, or on either of the two lines above, the site it covers. It is the
// right form when you hold the lane of the package that owns the site.
const ExemptDirective = "enumlint:exempt"

// exemptions is the OUT-OF-LANE form: sites this tree is deliberately partial
// at, recorded here because the agent that runs the linter usually does not hold
// the lane of the package that owns the site. Key is exemptKey(rule, pkgDir,
// owner); value is the reason, and it is REQUIRED.
//
// This table starts empty ON PURPOSE. #5935 ships as a RATCHET, not a refusal:
// today's debt is captured in baseline.txt as counts, which is a statement about
// what the tree looked like at a SHA and needs no per-site argument. An
// exemption is a stronger claim — "this site is CORRECT to be partial, forever"
// — and every entry here has to earn it at its own site. Filling this table from
// a baseline sweep would convert a measurement into a hundred unread assertions.
var exemptions = map[string]string{
	"literal|internal/harnessprofile|builtins.Repoint":                               "Each harness selects only the repoint mechanisms it actually supports; the slice is intentionally profile-specific, not an exhaustive enum catalog.",
	"switch|internal/modelperfobs|reasonWithRecoveryEvidence":                        "Request-counter resets need no recovery suffix because request correlation is handled separately; only causal cache/queue counters require guidance.",
	"switch|internal/modelperfobs|localWorkflowBackend.Apply":                        "Natural expiry is intentionally unsupported by the deterministic local workflow backend and is rejected by Supports before Apply.",
	"literal|internal/nativebench|contracts.Alternatives":                            "Each benchmark declares only applicable comparison classes; a first-class integration arm does not exist for every capability.",
	"literal|internal/fp4meta|allCapabilities.ScaleFormats":                          "ScaleNone is deliberately excluded: the fixture describes FP4 hardware that requires explicit scaling.",
	"literal|internal/kvquantmeta|testSupport.Precisions":                            "FP16 and BF16 are unquantized controls, deliberately outside the quantized-support fixture.",
	"literal|internal/devindex|DevOnlyPackages":                                      "The boundary registry intentionally lists only development-owned and shared packages; runtime ownership is the default complement checked by graph reachability, not an entry in this exception set.",
	"literal|internal/studyclass|mechanismRules":                                     "Keyword rules enumerate actionable mechanisms only; the explicit-non-candidate sentinel is assigned structurally from metadata dispositions before keyword matching.",
	"literal|internal/studylink|witnessSeeds":                                        "The seed registry contains only affirmative manually evidenced dispositions (landed, open-exact, or partial); conflict, obsolete, and uncovered are derived outcomes rather than valid witness inputs.",
	"switch|internal/gateway|referenceBatchBudgetFold":                               "The test reference fold verifies available vs exhausted states; default fallthrough is tested separately.",
	"switch|internal/ggufload|dequantF32Into":                                        "Dequantization to float32 only implements supported tensor types; unsupported types fall back to higher-level loaders.",
	"switch|internal/incidentrsi|TestConcurrentThresholdCrossingLaunchesExactlyOnce": "Concurrent incident test validates specific action transitions.",
	"switch|internal/metalgemm|q4kGEMMRequestedExecution":                            "Scalar mode is non-GEMM execution handled by scalar fallbacks outside this switcher.",
	"literal|internal/model|iq12ResidentCases":                                       "Unit test matrix specifically targets IQ1 and IQ2 resident k-quant variants.",
	"literal|internal/wipinventory|issueHistory.Transitions":                         "Test fixture models a partial session transition history covering only tested events.",
}

// LookupExemption is the default Config.Exempt. An entry with a blank reason is
// not an exemption.
func LookupExemption(key string) (string, bool) {
	r, ok := exemptions[key]
	if !ok || strings.TrimSpace(r) == "" {
		return "", false
	}
	return r, true
}

// NoExemptions is a Config.Exempt that judges every site. TestNoStaleExemptions
// and the realsite mutation test use it: an exemption must never be able to hide
// the very finding a test is asserting on.
func NoExemptions(string) (string, bool) { return "", false }

// ExemptionKeys returns the table's keys, sorted, for the staleness test.
func ExemptionKeys() []string {
	out := make([]string, 0, len(exemptions))
	for k := range exemptions {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ExemptionReason returns the recorded reason for a key.
func ExemptionReason(key string) string { return exemptions[key] }

// ErrBlankReason is what a caller building its own exemption table gets for an
// entry with no reason, so the rule is enforceable outside this package too.
func ErrBlankReason(key string) error {
	return fmt.Errorf("enumlint: exemption %q has no reason; a hatch nobody must "+
		"justify is a deny-list nobody enforces", key)
}

// collectExemptDirectives records the line ranges an in-place directive covers.
// A directive covers its own line and the two lines below it, so it may sit
// immediately above `var x = []T{` or trail the opening brace.
func (p *Package) collectExemptDirectives(f *ast.File) {
	for _, cg := range f.Comments {
		for _, c := range cg.List {
			txt := strings.TrimLeft(strings.TrimPrefix(strings.TrimPrefix(c.Text, "//"), "/*"), " \t")
			if !strings.HasPrefix(txt, ExemptDirective) {
				continue
			}
			if strings.TrimSpace(strings.TrimPrefix(txt, ExemptDirective)) == "" {
				continue // no reason given: the hatch does not open
			}
			pos := p.fset.Position(c.Pos())
			end := p.fset.Position(c.End())
			file := p.relPath(pos.Filename)
			p.exemptLines[file] = append(p.exemptLines[file], lineRange{lo: pos.Line, hi: end.Line + 2})
		}
	}
}

func (p *Package) hasExemptDirective(file string, line int) bool {
	for _, r := range p.exemptLines[file] {
		if line >= r.lo && line <= r.hi {
			return true
		}
	}
	return false
}
