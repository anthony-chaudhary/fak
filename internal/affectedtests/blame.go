package affectedtests

// blame.go attributes each RED package to whose change made it red (#2138). On a shared
// trunk a package can fail from a PEER's in-flight WIP unrelated to my diff; `fak
// affected` reporting the red without saying whose it is makes the agent mis-diagnose
// "I broke it" and burn a stash-and-rerun verify cycle. This file is the pure
// attribution fold; the impure shell (cmd/fak/affected.go --blame) gathers the two
// pieces of evidence it folds:
//
//   - the BASELINE outcome: is the package already red at a CLEAN checkout of the diff
//     base (HEAD / --base), before any working-tree change? Red there is positive
//     evidence the red pre-dates my diff.
//   - the MINE closure: the affected-set closure (Select) of the files the agent
//     DECLARES as its own (--mine). A red package the closure cannot reach did not come
//     from my diff — some other working-tree change (a peer's WIP) made it red.
//
// THE FAIL-CLOSED RULE: exoneration needs positive evidence. A package inside my
// closure with no baseline evidence (the baseline rerun was unavailable) stays MINE —
// the gate never turns green on a guess. The one honest caveat, named in the evidence
// string: a baseline that is ALREADY red cannot also prove my diff added no FURTHER
// breakage to the same package; the class is still peer-preexisting (the red I am
// looking at existed before me), and the full `make ci` on the merged tree remains the
// authoritative oracle, exactly as the package doc's stated limit says.

import (
	"fmt"
	"sort"
	"strings"
)

// The closed blame vocabulary (#2138). String constants in the same shape as the
// leaseref liveness classes; they are the JSON contract a calling loop routes on.
const (
	// BlameMine: the red is attributable to the caller's declared change — in the
	// --mine closure (or nothing was declared) and not exonerated by the baseline.
	// The only class that fails the gate.
	BlameMine = "mine"
	// BlamePeerWIP: the failing package is OUTSIDE the closure of the caller's
	// declared files — the red comes from some other working-tree change (a peer's
	// uncommitted WIP), not from the caller's diff.
	BlamePeerWIP = "peer-wip"
	// BlamePeerPreexisting: the package is red at a CLEAN checkout of the base ref —
	// the red pre-dates the caller's diff entirely.
	BlamePeerPreexisting = "peer-preexisting"
)

// Blame is one failing package with its attribution class and the evidence sentence
// naming the comparison that decided it.
type Blame struct {
	Package  string `json:"package"`
	Class    string `json:"class"`
	Evidence string `json:"evidence"`
}

// Attribute classifies each failing package. mineClosure is the affected-set closure of
// the caller's declared --mine files (nil = nothing declared, so every red is
// closure-attributable to the caller); baselineRed is the set of packages red at a
// clean checkout of baselineRef and baselineSeen the set the baseline actually PRODUCED
// A VERDICT for (red or ok) — nil on both means the baseline rerun was unavailable, so
// no exoneration from that rung, fail-closed. Precedence: a baseline red wins
// (peer-preexisting), then the closure rung (peer-wip), else mine; the mine evidence
// distinguishes "green at the baseline" from "the baseline never tested it" (a package
// new in the diff does not exist at the base ref) so the sentence never claims evidence
// that was not gathered. The result is sorted by package and deduplicated. Pure and
// deterministic.
func Attribute(failing []string, mineClosure, baselineRed, baselineSeen map[string]bool, baselineRef string) []Blame {
	seen := make(map[string]bool, len(failing))
	pkgs := make([]string, 0, len(failing))
	for _, p := range failing {
		if p != "" && !seen[p] {
			seen[p] = true
			pkgs = append(pkgs, p)
		}
	}
	sort.Strings(pkgs)

	out := make([]Blame, 0, len(pkgs))
	for _, p := range pkgs {
		switch {
		case baselineRed != nil && baselineRed[p]:
			out = append(out, Blame{
				Package: p,
				Class:   BlamePeerPreexisting,
				Evidence: fmt.Sprintf(
					"red at a clean checkout of %s — failing before your diff (caveat: a pre-existing red cannot also prove your diff added no further breakage; make ci stays the oracle)",
					baselineRef),
			})
		case mineClosure != nil && !mineClosure[p]:
			out = append(out, Blame{
				Package:  p,
				Class:    BlamePeerWIP,
				Evidence: "outside the affected-set closure of your declared --mine files — the red comes from other working-tree changes (a peer's WIP), not your diff",
			})
		case baselineSeen != nil && baselineSeen[p]:
			out = append(out, Blame{
				Package:  p,
				Class:    BlameMine,
				Evidence: fmt.Sprintf("green at a clean checkout of %s and reachable from your declared change — the red arrives with your diff", baselineRef),
			})
		case baselineSeen != nil:
			out = append(out, Blame{
				Package:  p,
				Class:    BlameMine,
				Evidence: fmt.Sprintf("not testable at a clean checkout of %s (the package is likely new in your diff) — attributed mine", baselineRef),
			})
		default:
			out = append(out, Blame{
				Package:  p,
				Class:    BlameMine,
				Evidence: "no baseline evidence available and reachable from your declared change — fail-closed to mine (exoneration needs positive evidence)",
			})
		}
	}
	return out
}

// FailedPackages parses the per-package result lines of plain `go test` output and
// returns the failing import paths, sorted and deduplicated. It reads only the
// package-level verdict lines go test always emits —
//
//	FAIL<tab>example.com/pkg<tab>0.42s
//	FAIL<tab>example.com/pkg [build failed]
//
// — matched by the "FAIL\t" COLUMN-ZERO prefix go itself prints, so test-level noise
// ("--- FAIL: TestX"), the bare trailing "FAIL" summary line, and INDENTED test-log
// lines that happen to start with FAIL are all excluded. (A test that itself prints a
// forged "FAIL\tpkg" at column zero to stdout can still spoof a row — un-forgeable
// attribution would need `go test -json`; named residual, not covered.) An output with
// no such lines yields an empty slice: the caller must treat "red run, nothing parsed"
// as unattributable and keep the red exit, never guess. Pure and deterministic.
func FailedPackages(output string) []string {
	return packageVerdicts(output, "FAIL\t")
}

// PassedPackages is FailedPackages' dual for the "ok  \texample.com/pkg\t0.01s" (or
// "(cached)") verdict lines. The union of the two is the set of packages a run actually
// PRODUCED A VERDICT for — the baseline-rerun coverage evidence Attribute needs to
// phrase a mine row honestly.
func PassedPackages(output string) []string {
	return packageVerdicts(output, "ok  \t")
}

// packageVerdicts collects field 2 of every line carrying go test's column-zero
// package-verdict prefix, sorted and deduplicated.
func packageVerdicts(output, prefix string) []string {
	set := map[string]bool{}
	for _, line := range strings.Split(output, "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		if fields := strings.Fields(line); len(fields) >= 2 {
			set[fields[1]] = true
		}
	}
	out := make([]string, 0, len(set))
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}
