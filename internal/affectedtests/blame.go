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
// clean checkout of baselineRef (nil = the baseline rerun was unavailable — no
// exoneration from that rung, fail-closed). Precedence: a baseline red wins
// (peer-preexisting), then the closure rung (peer-wip), else mine. The result is
// sorted by package and deduplicated. Pure and deterministic.
func Attribute(failing []string, mineClosure map[string]bool, baselineRed map[string]bool, baselineRef string) []Blame {
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
				Package: p,
				Class:   BlamePeerWIP,
				Evidence: "outside the affected-set closure of your declared --mine files — the red comes from other working-tree changes (a peer's WIP), not your diff",
			})
		case baselineRed != nil:
			out = append(out, Blame{
				Package: p,
				Class:   BlameMine,
				Evidence: fmt.Sprintf("green at a clean checkout of %s and reachable from your declared change — the red arrives with your diff", baselineRef),
			})
		default:
			out = append(out, Blame{
				Package: p,
				Class:   BlameMine,
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
//	ok  <tab>example.com/pkg<tab>0.01s
//
// — and ignores test-level noise ("--- FAIL: TestX"), so it works on the exact stream
// the shell already tees to the operator. An output with no such lines yields an empty
// slice: the caller must treat "red run, nothing parsed" as unattributable and keep the
// red exit, never guess. Pure and deterministic.
func FailedPackages(output string) []string {
	set := map[string]bool{}
	for _, line := range strings.Split(output, "\n") {
		fields := strings.Fields(line)
		// The package verdict line starts the line with the literal token FAIL; the
		// test-level "--- FAIL:" lines start with "---" and are skipped by this exact
		// match.
		if len(fields) >= 2 && fields[0] == "FAIL" {
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
