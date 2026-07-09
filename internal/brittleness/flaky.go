package brittleness

import (
	"fmt"
	"sort"
	"strings"
)

// flaky.go is the CAPTURE-WHEN-FRESH bridge from the transient rerun classifier to a
// durable brittleness finding.
//
// internal/affectedtests.ClassifyReruns runs on the inner loop: it reruns the
// initially-failing packages on the CURRENT working tree and, when a package failed
// once but then passed on a same-tree rerun, names it FLAKY_PASSED_ON_RETRY and drops
// the loop's exit code to 0. That verdict is the FRESHEST possible knowledge of a
// brittle test -- the exact tree that flaked is still checked out, the passing rerun
// just happened -- and today it is thrown away the instant the exit drops. The next
// agent to hit the same non-deterministic package re-derives it from scratch: another
// stash-and-rerun cycle proving a red that was never a regression.
//
// FromFlakyPackages turns that fresh verdict into a FLAKY_RETRY_PASS Finding so the
// brittleness card can rank it against the recurring-fix and reverted-landing seams,
// worst-first, for a remediation loop to de-flake. `when` is the caller-supplied
// freshness stamp (the tree sha the rerun ran on, or an ISO timestamp) recorded in
// Fresh -- the pure core takes no clock, exactly like the rest of the scorecard family,
// so the shell injects the moment. A blank package name is skipped; the input is
// de-duplicated and the output sorted, so the same package listed twice yields one
// finding of weight 2 (it flaked twice -- a heavier seam).
func FromFlakyPackages(pkgs []string, when string) []Finding {
	counts := map[string]int{}
	var order []string
	for _, p := range pkgs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if counts[p] == 0 {
			order = append(order, p)
		}
		counts[p]++
	}
	sort.Strings(order)

	when = strings.TrimSpace(when)
	var out []Finding
	for _, p := range order {
		n := counts[p]
		detail := "failed then passed on a same-tree rerun (non-deterministic; poisons the shared fast gate)"
		if n > 1 {
			detail = fmt.Sprintf("flaked %d times (failed then passed on a same-tree rerun; poisons the shared fast gate)", n)
		}
		f := Finding{
			Class:  ClassFlakyRetryPass,
			Ref:    p,
			Detail: detail,
			Weight: n,
		}
		if when != "" {
			f.Fresh = []string{when}
		}
		out = append(out, f)
	}
	SortFindings(out)
	return out
}
