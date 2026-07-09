package affectedtests

// flaky.go is the pure fold behind `fak affected --rerun-fail N`. On a shared trunk a
// non-deterministic test poisons the fast inner-loop gate for everyone: it fails on one
// run and passes on the next with the SAME code, so an agent reads a red that is not a
// regression from its diff and burns a stash-and-rerun cycle proving so. Worse, it
// defeats the --blame baseline rerun — a flaky package can read as green-at-baseline one
// moment and red the next, mis-attributing the flake as "mine". The rerun classifier
// answers a question --blame cannot: is this red REPRODUCIBLE on the same tree at all?
//
// The shell reruns the initially-failing packages up to N times on the CURRENT working
// tree and hands the outcome to ClassifyReruns. A package that failed the first run but
// produced a passing verdict on a later same-tree rerun is FLAKY by definition (the code
// did not change between the runs), so it is exonerated for the loop — loudly, recorded,
// and with `make ci` on the merged tree left as the authoritative oracle exactly as the
// package doc's stated limit says.
//
// THE FAIL-CLOSED RULE mirrors blame's: exoneration needs positive GREEN evidence. A
// package the reruns never saw pass — because it kept failing, or because a
// harness-execution failure made its FAIL lines untrustworthy so the shell gathered no
// verdict — stays in the still-failing set and keeps the red exit. Flakiness is never
// inferred from the mere ABSENCE of a repeated FAIL.

import "sort"

// FlakyPassedOnRetry is the closed verdict a run earns when EVERY initially-failing
// package passed on a same-tree rerun — non-deterministic, not a deterministic regression
// from the caller's diff. Named alongside the blame classes as the JSON contract a
// calling loop routes on; the `fak affected` shell promotes the run's verdict to this
// string and drops the exit to 0 when stillFailing is empty.
const FlakyPassedOnRetry = "FLAKY_PASSED_ON_RETRY"

// ClassifyReruns splits the initially-failing packages into the FLAKY ones (failed the
// first run, then produced a passing verdict on a later same-tree rerun) and the ones
// STILL FAILING after every rerun. passedOnRerun is the set of packages that produced a
// positive `ok` verdict in some rerun round; a package absent from it stays in
// stillFailing — fail-closed, because flakiness needs positive green evidence and never
// the mere absence of a repeated FAIL. Both slices are sorted and deduplicated. The
// union of the two is exactly the deduplicated input, so the caller can trust that an
// empty stillFailing means every red was exonerated as flaky. Pure and deterministic.
func ClassifyReruns(initialFailed []string, passedOnRerun map[string]bool) (flaky, stillFailing []string) {
	seen := make(map[string]bool, len(initialFailed))
	for _, p := range initialFailed {
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		if passedOnRerun[p] {
			flaky = append(flaky, p)
		} else {
			stillFailing = append(stillFailing, p)
		}
	}
	sort.Strings(flaky)
	sort.Strings(stillFailing)
	return flaky, stillFailing
}
