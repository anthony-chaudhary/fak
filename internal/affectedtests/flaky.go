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

import (
	"encoding/json"
	"sort"
	"strings"
)

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

// StillFailing is the per-test verdict for a test that failed the first run and
// never produced a positive PASS on any same-tree rerun — the fail-closed default
// (a red without green evidence is a real red, not a flake). It is the per-test
// counterpart of the package staying in ClassifyReruns' stillFailing set.
const StillFailing = "STILL_FAILING"

// TestEvent is the subset of one `go test -json` event line this fold reads: the
// Action ("run"|"pass"|"fail"|"output"|…), the Package it happened in, and the
// Test it names. A package-level event carries an empty Test; a test or subtest
// event carries "TestName" or "TestName/subtest". Only these three fields drive
// per-test flake identification, so the rest of the event is ignored.
type TestEvent struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Test    string `json:"Test"`
}

// testID identifies one test/subtest within one package — the grain a quarantine
// ledger, a "fix this flake" ticket, and a per-test trend all need but the
// package-level ClassifyReruns cannot express.
type testID struct {
	pkg  string
	name string
}

// Finding names one individual test/subtest that the reruns classified, carrying
// the specific Package/Test the package-level ClassifyReruns could only report as
// a whole poisoned package. Verdict is FlakyPassedOnRetry (failed first, then
// passed on a same-tree rerun) or StillFailing (never seen green — fail-closed).
type Finding struct {
	Package string `json:"package"`
	Test    string `json:"test"` // "TestName" or "TestName/subtest" — never just the package
	Verdict string `json:"verdict"`
}

// Qualified renders the finding as "package.Test" (or just the package when no
// test is named), the stable key a ledger or ticket dedupes on.
func (f Finding) Qualified() string {
	if f.Test == "" {
		return f.Package
	}
	return f.Package + "." + f.Test
}

// ParseTestEvents folds `go test -json` newline-delimited output into events,
// tolerantly skipping any non-JSON preamble a `go test` run can interleave (build
// errors, module-download notes, a bare "FAIL" trailer) so a malformed line never
// discards the whole stream. Pure and deterministic.
func ParseTestEvents(raw string) []TestEvent {
	var out []TestEvent
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line[0] != '{' {
			continue
		}
		var e TestEvent
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			continue // a non-event JSON object (or truncated line) is skipped, not fatal
		}
		out = append(out, e)
	}
	return out
}

// ClassifyRerunFindings is the per-test upgrade of ClassifyReruns: it names the
// individual test/subtest that flaked instead of the whole package. firstRun is
// the `-json` events of the initial (failing) run; reruns is the concatenated
// `-json` events of the same-tree rerun round(s). A leaf test that FAILED in
// firstRun and PASSED in some rerun is FlakyPassedOnRetry; one never seen green
// stays StillFailing — the SAME fail-closed rule as ClassifyReruns, because a
// flake needs positive green evidence, never the mere absence of a repeated FAIL.
//
// Only leaf tests are named: when a subtest fails, `go test` also emits a fail for
// its parent test and the package, but those fail only BECAUSE the subtest did, so
// an ancestor whose name is a "/"-prefix of another failed name in the same
// package is dropped — leaving the most specific unit ("TestFoo/case_b"), which is
// the point of this leaf. Findings are sorted by package then test. Pure.
func ClassifyRerunFindings(firstRun, reruns []TestEvent) (flaky, stillFailing []Finding) {
	failed := failedLeafTests(firstRun)
	passed := passedTests(reruns)

	ids := make([]testID, 0, len(failed))
	for id := range failed {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		if ids[i].pkg != ids[j].pkg {
			return ids[i].pkg < ids[j].pkg
		}
		return ids[i].name < ids[j].name
	})

	for _, id := range ids {
		f := Finding{Package: id.pkg, Test: id.name}
		if passed[id] {
			f.Verdict = FlakyPassedOnRetry
			flaky = append(flaky, f)
		} else {
			f.Verdict = StillFailing
			stillFailing = append(stillFailing, f)
		}
	}
	return flaky, stillFailing
}

// failedLeafTests collects the test-level FAIL events and keeps only the leaves —
// dropping any failed test that is a strict "/"-prefixed ancestor of another
// failed test in the same package, since a parent/umbrella fails only because its
// subtest did. Package-level FAILs (empty Test) are ignored: naming the package is
// exactly the granularity this leaf removes.
func failedLeafTests(events []TestEvent) map[testID]bool {
	failed := map[testID]bool{}
	for _, e := range events {
		if e.Action == "fail" && e.Test != "" {
			failed[testID{e.Package, e.Test}] = true
		}
	}
	leaves := make(map[testID]bool, len(failed))
	for id := range failed {
		isLeaf := true
		for other := range failed {
			if other.pkg == id.pkg && other.name != id.name &&
				strings.HasPrefix(other.name, id.name+"/") {
				isLeaf = false
				break
			}
		}
		if isLeaf {
			leaves[id] = true
		}
	}
	return leaves
}

// passedTests is the set of test/subtest units that produced a positive PASS in
// the given events — the green evidence a flake verdict requires.
func passedTests(events []TestEvent) map[testID]bool {
	passed := map[testID]bool{}
	for _, e := range events {
		if e.Action == "pass" && e.Test != "" {
			passed[testID{e.Package, e.Test}] = true
		}
	}
	return passed
}
