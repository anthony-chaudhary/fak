package affectedtests

import (
	"reflect"
	"testing"
)

// TestClassifyReruns pins the flaky/still-failing partition and its fail-closed rule:
// only a package with positive green evidence (in passedOnRerun) is called flaky, the
// input is deduplicated and both outputs sorted, and the union reconstructs the input so
// an empty stillFailing provably means every red was exonerated.
func TestClassifyReruns(t *testing.T) {
	for _, tc := range []struct {
		name      string
		failed    []string
		passed    map[string]bool
		wantFlaky []string
		wantStill []string
	}{
		{
			name:      "all flaky -> empty still-failing (the exonerating case)",
			failed:    []string{"m/b", "m/a"},
			passed:    map[string]bool{"m/a": true, "m/b": true},
			wantFlaky: []string{"m/a", "m/b"},
			wantStill: nil,
		},
		{
			name:      "partial: one flaky, one genuinely red",
			failed:    []string{"m/a", "m/b"},
			passed:    map[string]bool{"m/a": true},
			wantFlaky: []string{"m/a"},
			wantStill: []string{"m/b"},
		},
		{
			name:      "fail-closed: no green evidence -> nothing flaky",
			failed:    []string{"m/a", "m/b"},
			passed:    nil,
			wantFlaky: nil,
			wantStill: []string{"m/a", "m/b"},
		},
		{
			name:      "duplicates collapse and results sort",
			failed:    []string{"m/c", "m/a", "m/c", "m/a"},
			passed:    map[string]bool{"m/a": true},
			wantFlaky: []string{"m/a"},
			wantStill: []string{"m/c"},
		},
		{
			name:      "empty input -> empty outputs",
			failed:    nil,
			passed:    map[string]bool{"m/a": true},
			wantFlaky: nil,
			wantStill: nil,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			flaky, still := ClassifyReruns(tc.failed, tc.passed)
			if !reflect.DeepEqual(flaky, tc.wantFlaky) {
				t.Errorf("flaky = %v, want %v", flaky, tc.wantFlaky)
			}
			if !reflect.DeepEqual(still, tc.wantStill) {
				t.Errorf("stillFailing = %v, want %v", still, tc.wantStill)
			}
			// The partition is total: flaky ∪ stillFailing == deduped input.
			if len(flaky)+len(still) != countDistinctNonEmpty(tc.failed) {
				t.Errorf("partition lost entries: |flaky|=%d + |still|=%d != distinct(input)=%d",
					len(flaky), len(still), countDistinctNonEmpty(tc.failed))
			}
		})
	}
}

func countDistinctNonEmpty(in []string) int {
	seen := map[string]bool{}
	for _, s := range in {
		if s != "" {
			seen[s] = true
		}
	}
	return len(seen)
}

// firstRunJSON is a `go test -json` fixture where subtest TestFoo/case_b fails
// (so its parent TestFoo and the package also emit fail), a sibling subtest passes,
// and a separate top-level TestBar genuinely fails. Non-JSON preamble is included
// to prove ParseTestEvents tolerates it.
const firstRunJSON = `# m/pkg build note that is not JSON
{"Action":"run","Package":"m/pkg","Test":"TestFoo"}
{"Action":"run","Package":"m/pkg","Test":"TestFoo/case_a"}
{"Action":"pass","Package":"m/pkg","Test":"TestFoo/case_a"}
{"Action":"run","Package":"m/pkg","Test":"TestFoo/case_b"}
{"Action":"output","Package":"m/pkg","Test":"TestFoo/case_b","Output":"    foo_test.go:9: boom\n"}
{"Action":"fail","Package":"m/pkg","Test":"TestFoo/case_b"}
{"Action":"fail","Package":"m/pkg","Test":"TestFoo"}
{"Action":"run","Package":"m/pkg","Test":"TestBar"}
{"Action":"fail","Package":"m/pkg","Test":"TestBar"}
{"Action":"fail","Package":"m/pkg"}
`

// rerunJSON is the same-tree rerun: TestFoo/case_b now passes (so TestFoo passes),
// while TestBar still fails — the deterministic red.
const rerunJSON = `{"Action":"run","Package":"m/pkg","Test":"TestFoo/case_b"}
{"Action":"pass","Package":"m/pkg","Test":"TestFoo/case_b"}
{"Action":"pass","Package":"m/pkg","Test":"TestFoo"}
{"Action":"run","Package":"m/pkg","Test":"TestBar"}
{"Action":"fail","Package":"m/pkg","Test":"TestBar"}
{"Action":"fail","Package":"m/pkg"}
`

// TestClassifyRerunsPerTest is the per-test upgrade of the package-level partition
// (named to fall under the `-run TestClassifyReruns` witness): over a -json fixture
// where exactly one subtest flakes, the finding must name the individual subtest
// (TestFoo/case_b) — never the parent test or the package — and the genuinely-red
// top-level test must stay StillFailing (fail-closed).
func TestClassifyRerunsPerTest(t *testing.T) {
	first := ParseTestEvents(firstRunJSON)
	reruns := ParseTestEvents(rerunJSON)

	flaky, still := ClassifyRerunFindings(first, reruns)

	wantFlaky := []Finding{{Package: "m/pkg", Test: "TestFoo/case_b", Verdict: FlakyPassedOnRetry}}
	if !reflect.DeepEqual(flaky, wantFlaky) {
		t.Fatalf("flaky = %+v, want %+v (the individual subtest, not TestFoo or the package)", flaky, wantFlaky)
	}
	wantStill := []Finding{{Package: "m/pkg", Test: "TestBar", Verdict: StillFailing}}
	if !reflect.DeepEqual(still, wantStill) {
		t.Fatalf("stillFailing = %+v, want %+v", still, wantStill)
	}
	// The named unit is the subtest, and it is not the bare package.
	if flaky[0].Test != "TestFoo/case_b" || flaky[0].Qualified() != "m/pkg.TestFoo/case_b" {
		t.Fatalf("flaky finding did not name the subtest: %+v (%s)", flaky[0], flaky[0].Qualified())
	}
	// The parent umbrella test must be dropped as a non-leaf, never reported.
	for _, f := range append(append([]Finding{}, flaky...), still...) {
		if f.Test == "TestFoo" {
			t.Fatalf("parent test TestFoo leaked into findings; only the leaf subtest should be named: %+v", f)
		}
	}
}

// TestClassifyRerunsPerTestFailClosed pins the fail-closed rule at the per-test
// grain: a subtest that failed and was never seen green on any rerun stays
// StillFailing — flakiness is never inferred from the mere absence of a repeat FAIL.
func TestClassifyRerunsPerTestFailClosed(t *testing.T) {
	first := ParseTestEvents(`{"Action":"fail","Package":"m/pkg","Test":"TestX/edge"}
{"Action":"fail","Package":"m/pkg","Test":"TestX"}
`)
	// Rerun produced no PASS for TestX/edge (it kept failing, or the harness gave
	// no verdict) — no positive green evidence.
	flaky, still := ClassifyRerunFindings(first, nil)
	if len(flaky) != 0 {
		t.Fatalf("no green evidence must yield no flaky findings, got %+v", flaky)
	}
	want := []Finding{{Package: "m/pkg", Test: "TestX/edge", Verdict: StillFailing}}
	if !reflect.DeepEqual(still, want) {
		t.Fatalf("stillFailing = %+v, want %+v", still, want)
	}
}
