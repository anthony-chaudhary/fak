package affectedtests

import (
	"reflect"
	"strings"
	"testing"
)

// TestFailedPackages pins the go-test output parse: column-zero package-level FAIL
// lines (timed and [build failed] forms) are collected; ok lines, the bare trailing
// FAIL summary, test-level "--- FAIL:" noise, and INDENTED log lines that happen to
// start with FAIL are all ignored; the result is sorted + deduplicated. An unparseable
// output yields empty — the caller's "keep the red exit, never guess" contract.
func TestFailedPackages(t *testing.T) {
	out := strings.Join([]string{
		"=== RUN   TestX",
		"--- FAIL: TestX (0.00s)",
		"    FAIL example.com/m/indented 0.1s", // t.Log noise, not a verdict line
		"FAIL",
		"FAIL\texample.com/m/b\t0.42s",
		"ok  \texample.com/m/a\t0.01s",
		"FAIL\texample.com/m/c [build failed]",
		"FAIL\texample.com/m/b\t0.40s", // duplicate from a re-run chunk
		"",
	}, "\n")
	got := FailedPackages(out)
	want := []string{"example.com/m/b", "example.com/m/c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("FailedPackages = %v, want %v", got, want)
	}
	if got := FailedPackages("go: cannot find module\n"); len(got) != 0 {
		t.Fatalf("unparseable output = %v, want empty", got)
	}
}

// TestPassedPackages pins the dual parse: "ok  \tpkg\t0.01s" and "(cached)" verdict
// lines are collected; "FAIL" lines and prose lines starting with ok-ish words are not.
func TestPassedPackages(t *testing.T) {
	out := strings.Join([]string{
		"ok  \texample.com/m/a\t0.01s",
		"ok  \texample.com/m/b\t(cached)",
		"okay, this prose line is not a verdict",
		"FAIL\texample.com/m/c\t0.42s",
		"",
	}, "\n")
	got := PassedPackages(out)
	want := []string{"example.com/m/a", "example.com/m/b"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("PassedPackages = %v, want %v", got, want)
	}
}

// TestAttribute pins the closed vocabulary and its precedence: baseline red wins
// (peer-preexisting), then the mine-closure rung (peer-wip), else mine; a nil baseline
// never exonerates (fail-closed), a nil closure attributes everything to the caller,
// and the mine evidence never claims baseline coverage the run did not produce.
func TestAttribute(t *testing.T) {
	mine := map[string]bool{"m/a": true, "m/c": true, "m/new": true}
	baselineRed := map[string]bool{"m/a": true}
	// The baseline produced verdicts for a and c only; m/new does not exist at the ref.
	baselineSeen := map[string]bool{"m/a": true, "m/b": true, "m/c": true}

	cases := []struct {
		name         string
		failing      []string
		closure      map[string]bool
		baseline     map[string]bool
		seen         map[string]bool
		wantClass    map[string]string
		wantEvidence string
	}{
		{
			name:      "baseline red wins even inside the mine closure",
			failing:   []string{"m/a"},
			closure:   mine,
			baseline:  baselineRed,
			seen:      baselineSeen,
			wantClass: map[string]string{"m/a": BlamePeerPreexisting},
		},
		{
			name:      "outside closure and green at baseline is peer-wip",
			failing:   []string{"m/b"},
			closure:   mine,
			baseline:  map[string]bool{},
			seen:      baselineSeen,
			wantClass: map[string]string{"m/b": BlamePeerWIP},
		},
		{
			name:         "inside closure and green at baseline is mine",
			failing:      []string{"m/c"},
			closure:      mine,
			baseline:     map[string]bool{},
			seen:         baselineSeen,
			wantClass:    map[string]string{"m/c": BlameMine},
			wantEvidence: "green at a clean checkout",
		},
		{
			name:         "package the baseline never tested is mine with honest evidence",
			failing:      []string{"m/new"},
			closure:      mine,
			baseline:     map[string]bool{},
			seen:         baselineSeen,
			wantClass:    map[string]string{"m/new": BlameMine},
			wantEvidence: "not testable at a clean checkout",
		},
		{
			name:      "nil closure attributes everything reachable to the caller",
			failing:   []string{"m/b"},
			closure:   nil,
			baseline:  map[string]bool{},
			seen:      baselineSeen,
			wantClass: map[string]string{"m/b": BlameMine},
		},
		{
			name:         "nil baseline never exonerates inside the closure (fail-closed)",
			failing:      []string{"m/c"},
			closure:      mine,
			baseline:     nil,
			seen:         nil,
			wantClass:    map[string]string{"m/c": BlameMine},
			wantEvidence: "fail-closed",
		},
		{
			name:      "nil baseline still allows the closure rung to exonerate",
			failing:   []string{"m/b"},
			closure:   mine,
			baseline:  nil,
			seen:      nil,
			wantClass: map[string]string{"m/b": BlamePeerWIP},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Attribute(c.failing, c.closure, c.baseline, c.seen, "HEAD")
			if len(got) != len(c.wantClass) {
				t.Fatalf("Attribute = %+v, want %d row(s)", got, len(c.wantClass))
			}
			for _, b := range got {
				want, ok := c.wantClass[b.Package]
				if !ok {
					t.Fatalf("unexpected package %q in %+v", b.Package, got)
				}
				if b.Class != want {
					t.Fatalf("%s class = %s, want %s (evidence=%q)", b.Package, b.Class, want, b.Evidence)
				}
				if b.Evidence == "" {
					t.Fatalf("%s carries no evidence sentence", b.Package)
				}
				if c.wantEvidence != "" && !strings.Contains(b.Evidence, c.wantEvidence) {
					t.Fatalf("%s evidence %q missing %q", b.Package, b.Evidence, c.wantEvidence)
				}
			}
		})
	}
}

// TestAttributeOrderAndDedup pins the deterministic shape: input order and duplicates do
// not affect the sorted, deduplicated output.
func TestAttributeOrderAndDedup(t *testing.T) {
	got := Attribute([]string{"m/b", "m/a", "m/b", ""}, nil, map[string]bool{}, map[string]bool{"m/a": true, "m/b": true}, "origin/main")
	if len(got) != 2 || got[0].Package != "m/a" || got[1].Package != "m/b" {
		t.Fatalf("Attribute order/dedup = %+v, want [m/a m/b]", got)
	}
	if !strings.Contains(got[0].Evidence, "origin/main") {
		t.Fatalf("evidence %q does not name the baseline ref", got[0].Evidence)
	}
}
