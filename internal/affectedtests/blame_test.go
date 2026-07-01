package affectedtests

import (
	"reflect"
	"strings"
	"testing"
)

// TestFailedPackages pins the go-test output parse: package-level FAIL lines (timed and
// [build failed] forms) are collected, ok lines and test-level "--- FAIL:" noise are
// ignored, and the result is sorted + deduplicated. An unparseable output yields empty —
// the caller's "keep the red exit, never guess" contract.
func TestFailedPackages(t *testing.T) {
	out := strings.Join([]string{
		"=== RUN   TestX",
		"--- FAIL: TestX (0.00s)",
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

// TestAttribute pins the closed vocabulary and its precedence: baseline red wins
// (peer-preexisting), then the mine-closure rung (peer-wip), else mine; a nil baseline
// never exonerates (fail-closed), and a nil closure attributes everything to the caller.
func TestAttribute(t *testing.T) {
	mine := map[string]bool{"m/a": true, "m/c": true}
	baselineRed := map[string]bool{"m/a": true}

	cases := []struct {
		name      string
		failing   []string
		closure   map[string]bool
		baseline  map[string]bool
		wantClass map[string]string
	}{
		{
			name:     "baseline red wins even inside the mine closure",
			failing:  []string{"m/a"},
			closure:  mine,
			baseline: baselineRed,
			wantClass: map[string]string{
				"m/a": BlamePeerPreexisting,
			},
		},
		{
			name:     "outside closure and green at baseline is peer-wip",
			failing:  []string{"m/b"},
			closure:  mine,
			baseline: map[string]bool{},
			wantClass: map[string]string{
				"m/b": BlamePeerWIP,
			},
		},
		{
			name:     "inside closure and green at baseline is mine",
			failing:  []string{"m/c"},
			closure:  mine,
			baseline: map[string]bool{},
			wantClass: map[string]string{
				"m/c": BlameMine,
			},
		},
		{
			name:     "nil closure attributes everything reachable to the caller",
			failing:  []string{"m/z"},
			closure:  nil,
			baseline: map[string]bool{},
			wantClass: map[string]string{
				"m/z": BlameMine,
			},
		},
		{
			name:     "nil baseline never exonerates inside the closure (fail-closed)",
			failing:  []string{"m/c"},
			closure:  mine,
			baseline: nil,
			wantClass: map[string]string{
				"m/c": BlameMine,
			},
		},
		{
			name:     "nil baseline still allows the closure rung to exonerate",
			failing:  []string{"m/b"},
			closure:  mine,
			baseline: nil,
			wantClass: map[string]string{
				"m/b": BlamePeerWIP,
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Attribute(c.failing, c.closure, c.baseline, "HEAD")
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
			}
		})
	}
}

// TestAttributeOrderAndDedup pins the deterministic shape: input order and duplicates do
// not affect the sorted, deduplicated output.
func TestAttributeOrderAndDedup(t *testing.T) {
	got := Attribute([]string{"m/b", "m/a", "m/b", ""}, nil, map[string]bool{}, "origin/main")
	if len(got) != 2 || got[0].Package != "m/a" || got[1].Package != "m/b" {
		t.Fatalf("Attribute order/dedup = %+v, want [m/a m/b]", got)
	}
	if !strings.Contains(got[0].Evidence, "origin/main") {
		t.Fatalf("evidence %q does not name the baseline ref", got[0].Evidence)
	}
}
