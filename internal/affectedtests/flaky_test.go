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
