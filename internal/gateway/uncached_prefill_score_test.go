package gateway

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestUncachedPrefillScore(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name       string
		prompt     int
		known      bool
		candidates []uncachedPrefillCandidate
		want       uncachedPrefillChoice
		ok         bool
	}{
		// warm has TWO outstanding 4000-token requests, each 95% cached;
		// cold has only ONE, with no cached prompt tokens.
		{"cached_volume_beats_request_count", 4000, true, []uncachedPrefillCandidate{{"warm", 3800, 2 * 200, true}, {"cold", 0, 4000, true}}, uncachedPrefillChoice{"warm", 200, 600, 4000}, true},
		{"candidate_specific_cache", 4000, true, []uncachedPrefillCandidate{{"cold", 0, 0, true}, {"warm", 3900, 500, true}}, uncachedPrefillChoice{"warm", 100, 600, 600}, true},
		{"heavy_holder_sheds", 4000, true, []uncachedPrefillCandidate{{"warm", 3900, 8000, true}, {"cold", 0, 0, true}}, uncachedPrefillChoice{"cold", 4000, 4000, 8000}, true},
		{"lexical_tie", 10, true, []uncachedPrefillCandidate{{"z", 0, 0, true}, {"a", 0, 0, true}}, uncachedPrefillChoice{"a", 10, 10, 10}, true},
		{"known_zero", 0, true, []uncachedPrefillCandidate{{"a", 0, 0, true}}, uncachedPrefillChoice{"a", 0, 0, 0}, true},
		{"exact_integer_limit", 1, true, []uncachedPrefillCandidate{{"a", 0, maxInt - 1, true}}, uncachedPrefillChoice{"a", 1, maxInt, maxInt}, true},
		{"unknown_prompt", 0, false, []uncachedPrefillCandidate{{"a", 0, 0, true}}, uncachedPrefillChoice{}, false},
		{"unknown_candidate", 10, true, []uncachedPrefillCandidate{{"a", 0, 0, true}, {"b", 0, 0, false}}, uncachedPrefillChoice{}, false},
		{"negative_prompt", -1, true, []uncachedPrefillCandidate{{"a", 0, 0, true}}, uncachedPrefillChoice{}, false},
		{"negative_cache", 10, true, []uncachedPrefillCandidate{{"a", -1, 0, true}}, uncachedPrefillChoice{}, false},
		{"oversized_cache", 10, true, []uncachedPrefillCandidate{{"a", 11, 0, true}}, uncachedPrefillChoice{}, false},
		{"negative_pending", 10, true, []uncachedPrefillCandidate{{"a", 0, -1, true}}, uncachedPrefillChoice{}, false},
		{"overflow", 1, true, []uncachedPrefillCandidate{{"a", 0, maxInt, true}}, uncachedPrefillChoice{}, false},
		{"empty_worker", 10, true, []uncachedPrefillCandidate{{"", 0, 0, true}}, uncachedPrefillChoice{}, false},
		{"duplicate_worker", 10, true, []uncachedPrefillCandidate{{"a", 0, 0, true}, {"a", 1, 2, true}}, uncachedPrefillChoice{}, false},
		{"empty_fleet", 10, true, nil, uncachedPrefillChoice{}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := append([]uncachedPrefillCandidate(nil), tc.candidates...)
			got, ok := chooseUncachedPrefill(tc.prompt, tc.known, tc.candidates)
			if got != tc.want || ok != tc.ok {
				t.Fatalf("got (%+v, %v), want (%+v, %v)", got, ok, tc.want, tc.ok)
			}
			if !reflect.DeepEqual(before, tc.candidates) {
				t.Fatal("scorer mutated snapshot")
			}
			for i, j := 0, len(tc.candidates)-1; i < j; i, j = i+1, j-1 {
				tc.candidates[i], tc.candidates[j] = tc.candidates[j], tc.candidates[i]
			}
			if reversed, reversedOK := chooseUncachedPrefill(tc.prompt, tc.known, tc.candidates); reversed != got || reversedOK != ok {
				t.Fatal("decision depends on candidate order")
			}
		})
	}
}
