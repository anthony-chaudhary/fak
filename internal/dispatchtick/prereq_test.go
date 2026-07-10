package dispatchtick

import (
	"reflect"
	"testing"
)

// TestCandidateBlockedBy mirrors the Python CandidateBlockedByTest case-for-case
// (tools/issue_resolve_dispatch_test.py:4851) so the Go and Python pickers parse identical edges.
func TestCandidateBlockedBy(t *testing.T) {
	cases := []struct {
		name string
		body string
		want []string
	}{
		// Both marker verbs, hyphenated or spaced, colon optional.
		{"depends-on hyphen colon", "depends-on: #120", []string{"120"}},
		{"depends on spaced", "Depends on #120", []string{"120"}},
		{"blocked-by hyphen colon", "blocked-by: #120", []string{"120"}},
		{"blocked by spaced", "Blocked by #120", []string{"120"}},
		// Comma/and/&-separated refs on one marker, and across markers, deduped in first-seen order.
		{"multi-ref one marker", "blocked-by: #120, #121 and #122", []string{"120", "121", "122"}},
		{"across markers deduped", "Depends on #7.\nAlso blocked by #7 & #9.", []string{"7", "9"}},
		// Prose that merely contains the words never matches (marker must be immediately followed
		// by #N); a marker-free / body-free issue carries no prerequisite edge.
		{"prose no ref", "it depends on the weather", nil},
		{"marker no number", "no markers here #notanumber", nil},
		{"empty body", "", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := CandidateBlockedBy(tc.body)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("CandidateBlockedBy(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}
