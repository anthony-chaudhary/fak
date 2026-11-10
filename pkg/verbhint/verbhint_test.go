package verbhint

import "testing"

func TestNearest(t *testing.T) {
	verbs := []string{"repo", "push", "preflight", "provenance", "model-quirks", "opencode"}

	cases := []struct {
		token string
		want  string
		ok    bool
	}{
		{"provenence", "provenance", true}, // transposition/typo
		{"provenancee", "provenance", true},
		{"modle-quirks", "model-quirks", true},
		{"opencode", "opencode", true}, // exact still resolves
		{"rep", "repo", true},          // distance 1 (delete)
		{"completely-unrelated", "", false},
		{"", "", false},
	}
	for _, tc := range cases {
		got, ok := Nearest(tc.token, verbs, DefaultMaxDistance)
		if ok != tc.ok || got != tc.want {
			t.Errorf("Nearest(%q) = (%q,%v); want (%q,%v)", tc.token, got, ok, tc.want, tc.ok)
		}
	}
}

func TestNearestDeterministicTieBreak(t *testing.T) {
	// "board" is distance 1 from both "board" and... use two equal-distance
	// candidates to pin the shorter-then-lexical tie-break.
	got, ok := Nearest("cut", []string{"cute", "cuts"}, DefaultMaxDistance)
	if !ok || got != "cute" {
		t.Fatalf("Nearest tie-break = (%q,%v); want (cute,true)", got, ok)
	}
}

func TestHintShape(t *testing.T) {
	h := Hint("provenence", []string{"provenance"}, DefaultMaxDistance)
	want := `did you mean "provenance"? run 'provenance -h' for its flags`
	if h != want {
		t.Fatalf("Hint = %q; want %q", h, want)
	}
	if Hint("zzzzzz", []string{"provenance"}, DefaultMaxDistance) != "" {
		t.Fatal("expected empty hint for an unrelated token")
	}
}

func TestLevenshteinLimit(t *testing.T) {
	if d := Levenshtein("kitten", "sitting", -1); d != 3 {
		t.Fatalf("Levenshtein = %d; want 3", d)
	}
	if d := Levenshtein("kitten", "sitting", 1); d != 2 {
		t.Fatalf("Levenshtein with limit = %d; want limit+1 = 2", d)
	}
}
