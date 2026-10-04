package coherence

import "testing"

// fak-test:runtime fast est=1s lane=default
func TestCheckClosedVerdicts(t *testing.T) {
	cases := []struct {
		name string
		text string
		n    int
		want Verdict
	}{
		{"spaces", "1 2 3 4 5 6 7 8 9 10", 10, Coherent},
		{"commas and preamble", "Sure! 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.", 10, Coherent},
		{"newlines", "1\n2\n3\n4\n5", 5, Coherent},
		{"closed reasoning ignored", "<think>count 1 to 3</think>1 2 3", 3, Coherent},
		{"run after noise", "7 1 2 3", 3, Coherent},
		{"empty", "   ", 10, Empty},
		{"unclosed reasoning only", "<think>1 2 3", 3, Empty},
		{"fluent garbage", "ledger ocean 4 window 9 sparrow", 10, SequenceMissing},
		{"truncated run", "1 2 3 4 5 6 7 8 9", 10, SequenceMissing},
		{"skipped number", "1 2 3 5 6", 5, SequenceMissing},
		{"out of order", "2 1 3", 3, SequenceMissing},
		{"repeated token", "1 1 1 1 1", 3, SequenceMissing},
		{"reasoning echo only", "<think>x</think>count from 1 to 10", 10, SequenceMissing},
		{"zero count", "1", 0, InvalidCount},
		{"over max", "1", MaxCount + 1, InvalidCount},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Check(tc.text, tc.n); got != tc.want {
				t.Fatalf("Check(%q, %d) = %q, want %q", tc.text, tc.n, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestVerdictsClosedSet(t *testing.T) {
	seen := map[Verdict]bool{}
	for _, v := range Verdicts() {
		if v == "" || seen[v] {
			t.Fatalf("verdict vocabulary has empty or duplicate member %q", v)
		}
		seen[v] = true
	}
	if len(seen) != 4 {
		t.Fatalf("verdict vocabulary size = %d, want 4", len(seen))
	}
}

// fak-test:runtime fast est=1s lane=default
func TestCountPromptAnswerableAndBounded(t *testing.T) {
	// The prompt is code-authored and must name both ends of the run.
	p := CountPrompt(DefaultCount)
	if Check(p, DefaultCount) == Coherent {
		t.Fatalf("the prompt itself must not satisfy the check: %q", p)
	}
	if MaxTokens(DefaultCount) <= DefaultCount {
		t.Fatalf("MaxTokens(%d) = %d leaves no room for the answer", DefaultCount, MaxTokens(DefaultCount))
	}
}
