package answershape

import (
	"strings"
	"testing"
)

// A coherent, non-repetitive paragraph well above the floor. It must never trip
// the repeat verdict at the default threshold — the conservatism contract.
const coherent = "The gateway adjudicates every tool call in-process before it runs. " +
	"A denied call comes back with a structured reason from a closed vocabulary, " +
	"a malformed one is repaired without a model turn, and a poisoned result is " +
	"held out of context by the write-time admission gate. None of this needs a " +
	"network round trip or a spawned subprocess on the decision path."

func def() Limits { return Limits{MaxRepeat: DefaultMaxRepeat, NGram: DefaultNGram} }

func TestCoherentTextIsNotDegenerate(t *testing.T) {
	r := Measure([]byte(coherent), def())
	if r.Degenerate {
		t.Fatalf("coherent prose flagged degenerate: RepeatFraction=%.3f reasons=%v\n  ngram=%.3f line=%.3f period=%.3f",
			r.RepeatFraction, r.Reasons, r.NGramRepeat, r.LineBlockRepeat, r.PeriodRepeat)
	}
	if r.RepeatFraction >= DefaultMaxRepeat {
		t.Fatalf("coherent prose RepeatFraction=%.3f should sit well below %.2f", r.RepeatFraction, DefaultMaxRepeat)
	}
}

func TestSignals(t *testing.T) {
	// The verdict only: which signal wins the max is exercised separately
	// (TestEachSignalFires), because overlapping degeneration modes legitimately
	// fire several signals at once (a single-word loop is also byte-periodic).
	cases := []struct {
		name  string
		text  string
		degen bool
	}{
		{"word-loop", strings.Repeat("yes ", 40), true},
		{"phrase-loop", strings.Repeat("the system is fine. ", 12), true},
		{"punctuated-loop", "yes, yes, yes, yes, yes, yes, yes, yes, yes, yes!", true},
		{"repeated-line-block", strings.Repeat("ALERT: disk full\n", 20), true},
		{"single-rune-runaway", strings.Repeat("A", 200), true},
		{"tiled-unit", strings.Repeat("abc", 80), true},
		{"role-header-loop", strings.Repeat(".assistant", 30), true},
		{"coherent", coherent, false},
		{"short-curt-answer", "Yes, that works.", false}, // below floor → never degen by repeat
		{"short-but-repetitive-below-floor", "no no no", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := Measure([]byte(c.text), def())
			if r.Degenerate != c.degen {
				t.Fatalf("Degenerate=%v want %v (RepeatFraction=%.3f ngram=%.3f line=%.3f period=%.3f reasons=%v)",
					r.Degenerate, c.degen, r.RepeatFraction, r.NGramRepeat, r.LineBlockRepeat, r.PeriodRepeat, r.Reasons)
			}
		})
	}
}

// TestEachSignalFires isolates each sub-signal with an archetype that drives it
// (and not necessarily the others) above 0.5, so a regression that silently zeroes
// one signal is caught even though the headline max would still trip on the rest.
func TestEachSignalFires(t *testing.T) {
	// N-gram: a sentence LONGER than maxPeriod bytes repeated, single line — so
	// period-tiling can't catch the full-sentence unit and line-block sees one
	// line; only word-trigram repetition survives to flag it.
	ngramText := strings.Repeat("the kernel adjudicates every tool call before it dispatches anything. ", 8)
	if r := Measure([]byte(ngramText), def()); r.NGramRepeat <= 0.5 {
		t.Fatalf("NGramRepeat=%.3f, want > 0.5 for a long repeated sentence", r.NGramRepeat)
	}
	// Line-block: identical lines repeated.
	if r := Measure([]byte(strings.Repeat("ALERT: disk full\n", 20)), def()); r.LineBlockRepeat <= 0.5 {
		t.Fatalf("LineBlockRepeat=%.3f, want > 0.5 for a repeated line", r.LineBlockRepeat)
	}
	// Period: a tiled multi-char unit with no whitespace and one line.
	if r := Measure([]byte(strings.Repeat("abc", 80)), def()); r.PeriodRepeat <= 0.5 {
		t.Fatalf("PeriodRepeat=%.3f, want > 0.5 for abcabc…", r.PeriodRepeat)
	}
	// Period reason is unambiguous for a single-rune runaway (n-gram=0, line=0).
	r := Measure([]byte(strings.Repeat("A", 200)), def())
	if !strings.Contains(strings.Join(r.Reasons, " "), "period-1") {
		t.Fatalf("single-rune runaway reason %v should name period-1", r.Reasons)
	}
}

func TestMaxCharsVerbosity(t *testing.T) {
	// A coherent (non-repetitive) but long answer trips ONLY the length check.
	long := coherent + " " + coherent // ~2x, still low repeat
	r := Measure([]byte(long), Limits{MaxRepeat: DefaultMaxRepeat, MaxChars: 100, NGram: DefaultNGram})
	if !r.Degenerate {
		t.Fatalf("text of %d chars should exceed max-chars 100", r.Chars)
	}
	foundVerbose := false
	for _, rs := range r.Reasons {
		if strings.Contains(rs, "verbose") {
			foundVerbose = true
		}
	}
	if !foundVerbose {
		t.Fatalf("expected a verbose reason, got %v", r.Reasons)
	}
}

func TestDisabledChecksNeverTrip(t *testing.T) {
	// MaxRepeat<=0 disables the repeat check even on a blatant loop; MaxChars<=0
	// disables the length check even on a long text.
	loop := strings.Repeat("A", 500)
	r := Measure([]byte(loop), Limits{MaxRepeat: 0, MaxChars: 0, NGram: DefaultNGram})
	if r.Degenerate {
		t.Fatalf("all checks disabled but text flagged degenerate: %v", r.Reasons)
	}
	// The fraction is still REPORTED (informational) even with the check disabled.
	if r.PeriodRepeat == 0 {
		t.Fatalf("PeriodRepeat should still be reported for a runaway even when the check is disabled")
	}
}

func TestRepeatFractionIsMaxOfSignals(t *testing.T) {
	r := Measure([]byte(strings.Repeat("abc", 80)), def())
	if r.RepeatFraction < r.NGramRepeat || r.RepeatFraction < r.LineBlockRepeat || r.RepeatFraction < r.PeriodRepeat {
		t.Fatalf("RepeatFraction=%.3f is not >= each signal (ngram=%.3f line=%.3f period=%.3f)",
			r.RepeatFraction, r.NGramRepeat, r.LineBlockRepeat, r.PeriodRepeat)
	}
}

func TestNGramFloorOverride(t *testing.T) {
	// NGram<=0 falls back to DefaultNGram and is echoed in the report.
	r := Measure([]byte(coherent), Limits{MaxRepeat: DefaultMaxRepeat, NGram: 0})
	if r.NGram != DefaultNGram {
		t.Fatalf("NGram=%d, want fallback %d", r.NGram, DefaultNGram)
	}
}

func TestEmptyAndTinyInputs(t *testing.T) {
	for _, in := range []string{"", " ", "\n\n", "ok"} {
		r := Measure([]byte(in), def())
		if r.Degenerate {
			t.Fatalf("tiny/empty input %q flagged degenerate: %v", in, r.Reasons)
		}
	}
}
