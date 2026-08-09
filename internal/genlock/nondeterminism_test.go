package genlock_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/genlock"
	"github.com/anthony-chaudhary/fak/internal/marketing"
)

// TestNaiveOutputCheckIsWrongForARealFakGenerator is the premise of the whole package,
// checked against a generator that is actually in this repo rather than asserted.
//
// `fak marketing aeo` (cmd/fak/marketing.go) takes `now := time.Now()` and hands it to
// four renderers, each of which stamps it into the artifact it produces. Two of those
// renderers take NO other input at all: marketing.LlmsTermsText and
// marketing.DisambiguationTermsFeed render the hard-coded AEODisambiguationTerms roster.
// So across two runs the input is not merely unchanged, it is bit-for-bit the same Go
// slice — and the output bytes still differ.
//
// That is what makes "did the output change?" the wrong question. Asked of these files
// it always answers yes, so a freshness check keyed on the output can only ever say
// "rebuild", and every run rewrites every artifact. Ask it of the input and the answer
// is decidable: the clock-free rendering (the same functions with the zero time, which
// their own when.IsZero() guards already make clock-free) is stable, so that is what the
// lock hashes.
func TestNaiveOutputCheckIsWrongForARealFakGenerator(t *testing.T) {
	// Mimic the real caller: two invocations, each taking its own time.Now(). RFC3339 is
	// second-resolution, so the runs have to straddle a second boundary the way two
	// separate `fak marketing aeo` invocations in a session do.
	first := time.Now()
	runA := map[string][]byte{
		"llms-terms.txt":   []byte(marketing.LlmsTermsText(first)),
		"llms-updates.txt": []byte(marketing.LlmsUpdatesText(nil, first)),
	}
	termsA, err := marketing.DisambiguationTermsFeed(first)
	if err != nil {
		t.Fatal(err)
	}
	runA["docs/marketing/disambiguation-terms.json"] = termsA

	time.Sleep(1100 * time.Millisecond)

	second := time.Now()
	runB := map[string][]byte{
		"llms-terms.txt":   []byte(marketing.LlmsTermsText(second)),
		"llms-updates.txt": []byte(marketing.LlmsUpdatesText(nil, second)),
	}
	termsB, err := marketing.DisambiguationTermsFeed(second)
	if err != nil {
		t.Fatal(err)
	}
	runB["docs/marketing/disambiguation-terms.json"] = termsB

	total := 0
	for artifact, a := range runA {
		b := runB[artifact]
		if bytes.Equal(a, b) {
			t.Errorf("%s was byte-identical across two runs. If this generator became "+
				"reproducible that is good news, but re-derive the package doc before "+
				"trusting it — the input-hash design is justified by this failing.", artifact)
			continue
		}
		total += len(a)
		t.Logf("%s: %d bytes, differs across two runs over identical input (%s vs %s)",
			artifact, len(a), first.UTC().Format(time.RFC3339), second.UTC().Format(time.RFC3339))
	}
	t.Logf("naive output-comparison freshness would rewrite %d bytes per no-op run across %d artifacts",
		total, len(runA))

	// And the other half: the CLOCK-FREE identity the lock actually hashes is stable
	// across exactly those two runs. Without this the package would have replaced one
	// undefined comparison with another.
	zeroTerms, err := marketing.DisambiguationTermsFeed(time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	zeroTermsAgain, err := marketing.DisambiguationTermsFeed(time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ name, a, b string }{
		{"llms-terms.txt", marketing.LlmsTermsText(time.Time{}), marketing.LlmsTermsText(time.Time{})},
		{"llms-updates.txt", marketing.LlmsUpdatesText(nil, time.Time{}), marketing.LlmsUpdatesText(nil, time.Time{})},
		{"disambiguation-terms.json", string(zeroTerms), string(zeroTermsAgain)},
	} {
		if genlock.Sum([]byte(tc.a)) != genlock.Sum([]byte(tc.b)) {
			t.Errorf("the clock-free input identity for %s is itself unstable; there is nothing "+
				"left to key freshness on", tc.name)
		}
	}
}

// TestLockSkipsTheClockStampedGenerator closes the loop: the same real renderers, driven
// through the lock, write once and then stop — even though every subsequent render would
// have produced different bytes.
func TestLockSkipsTheClockStampedGenerator(t *testing.T) {
	root := t.TempDir()
	const artifact = "llms-terms.txt"

	// The input is the clock-free rendering. This is the fak analogue of hashing the HTML
	// that produced a PDF rather than the PDF.
	input := []byte(marketing.LlmsTermsText(time.Time{}))

	l := openLock(t, root)
	if got, err := l.Sync(artifact, input, func() ([]byte, error) {
		return []byte(marketing.LlmsTermsText(time.Now())), nil
	}); err != nil || got != genlock.Wrote {
		t.Fatalf("first Sync = %v, %v; want Wrote", got, err)
	}
	if err := l.Save(); err != nil {
		t.Fatal(err)
	}
	first := read(t, root, artifact)

	// Ten more runs, each of which a naive output comparison would call stale.
	for i := 0; i < 10; i++ {
		l2 := openLock(t, root)
		got, err := l2.Sync(artifact, input, func() ([]byte, error) {
			return []byte(marketing.LlmsTermsText(time.Now())), nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if got != genlock.Skipped {
			t.Fatalf("run %d = %v, want Skipped", i+2, got)
		}
		if err := l2.Save(); err != nil {
			t.Fatal(err)
		}
	}
	if body := read(t, root, artifact); body != first {
		t.Error("the artifact changed across ten skipped runs")
	}
	t.Logf("11 runs, %d bytes written: the first one. Naive output comparison would have "+
		"written %d.", len(first), 11*len(first))
}
