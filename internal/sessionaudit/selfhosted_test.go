package sessionaudit

import (
	"strings"
	"testing"
	"time"
)

func outTok(n int64) ModelCounts { return ModelCounts{Turns: 1, Output: n, CacheRead: n * 10} }

func wantPct(t *testing.T, v *float64, want float64) {
	t.Helper()
	if v == nil {
		t.Fatalf("fraction is absent, want %.3f", want)
	}
	if diff := *v - want; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("fraction = %.6f, want %.6f", *v, want)
	}
}

// TestSelfHostedShareIsTheFractionOfKNOWNPlacement is the headline claim: with 300
// output tokens on our own hardware and 100 at a vendor, the share is 75% — and the
// 600 tokens whose placement nobody recorded change neither side of that fraction.
func TestSelfHostedShareIsTheFractionOfKNOWNPlacement(t *testing.T) {
	s := FoldSelfHostedShare(map[string]ModelCounts{
		BucketDevice:      outTok(100),
		BucketFleet:       outTok(200),
		BucketAnthropic:   outTok(100),
		BucketOpenWeights: outTok(600),
	})
	wantPct(t, s.OutputShare, 0.75)
	wantPct(t, s.Coverage, 400.0/1000.0)
	if s.SelfHosted.Output != 300 || s.Vendor.Output != 100 || s.Unattributable.Output != 600 {
		t.Fatalf("fold = %+v", s)
	}
	if len(s.UnattributableBuckets) != 1 || s.UnattributableBuckets[0] != BucketOpenWeights {
		t.Fatalf("unattributable buckets = %v, want just %q — an operator needs to know WHICH signal is missing",
			s.UnattributableBuckets, BucketOpenWeights)
	}
}

// TestUnattributableVolumeIsNeverScoredEitherWay is the property the whole design of
// this fold exists for. Pour unattributable volume into the corpus: the reported
// self-hosted share must not move ONE BIT, and coverage must fall to say so.
//
// If unattributable volume were folded into the denominator (the obvious, wrong
// implementation), this share would collapse from 75% toward 7.5% purely because fak
// does not yet record a placement — punishing us for our own missing wiring and making
// later attribution work look like a self-hosting win it isn't.
func TestUnattributableVolumeIsNeverScoredEitherWay(t *testing.T) {
	base := map[string]ModelCounts{
		BucketDevice:    outTok(300),
		BucketAnthropic: outTok(100),
	}
	before := FoldSelfHostedShare(base)
	wantPct(t, before.OutputShare, 0.75)
	wantPct(t, before.Coverage, 1.0)

	base[BucketOpenWeights] = outTok(3600)
	base[BucketUnknownPrices] = outTok(400)
	after := FoldSelfHostedShare(base)

	wantPct(t, after.OutputShare, 0.75) // unmoved
	if after.Coverage == nil || *after.Coverage >= *before.Coverage {
		t.Fatalf("coverage did not fall (%v -> %v) after adding unattributable volume",
			*before.Coverage, after.Coverage)
	}
	wantPct(t, after.Coverage, 400.0/4400.0)
}

// TestACorpusWithNoPlacementSignalHasNoShareAtAll pins the difference between "we
// self-host nothing" and "we cannot tell". Reporting 0% for the second is a claim fak
// has no evidence for, and it is exactly the claim today's corpus would produce.
func TestACorpusWithNoPlacementSignalHasNoShareAtAll(t *testing.T) {
	s := FoldSelfHostedShare(map[string]ModelCounts{
		BucketOpenWeights:   outTok(500),
		BucketUnknownPrices: outTok(500),
	})
	if s.OutputShare != nil {
		t.Fatalf("share = %.3f over a corpus with zero attributed output; want absent", *s.OutputShare)
	}
	wantPct(t, s.Coverage, 0.0) // zero coverage IS knowable, and it is the finding
	if fmtPct(s.OutputShare) != "-" {
		t.Fatalf("an absent share must render as %q, not %q", "-", fmtPct(s.OutputShare))
	}
	// And an empty corpus has neither number.
	if e := FoldSelfHostedShare(nil); e.OutputShare != nil || e.Coverage != nil {
		t.Fatalf("empty corpus = %+v, want both fractions absent", e)
	}
}

// TestZeroSelfHostedIsReportedWhenItIsActuallyKNOWN is the other side of the same coin:
// when every attributed token demonstrably went to a vendor, 0% is the truth and must be
// reported as a number, not withheld.
func TestZeroSelfHostedIsReportedWhenItIsActuallyKNOWN(t *testing.T) {
	s := FoldSelfHostedShare(map[string]ModelCounts{
		BucketAnthropic:  outTok(700),
		BucketVendorOpen: outTok(300),
	})
	wantPct(t, s.OutputShare, 0.0)
	wantPct(t, s.Coverage, 1.0)
}

// TestHarnessTurnsAreNotInferenceVolume pins that non-billed `<synthetic>` turns move
// neither fraction. No model served them, so they are not evidence about placement in
// either direction — but they are still counted, so the rows reconcile against
// PerBucket rather than silently vanishing.
func TestHarnessTurnsAreNotInferenceVolume(t *testing.T) {
	base := map[string]ModelCounts{BucketFleet: outTok(300), BucketAnthropic: outTok(100)}
	before := FoldSelfHostedShare(base)
	base[BucketNonBilled] = outTok(9000)
	after := FoldSelfHostedShare(base)

	wantPct(t, after.OutputShare, *before.OutputShare)
	wantPct(t, after.Coverage, *before.Coverage)
	if after.NonBilled.Output != 9000 {
		t.Fatalf("harness volume = %d, want it counted and reported (%+v)", after.NonBilled.Output, after)
	}
	if after.InferenceOutput() != 400 {
		t.Fatalf("inference output = %d, want 400 — harness turns are not inference", after.InferenceOutput())
	}
}

// TestTheFoldIsTotalOverTheBucketVocabulary keeps this file honest as the vocabulary
// grows: every bucket must land in exactly one category, so adding a bucket without
// teaching BucketIsSelfHosted about it fails here instead of silently disappearing from
// a report. An unrecognized bucket must land on the SAFE side (unattributable), never be
// counted as a self-hosted saving.
func TestTheFoldIsTotalOverTheBucketVocabulary(t *testing.T) {
	all := []string{
		BucketNonBilled, BucketAnthropic, BucketGoogle, BucketOpenAI,
		BucketOpenWeights, BucketDevice, BucketFleet, BucketVendorOpen, BucketUnknownPrices,
	}
	corpus := make(map[string]ModelCounts, len(all)+1)
	for _, b := range all {
		corpus[b] = ModelCounts{Turns: 1, Output: 1}
	}
	corpus["a bucket invented next quarter"] = ModelCounts{Turns: 1, Output: 1}

	s := FoldSelfHostedShare(corpus)
	total := s.SelfHosted.Output + s.Vendor.Output + s.Unattributable.Output + s.NonBilled.Output
	if total != int64(len(corpus)) {
		t.Fatalf("fold accounted for %d of %d buckets: %+v", total, len(corpus), s)
	}
	if s.SelfHosted.Output != 2 {
		t.Fatalf("self-hosted output = %d, want exactly device+fleet = 2 (%+v)", s.SelfHosted.Output, s)
	}
	if !containsString(s.UnattributableBuckets, "a bucket invented next quarter") {
		t.Fatalf("an unknown bucket must land in unattributable, got %v", s.UnattributableBuckets)
	}
}

// TestCacheReadNeverEntersTheFraction pins the measure. Prompt-cache reuse is a
// provider-side billing artifact with no counterpart on a self-hosted rung, so a corpus
// whose cache-read volume is lopsided must not shift the share.
func TestCacheReadNeverEntersTheFraction(t *testing.T) {
	s := FoldSelfHostedShare(map[string]ModelCounts{
		BucketDevice:    {Turns: 1, Output: 300, CacheRead: 0},
		BucketAnthropic: {Turns: 1, Output: 100, CacheRead: 10_000_000},
	})
	wantPct(t, s.OutputShare, 0.75)
}

// TestTheShareIsRenderedWithItsCoverage pins that the number never appears in a report
// without the caveat that qualifies it. A bare "62% self-hosted" over 4% coverage is a
// misleading headline, and the two must travel together.
func TestTheShareIsRenderedWithItsCoverage(t *testing.T) {
	agg := Aggregate{PerBucket: map[string]ModelCounts{
		BucketDevice:      outTok(300),
		BucketAnthropic:   outTok(100),
		BucketOpenWeights: outTok(3600),
	}}
	var b strings.Builder
	renderSelfHostedShare(&b, agg)
	md := b.String()
	for _, want := range []string{"75.0%", "10.0%", BucketOpenWeights} {
		if !strings.Contains(md, want) {
			t.Fatalf("rendered share is missing %q:\n%s", want, md)
		}
	}

	// With no placement signal anywhere, the report must say so in words rather than
	// print a zero.
	var b2 strings.Builder
	renderSelfHostedShare(&b2, Aggregate{PerBucket: map[string]ModelCounts{BucketOpenWeights: outTok(500)}})
	if md2 := b2.String(); !strings.Contains(md2, "no turn in this corpus") {
		t.Fatalf("a corpus with no placement signal must say so:\n%s", md2)
	} else if strings.Contains(md2, "0.0% self-hosted") {
		t.Fatalf("unknown placement must not be rendered as 0%%:\n%s", md2)
	}
}

// TestTheFullReportCarriesTheHeadlineNumber pins the wiring, not just the arithmetic:
// the fold reaches the report a human actually reads.
func TestTheFullReportCarriesTheHeadlineNumber(t *testing.T) {
	agg := AggregateSessions([]Session{{
		Session:  "s",
		Path:     "/x/s.jsonl",
		Models:   map[string]int64{"claude-opus-5": 1},
		PerModel: map[string]ModelCounts{"claude-opus-5": outTok(100)},
		Tokens:   TokenCounts{Output: 100},
	}})
	md := ReportMarkdown([]Session{{Session: "s", Path: "/x/s.jsonl"}}, agg, "", nil, false, 0, 1, nil, time.Now())
	if !strings.Contains(md, "## Self-hosted share") {
		t.Fatal("the fleet report does not carry the self-hosted share section")
	}
}

func containsString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
