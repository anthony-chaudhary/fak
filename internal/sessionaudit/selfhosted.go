package sessionaudit

import "sort"

// THE HEADLINE NUMBER (epic #5416): what fraction of a corpus's tokens were served on
// hardware the engineer or their organization operates?
//
// The whole three-stratum design — local models for small work, company-hosted models
// for the middle, a frontier vendor only for what genuinely needs it — is a claim about
// this one fraction. So it gets COMPUTED, over a real corpus, rather than asserted; and
// it gets computed in the one place a corpus is already rolled up by placement bucket.
//
// The honest part is the denominator. It is NOT every token:
//
//   - Volume with NO placement signal (an open-weights model id that could have been
//     served from a laptop or bought from Groq — see BucketOpenWeights) is
//     UNATTRIBUTABLE. Folding it into the denominator would bias the number down by
//     our own ignorance and make adoption look like it was rising whenever attribution
//     improved; folding it into the numerator would claim a saving we cannot show.
//     So it goes in NEITHER, and Coverage reports how much of the corpus the fraction
//     actually speaks for.
//   - Non-billed harness turns (`<synthetic>`) are excluded entirely. No model served
//     them, so they are not inference volume and belong on no side of a placement
//     question. They are still reported, so the rows reconcile against PerBucket.
//
// This is BucketIsSelfHosted's two-value contract carried through to arithmetic: a
// caller must decide explicitly what to do with the unknown remainder, and it must
// never be scored as a saving.

// SelfHostedShare is the placement fold of a bucket rollup: every bucket's counts
// sorted into self-hosted, vendor, unattributable, and non-billed, plus the two
// fractions that summarize them.
//
// The fractions are pointers because this package distinguishes "the answer is zero"
// from "there is no answer" everywhere else it reports a ratio (ReadOnlyFrac, IORatio,
// CacheHitFrac). A corpus in which nothing carries a placement signal has NO self-hosted
// share — reporting 0% there would read as "we self-host nothing", which is a different
// and much stronger claim than "we cannot tell".
type SelfHostedShare struct {
	// SelfHosted is volume served on device or fleet hardware — the numerator.
	SelfHosted ModelCounts `json:"self_hosted"`
	// Vendor is volume a third-party API served, including open-weights models bought
	// from a vendor. Known placement, just not ours.
	Vendor ModelCounts `json:"vendor"`
	// Unattributable is volume whose bucket names a model but not a place. Excluded
	// from both sides of Share; it is what Coverage is missing.
	Unattributable ModelCounts `json:"unattributable"`
	// NonBilled is harness-injected volume, excluded from every fraction.
	NonBilled ModelCounts `json:"non_billed"`

	// UnattributableBuckets names the buckets that produced Unattributable, sorted by
	// output volume, so an operator sees WHICH signal is missing rather than only how
	// much. This is the actionable half: it names the wiring left to do.
	UnattributableBuckets []string `json:"unattributable_buckets,omitempty"`

	// OutputShare is self-hosted output / attributed output — the headline. Output is
	// the measure because it is the volume a GPU actually generated. Cache-read tokens
	// are deliberately excluded from the fractions: prompt-cache reuse is a
	// provider-side artifact whose accounting has no counterpart on a self-hosted rung,
	// so mixing it in would compare unlike things.
	OutputShare *float64 `json:"output_share"`
	// Coverage is attributed output / (attributed + unattributable) output: the share
	// of real inference volume OutputShare speaks for. A high share over low coverage
	// is not evidence of anything.
	Coverage *float64 `json:"coverage"`
}

// AttributedOutput is the denominator of OutputShare — output volume whose placement is
// known either way.
func (s SelfHostedShare) AttributedOutput() int64 {
	return s.SelfHosted.Output + s.Vendor.Output
}

// InferenceOutput is all output a model actually generated: attributed plus
// unattributable, excluding harness turns.
func (s SelfHostedShare) InferenceOutput() int64 {
	return s.AttributedOutput() + s.Unattributable.Output
}

// FoldSelfHostedShare sorts a bucket rollup (Aggregate.PerBucket) by placement and
// computes the self-hosted fraction over the volume whose placement is known.
//
// It is total over the bucket vocabulary by construction: every bucket lands in exactly
// one of the four categories, and a bucket BucketIsSelfHosted does not recognize lands
// in Unattributable — the safe side — rather than being dropped.
func FoldSelfHostedShare(perBucket map[string]ModelCounts) SelfHostedShare {
	var s SelfHostedShare
	for bucket, c := range perBucket {
		if bucket == BucketNonBilled {
			// Not inference volume. Reported, never scored.
			addCounts(&s.NonBilled, c)
			continue
		}
		selfHosted, known := BucketIsSelfHosted(bucket)
		switch {
		case !known:
			addCounts(&s.Unattributable, c)
			if c.Output > 0 || c.Turns > 0 {
				s.UnattributableBuckets = append(s.UnattributableBuckets, bucket)
			}
		case selfHosted:
			addCounts(&s.SelfHosted, c)
		default:
			addCounts(&s.Vendor, c)
		}
	}
	sort.Slice(s.UnattributableBuckets, func(i, j int) bool {
		a, b := perBucket[s.UnattributableBuckets[i]], perBucket[s.UnattributableBuckets[j]]
		if a.Output == b.Output {
			return s.UnattributableBuckets[i] < s.UnattributableBuckets[j]
		}
		return a.Output > b.Output
	})
	s.OutputShare = ratio(s.SelfHosted.Output, s.AttributedOutput())
	s.Coverage = ratio(s.AttributedOutput(), s.InferenceOutput())
	return s
}

func addCounts(dst *ModelCounts, src ModelCounts) {
	dst.Turns += src.Turns
	dst.Input += src.Input
	dst.Output += src.Output
	dst.CacheRead += src.CacheRead
	dst.CacheCreate += src.CacheCreate
}
