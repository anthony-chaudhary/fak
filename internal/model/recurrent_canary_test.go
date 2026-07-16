package model

import (
	"strings"
	"testing"
	"time"
)

// passingEvidence builds a complete, current, all-arms-present evidence set that
// clears every promotion floor, with the long/greedy bucket doubling as the retained
// #4273 regression canary. Tests mutate one field to prove each fail-closed path.
func passingEvidence(now time.Time) []BucketEvidence {
	fx := Canary4273Fixture()
	var ev []BucketEvidence
	for _, b := range RequiredBuckets() {
		e := BucketEvidence{
			Bucket:            b,
			PromptDigest:      "sha256:" + b.ID(),
			TokenCount:        b.MinPromptTokens,
			ModelDigest:       "gguf:qwen35-27b-q4km",
			FakCommit:         "abc1234",
			ReferenceCommit:   "hf-transformers@def5678",
			Sampling:          SamplingParams{Profile: b.Profile, Temperature: 0.7, TopP: 0.95, TopK: 40, Seed: 7},
			ArtifactLocations: []string{"blob://canary/" + b.ID() + "/logits.f32"},
			RealWeightArm:     true,
			ReferenceArm:      true,
			ObservedAt:        now,
			Metrics: ComparisonMetrics{
				TopLogitOverlap:       1.0,
				ArgmaxAgreement:       1.0,
				RepetitionScore:       1.0,
				RequiredLabelsPresent: true,
				GroundingScore:        1.0,
			},
		}
		if b.Kind == BucketLong && b.Profile == DecodeGreedy {
			e.Regression4273 = true
			e.PromptDigest = fx.PromptDigest
			e.TokenCount = fx.TokenCount
		}
		ev = append(ev, e)
	}
	return ev
}

func TestPromotionCleanWhenAllBucketsCurrent(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	v := EvaluatePromotion("qwen35", RequiredBuckets(), passingEvidence(now), DefaultPromotionThresholds(), now)
	if v.Blocked {
		t.Fatalf("expected clean promotion, blocked with reasons:\n%s", strings.Join(v.Reasons, "\n"))
	}
}

// mutateBucket returns the evidence set with fn applied to the first bucket matching id.
func mutateBucket(ev []BucketEvidence, id string, fn func(*BucketEvidence)) []BucketEvidence {
	out := make([]BucketEvidence, len(ev))
	copy(out, ev)
	for i := range out {
		if out[i].Bucket.ID() == id {
			fn(&out[i])
			break
		}
	}
	return out
}

func assertBlockedBecause(t *testing.T, v PromotionVerdict, want string) {
	t.Helper()
	if !v.Blocked {
		t.Fatalf("expected promotion blocked, got clean")
	}
	for _, r := range v.Reasons {
		if strings.Contains(r, want) {
			return
		}
	}
	t.Fatalf("no reason contained %q; reasons:\n%s", want, strings.Join(v.Reasons, "\n"))
}

func TestPromotionFailsClosedOnMissingRealWeightArm(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	ev := mutateBucket(passingEvidence(now), "medium/greedy", func(e *BucketEvidence) { e.RealWeightArm = false })
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "real-weight arm missing")
}

func TestPromotionFailsClosedOnMissingReferenceArm(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	ev := mutateBucket(passingEvidence(now), "long/greedy", func(e *BucketEvidence) { e.ReferenceArm = false })
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "reference arm missing")
}

func TestPromotionFailsClosedOnMissingBucket(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	full := passingEvidence(now)
	// Drop the long/sampled bucket entirely.
	var ev []BucketEvidence
	for _, e := range full {
		if e.Bucket.ID() == "long/sampled" {
			continue
		}
		ev = append(ev, e)
	}
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "no observed evidence")
}

func TestPromotionBlocksStaleEvidence(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	stale := now.Add(-90 * 24 * time.Hour)
	ev := mutateBucket(passingEvidence(now), "medium/sampled", func(e *BucketEvidence) { e.ObservedAt = stale })
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "stale")
}

func TestPromotionBlocksIncompleteMetadata(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	ev := mutateBucket(passingEvidence(now), "short/greedy", func(e *BucketEvidence) { e.ModelDigest = "" })
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "model-digest missing")

	ev2 := mutateBucket(passingEvidence(now), "short/greedy", func(e *BucketEvidence) { e.ArtifactLocations = nil })
	v2 := EvaluatePromotion("qwen35", RequiredBuckets(), ev2, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v2, "no raw artifact locations")
}

func TestPromotionBlocksSubHorizonPrompt(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	ev := mutateBucket(passingEvidence(now), "long/sampled", func(e *BucketEvidence) { e.TokenCount = 10 })
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "below horizon floor")
}

// TestGenericGrammaticalFailsGroundingDespiteRepetition pins acceptance §3: a generic
// but grammatical answer (perfect repetition score, but few grounded source entities)
// is blocked ON GROUNDING — passing repetition alone is not sufficient.
func TestGenericGrammaticalFailsGroundingDespiteRepetition(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	ev := mutateBucket(passingEvidence(now), "long/sampled", func(e *BucketEvidence) {
		e.Metrics.RepetitionScore = 1.0 // grammatical, non-repetitive — repetition passes
		e.Metrics.GroundingScore = 0.10 // but grounds almost no source entities
	})
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "grounding")
	for _, r := range v.Reasons {
		if strings.Contains(r, "long/sampled") && strings.Contains(r, "repetition score") {
			t.Fatalf("grounding failure must not masquerade as a repetition failure: %q", r)
		}
	}
}

func TestRetained4273FixtureRequired(t *testing.T) {
	now := time.Unix(1_700_000_000, 0)
	// Strip the regression flag/digest from the canary bucket.
	ev := mutateBucket(passingEvidence(now), "long/greedy", func(e *BucketEvidence) {
		e.Regression4273 = false
		e.PromptDigest = "sha256:long/greedy"
	})
	v := EvaluatePromotion("qwen35", RequiredBuckets(), ev, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v, "#4273 regression canary")

	// A digest drift on the retained canary must also block.
	ev2 := mutateBucket(passingEvidence(now), "long/greedy", func(e *BucketEvidence) {
		e.PromptDigest = "sha256:drifted"
	})
	v2 := EvaluatePromotion("qwen35", RequiredBuckets(), ev2, DefaultPromotionThresholds(), now)
	assertBlockedBecause(t, v2, "#4273 regression canary")
}

func TestRequiredBucketsMatrixShape(t *testing.T) {
	bs := RequiredBuckets()
	if len(bs) != 6 {
		t.Fatalf("want 6 buckets (3 lengths x 2 profiles), got %d", len(bs))
	}
	seen := map[string]bool{}
	for _, b := range bs {
		seen[b.ID()] = true
	}
	for _, id := range []string{"short/greedy", "short/sampled", "medium/greedy", "medium/sampled", "long/greedy", "long/sampled"} {
		if !seen[id] {
			t.Fatalf("missing required bucket %s", id)
		}
	}
}

func TestRepetitionScoreDetectsDegenerateLoop(t *testing.T) {
	clean := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	cs := RepetitionScore(clean, 2)
	if cs < 0.8 {
		t.Fatalf("clean stream should score high, got %.3f", cs)
	}
	// A degenerate repeat loop (the #4273 collapse shape) scores near 0.
	loop := []int{9, 9, 9, 9, 9, 9, 9, 9}
	ls := RepetitionScore(loop, 2)
	if ls > 0.2 {
		t.Fatalf("degenerate loop should score low, got %.3f", ls)
	}
	if cs <= ls {
		t.Fatalf("clean (%.3f) must score strictly above a degenerate loop (%.3f)", cs, ls)
	}
}

func TestGroundingScoreTiedToSourceEntities(t *testing.T) {
	entities := []string{"Acme", "Q3", "revenue", "Bangalore"}
	grounded := "Acme Q3 revenue rose in Bangalore by twelve percent."
	if s := GroundingScore(grounded, entities); s < 0.99 {
		t.Fatalf("grounded answer should score ~1, got %.3f", s)
	}
	// Generic but grammatical: fluent, names no source entities.
	generic := "The company performed well this quarter and results were positive overall."
	if s := GroundingScore(generic, entities); s > 0.25 {
		t.Fatalf("generic answer must fail grounding, got %.3f", s)
	}
	// Fail-closed: non-empty entities against empty output.
	if s := GroundingScore("", entities); s != 0 {
		t.Fatalf("empty output must ground nothing, got %.3f", s)
	}
}

func TestArgmaxAgreementAndTopLogitOverlap(t *testing.T) {
	if p := ArgmaxAgreement([]int{1, 2, 3}, []int{1, 2, 3}); p != 1 {
		t.Fatalf("identical ids should have parity 1, got %.3f", p)
	}
	if p := ArgmaxAgreement([]int{1, 2, 3}, []int{1, 9, 3}); p >= 1 {
		t.Fatalf("one mismatch should drop parity below 1, got %.3f", p)
	}
	// Length mismatch is fail-closed on the missing tail.
	if p := ArgmaxAgreement([]int{1, 2, 3, 4}, []int{1, 2}); p > 0.6 {
		t.Fatalf("missing tail should drag parity down, got %.3f", p)
	}
	over := TopLogitOverlap([][]int{{1, 2, 3}}, [][]int{{1, 2, 3}})
	if over != 1 {
		t.Fatalf("identical top-k should overlap 1, got %.3f", over)
	}
	if o := TopLogitOverlap([][]int{{1, 2, 3}}, [][]int{{4, 5, 6}}); o != 0 {
		t.Fatalf("disjoint top-k should overlap 0, got %.3f", o)
	}
}

func TestDigestPromptStableAndDistinct(t *testing.T) {
	a := DigestPrompt([]int{1, 2, 3})
	b := DigestPrompt([]int{1, 2, 3})
	c := DigestPrompt([]int{1, 2, 4})
	if a != b {
		t.Fatalf("digest must be stable: %s != %s", a, b)
	}
	if a == c {
		t.Fatalf("distinct prompts must digest differently")
	}
	if !strings.HasPrefix(a, "sha256:") {
		t.Fatalf("digest must carry sha256: prefix, got %s", a)
	}
}
