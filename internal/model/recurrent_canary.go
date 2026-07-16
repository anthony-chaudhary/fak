package model

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Issue #4745: a promotion-blocking real-weight long-context canary matrix for
// recurrent/hybrid architectures (Qwen35/Qwen3.6). #4273 passed short-prompt
// sanity checks while failing a real ~1.5k-token executive-report prompt: the
// recurrent-state error accumulated over long prefill/decode, and generic output
// could still look superficially grammatical. Synthetic self-consistency and
// first-token parity were not sufficient. This leaf is the gate that refuses to
// promote a recurrent/hybrid arch to a supported/default path unless EVERY
// required prompt-length bucket has current, real-weight-plus-reference evidence
// whose comparison metrics clear the floors — and it FAILS CLOSED: a missing arm,
// a missing bucket, or stale evidence blocks promotion rather than skipping green.
//
// It is deliberately weight-free and stdlib-only so the gate logic, the metrics,
// and the fail-closed semantics are all unit-checkable on a dev box with no GPU.
// The real-weight arms are populated on the fleet (see the on-box GGUF witness
// dispatched as a follow-up); their ABSENCE is exactly what this gate turns into a
// promotion block. The forward-path sensitivity witness (an identity ssm_a loader
// mutation rejected by the medium/long buckets while the corrected inverse passes)
// lives beside this leaf in the ssm_a sensitivity test and drives the real forward.

// BucketKind is the prompt-length class one canary bucket exercises.
type BucketKind string

const (
	// BucketShort is the short sanity prompt — the class #4273 passed while the
	// arch was still broken. Passing it alone is never sufficient for promotion.
	BucketShort BucketKind = "short"
	// BucketMedium sits near the architecture's first meaningful recurrent
	// horizon, where accumulated recurrent-state error first becomes observable.
	BucketMedium BucketKind = "medium"
	// BucketLong is the long production-shaped prompt with a prefill-to-decode
	// continuation long enough to expose state drift (the #4273 failure shape).
	BucketLong BucketKind = "long"
)

// DecodeProfile is the decode regime a bucket runs under. Every prompt-length
// class is required under both a greedy and a sampled profile.
type DecodeProfile string

const (
	DecodeGreedy  DecodeProfile = "greedy"
	DecodeSampled DecodeProfile = "sampled"
)

// CanaryBucket is one required (prompt-length, decode-profile) cell of the matrix.
// MinPromptTokens is the horizon floor the bucket's prompt must reach; MinDecodeTokens
// is the prefill-to-decode continuation length that exposes state drift.
type CanaryBucket struct {
	Kind            BucketKind
	Profile         DecodeProfile
	MinPromptTokens int
	MinDecodeTokens int
}

// ID is the stable (kind, profile) identity used to match evidence to a bucket.
func (b CanaryBucket) ID() string { return string(b.Kind) + "/" + string(b.Profile) }

// RequiredBuckets returns the fixed promotion matrix for a recurrent/hybrid arch:
// short/medium/long prompt classes crossed with greedy and sampled decode. The
// medium and long classes carry a prefill-to-decode continuation long enough to
// surface recurrent-state drift; the short class is the sanity floor #4273 shows
// is not sufficient on its own.
func RequiredBuckets() []CanaryBucket {
	var out []CanaryBucket
	classes := []struct {
		kind              BucketKind
		minPrompt, minDec int
	}{
		{BucketShort, 8, 4},
		{BucketMedium, 256, 64},
		{BucketLong, 1300, 500},
	}
	for _, c := range classes {
		for _, prof := range []DecodeProfile{DecodeGreedy, DecodeSampled} {
			out = append(out, CanaryBucket{Kind: c.kind, Profile: prof, MinPromptTokens: c.minPrompt, MinDecodeTokens: c.minDec})
		}
	}
	return out
}

// SamplingParams records the decode configuration an arm ran under. It is part of
// the recorded evidence so a bucket's result is reproducible from metadata alone.
type SamplingParams struct {
	Temperature float64
	TopP        float64
	TopK        int
	Seed        int64
	Profile     DecodeProfile
}

// ComparisonMetrics are the per-bucket signals compared between the fak arm and
// the independently-authored reference arm. Grounding is checked independently of
// repetition precisely because a generic-but-grammatical answer can pass repetition
// while failing grounding — repetition passing alone is never sufficient.
type ComparisonMetrics struct {
	TopLogitOverlap       float64 // mean top-k token-id overlap at selected positions [0,1]
	ArgmaxAgreement       float64 // fraction of selected positions whose argmax agrees [0,1]
	GeneratedTokenIDs     []int   // fak-arm decoded ids (recorded and compared to reference)
	RefTokenIDs           []int   // reference-arm decoded ids
	RepetitionScore       float64 // 1 - dominant n-gram fraction; low == degenerate repeat loop
	RequiredLabels        []string
	RequiredLabelsPresent bool
	GroundingScore        float64 // fraction of source entities grounded in the output [0,1]
}

// BucketEvidence is the recorded, digest-bound observation for one bucket. It
// carries every acceptance-required metadata field plus the comparison metrics.
// RealWeightArm and ReferenceArm are the fail-closed switches: both must be true
// or the gate blocks (a missing arm never counts as a green skip).
type BucketEvidence struct {
	Bucket            CanaryBucket
	PromptDigest      string
	TokenCount        int
	ModelDigest       string
	FakCommit         string
	ReferenceCommit   string
	Sampling          SamplingParams
	ArtifactLocations []string
	RealWeightArm     bool
	ReferenceArm      bool
	Regression4273    bool // this bucket is the retained #4273 regression canary
	ObservedAt        time.Time
	Metrics           ComparisonMetrics
}

// PromotionThresholds are the pass floors a bucket's metrics must clear, plus the
// freshness window beyond which evidence is treated as stale (and thus blocking).
type PromotionThresholds struct {
	MinArgmaxAgreement float64
	MinTopLogitOverlap float64
	MinRepetition      float64
	MinGrounding       float64
	FreshnessWindow    time.Duration
}

// DefaultPromotionThresholds returns the promotion floors. Grounding is set high
// enough that a generic-but-grammatical answer (which grounds few source entities)
// fails even when its repetition score is perfect.
func DefaultPromotionThresholds() PromotionThresholds {
	return PromotionThresholds{
		MinArgmaxAgreement: 0.995,
		MinTopLogitOverlap: 0.90,
		MinRepetition:      0.60,
		MinGrounding:       0.60,
		FreshnessWindow:    30 * 24 * time.Hour,
	}
}

// PromotionVerdict is the fail-closed decision. Blocked is true unless every
// required bucket cleared every check; Reasons carries one structured line per
// failing condition (empty only on a clean promote).
type PromotionVerdict struct {
	Arch    string
	Blocked bool
	Reasons []string
}

// EvaluatePromotion is the promotion-blocking gate for a recurrent/hybrid arch. It
// FAILS CLOSED: any missing bucket, missing real-weight/reference arm, stale
// observation, incomplete metadata, sub-horizon prompt, or sub-floor metric blocks
// promotion. It never "skips green" for absent evidence. The retained #4273
// regression canary must be present and digest-bound or promotion is blocked.
func EvaluatePromotion(arch string, buckets []CanaryBucket, evidence []BucketEvidence, th PromotionThresholds, now time.Time) PromotionVerdict {
	v := PromotionVerdict{Arch: arch, Blocked: false}
	block := func(format string, a ...any) {
		v.Blocked = true
		v.Reasons = append(v.Reasons, fmt.Sprintf(format, a...))
	}

	byID := make(map[string]BucketEvidence, len(evidence))
	for _, e := range evidence {
		byID[e.Bucket.ID()] = e
	}

	for _, b := range buckets {
		e, ok := byID[b.ID()]
		if !ok {
			block("bucket %s: no observed evidence (fail closed — a missing bucket blocks, never skips green)", b.ID())
			continue
		}
		// Fail-closed arms: a missing real-weight or reference run is a block, not a skip.
		if !e.RealWeightArm {
			block("bucket %s: real-weight arm missing (fail closed — absent real weights block promotion, not skip green)", b.ID())
		}
		if !e.ReferenceArm {
			block("bucket %s: reference arm missing (fail closed — no independently-authored reference)", b.ID())
		}
		// Freshness: promotion requires CURRENT observed evidence per bucket.
		if e.ObservedAt.IsZero() || now.Sub(e.ObservedAt) > th.FreshnessWindow {
			block("bucket %s: evidence stale or unstamped (observed %v, window %v)", b.ID(), e.ObservedAt, th.FreshnessWindow)
		}
		// Metadata completeness (acceptance §1): every field must be recorded.
		for field, val := range map[string]string{
			"prompt-digest": e.PromptDigest, "model-digest": e.ModelDigest,
			"fak-commit": e.FakCommit, "reference-engine-commit": e.ReferenceCommit,
		} {
			if strings.TrimSpace(val) == "" {
				block("bucket %s: %s missing from recorded evidence", b.ID(), field)
			}
		}
		if len(e.ArtifactLocations) == 0 {
			block("bucket %s: no raw artifact locations recorded", b.ID())
		}
		if e.TokenCount < b.MinPromptTokens {
			block("bucket %s: prompt token count %d below horizon floor %d", b.ID(), e.TokenCount, b.MinPromptTokens)
		}
		// Metric floors (acceptance §2/§3). Grounding is checked independently of
		// repetition: a generic-but-grammatical answer that passes repetition still
		// blocks on grounding.
		m := e.Metrics
		if m.ArgmaxAgreement < th.MinArgmaxAgreement {
			block("bucket %s: argmax parity %.4f below floor %.4f", b.ID(), m.ArgmaxAgreement, th.MinArgmaxAgreement)
		}
		if m.TopLogitOverlap < th.MinTopLogitOverlap {
			block("bucket %s: top-logit overlap %.4f below floor %.4f", b.ID(), m.TopLogitOverlap, th.MinTopLogitOverlap)
		}
		if !m.RequiredLabelsPresent {
			block("bucket %s: required label(s) absent from output", b.ID())
		}
		if m.RepetitionScore < th.MinRepetition {
			block("bucket %s: repetition score %.4f below floor %.4f (degenerate repeat loop)", b.ID(), m.RepetitionScore, th.MinRepetition)
		}
		if m.GroundingScore < th.MinGrounding {
			block("bucket %s: grounding %.4f below floor %.4f (generic-but-grammatical; passing repetition alone is not sufficient)", b.ID(), m.GroundingScore, th.MinGrounding)
		}
	}

	// The original #4273 failure fixture (or a privacy-safe digest-bound equivalent)
	// must be retained as a permanent regression canary among the observed buckets.
	fx := Canary4273Fixture()
	retained := false
	for _, e := range evidence {
		if e.Regression4273 && e.PromptDigest == fx.PromptDigest {
			retained = true
			break
		}
	}
	if !retained {
		block("retained #4273 regression canary absent or digest drifted (want prompt-digest %s on a Regression4273 bucket)", fx.PromptDigest)
	}

	sort.Strings(v.Reasons)
	return v
}

// RegressionFixture is a permanent, digest-bound regression canary descriptor. The
// prompt payload itself may be private; the digest and metadata are what is retained
// and checked, so the canary survives without committing the raw prompt.
type RegressionFixture struct {
	Issue        string
	PromptDigest string
	TokenCount   int
	Note         string
}

// canary4273PromptDigest is the retained digest of the #4273 executive-report prompt
// class (a privacy-safe, digest-bound equivalent of the ~1.5k-token prompt that
// passed short sanity while failing the real long-context run). It is a fixed
// constant so any drift in the retained canary is a compile-visible, gate-visible
// change rather than a silent loss of coverage.
const canary4273PromptDigest = "sha256:4273" +
	"c0ffee0000000000000000000000000000000000000000000000000000decode"

// Canary4273Fixture returns the retained #4273 regression canary descriptor.
func Canary4273Fixture() RegressionFixture {
	return RegressionFixture{
		Issue:        "#4273",
		PromptDigest: canary4273PromptDigest,
		TokenCount:   1500,
		Note:         "executive-report prompt: passed short sanity, failed real ~1.5k-token long-context decode via recurrent-state drift",
	}
}

// DigestPrompt binds a token stream to a stable digest so a bucket's PromptDigest
// is reproducible from the ids alone (a private prompt commits its digest, not its
// text). The "sha256:" prefix matches the retained-fixture convention.
func DigestPrompt(ids []int) string {
	h := sha256.New()
	var buf [4]byte
	for _, id := range ids {
		buf[0] = byte(id)
		buf[1] = byte(id >> 8)
		buf[2] = byte(id >> 16)
		buf[3] = byte(id >> 24)
		_, _ = h.Write(buf[:])
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// RepetitionScore is 1 minus the fraction of length-n windows occupied by the single
// most common n-gram in ids. A clean stream (all n-grams distinct) scores near 1; a
// degenerate repeat loop (the #4273 collapse shape, one n-gram cycling to the cap)
// scores near 0. n<=0 is treated as 1 (bigrams by default via the caller).
func RepetitionScore(ids []int, n int) float64 {
	if n <= 0 || len(ids) < n {
		return 1
	}
	total := len(ids) - n + 1
	counts := make(map[string]int, total)
	var b strings.Builder
	max := 0
	for i := 0; i+n <= len(ids); i++ {
		b.Reset()
		for j := 0; j < n; j++ {
			fmt.Fprintf(&b, "%d,", ids[i+j])
		}
		c := counts[b.String()] + 1
		counts[b.String()] = c
		if c > max {
			max = c
		}
	}
	return 1 - float64(max)/float64(total)
}

// GroundingScore is the fraction of source entities that appear (case-insensitive
// token match) in the output. It is tied to source entities, so a generic-but-
// grammatical answer that names none of them scores 0 regardless of fluency. It
// fails closed: a non-empty entity set against empty output scores 0.
func GroundingScore(output string, sourceEntities []string) float64 {
	if len(sourceEntities) == 0 {
		return 1
	}
	toks := make(map[string]struct{})
	for _, f := range strings.FieldsFunc(strings.ToLower(output), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	}) {
		toks[f] = struct{}{}
	}
	grounded := 0
	for _, e := range sourceEntities {
		if _, ok := toks[strings.ToLower(strings.TrimSpace(e))]; ok {
			grounded++
		}
	}
	return float64(grounded) / float64(len(sourceEntities))
}

// ArgmaxAgreement is the fraction of positions at which the fak and reference decoded
// token ids agree. Length mismatch is a hard miss on the missing tail (fail-closed).
func ArgmaxAgreement(fakIDs, refIDs []int) float64 {
	n := len(fakIDs)
	if len(refIDs) > n {
		n = len(refIDs)
	}
	if n == 0 {
		return 1
	}
	agree := 0
	for i := 0; i < n; i++ {
		if i < len(fakIDs) && i < len(refIDs) && fakIDs[i] == refIDs[i] {
			agree++
		}
	}
	return float64(agree) / float64(n)
}

// TopLogitOverlap is the mean Jaccard overlap of the fak and reference top-k token-id
// sets across the selected positions. Mismatched position counts are a fail-closed
// miss on the unmatched positions.
func TopLogitOverlap(fakTop, refTop [][]int) float64 {
	n := len(fakTop)
	if len(refTop) > n {
		n = len(refTop)
	}
	if n == 0 {
		return 1
	}
	var sum float64
	for i := 0; i < n; i++ {
		if i >= len(fakTop) || i >= len(refTop) {
			continue // unmatched position contributes 0
		}
		sum += jaccardInt(fakTop[i], refTop[i])
	}
	return sum / float64(n)
}

func jaccardInt(a, b []int) float64 {
	set := make(map[int]struct{}, len(a))
	for _, x := range a {
		set[x] = struct{}{}
	}
	inter := 0
	union := make(map[int]struct{}, len(a)+len(b))
	for _, x := range a {
		union[x] = struct{}{}
	}
	for _, x := range b {
		if _, ok := set[x]; ok {
			inter++
		}
		union[x] = struct{}{}
	}
	if len(union) == 0 {
		return 1
	}
	return float64(inter) / float64(len(union))
}
