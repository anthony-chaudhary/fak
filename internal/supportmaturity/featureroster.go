// featureroster.go — C6 of the support-maturity epic (#1249/#1243): score non-model
// FEATURES on the SAME M0–M7 ladder the model×backend grid reads off, so "feature
// support" and "architecture support" come off one instrument instead of two prose
// tables.
//
// The model grid lowers covmatrix.Support / preflight verdicts onto the ladder
// (supportmaturity.go). A feature is not a (family×backend) cell — it is a cache
// residency tier, an attention variant, or a serving capability — so it carries no
// covmatrix.Support value to lower. Its witness is instead a CLAIMS-register entry:
// CLAIMS.md is the repo's honesty ledger, where every capability line carries exactly
// one of [SHIPPED]/[SIMULATED]/[STUB] (the unit-96 lint contract). That tag IS a
// maturity claim closed by a non-author witness — [SHIPPED] means "closed by a
// mechanical witness (a go test / build / benchmark field / file read-back),
// reproducible now"; [SIMULATED] means "modeled with labeled stand-in data, the seam
// real, the numbers illustrative"; [STUB] means "plumbing present, behavior deferred,
// returns a no-op". So FromClaimTag lowers the register tag onto the ladder exactly the
// way FromSupport lowers covmatrix.Support, and the feature's rung is READ from the
// ledger, never self-declared (the epic's honesty fence, C13/#1256, applied to
// features). A feature whose anchor is absent from the ledger has no honest support to
// stand on and floors to M0 — the instrument SURFACING the gap rather than claiming it,
// and the line flips up automatically the day a real witness lands.
//
// Seed roster: the families come from the concept-disambiguation-scorecard
// (tools/concept_disambiguation_scorecard.data/rows-{cache,attention,…}.json) — the dual
// instrument that disambiguates feature NAMES; this one grades their maturity. The
// attention family carries the variants the issue names — softmax / linear / MLA / DSA /
// paged — so a reader sees all five on one ladder, immature variants included.
package supportmaturity

import "strings"

// FeatureFamily is the non-model feature family a rostered feature belongs to, seeded
// from the concept-disambiguation-scorecard families. The set is closed; the witness
// test fails if a feature declares a family outside it.
type FeatureFamily string

const (
	// FamilyCache covers cache residency tiers — the CXL/NUMA-far placement tiers,
	// KV-eviction/weight residency, and the residency telemetry plane.
	FamilyCache FeatureFamily = "cache"
	// FamilyAttention covers attention variants — softmax / linear / MLA / DSA / paged.
	FamilyAttention FeatureFamily = "attention"
	// FamilyServing covers serving features — the poly-model serving core, latency
	// observability, OOM recovery, and the multi-model dispatch / external-KV transports.
	FamilyServing FeatureFamily = "serving"
)

// FeatureFamilies is the closed roster of non-model feature families, in display order.
var FeatureFamilies = []FeatureFamily{FamilyCache, FamilyAttention, FamilyServing}

// ClaimTag is one of the three closed tags of the CLAIMS.md honesty ledger (unit 96).
type ClaimTag string

const (
	// ClaimShipped: real code on the critical path, closed by a mechanical witness
	// (a go test / build / benchmark field / file read-back); reproducible now.
	ClaimShipped ClaimTag = "SHIPPED"
	// ClaimSimulated: modeled with labeled stand-in data — the seam is real, the numbers
	// are illustrative (no GPU / no live engine on the build box).
	ClaimSimulated ClaimTag = "SIMULATED"
	// ClaimStub: plumbing present, behavior deferred; clearly labeled, returns a no-op.
	ClaimStub ClaimTag = "STUB"
)

// FromClaimTag lowers a CLAIMS-register tag onto the M0–M7 ladder. The register is the
// FEATURE-band vocabulary — "what does the honesty ledger witness about this capability?"
// — so it spans the M1–M4 band:
//
//	STUB      → M1Fenced  (an honest no-op: plumbing present, behavior deferred — it does
//	                       not run, but it refuses out loud rather than silently diverging)
//	SIMULATED → M3Runs    (runs on a proof path with labeled stand-in data; the numeric
//	                       claim is asserted, not proven by a CI oracle)
//	SHIPPED   → M4Correct (correctness witnessed by a CI-runnable mechanical witness —
//	                       the ledger's own definition of the tag)
//
// The lowering is order-preserving over the ledger's own honesty order
// (STUB < SIMULATED < SHIPPED). A [SHIPPED] bench that PROVES a speedup witnesses M5, and
// a turnbench parity class witnesses M6 — but the tag alone witnesses only correctness,
// so SHIPPED floors honestly at M4; binding the bench → M5 / parity → M6 promotions for
// a feature is the same follow-on C2 (#1245) is for the model grid. An unrecognized tag
// floors to M0None — the honest "we can witness nothing" default.
func FromClaimTag(t ClaimTag) Rung {
	switch t {
	case ClaimShipped:
		return M4Correct
	case ClaimSimulated:
		return M3Runs
	case ClaimStub:
		return M1Fenced
	default:
		return M0None
	}
}

// Feature is one rostered non-model feature scored on the ladder. ClaimAnchor is a unique
// substring of the CLAIMS.md capability line that witnesses the feature; an empty anchor
// means no register witness exists yet (the feature floors to M0None — see FeatureRung).
// The rung is never stored on the Feature: it is always re-read from the live ledger, so
// it cannot drift from the witness.
type Feature struct {
	ID          string
	Name        string
	Family      FeatureFamily
	ClaimAnchor string
}

// FeatureRoster is the closed roster of non-model features scored on the ladder, seeded
// from the concept-disambiguation-scorecard families. Each ClaimAnchor is a verified-unique
// substring of a real CLAIMS.md capability line (the witness test pins presence + uniqueness).
var FeatureRoster = []Feature{
	// Cache residency tiers.
	{ID: "cache-residency-tiers", Name: "Hardware-aware cache residency tiers (CXL/NUMA-far)", Family: FamilyCache, ClaimAnchor: "Hardware-aware cache placement & lifecycle"},
	{ID: "cache-ddr-tiers", Name: "DDR cache tiers (engine cache-event visibility)", Family: FamilyCache, ClaimAnchor: "DDR cache tiers"},
	{ID: "cache-kv-eviction-residency", Name: "Planned-elision → KV-eviction residency bridge", Family: FamilyCache, ClaimAnchor: "KV-eviction residency bridge"},
	{ID: "cache-weight-residency", Name: "Multi-model weight-residency layer", Family: FamilyCache, ClaimAnchor: "Multi-model weight-residency layer"},
	{ID: "cache-kv-residency-telemetry", Name: "KV-residency telemetry (token-per-watt)", Family: FamilyCache, ClaimAnchor: "metrics-service scrape adapter / KV-residency / token-per-watt"},

	// Attention variants — softmax / linear / MLA / DSA / paged (the five the issue names).
	{ID: "attn-softmax", Name: "Softmax attention (reference forward pass)", Family: FamilyAttention, ClaimAnchor: "A pure-Go SmolLM2-135M forward pass"},
	{ID: "attn-paged", Name: "Paged attention (native paged KV)", Family: FamilyAttention, ClaimAnchor: "Native paged KV opt-in path"},
	{ID: "attn-dsa", Name: "DeepSeek sparse attention (GLM glm_moe_dsa)", Family: FamilyAttention, ClaimAnchor: "glm_moe_dsa"},
	// MLA and linear attention have no register witness yet — the in-kernel runtime still
	// requires explicit MLA projection wiring (internal/model/config.go), and linear
	// attention (DeltaNet) is unwitnessed in the ledger. Empty anchor ⇒ M0None: the
	// instrument reports the gap honestly rather than claiming it, and the line rises the
	// day a [SHIPPED]/[SIMULATED]/[STUB] register entry names the variant.
	{ID: "attn-mla", Name: "Multi-head latent attention (MLA)", Family: FamilyAttention, ClaimAnchor: ""},
	{ID: "attn-linear", Name: "Linear attention (DeltaNet)", Family: FamilyAttention, ClaimAnchor: ""},

	// Serving features.
	{ID: "serving-polymodel-core", Name: "Poly-model serving core", Family: FamilyServing, ClaimAnchor: "Poly-model serving core"},
	{ID: "serving-latency-observability", Name: "Serving-latency observability (TTFT/TPOT histograms)", Family: FamilyServing, ClaimAnchor: "Serving-latency observability"},
	{ID: "serving-oom-recovery", Name: "Classed runtime device OOM recovery", Family: FamilyServing, ClaimAnchor: "Classed runtime device OOM recovery"},
	{ID: "serving-multimodel-dispatch", Name: "LIVE multi-model dispatch", Family: FamilyServing, ClaimAnchor: "LIVE multi-model DISPATCH is not wired"},
	{ID: "serving-external-kv-transport", Name: "External serving-engine KV transport", Family: FamilyServing, ClaimAnchor: "No LIVE transport attaching a real"},
}

// FeatureRung resolves a feature to its ladder rung by finding its ClaimAnchor in the
// CLAIMS-register text and lowering the matched capability line's tag via FromClaimTag.
// It returns (rung, tag, true) when a bound witness is found, and (M0None, "", false)
// when the anchor is empty (no witness declared) or absent from the ledger (unwitnessed)
// — the honest "no support to stand on" floor. The witness is the ledger, never the
// caller: a feature cannot self-report a rung the register does not back.
func FeatureRung(claims string, f Feature) (Rung, ClaimTag, bool) {
	if f.ClaimAnchor == "" {
		return M0None, "", false
	}
	for _, line := range strings.Split(claims, "\n") {
		if !strings.Contains(line, f.ClaimAnchor) {
			continue
		}
		tag, ok := claimTagOf(line)
		if !ok {
			continue // anchor matched a non-capability line; keep scanning
		}
		return FromClaimTag(tag), tag, true
	}
	return M0None, "", false
}

// ScoredFeature is a feature resolved against a CLAIMS-register snapshot: its rung, the
// witnessing tag (empty when unwitnessed), and whether a bound witness was found.
type ScoredFeature struct {
	Feature
	Rung      Rung
	Tag       ClaimTag
	Witnessed bool
}

// ScoreFeatures lowers the whole roster against a CLAIMS-register snapshot, in roster
// order. It is the feature-side dual of supportmaturityscore.Build's fold over the model
// grid: one ordered list of (feature, rung, witness) the higher children (the scorecard
// fold, the `fak support` read-out) consume.
func ScoreFeatures(claims string) []ScoredFeature {
	out := make([]ScoredFeature, 0, len(FeatureRoster))
	for _, f := range FeatureRoster {
		rung, tag, ok := FeatureRung(claims, f)
		out = append(out, ScoredFeature{Feature: f, Rung: rung, Tag: tag, Witnessed: ok})
	}
	return out
}

// claimTagOf parses the leading "[TAG]" of a CLAIMS.md capability line (a line that, once
// trimmed, begins "- [TAG] …" per the unit-96 lint contract). It returns ("", false) for
// any line that is not a tagged capability line, so a prose line that merely contains an
// anchor substring is never mistaken for a witness.
func claimTagOf(line string) (ClaimTag, bool) {
	s := strings.TrimSpace(line)
	if !strings.HasPrefix(s, "-") {
		return "", false
	}
	s = strings.TrimSpace(s[1:])
	if !strings.HasPrefix(s, "[") {
		return "", false
	}
	end := strings.IndexByte(s, ']')
	if end < 0 {
		return "", false
	}
	switch tag := ClaimTag(s[1:end]); tag {
	case ClaimShipped, ClaimSimulated, ClaimStub:
		return tag, true
	default:
		return "", false
	}
}
