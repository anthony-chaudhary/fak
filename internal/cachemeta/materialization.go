package cachemeta

import "strings"

// materialization.go bridges a SELECTED semantic view to a runtime KV/provider
// materialization without collapsing trust (issue #432).
//
// A semantic view is a model-agnostic, source-linked artifact (a cachemeta
// MemoryView). Its identity (DigestMemoryView) carries no model/tokenizer/
// position axis, so the SAME view digest is reusable under ANY backend — a
// semantic view can be SHARED across model backends. Its RUNTIME materialization —
// a provider prompt-prefix cache, a local KV prefix/span, or a compressed KV
// span — is model/tokenizer/position-specific and may ONLY be reused when every
// axis of its materialization key matches; a compressed/approximate span
// additionally needs quality/fault evidence before it may be reported as a hit;
// and a provider prefix is cost/latency telemetry only, never local trust. This
// file makes those rules mechanical so a benchmark or a runtime cannot
// accidentally treat a backend-specific KV span as cross-agent semantic memory.

// MaterializationView names a runtime form a selected semantic view MAY be
// lowered into. It is DISTINCT from the semantic ViewType (snippet/summary/…):
// the semantic view says WHAT content was selected; the materialization view says
// HOW that content is held at runtime. (Acceptance #1.)
type MaterializationView string

const (
	// MatProviderPrefix is a remote provider prompt-prefix cache. It is cost/
	// latency telemetry only, never a re-serveable local trust artifact
	// (provider.go / refusal rule 6).
	MatProviderPrefix MaterializationView = "provider_prefix"
	// MatKVPrefix is a local, prefix-aligned KV prefix (exact, bit-faithful).
	MatKVPrefix MaterializationView = "kv_prefix"
	// MatKVSpan is a local KV span — a kernel-owned prefix/middle span, exact.
	MatKVSpan MaterializationView = "kv_span"
	// MatCompressedKV is an APPROXIMATE / quantized KV span. It is lossy, so it
	// may be reported as a hit only once quality/fault metrics bound its error.
	MatCompressedKV MaterializationView = "compressed_kv"
)

// MaterializationViews is the full runtime-materialization set this bridge
// recognizes — the four views acceptance #1 names.
var MaterializationViews = []MaterializationView{
	MatProviderPrefix, MatKVPrefix, MatKVSpan, MatCompressedKV,
}

// IsLocalKV reports whether a view is a LOCAL KV form (kv_prefix/kv_span/
// compressed_kv): model/tokenizer/position-bound and NOT shareable across
// backends. provider_prefix is remote telemetry and is handled separately.
func (m MaterializationView) IsLocalKV() bool {
	return m == MatKVPrefix || m == MatKVSpan || m == MatCompressedKV
}

// IsApproximate reports whether a view is lossy/approximate and therefore gated
// on quality evidence before a hit (compressed_kv today).
func (m MaterializationView) IsApproximate() bool { return m == MatCompressedKV }

// MaterializationKey is the binding identity a runtime KV/provider materialization
// MUST match to be reused. Every axis is load-bearing for KV correctness: a KV
// span materialized under one model / tokenizer / serializer / RoPE-position
// regime is GARBAGE under another, and a policy- or admitter-version bump can
// change what was admitted into the span. (Acceptance #2.)
type MaterializationKey struct {
	ModelID         string
	TokenizerID     string
	SerializerID    string
	PositionRegime  string // RoPE / position regime (theta, scaling, prefix-alignment)
	PolicyVersion   string
	AdmitterVersion string
}

// Complete reports whether every axis the issue enumerates is populated. An
// incomplete key cannot key a KV materialization — a missing axis is an
// unprovable match — so callers fail closed on it rather than serve.
func (k MaterializationKey) Complete() bool {
	return k.ModelID != "" && k.TokenizerID != "" && k.SerializerID != "" &&
		k.PositionRegime != "" && k.PolicyVersion != "" && k.AdmitterVersion != ""
}

// Matches reports whether a stored materialization keyed by k may be reused for a
// request keyed by want, returning the FIRST axis that diverges as a typed reason.
// Equal keys match with ReasonNone.
func (k MaterializationKey) Matches(want MaterializationKey) (bool, LookupReason) {
	switch {
	case k.ModelID != want.ModelID:
		return false, ReasonModelMismatch
	case k.TokenizerID != want.TokenizerID:
		return false, ReasonTokenizerMismatch
	case k.SerializerID != want.SerializerID:
		return false, ReasonSerializerMismatch
	case k.PositionRegime != want.PositionRegime:
		return false, ReasonPositionMismatch
	case k.PolicyVersion != want.PolicyVersion:
		return false, ReasonPolicyMismatch
	case k.AdmitterVersion != want.AdmitterVersion:
		return false, ReasonAdmitterMismatch
	}
	return true, ReasonNone
}

// String renders the key as a stable, order-fixed identity suitable for a
// cache-key digest. Equal keys render equal; any axis change changes the string.
func (k MaterializationKey) String() string {
	return strings.Join([]string{
		"model=" + k.ModelID,
		"tok=" + k.TokenizerID,
		"ser=" + k.SerializerID,
		"pos=" + k.PositionRegime,
		"policy=" + k.PolicyVersion,
		"admitter=" + k.AdmitterVersion,
	}, ";")
}

// MaterializationKeyOf extracts the materialization key a stored entry was built
// under. model/tokenizer/serializer/position come from Derivation; the policy
// version from Validity; the position regime from a "position_regime" label
// (falling back to the entry's PositionMode); the admitter version from an
// "admitter_version" label (falling back to the admitting authority).
func MaterializationKeyOf(e Entry) MaterializationKey {
	pos := e.Labels["position_regime"]
	if pos == "" {
		pos = string(e.Derivation.PositionMode)
	}
	admitter := e.Labels["admitter_version"]
	if admitter == "" {
		admitter = e.Security.AdmittedBy
	}
	return MaterializationKey{
		ModelID:         e.Derivation.ModelID,
		TokenizerID:     e.Derivation.TokenizerID,
		SerializerID:    e.Derivation.SerializerID,
		PositionRegime:  pos,
		PolicyVersion:   e.Validity.PolicyVersion,
		AdmitterVersion: admitter,
	}
}

// QualityEvidence bounds the error of an APPROXIMATE materialization. A
// compressed/approximate KV span may be reported as a hit ONLY when Measured is
// true — a quality delta and a fault count have actually been observed — and the
// measured delta clears the admission bound. An UNMEASURED approximate span is an
// unproven hit and is refused, never served. (Acceptance #4.)
type QualityEvidence struct {
	Measured        bool    // a quality/fault measurement was actually taken
	QualityDelta    float64 // observed quality regression vs exact KV (0 = none)
	FaultsObserved  uint64  // false-hit faults attributed to this span
	MaxQualityDelta float64 // admission bound; a delta above this fails (<=0 = bound unset)
}

// Acceptable reports whether measured quality clears the admission bound. An
// unmeasured span is never acceptable.
func (q QualityEvidence) Acceptable() bool {
	if !q.Measured {
		return false
	}
	if q.MaxQualityDelta > 0 && q.QualityDelta > q.MaxQualityDelta {
		return false
	}
	return true
}

// SemanticShareable reports whether an entry is a model-AGNOSTIC semantic view —
// a memory_view whose identity carries no model/tokenizer/position axis and so is
// reusable under ANY backend. It is the formal counterpart of the materialization
// key's asymmetry: the semantic view crosses backends; its KV materialization,
// keyed by MaterializationKey, does not. (Acceptance #5.)
func SemanticShareable(e Entry) bool {
	return e.ID.MediaType == MediaMemoryView
}

// MaterializeVerdict is the central gate: given the runtime materialization view,
// the stored materialization entry, the key the current request needs, and any
// quality evidence, it returns whether the materialization may be served.
//
//   - provider_prefix never serves as local trust — it is cost/latency telemetry
//     only (a non-serveable Transform verdict). (Acceptance #3.)
//   - a local KV view (kv_prefix/kv_span/compressed_kv) serves only when the
//     stored key MATCHES the requested key on every axis, so a KV span built
//     under model A is REFUSED under model B. An incomplete key fails closed.
//     (Acceptance #2 + #5.)
//   - an approximate view (compressed_kv) additionally needs acceptable quality
//     evidence; without it the verdict is an approximate-fault MISS, never a hit.
//     (Acceptance #4.)
func MaterializeVerdict(view MaterializationView, stored Entry, want MaterializationKey, q QualityEvidence) LookupVerdict {
	if view == MatProviderPrefix {
		// Provider residency is observational; ProviderCacheVerdict makes the
		// no-local-trust rule mechanical (CanServe() == false).
		return ProviderCacheVerdict(stored)
	}
	have := MaterializationKeyOf(stored)
	if !have.Complete() || !want.Complete() {
		// A missing key axis cannot prove a match — refuse rather than serve a
		// possibly-mismatched KV span.
		return Miss(ReasonModelMismatch)
	}
	if ok, reason := have.Matches(want); !ok {
		return Miss(reason)
	}
	if view.IsApproximate() && !q.Acceptable() {
		return Miss(ReasonApproxFault)
	}
	return Hit(stored)
}

// MaterializationBenchRow is ONE reportable benchmark row that places runtime
// materialization savings NEXT TO the semantic-view witness they were selected
// under. A benchmark folds its observed entries through a SavingsSplit (local vs
// provider tokens, never double-counted) and stamps the underlying semantic
// view's witness/digest, so a reader sees cached provider tokens / local KV reuse
// ALONGSIDE the source-linked semantic view that drove the materialization —
// never a bare token number with no provenance. (Acceptance #6.)
type MaterializationBenchRow struct {
	SemanticViewID     string
	SemanticViewDigest string
	SemanticWitness    string
	View               MaterializationView
	LocalReuseTokens   int64
	ProviderReadTokens int64
	ProviderHits       int64
	LocalKVHits        int64
}

// NewMaterializationBenchRow builds a row from a semantic-view entry, the runtime
// materialization view it was lowered into, the SavingsSplit accumulated over the
// materialization's cache entries, and the local KV hit count.
func NewMaterializationBenchRow(semView Entry, view MaterializationView, split SavingsSplit, localKVHits int64) MaterializationBenchRow {
	return MaterializationBenchRow{
		SemanticViewID:     semView.Labels["view_id"],
		SemanticViewDigest: semView.ID.Digest,
		SemanticWitness:    semView.Validity.Witness,
		View:               view,
		LocalReuseTokens:   split.LocalReuseTokens,
		ProviderReadTokens: split.ProviderReadTokens,
		ProviderHits:       split.ProviderHits,
		LocalKVHits:        localKVHits,
	}
}
