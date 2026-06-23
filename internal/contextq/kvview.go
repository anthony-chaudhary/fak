package contextq

import (
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// kvview.go closes epic #437 acceptance item #7 (filed as #514): KV/provider
// materialization is represented as a model/tokenizer/position-SCOPED view, never
// as model-agnostic memory.
//
// The snippet/summary/descriptor/facts/timeline/QA views this package emits are
// model-AGNOSTIC, source-linked artifacts: the same view digest is reusable under
// any backend, because its identity carries no model/tokenizer/position axis. A KV
// view is the opposite kind of object. A KV span computed under model A's weights,
// tokenizer, serializer, and RoPE/position regime is GARBAGE under model B, so a KV
// view is model/tokenizer/position-BOUND and may be reused only when every axis of
// its binding identity matches the consuming session.
//
// The binding identity and its axis-by-axis match test already ship as the proven
// cachemeta KV gate (issue #432: MaterializationKey / MaterializeVerdict). This file
// does NOT reimplement that gate — it binds it into the contextq view layer: a
// KVView carries a cachemeta.MaterializationKey, and GateKVView lowers the kernel's
// match verdict into contextq's MaterializationVerdict vocabulary so a cross-model
// reuse surfaces as a typed REFUSE in the same verdict stream as every other view,
// never as a silent (and wrong) HIT.

// ViewKV is the runtime KV materialization of a semantic view. Unlike the other
// view types — model-agnostic, source-linked, shareable across backends — a ViewKV
// is model/tokenizer/position-bound and is gated by GateKVView before any reuse.
const ViewKV ViewType = "kv"

// KVView is a provenance-bound MemoryViewRecord that ALSO carries the runtime
// KV-materialization binding (which model/tokenizer/serializer/position regime the
// span was computed under). It is the view-layer representation of acceptance #7:
// the semantic provenance (source pages, producer, scope, taint) lives on the
// embedded record; the model-bound runtime identity lives in Materialization + Key.
type KVView struct {
	MemoryViewRecord
	// Materialization names the runtime form the span is held in: a local KV form
	// (kv_prefix/kv_span/compressed_kv) — model-bound, gated for reuse — or
	// provider_prefix, which is remote cost/latency telemetry and never serves as a
	// local-trust hit.
	Materialization cachemeta.MaterializationView `json:"materialization"`
	// Key is the binding identity the span was materialized under. A reuse must
	// match it on every axis (model, tokenizer, serializer, position regime, policy,
	// admitter) or the gate fails closed.
	Key cachemeta.MaterializationKey `json:"kv_key"`
	// Quality bounds an APPROXIMATE (compressed_kv) span's error. An exact span
	// leaves it zero; an approximate span without acceptable evidence is refused,
	// never served as a hit.
	Quality cachemeta.QualityEvidence `json:"-"`
}

// NewKVView lowers a source page's KV materialization into a provenance-bound KV
// view. source is the page's cachemeta entry (carrying its digest, taint, and
// witness); mat is the runtime form; key is the model/tokenizer/position binding.
// The view's faithfulness is 1.0 for an exact local KV form (a bit-faithful span)
// and is left to the caller's QualityEvidence for an approximate one. The binding
// axes are also stamped into Labels so the view is legible without the typed key.
func NewKVView(producer string, sourceStep int, source cachemeta.Entry, scope abi.ShareScope, mat cachemeta.MaterializationView, key cachemeta.MaterializationKey, q cachemeta.QualityEvidence) KVView {
	if producer == "" {
		producer = "contextq-kv"
	}
	viewID := "view-kv-step-" + strconv.Itoa(sourceStep) + "-" + short(source.ID.Digest)
	faithful := 1.0
	if mat.IsApproximate() {
		faithful = 0.0 // unproven until QualityEvidence bounds it; never a silent 1.0
	}
	rec := MemoryViewRecord{
		ViewID:            viewID,
		ViewType:          ViewKV,
		SourcePageIDs:     []int{sourceStep},
		SourceDigests:     []string{source.ID.Digest},
		SourceLen:         source.ID.Length,
		Producer:          producer,
		PolicyVersion:     key.PolicyVersion,
		Scope:             scope,
		Taint:             source.Security.Taint,
		Coverage:          1.0,
		FaithfulnessProbe: faithful,
		CacheEntry:        source,
		Labels: map[string]string{
			"view_type":       string(ViewKV),
			"materialization": string(mat),
			"model":           key.ModelID,
			"tokenizer":       key.TokenizerID,
			"serializer":      key.SerializerID,
			"position_regime": key.PositionRegime,
		},
	}
	return KVView{MemoryViewRecord: rec, Materialization: mat, Key: key, Quality: q}
}

// GateKVView decides whether a stored KV view may be reused for a session that
// needs `want`, returning a contextq MaterializationVerdict. It delegates the
// model/tokenizer/serializer/position/policy/admitter binding to the proven
// cachemeta primitives (MaterializationKey.Matches / .Complete, QualityEvidence) so
// the contextq view layer inherits exactly the #432 KV-reuse discipline:
//
//   - provider_prefix is remote cost/latency telemetry, never local trust -> REFUSE.
//   - a non-KV materialization is not a reuse candidate at all -> ABSTAIN.
//   - an incomplete binding key (any axis unproven) fails closed -> REFUSE.
//   - a cross-model / cross-tokenizer / cross-position reuse fails closed -> REFUSE,
//     carrying the first divergent axis as the reason (never a silent HIT).
//   - an approximate span without acceptable quality evidence -> REFUSE.
//   - every axis matches (and quality, if approximate, clears its bound) -> HIT.
func GateKVView(v KVView, want cachemeta.MaterializationKey) MaterializationVerdict {
	base := MaterializationVerdict{
		Step:   firstSourcePage(v.MemoryViewRecord),
		ViewID: v.ViewID,
		Entry:  v.CacheEntry.ID,
	}

	// provider_prefix residency is observational only — it can never be promoted to
	// a local-trust hit (provider.go refusal rule 6 / cachemeta acceptance #3).
	if v.Materialization == cachemeta.MatProviderPrefix {
		base.Kind = MaterializationRefuse
		base.Reason = "kv_provider_prefix_not_local_trust"
		return base
	}
	// Only a local KV form is a reuse candidate; anything else is out of scope for
	// this gate rather than a trust violation.
	if !v.Materialization.IsLocalKV() {
		base.Kind = MaterializationAbstain
		base.Reason = "kv_view_not_local_kv"
		return base
	}
	// A missing key axis cannot prove a match — refuse rather than serve a
	// possibly-mismatched span.
	if !v.Key.Complete() || !want.Complete() {
		base.Kind = MaterializationRefuse
		base.Reason = "kv_key_incomplete"
		return base
	}
	// The load-bearing fail-closed step: a span built under one model/tokenizer/
	// position regime is REFUSED under another, carrying the divergent axis.
	if ok, reason := v.Key.Matches(want); !ok {
		base.Kind = MaterializationRefuse
		base.Reason = "kv_" + string(reason)
		return base
	}
	// An approximate span is a hit only once measured quality clears its bound.
	if v.Materialization.IsApproximate() && !v.Quality.Acceptable() {
		base.Kind = MaterializationRefuse
		base.Reason = "kv_approx_unproven"
		return base
	}
	base.Kind = MaterializationHit
	base.Reason = "kv_materialization_match"
	return base
}
