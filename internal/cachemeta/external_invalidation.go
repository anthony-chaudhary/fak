package cachemeta

import "strconv"

// ExternalInvalidationKind names the kind of remote-engine cache object that must
// be dropped when fak has quarantined or refuted a parent K/V span.
type ExternalInvalidationKind string

const (
	ExternalInvalidateKVSpan         ExternalInvalidationKind = "kv_span"
	ExternalInvalidateAttentionIndex ExternalInvalidationKind = "attention_index"
)

// ExternalInvalidationDirective is the provider/engine-facing invalidation plan
// for a cachemeta entry. cachemeta still never calls an engine API; concrete
// SGLang/vLLM/llama.cpp adapters translate these directives into their wire calls.
type ExternalInvalidationDirective struct {
	Kind       ExternalInvalidationKind
	Entry      EntryID
	Plane      Plane
	Residency  Residency
	Provider   string
	Engine     string
	Reason     string
	Governance KVGovernance
}

// PlanExternalInvalidations returns the engine-cache entries that must be dropped
// when a K/V span is poisoned. It covers the DSA-specific dependency: any
// attention_index entry whose ParentKV references the poisoned span is invalidated
// with the K/V. Provider prompt-cache telemetry is deliberately ignored because
// it is cost-only metadata, not an engine-owned reusable K/V handle.
func PlanExternalInvalidations(poisonedKV EntryID, entries []Entry) []ExternalInvalidationDirective {
	if !poisonedKV.Valid() {
		return nil
	}
	var out []ExternalInvalidationDirective
	poisonedGovernance := KVGovernance{}
	for _, e := range entries {
		if sameEntryID(e.ID, poisonedKV) {
			poisonedGovernance = GovernanceFromEntry(e)
			break
		}
	}
	seen := map[string]bool{}
	add := func(kind ExternalInvalidationKind, e Entry, reason string, gov KVGovernance) {
		key := string(kind) + "\x00" + e.ID.Digest + "\x00" + string(e.ID.MediaType) +
			"\x00" + strconv.FormatInt(e.ID.Length, 10) + "\x00" + string(e.ID.Unit)
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, ExternalInvalidationDirective{
			Kind:       kind,
			Entry:      e.ID,
			Plane:      e.Plane,
			Residency:  e.Residency,
			Provider:   e.Labels["provider"],
			Engine:     e.Labels["engine"],
			Reason:     reason,
			Governance: gov,
		})
	}
	for _, e := range entries {
		switch {
		case e.Plane == PlaneProvider:
			continue
		case sameEntryID(e.ID, poisonedKV) && isExternalResidency(e.Residency.Tier):
			add(ExternalInvalidateKVSpan, e, "poisoned_kv", poisonedGovernance)
		case AttentionIndexReferences(e, poisonedKV):
			add(ExternalInvalidateAttentionIndex, e, "parent_kv_poisoned", poisonedGovernance)
		}
	}
	return out
}

func isExternalResidency(t ResidencyTier) bool {
	return t == TierRemote || t == TierProvider
}

// ExactSpanTarget is the payload-free identity of one precise cache object an
// exact-span-capable engine must evict: the content-addressed K/V span (or its
// dependent DSA attention_index), never the bytes. It is the projection of a
// planned ExternalInvalidationDirective that an engine adapter serializes into an
// exact-span eviction request.
type ExactSpanTarget struct {
	Kind       ExternalInvalidationKind
	Digest     string
	MediaType  MediaType
	Length     int64
	Unit       LengthUnit
	Reason     string
	Governance KVGovernance
}

// ExactSpanTargets projects planned invalidation directives into the precise span
// targets an exact-span-capable engine evicts. Only directives that name a valid,
// content-addressed entry are included: a directive without span identity (e.g. a
// coarse whole-cache reset request that carries no Entry) yields no exact-span
// target. That is the fail-closed seam — a caller that requires exact-span
// eviction but holds no named span gets an empty target set and must refuse,
// never silently "precisely evict nothing." Directive order is preserved.
func ExactSpanTargets(dirs []ExternalInvalidationDirective) []ExactSpanTarget {
	var out []ExactSpanTarget
	for _, d := range dirs {
		if !d.Entry.Valid() {
			continue
		}
		out = append(out, ExactSpanTarget{
			Kind:       d.Kind,
			Digest:     d.Entry.Digest,
			MediaType:  d.Entry.MediaType,
			Length:     d.Entry.Length,
			Unit:       d.Entry.Unit,
			Reason:     d.Reason,
			Governance: d.Governance,
		})
	}
	return out
}

// KVEvictionScope names how an invalidation directive was physically lowered.
// exact_span is the native/referee-owned guarantee; whole_prefix_cache is the
// safe over-invalidation used when a ridden engine has no public exact-span API.
type KVEvictionScope string

const (
	KVEvictionScopeWholePrefixCache KVEvictionScope = "whole_prefix_cache"
	KVEvictionScopeExactSpan        KVEvictionScope = "exact_span"
)

// KVEvictionAttestation is the common receipt both Track A and Track B produce
// after lowering a governed K/V eviction. It carries the target identity, the
// enforcement scope, the degradation bit, and the shared governance descriptor.
type KVEvictionAttestation struct {
	Kind               ExternalInvalidationKind
	Target             EntryID
	Scope              KVEvictionScope
	ExactSpanSupported bool
	Degraded           bool
	DegradeReason      string
	Governance         KVGovernance
	RefereeAdmitted    bool
	RefereeReason      KVRefereeReason
}

// AttestInvalidations converts planned invalidations into payload-free eviction
// receipts. degradeReason is intentionally explicit: callers must surface a
// whole-prefix fallback as a named degradation rather than letting a precise span
// directive silently collapse to a coarse reset.
func AttestInvalidations(dirs []ExternalInvalidationDirective, scope KVEvictionScope, exactSpanSupported bool, degradeReason string) []KVEvictionAttestation {
	if len(dirs) == 0 {
		return nil
	}
	out := make([]KVEvictionAttestation, 0, len(dirs))
	for _, d := range dirs {
		out = append(out, DefaultKVGovernanceReferee.AttestEviction(d.Kind, d.Entry, scope, exactSpanSupported, degradeReason, d.Governance))
	}
	return out
}
