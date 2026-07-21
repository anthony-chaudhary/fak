package cachemeta

import "github.com/anthony-chaudhary/fak/internal/abi"

// This adapter maps LMCache's offload/restore events onto the SAME kv_transfer
// residency stream every other KV movement uses (FromKVTransfer / KVTransferVerdict).
// It is a MAPPING adapter only: cachemeta never links or calls LMCache, vendors no
// external code, and rebuilds no store — a concrete LMCache-facing adapter populates
// the field-only LMCTransferEvent shape and lowers it here, exactly as
// mooncake_transfer.go does for the Mooncake Transfer Engine and nixl_lease.go does
// for vLLM NIXL leases.
//
// LMCache is the disaggregated-prefill KV path for vLLM: it appends produced K/V spans
// into an offload store (CPU DRAM / disk / remote) and later looks a span up and
// re-materializes it back onto a serving tier. Per the KV-transport governance doc
// (docs/serving/kv-transport-governance-nixl-mooncake-lmcache.md §2.3) the LMCache
// integration mapping is direct — only two operations project onto the existing
// KVTransferDirection vocabulary:
//
//   - lmcache.append() (offload)              -> KVOffload (residency spills to a colder tier)
//   - lmcache.lookup() + re-materialization   -> KVRestore (a tier restores a wanted span)
//
// plus the shared outcome trichotomy and a fault path. §6 marked the live wiring a GAP:
// fak's seams (FromKVTransfer, KVTransferVerdict, the ExternalInvalidationDirective
// plane) shipped, but nothing actually emitted LMCache events into them. This file
// closes that gap.
//
// House posture is preserved: the event is field-only and its lowering is a
// deterministic, wall-clock-FREE function of its fields (an offload/restore event is a
// point-in-time residency transition with no TTL, so no clock is threaded — replay is
// reproducible from the event alone). A lookup that finds nothing usable is a typed
// MISS (may trigger recompute, never a silent hit); a transfer error is a typed
// residency FAULT (never a silent recompute); and a fault inside a DSA (disaggregated
// sparse attention) path is escalated to QUARANTINE because a dependent attention index
// may reference the poisoned span (§1.2).

// LMCEventKind names the LMCache operation fak observed. The two kinds are the §2.3
// mapping targets; each projects onto one existing KVTransferDirection.
type LMCEventKind string

const (
	// LMCAppend — lmcache.append(): a produced K/V span is offloaded from the serving
	// tier into the LMCache store (a colder residency tier). Projects onto KVOffload.
	LMCAppend LMCEventKind = "append"
	// LMCLookup — lmcache.lookup() followed by re-materialization: the store restores a
	// wanted span back onto a serving tier. Projects onto KVRestore. A lookup that finds
	// nothing usable carries Outcome KVTransferMissed on this same kind (a restore MISS).
	LMCLookup LMCEventKind = "lookup"
)

// Direction projects the observed LMCache operation onto the existing KVTransferDirection
// the kv_transfer plane already speaks. An unrecognized kind yields the empty direction
// (fail-closed: an unknown operation is never silently treated as one of the two governed
// movements — LMCTransferVerdict turns it into a Miss rather than a phantom hit).
func (k LMCEventKind) Direction() KVTransferDirection {
	switch k {
	case LMCAppend:
		return KVOffload
	case LMCLookup:
		return KVRestore
	default:
		return ""
	}
}

// LMCTransferEvent is the field-only witness of one LMCache offload/restore event. It
// carries exactly the §4 external-system requirements: a stable content-addressed
// SpanDigest, the FromTier/ToTier residency tiers the bytes moved between, the typed
// Outcome (OK / Missed / Fault), and BytesMoved. DSAPath marks that the transfer served a
// DSA (disaggregated sparse attention) index so a fault must quarantine, not merely fault
// (§1.2). cachemeta never calls an LMCache API; an LMCache-facing adapter fills this shape.
type LMCTransferEvent struct {
	// Kind is which of the two §2.3 operations this event reports (append or lookup).
	Kind LMCEventKind
	// Backend is the transport/medium that moved the bytes (cpu, disk, redis, object, rdma).
	Backend string
	// SpanDigest is the content-addressed identity of the K/V span being offloaded or
	// restored. Empty = an unidentified event (fail-closed: it can never back a hit —
	// see LMCTransferVerdict).
	SpanDigest string
	// Tokens is the span length in positions.
	Tokens int64
	// ModelID / TokenizerID bind the span to the model that produced it.
	ModelID     string
	TokenizerID string
	// FromTier / ToTier are the residency tiers the transfer moved between; ToTier is
	// where the span now lives (a colder tier after append, a serving tier after restore).
	FromTier ResidencyTier
	ToTier   ResidencyTier
	// Owner is the instance/store node that now owns the residency (defaults to the
	// generic engine owner inside FromKVTransfer when empty).
	Owner string
	// Lease is the store's own residency handle for the span, when it exposes one.
	Lease string
	// Outcome is the typed result: OK, a MISS (lookup found nothing usable), or a FAULT
	// (the transfer errored). The empty value lowers to OK inside FromKVTransfer.
	Outcome KVTransferOutcome
	// FaultReason is free text carried when Outcome == fault.
	FaultReason string
	// BytesMoved records transfer volume when known.
	BytesMoved int64
	// DSAPath marks that this transfer backed a DSA attention index, so a fault must
	// escalate to quarantine (§1.2) rather than a plain residency fault.
	DSAPath bool
}

// FromLMCTransfer lowers an LMCache offload/restore event onto the kv_transfer plane
// by reusing FromKVTransfer, so an LMCache movement flows through the SAME residency
// stream and label schema as every other KV event — never a parallel record. The
// direction comes from the §2.3 mapping; the outcome, tiers, digest, and bytes pass
// straight through. The event kind, backend, and DSA bit are recorded as labels so an
// observing sink separates them without re-deriving. An LMCache span is externally owned,
// so its coherence is external refutation (a freed/evicted store span is refuted the
// moment the store drops it, never a policy-governed local borrow). When the event
// faulted inside a DSA path the entry itself is quarantined (taint + admission), so a
// sink that stores it holds it back from serving (§1.2).
func FromLMCTransfer(ev LMCTransferEvent, opts ...Option) Entry {
	e := FromKVTransfer(KVTransfer{
		Direction:   ev.Kind.Direction(),
		Backend:     ev.Backend,
		SpanDigest:  ev.SpanDigest,
		Tokens:      ev.Tokens,
		ModelID:     ev.ModelID,
		TokenizerID: ev.TokenizerID,
		FromTier:    ev.FromTier,
		ToTier:      ev.ToTier,
		Owner:       ev.Owner,
		Lease:       ev.Lease,
		Outcome:     ev.Outcome,
		FaultReason: ev.FaultReason,
		BytesMoved:  ev.BytesMoved,
	}, opts...)
	if e.Labels == nil {
		e.Labels = map[string]string{}
	}
	e.Labels["lmc_event"] = string(ev.Kind)
	if ev.DSAPath {
		e.Labels["dsa_path"] = "true"
	}
	// An externally owned span is refuted the moment the store frees it.
	e.Coherence.InvalidationMode = InvalidationExternalRefutation
	// §1.2: a fault inside a DSA path poisons any dependent attention index, so the entry
	// is quarantined at the source rather than left as a plain fault.
	if ev.Outcome == KVTransferFault && ev.DSAPath {
		e.Security.Taint = abi.TaintQuarantined
		e.Security.AdmissionVerdict = AdmissionQuarantine
	}
	return e
}

// LMCTransferVerdict folds an LMCache event into the typed lookup verdict a serving
// path MUST consult before treating an offloaded span as usable. It reuses
// KVTransferVerdict for the ordinary trichotomy (OK -> HIT, missed -> MISS(restore_miss),
// fault -> FAULT(residency_fault)) so the mapping is literally the shared one, and only
// special-cases the DSA fault: per §1.2 a fault inside a DSA path is QUARANTINE (held
// back, dependent index invalid), not a plain residency fault. An event that names no
// span or an unrecognized kind is a Miss (never a phantom hit) — so a lookup miss cleanly
// becomes a MISS that may trigger recompute rather than a silent zero.
func LMCTransferVerdict(ev LMCTransferEvent) LookupVerdict {
	if ev.SpanDigest == "" || ev.Kind.Direction() == "" {
		return Miss(ReasonAbsent)
	}
	e := FromLMCTransfer(ev)
	if ev.Outcome == KVTransferFault && ev.DSAPath {
		return Quarantine(e, ReasonResidencyFault)
	}
	return KVTransferVerdict(e)
}

// LMCResetTarget is the field-only instruction an LMCache-facing adapter turns into a
// store reset when fak refutes a span (§3.2, §4.5). It is the LMCache projection of a
// planned ExternalInvalidationDirective: a named span reset when fak holds span identity,
// or a whole-store reset fallback when the directive carries no span (the store has no
// exact-span API to target). Payload-free — it names the span identity and the reason,
// never the bytes.
type LMCResetTarget struct {
	Kind       ExternalInvalidationKind
	SpanDigest string
	MediaType  MediaType
	Length     int64
	Unit       LengthUnit
	// WholeReset is set when the directive named no exact span, so the only honest
	// response is a coarse whole-store reset (attested over-invalidation) rather than
	// silently resetting nothing.
	WholeReset bool
	Reason     string
}

// LMCResetTargets maps fak's planned invalidation directives onto LMCache reset
// instructions — this is the adapter "responding to" fak's ExternalInvalidationDirective
// for cache reset (§3.2, §4.5). A directive that names a valid content-addressed span
// becomes a precise exact-span reset; a directive with no span identity becomes a
// whole-store reset fallback so the store is never left holding a refuted span merely
// because fak could not name it exactly. Directive order is preserved.
func LMCResetTargets(dirs []ExternalInvalidationDirective) []LMCResetTarget {
	if len(dirs) == 0 {
		return nil
	}
	out := make([]LMCResetTarget, 0, len(dirs))
	for _, d := range dirs {
		if d.Entry.Valid() {
			out = append(out, LMCResetTarget{
				Kind:       d.Kind,
				SpanDigest: d.Entry.Digest,
				MediaType:  d.Entry.MediaType,
				Length:     d.Entry.Length,
				Unit:       d.Entry.Unit,
				Reason:     d.Reason,
			})
			continue
		}
		out = append(out, LMCResetTarget{
			Kind:       d.Kind,
			WholeReset: true,
			Reason:     d.Reason,
		})
	}
	return out
}
