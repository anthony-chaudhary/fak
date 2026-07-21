package cachemeta

import "github.com/anthony-chaudhary/fak/internal/abi"

// mooncake_transfer.go maps Mooncake's Transfer Engine events onto the SAME
// kv_transfer residency stream every other KV movement uses (FromKVTransfer /
// KVTransferVerdict). It is a MAPPING adapter only: cachemeta never links or calls
// Mooncake, vendors no external code, and rebuilds no store — a concrete
// Mooncake-facing adapter populates the field-only MooncakeTransferEvent shape and
// lowers it here, exactly as nixl_lease.go does for vLLM NIXL leases.
//
// Mooncake exposes a KVCache-centric store fronted by a Transfer Engine that moves
// K/V blocks over RDMA / TCP / NVMe-oF. Per the KV-transport governance doc
// (docs/serving/kv-transport-governance-nixl-mooncake-lmcache.md §2.2) three engine
// operations map onto the existing KVTransferDirection vocabulary:
//
//   - prefill->decode KV transfer  -> KVMigrate (residency moves between instances)
//   - distributed store lookup     -> KVRoute   (a KV-aware route pins to the holder)
//   - remote KV materialization    -> KVRestore (a tier re-materializes a wanted span)
//
// plus a fault path. §6 marked the live wiring a GAP: fak's seams (FromKVTransfer,
// KVTransferVerdict, the ExternalInvalidationDirective plane) shipped, but nothing
// actually emitted Mooncake events into them. This file closes that gap.
//
// House posture is preserved: the event is field-only and its lowering is a
// deterministic, wall-clock-FREE function of its fields (unlike a NIXL lease, a
// Transfer Engine event is a point-in-time transition with no TTL, so no clock is
// threaded — replay is reproducible from the event alone). A failed restore/migrate
// is a typed FAULT (never a silent recompute), and a fault inside a DSA path is
// escalated to QUARANTINE because a dependent attention index may reference the
// poisoned span (§1.2).

// MooncakeEventKind names the Transfer Engine operation fak observed. The three kinds
// are the §2.2 mapping targets; each projects onto one existing KVTransferDirection.
type MooncakeEventKind string

const (
	// MooncakeMigrate — a prefill->decode K/V transfer: residency moves from the
	// prefill instance to the decode instance. Projects onto KVMigrate.
	MooncakeMigrate MooncakeEventKind = "migrate"
	// MooncakeRoute — a distributed-store lookup that pins a request to the instance
	// already holding the span (a KV-aware route). Projects onto KVRoute.
	MooncakeRoute MooncakeEventKind = "route"
	// MooncakeRestore — a remote K/V materialization: a tier re-materializes a span
	// the engine wants back on a faster tier. Projects onto KVRestore.
	MooncakeRestore MooncakeEventKind = "restore"
)

// Direction projects the observed Mooncake operation onto the existing
// KVTransferDirection the kv_transfer plane already speaks. An unrecognized kind
// yields the empty direction (fail-closed: an unknown operation is never silently
// treated as one of the three governed movements — MooncakeTransferVerdict turns it
// into a Miss rather than a phantom hit).
func (k MooncakeEventKind) Direction() KVTransferDirection {
	switch k {
	case MooncakeMigrate:
		return KVMigrate
	case MooncakeRoute:
		return KVRoute
	case MooncakeRestore:
		return KVRestore
	default:
		return ""
	}
}

// MooncakeTransferEvent is the field-only witness of one Transfer Engine event. It
// carries exactly the §4 external-system requirements: a stable content-addressed
// SpanDigest, the FromTier/ToTier residency tiers the bytes moved between, the typed
// Outcome, and BytesMoved. DSAPath marks that the transfer served a DSA (disaggregated
// sparse attention) index so a fault must quarantine, not merely fault (§1.2).
// cachemeta never calls a Mooncake API; a Mooncake-facing adapter fills this shape.
type MooncakeTransferEvent struct {
	// Kind is which of the three §2.2 operations this event reports.
	Kind MooncakeEventKind
	// Backend is the transport that moved the bytes (rdma, tcp, nvme-of).
	Backend string
	// SpanDigest is the content-addressed identity of the K/V span being moved,
	// looked up, or restored. Empty = an unidentified event (fail-closed: it can
	// never back a hit — see MooncakeTransferVerdict).
	SpanDigest string
	// Tokens is the span length in positions.
	Tokens int64
	// ModelID / TokenizerID bind the span to the model that produced it.
	ModelID     string
	TokenizerID string
	// FromTier / ToTier are the residency tiers the transfer moved between; ToTier is
	// where the span now lives.
	FromTier ResidencyTier
	ToTier   ResidencyTier
	// Owner is the instance/store node that now owns the residency (defaults to the
	// generic engine owner inside FromKVTransfer when empty).
	Owner string
	// Lease is the store's own residency handle for the span, when it exposes one.
	Lease string
	// Outcome is the typed result: OK, a MISS (lookup found nothing usable), or a
	// FAULT (the transfer errored). The empty value lowers to OK inside FromKVTransfer.
	Outcome KVTransferOutcome
	// FaultReason is free text carried when Outcome == fault.
	FaultReason string
	// BytesMoved records transfer volume when known.
	BytesMoved int64
	// DSAPath marks that this transfer backed a DSA attention index, so a fault must
	// escalate to quarantine (§1.2) rather than a plain residency fault.
	DSAPath bool
}

// FromMooncakeTransfer lowers a Transfer Engine event onto the kv_transfer plane by
// reusing FromKVTransfer, so a Mooncake movement flows through the SAME residency
// stream and label schema as every other KV event — never a parallel record. The
// direction comes from the §2.2 mapping; the outcome, tiers, digest, and bytes pass
// straight through. The event kind, backend, and DSA bit are recorded as labels so an
// observing sink separates them without re-deriving. A Mooncake span is externally
// owned, so its coherence is external refutation (a freed/evicted remote span is
// refuted the moment the store drops it, never a policy-governed local borrow). When
// the event faulted inside a DSA path the entry itself is quarantined (taint +
// admission), so a pool/sink that stores it holds it back from serving (§1.2).
func FromMooncakeTransfer(ev MooncakeTransferEvent, opts ...Option) Entry {
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
	e.Labels["mooncake_event"] = string(ev.Kind)
	if ev.DSAPath {
		e.Labels["dsa_path"] = "true"
	}
	// An externally owned span is refuted the moment the store frees it.
	e.Coherence.InvalidationMode = InvalidationExternalRefutation
	// §1.2: a fault inside a DSA path poisons any dependent attention index, so the
	// entry is quarantined at the source rather than left as a plain fault.
	if ev.Outcome == KVTransferFault && ev.DSAPath {
		e.Security.Taint = abi.TaintQuarantined
		e.Security.AdmissionVerdict = AdmissionQuarantine
	}
	return e
}

// MooncakeTransferVerdict folds a Transfer Engine event into the typed lookup verdict
// a router MUST consult before taking a cache-aware route to the span. It reuses
// KVTransferVerdict for the ordinary trichotomy (OK -> HIT, missed -> MISS(restore),
// fault -> FAULT(residency_fault)) so the mapping is literally the shared one, and
// only special-cases the DSA fault: per §1.2 a fault inside a DSA path is QUARANTINE
// (held back, dependent index invalid), not a plain residency fault. An event that
// names no span or an unrecognized kind is a Miss (never a phantom hit).
func MooncakeTransferVerdict(ev MooncakeTransferEvent) LookupVerdict {
	if ev.SpanDigest == "" || ev.Kind.Direction() == "" {
		return Miss(ReasonAbsent)
	}
	e := FromMooncakeTransfer(ev)
	if ev.Outcome == KVTransferFault && ev.DSAPath {
		return Quarantine(e, ReasonResidencyFault)
	}
	return KVTransferVerdict(e)
}

// MooncakeResetTarget is the field-only instruction a Mooncake-facing adapter turns
// into a Transfer-Engine store reset when fak refutes a span (§3.2, §4.5). It is the
// Mooncake projection of a planned ExternalInvalidationDirective: a named span reset
// when fak holds span identity, or a whole-store reset fallback when the directive
// carries no span (the store has no exact-span API to target). Payload-free — it names
// the span identity and the reason, never the bytes.
type MooncakeResetTarget struct {
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

// MooncakeResetTargets maps fak's planned invalidation directives onto Mooncake reset
// instructions — this is the adapter "responding to" fak's ExternalInvalidationDirective
// for cache reset (§3.2, §4.5). A directive that names a valid content-addressed span
// becomes a precise exact-span reset; a directive with no span identity becomes a
// whole-store reset fallback so the store is never left holding a refuted span merely
// because fak could not name it exactly. Directive order is preserved.
func MooncakeResetTargets(dirs []ExternalInvalidationDirective) []MooncakeResetTarget {
	if len(dirs) == 0 {
		return nil
	}
	out := make([]MooncakeResetTarget, 0, len(dirs))
	for _, d := range dirs {
		if d.Entry.Valid() {
			out = append(out, MooncakeResetTarget{
				Kind:       d.Kind,
				SpanDigest: d.Entry.Digest,
				MediaType:  d.Entry.MediaType,
				Length:     d.Entry.Length,
				Unit:       d.Entry.Unit,
				Reason:     d.Reason,
			})
			continue
		}
		out = append(out, MooncakeResetTarget{
			Kind:       d.Kind,
			WholeReset: true,
			Reason:     d.Reason,
		})
	}
	return out
}
