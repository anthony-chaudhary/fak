package cachemeta

import "github.com/anthony-chaudhary/fak/internal/abi"

// KVTransferDirection names a live-engine KV residency event: offload (HBM->DRAM/
// disk/remote), restore (a tier re-materializing a span the engine wants), route
// (a KV-aware router pinning a request to the replica holding the span), or migrate
// (moving residency between instances). §2.2's gap was that these events were not
// in the same cache-entry stream as tool/context entries.
type KVTransferDirection string

const (
	KVOffload KVTransferDirection = "offload"
	KVRestore KVTransferDirection = "restore"
	KVRoute   KVTransferDirection = "route"
	KVMigrate KVTransferDirection = "migrate"
)

// KVTransferOutcome records whether a residency transition succeeded. A restore
// that found nothing usable is a typed MISS, and a load/restore error is a typed
// FAULT — never silent recompute (§2.2 parity requirement).
type KVTransferOutcome string

const (
	KVTransferOK     KVTransferOutcome = "ok"
	KVTransferMissed KVTransferOutcome = "missed"
	KVTransferFault  KVTransferOutcome = "fault"
)

// KVTransfer is the field-only shape a live-engine KV-residency adapter lowers
// into. It describes a residency transition for a KV span, keeping residency tier
// and owner separate from the payload (§2.2: "residency tier and owner recorded
// separately from payload").
type KVTransfer struct {
	Direction           KVTransferDirection
	SpanDigest          string // identity of the KV span being moved/restored/routed
	Tokens              int64  // span length in positions
	ModelID             string
	TokenizerID         string
	SerializerID        string
	PositionMode        PositionMode
	FromTier            ResidencyTier
	ToTier              ResidencyTier
	Owner               string
	Lease               string
	SecuritySet         bool
	Taint               abi.TaintLabel
	Scope               abi.ShareScope
	AdmissionVerdict    AdmissionVerdict
	AdmittedBy          string
	DeletionCertificate DeletionCertificate
	Outcome             KVTransferOutcome
	FaultReason         string // free-text when Outcome == fault
	BytesMoved          int64
}

// FromKVTransfer normalizes a live-engine KV residency event into a cache entry on
// the kv_transfer plane. Residency records where the span now lives (ToTier); the
// outcome is carried in Labels so an observing sink can separate ok/missed/fault
// without re-deriving it.
func FromKVTransfer(t KVTransfer, opts ...Option) Entry {
	owner := t.Owner
	if owner == "" {
		owner = "engine"
	}
	to := t.ToTier
	if to == "" {
		to = TierUnknown
	}
	pm := t.PositionMode
	if pm == "" {
		pm = PositionPrefixAligned
	}
	outcome := t.Outcome
	if outcome == "" {
		outcome = KVTransferOK
	}
	digest := t.SpanDigest
	if digest == "" {
		digest = DigestBytes([]byte(string(t.Direction) + "\x00" + t.ModelID + "\x00" + t.TokenizerID))
	}
	e := Entry{
		ID: EntryID{
			Digest:    digest,
			MediaType: MediaKVSpan,
			Length:    t.Tokens,
			Unit:      UnitPositions,
		},
		Plane: PlaneKVTransfer,
		Derivation: Derivation{
			Producer:     owner,
			ModelID:      t.ModelID,
			TokenizerID:  t.TokenizerID,
			SerializerID: t.SerializerID,
			PositionMode: pm,
		},
		Security: Security{
			Taint:            abi.TaintTrusted,
			Scope:            abi.ScopeFleet,
			AdmissionVerdict: AdmissionAllow,
			AdmittedBy:       owner,
		},
		Residency: Residency{Tier: to, Owner: owner, Lease: t.Lease},
		Coherence: Coherence{InvalidationMode: InvalidationNone},
		Metrics: Metrics{
			BytesTransferred: t.BytesMoved,
		},
		Labels: map[string]string{
			"direction": string(t.Direction),
			"outcome":   string(outcome),
		},
	}
	if t.SecuritySet {
		admittedBy := t.AdmittedBy
		if admittedBy == "" {
			admittedBy = owner
		}
		e.Security = Security{
			Taint:            t.Taint,
			Scope:            t.Scope,
			AdmissionVerdict: t.AdmissionVerdict,
			AdmittedBy:       admittedBy,
		}
	}
	if t.FromTier != "" {
		e.Labels["from_tier"] = string(t.FromTier)
	}
	if t.ToTier != "" {
		e.Labels["to_tier"] = string(t.ToTier)
	}
	if t.FaultReason != "" {
		e.Labels["fault_reason"] = t.FaultReason
	}
	putDeletionCertificateLabels(e.Labels, t.DeletionCertificate)
	apply(&e, opts)
	return e
}

// KVTransferVerdict turns a transfer entry's recorded outcome into a typed lookup
// verdict: fault -> FAULT (residency_fault), missed -> MISS (restore_miss), ok ->
// HIT. This is the §2.2 guarantee that a failed restore is never silent recompute.
func KVTransferVerdict(e Entry) LookupVerdict {
	switch KVTransferOutcome(e.Labels["outcome"]) {
	case KVTransferFault:
		return Fault(e, ReasonResidencyFault)
	case KVTransferMissed:
		return Miss(ReasonRestoreMiss)
	case KVTransferOK:
		return Hit(e)
	default:
		return Miss(ReasonAbsent)
	}
}
