package abi

// KVResidencyOutcome is the typed result of a residency transfer on a span fak does
// NOT host locally — the disaggregated / remote-L3 KV direction (the agent-memory
// off-box tier). It is the ok | MISS | FAULT trichotomy cachemeta.KVTransferVerdict
// already speaks, declared HERE because internal/abi imports nothing internal (so the
// widened KVBackend seam can return it without a dependency cycle: cachemeta imports
// abi, never the reverse, and bridges this back to a LookupVerdict via FromKVResidency).
//
// The point of a typed outcome — instead of the dense []float32 logits Prefill returns
// — is that a failed restore is NEVER a silent recompute: a MISS is reported so the
// caller knows to recompute, a FAULT is reported with a reason, and neither hangs.
type KVResidencyOutcome uint8

const (
	// KVResidencyUnknown is the zero value: the outcome was not set. It maps to a MISS
	// (fail-closed), never an accidental HIT — a half-built result cannot be mistaken
	// for a successful transfer.
	KVResidencyUnknown KVResidencyOutcome = iota
	// KVResidencyOK: the span moved — staged off-box, or restored into the live cache.
	KVResidencyOK
	// KVResidencyMiss: the tier does not hold the span. The caller must recompute, but
	// it is TOLD (not a silent fallthrough).
	KVResidencyMiss
	// KVResidencyFault: a transport / store error, or a ctx deadline / cancel — also
	// typed, never a hang.
	KVResidencyFault
)

// String renders the outcome for logs and metrics.
func (o KVResidencyOutcome) String() string {
	switch o {
	case KVResidencyOK:
		return "ok"
	case KVResidencyMiss:
		return "miss"
	case KVResidencyFault:
		return "fault"
	default:
		return "unknown"
	}
}

// KVResidency is the typed result a KVBackend's residency-transfer methods
// (StageSpan / RestoreSpan) return for a span addressed by DIGEST. It carries the
// outcome plus enough provenance for cachemeta to lower it onto the kv_transfer plane
// (cachemeta.FromKVResidency): the span digest, its length in positions, the bytes
// actually moved off / on box, and a typed reason on a MISS / FAULT (empty on OK).
type KVResidency struct {
	Outcome    KVResidencyOutcome
	Digest     string // identity of the span staged / restored
	Positions  int    // span length in positions, when known
	BytesMoved int64  // bytes moved off / on box (0 for the in-process local no-op)
	Reason     string // typed detail on MISS / FAULT; empty on OK
}
