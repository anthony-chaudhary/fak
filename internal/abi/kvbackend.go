package abi

import "context"

// KVResidencyOutcome is the typed result of a residency transfer on a span fak does NOT
// host locally (disaggregated / remote-L3 KV): the ok | MISS | FAULT trichotomy
// cachemeta.KVTransferVerdict speaks, declared HERE because internal/abi imports nothing
// internal, so the widened KVBackend seam returns it without a dependency cycle. A typed
// outcome means a failed restore is NEVER a silent recompute: MISS tells the caller to
// recompute, FAULT carries a reason.
type KVResidencyOutcome uint8

const (
	// KVResidencyUnknown is the zero value (outcome not set); it maps to a MISS (fail-closed),
	// never an accidental HIT — a half-built result cannot be mistaken for a successful transfer.
	KVResidencyUnknown KVResidencyOutcome = iota
	// KVResidencyOK: the span moved — staged off-box, or restored into the live cache.
	KVResidencyOK
	// KVResidencyMiss: the tier does not hold the span; the caller must recompute (told, not a silent fallthrough).
	KVResidencyMiss
	// KVResidencyFault: a transport/store error or ctx deadline/cancel — typed, never a hang.
	KVResidencyFault
)

// String renders the residency outcome as its cachemeta plane token: ok, miss, fault, or unknown.
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

// KVResidency is the typed result a KVBackend's residency-transfer methods return for a
// span addressed by DIGEST: the outcome plus the provenance cachemeta lowers onto the
// kv_transfer plane (digest, positions, bytes moved off/on box, typed reason on MISS/FAULT).
type KVResidency struct {
	Outcome    KVResidencyOutcome
	Digest     string
	Positions  int
	BytesMoved int64
	Reason     string
}

// KVPlacementStager is an optional extension for backends whose staging target matters:
// CapacityAdapter uses it when available so a backend cannot stage to host DRAM while the
// placement record falsely names disk or a remote tier.
type KVPlacementStager interface {
	StageSpanTo(ctx context.Context, digest string, from, n int, tier string) (KVResidency, error)
}

// KVDigestEvictor is an optional extension for caches whose resident entries are complete digest-addressed owners rather than mutable ranges in one sequence.
type KVDigestEvictor interface {
	EvictDigest(digest string, from, n int) int
}
