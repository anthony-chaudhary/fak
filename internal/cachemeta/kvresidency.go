package cachemeta

import "github.com/anthony-chaudhary/fak/internal/abi"

// FromKVResidency lowers the abi.KVBackend seam's typed residency outcome (the result
// of a StageSpan / RestoreSpan on a span fak does not host locally) onto the SAME
// LookupVerdict trichotomy KVTransferVerdict produces — so a remote / disaggregated L3
// KV backend's outcome flows into the cache-meta lookup path identically to an
// in-stream kv_transfer entry, and a failed restore is a typed MISS / FAULT, never a
// silent recompute (the §2.2 parity requirement applied to the off-box tier).
//
// OK -> HIT (via a kv_transfer entry keyed by the span digest), MISS -> restore_miss,
// FAULT -> residency_fault, and the unset zero value -> absent (fail-closed: a
// half-built outcome is never mistaken for a hit). Reusing FromKVTransfer +
// KVTransferVerdict for the OK / FAULT cases is what makes "KVTransferVerdict-shaped"
// literally true rather than a parallel re-implementation.
func FromKVResidency(r abi.KVResidency) LookupVerdict {
	switch r.Outcome {
	case abi.KVResidencyOK:
		return KVTransferVerdict(FromKVTransfer(KVTransfer{
			Direction:  KVRestore,
			SpanDigest: r.Digest,
			Tokens:     int64(r.Positions),
			ToTier:     TierRemote,
			Outcome:    KVTransferOK,
			BytesMoved: r.BytesMoved,
		}))
	case abi.KVResidencyMiss:
		return Miss(ReasonRestoreMiss)
	case abi.KVResidencyFault:
		return KVTransferVerdict(FromKVTransfer(KVTransfer{
			Direction:   KVRestore,
			SpanDigest:  r.Digest,
			ToTier:      TierRemote,
			Outcome:     KVTransferFault,
			FaultReason: r.Reason,
		}))
	default:
		return Miss(ReasonAbsent)
	}
}
