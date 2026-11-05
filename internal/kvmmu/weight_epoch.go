package kvmmu

// weight_epoch.go — weight-epoch accounting for KV spans (continual-engine §4 P2,
// phase 1): every span records the weight state it was computed under (stamped in
// Append through the optional abi.KVBackend seam); serving a span under a changed
// weight state fails closed instead of silently recomputing — the
// never-silent-recompute doctrine (internal/engine/cacheevents.go SilentRecompute;
// the engine package cites it, it does not own this seam). Page-level coupling is
// deferred (§10.2): the epoch rides the span ledger, not the pager ABI.

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ErrWeightEpochMismatch is the fail-closed refusal for serving a KV span whose
// recorded weight state does not cover the live one. Callers assert it with
// errors.Is; it is the sentinel, never the prose, that gates a serve.
var ErrWeightEpochMismatch = errors.New("kvmmu: weight-epoch mismatch: refusing to serve a KV span under a different weight state")

// CheckSpanEpoch decides whether a span stamped under `stamped` may be served under
// `current`. Servable when the weight-state digest still matches — fresh spans, and
// spans restored by a revert to the same content — or when EVERY weight transition
// committed between the two generations declared itself compatible (representation-
// only changes such as quant re-packs). Anything else refuses: an epoch_bump anywhere
// in the window, a rewound stamp, or an incomplete accounting window. decls lists the
// declaration of each transition committed after the span's generation, oldest first
// (the weight layer owns that history: model.WeightDeltaSource.DeclarationsSince).
func CheckSpanEpoch(stamped, current model.WeightEpochStamp, decls []model.WeightDeltaKVCompat) error {
	if stamped.Digest == current.Digest {
		return nil
	}
	if stamped.Generation > current.Generation {
		return ErrWeightEpochMismatch // a rewind is impossible: corrupt or foreign stamp
	}
	if int(current.Generation-stamped.Generation) != len(decls) {
		return ErrWeightEpochMismatch // accounting gap: the window is not complete
	}
	for _, d := range decls {
		if d == model.KVCompatEpochBump {
			return ErrWeightEpochMismatch
		}
	}
	return nil
}
