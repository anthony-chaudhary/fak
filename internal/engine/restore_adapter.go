package engine

import (
	"context"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// RestoreAdapter executes one explicitly demanded digest restore. It deliberately
// does not discover demand: the generic digest-to-live-miss ledger remains the
// production follow-on for #1469.
type RestoreAdapter struct {
	KV       abi.KVBackend
	Recorder *CacheEventRecorder
}

type RestoreMove struct {
	SpanDigest   string
	ModelID      string
	TokenizerID  string
	PositionMode cachemeta.PositionMode
	FromTier     cachemeta.ResidencyTier
	ToTier       cachemeta.ResidencyTier
	Owner        string
	Lease        string
}

type RestoreResult struct {
	Residency abi.KVResidency
	Verdict   cachemeta.LookupVerdict
	Recorded  CacheEventResult
	Restored  bool
}

// Restore pages one known demoted span into the live backend. Backend transport
// errors are lowered to the same typed residency-fault plane as explicit faults;
// only a missing adapter/backend is a programming error.
func (a *RestoreAdapter) Restore(ctx context.Context, mv RestoreMove) (RestoreResult, error) {
	if a == nil || a.KV == nil {
		return RestoreResult{}, fmt.Errorf("engine: restore adapter has no live KV backend")
	}
	residency, err := a.KV.RestoreSpan(ctx, mv.SpanDigest)
	if err != nil {
		residency = abi.KVResidency{Outcome: abi.KVResidencyFault, Digest: mv.SpanDigest, Reason: err.Error()}
	}
	if residency.Digest == "" {
		residency.Digest = mv.SpanDigest
	} else if residency.Digest != mv.SpanDigest {
		residency = abi.KVResidency{Outcome: abi.KVResidencyFault, Digest: mv.SpanDigest, Reason: "restore returned a different span digest"}
	}
	if residency.Outcome == abi.KVResidencyOK && residency.Positions <= 0 {
		residency = abi.KVResidency{Outcome: abi.KVResidencyFault, Digest: mv.SpanDigest, Reason: "restore reported OK without installed positions"}
	}
	out := RestoreResult{
		Residency: residency,
		Verdict:   cachemeta.FromKVResidency(residency),
		Restored:  residency.Outcome == abi.KVResidencyOK && residency.Positions > 0,
	}
	if a.Recorder == nil {
		return out, nil
	}
	outcome := cachemeta.KVTransferMissed
	switch residency.Outcome {
	case abi.KVResidencyOK:
		outcome = cachemeta.KVTransferOK
	case abi.KVResidencyFault:
		outcome = cachemeta.KVTransferFault
	}
	out.Recorded = a.Recorder.Record(CacheEvent{
		Direction:    cachemeta.KVRestore,
		SpanDigest:   residency.Digest,
		Tokens:       int64(residency.Positions),
		ModelID:      mv.ModelID,
		TokenizerID:  mv.TokenizerID,
		PositionMode: mv.PositionMode,
		FromTier:     mv.FromTier,
		ToTier:       mv.ToTier,
		Owner:        mv.Owner,
		Lease:        mv.Lease,
		Outcome:      outcome,
		FaultReason:  residency.Reason,
		BytesMoved:   residency.BytesMoved,
	})
	return out, nil
}
