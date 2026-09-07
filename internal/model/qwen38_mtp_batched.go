package model

import (
	"errors"
	"fmt"
	"time"
)

// Qwen38MTPBatchedReceiptSchema is the canonical schema for batched MTP verification receipts.
const Qwen38MTPBatchedReceiptSchema = "fak/qwen38-mtp-batched-receipt/v1"

// Qwen38MTPBatchedReceipt records the outcome, token disposition, and end-to-end latency
// accounting of a single-operation target verification for Qwen3.8 MTP blocks.
type Qwen38MTPBatchedReceipt struct {
	SchemaVersion                string                    `json:"schema_version"`
	Engine                       Qwen38MTPEngine           `json:"engine"`
	FallbackEngine               Qwen38MTPEngine           `json:"fallback_engine,omitempty"`
	OneOperation                 bool                      `json:"one_operation"`
	TargetVerificationOperations int                       `json:"target_verification_operations"`
	TargetDecodeSteps            int                       `json:"target_decode_steps"`
	DraftTokens                  []int                     `json:"draft_tokens"`
	AcceptedTokens               []int                     `json:"accepted_tokens"`
	AcceptedCount                int                       `json:"accepted_count"`
	RejectedCount                int                       `json:"rejected_count"`
	BonusToken                   int                       `json:"bonus_token"`
	DowngradeReason              Qwen38MTPDowngradeReason  `json:"downgrade_reason,omitempty"`
	DraftTimeNS                  time.Duration             `json:"draft_time_ns"`
	VerifyTimeNS                 time.Duration             `json:"verify_time_ns"`
	RejectionTimeNS              time.Duration             `json:"rejection_time_ns"`
	RollbackTimeNS               time.Duration             `json:"rollback_time_ns"`
	SyncTimeNS                   time.Duration             `json:"sync_time_ns"`
	TotalWallTimeNS              time.Duration             `json:"total_wall_time_ns"`
	Accounting                   SpeculativeCostAccounting `json:"accounting"`
	AcceptedLogits               []float32                 `json:"-"`
}

// Validate verifies that the receipt represents a strictly fak-native execution path
// and internally consistent token and timing accounting. It strictly rejects foreign runtimes.
func (r Qwen38MTPBatchedReceipt) Validate() error {
	if r.SchemaVersion != Qwen38MTPBatchedReceiptSchema {
		return fmt.Errorf("model: qwen3.8 batched receipt schema %q, want %q", r.SchemaVersion, Qwen38MTPBatchedReceiptSchema)
	}
	if r.Engine != Qwen38EngineMTP {
		return fmt.Errorf("model: qwen3.8 batched receipt engine %q is not fak-native MTP (%q)", r.Engine, Qwen38EngineMTP)
	}
	if r.FallbackEngine != "" && r.FallbackEngine != Qwen38EngineTargetDecode {
		return fmt.Errorf("model: qwen3.8 batched receipt fallback engine %q is not fak-native target decode", r.FallbackEngine)
	}
	if len(r.DraftTokens) != r.AcceptedCount+r.RejectedCount {
		return fmt.Errorf("model: draft token count %d != accepted %d + rejected %d", len(r.DraftTokens), r.AcceptedCount, r.RejectedCount)
	}
	if r.AcceptedCount != len(r.AcceptedTokens) {
		return fmt.Errorf("model: accepted count %d != len(accepted_tokens) %d", r.AcceptedCount, len(r.AcceptedTokens))
	}
	if r.OneOperation && r.TargetVerificationOperations != 1 {
		return fmt.Errorf("model: one_operation=true requires target_verification_operations=1, got %d", r.TargetVerificationOperations)
	}
	if !r.OneOperation && r.TargetDecodeSteps == 0 && len(r.DraftTokens) > 0 {
		return fmt.Errorf("model: downgraded target decode requires non-zero target_decode_steps")
	}
	return nil
}

// Qwen38BatchedVerifier evaluates proposed MTP blocks with a single target-side
// verification operation rather than N serial target decode steps. On unsupported
// configurations, it safely downgrades to ordinary fak-native target decode.
type Qwen38BatchedVerifier struct {
	Target *Session
}

// NewQwen38BatchedVerifier constructs a batched verifier bound to a target Session.
func NewQwen38BatchedVerifier(target *Session) (*Qwen38BatchedVerifier, error) {
	if target == nil {
		return nil, errors.New("model: qwen3.8 batched verifier requires non-nil target session")
	}
	return &Qwen38BatchedVerifier{Target: target}, nil
}

// VerifyBlock evaluates draft tokens in a single batched target forward operation.
func (v *Qwen38BatchedVerifier) VerifyBlock(draftTokens []int, boundaryLogits []float32) (*Qwen38MTPBatchedReceipt, error) {
	return v.VerifyBlockWithDraftTime(draftTokens, boundaryLogits, 0)
}

// VerifyBlockWithDraftTime evaluates draft tokens in a single batched target forward operation,
// incorporating the measured draft time into total wall time accounting.
func (v *Qwen38BatchedVerifier) VerifyBlockWithDraftTime(draftTokens []int, boundaryLogits []float32, draftTime time.Duration) (*Qwen38MTPBatchedReceipt, error) {
	if v == nil || v.Target == nil || v.Target.M == nil {
		return nil, errors.New("model: qwen3.8 batched verifier requires active target session")
	}

	setupStart := time.Now()
	if len(draftTokens) == 0 {
		return &Qwen38MTPBatchedReceipt{
			SchemaVersion:                Qwen38MTPBatchedReceiptSchema,
			Engine:                       Qwen38EngineMTP,
			OneOperation:                 true,
			TargetVerificationOperations: 1,
			BonusToken:                   -1,
			AcceptedLogits:               append([]float32(nil), boundaryLogits...),
		}, nil
	}

	// Capture pre-round verified snapshot for exact rollback.
	snap, err := v.Target.PrefixSnapshot()
	if err != nil {
		return nil, fmt.Errorf("model: snapshot target before batched verification: %w", err)
	}
	defer snap.Close()
	setupDuration := time.Since(setupStart)

	// Single-operation target forward verification.
	verifyStart := time.Now()
	rows, _, vErr := v.Target.VerifyForwardOneOperation(draftTokens, boundaryLogits)

	isOneOp := true
	targetOps := 1
	targetSteps := 0
	fallbackEngine := Qwen38MTPEngine("")
	downgradeReason := Qwen38MTPDowngradeReason("")

	if vErr != nil {
		// Clean fail-closed downgrade to ordinary fak-native target decode.
		// Never delegates to foreign runtimes (llama.cpp).
		isOneOp = false
		targetOps = 0
		targetSteps = len(draftTokens)
		fallbackEngine = Qwen38EngineTargetDecode

		var dgErr *TargetVerificationDowngradeError
		if errors.As(vErr, &dgErr) {
			downgradeReason = Qwen38MTPDowngradeReason(dgErr.Reason)
		} else {
			downgradeReason = Qwen38MTPAttemptFailed
		}

		// Restore pre-round snapshot before sequential step decode.
		clone, cErr := snap.Clone()
		if cErr != nil {
			return nil, fmt.Errorf("model: clone pre-round snapshot for downgrade: %w", cErr)
		}
		if rErr := clone.Restore(v.Target); rErr != nil {
			clone.Close()
			return nil, fmt.Errorf("model: restore pre-round snapshot for downgrade: %w", rErr)
		}

		rows = v.Target.verifyForwardSequential(draftTokens)
	}
	verifyDuration := time.Since(verifyStart)
	if verifyDuration == 0 {
		verifyDuration = 1 * time.Nanosecond
	}

	// Compute acceptance count: all matching tokens up to the first divergence.
	rejectionStart := time.Now()
	accepted := 0
	if len(boundaryLogits) > 0 && len(draftTokens) > 0 {
		target0 := argmaxF32(boundaryLogits)
		if draftTokens[0] == target0 {
			accepted = 1
			for i := 1; i < len(draftTokens); i++ {
				if i-1 < len(rows) {
					targetI := argmaxF32(rows[i-1])
					if draftTokens[i] == targetI {
						accepted++
					} else {
						break
					}
				}
			}
		}
	}
	rejected := len(draftTokens) - accepted
	rejectionDuration := time.Since(rejectionStart)

	// Handle atomic commit or rollback of target state.
	var rollbackDuration time.Duration
	var syncDuration time.Duration
	var lastLogits []float32
	bonusToken := -1

	if accepted == len(draftTokens) {
		// Full accept: advance/commit all draft tokens into target session.
		syncStart := time.Now()
		if isOneOp {
			// Cacheless operation didn't mutate live target; step accepted prefix.
			for _, tok := range draftTokens {
				lastLogits = v.Target.Step(tok)
			}
		} else {
			// Sequential decode already stepped all draft tokens.
			if len(rows) > 0 {
				lastLogits = rows[len(rows)-1]
			}
		}
		syncDuration = time.Since(syncStart)

		if len(rows) > 0 && len(rows[len(rows)-1]) > 0 {
			bonusToken = argmaxF32(rows[len(rows)-1])
		}
	} else if accepted == 0 {
		// Total rejection: restore exact pre-round snapshot if mutated.
		rollbackStart := time.Now()
		if !isOneOp {
			clone, _ := snap.Clone()
			_ = clone.Restore(v.Target)
		}
		rollbackDuration = time.Since(rollbackStart)
		if rollbackDuration == 0 {
			rollbackDuration = 1 * time.Nanosecond
		}
		lastLogits = append([]float32(nil), boundaryLogits...)
	} else {
		// Partial accept: rollback unverified suffix, advance only accepted prefix.
		rollbackStart := time.Now()
		if !isOneOp {
			clone, _ := snap.Clone()
			_ = clone.Restore(v.Target)
		}
		rollbackDuration = time.Since(rollbackStart)
		if rollbackDuration == 0 {
			rollbackDuration = 1 * time.Nanosecond
		}

		syncStart := time.Now()
		for _, tok := range draftTokens[:accepted] {
			lastLogits = v.Target.Step(tok)
		}
		syncDuration = time.Since(syncStart)
	}

	totalDuration := draftTime + verifyDuration + rejectionDuration + rollbackDuration + syncDuration

	receipt := &Qwen38MTPBatchedReceipt{
		SchemaVersion:                Qwen38MTPBatchedReceiptSchema,
		Engine:                       Qwen38EngineMTP,
		FallbackEngine:               fallbackEngine,
		OneOperation:                 isOneOp,
		TargetVerificationOperations: targetOps,
		TargetDecodeSteps:            targetSteps,
		DraftTokens:                  append([]int(nil), draftTokens...),
		AcceptedTokens:               append([]int(nil), draftTokens[:accepted]...),
		AcceptedCount:                accepted,
		RejectedCount:                rejected,
		BonusToken:                   bonusToken,
		DowngradeReason:              downgradeReason,
		DraftTimeNS:                  draftTime,
		VerifyTimeNS:                 verifyDuration,
		RejectionTimeNS:              rejectionDuration,
		RollbackTimeNS:               rollbackDuration,
		SyncTimeNS:                   syncDuration,
		TotalWallTimeNS:              totalDuration,
		AcceptedLogits:               lastLogits,
		Accounting: SpeculativeCostAccounting{
			Setup:              SpeculativeCostComponent{Nanoseconds: setupDuration.Nanoseconds(), Measured: true},
			Drafting:           SpeculativeCostComponent{Nanoseconds: draftTime.Nanoseconds(), Measured: draftTime > 0},
			TargetVerification: SpeculativeCostComponent{Nanoseconds: verifyDuration.Nanoseconds(), Measured: true},
			Rejection:          SpeculativeCostComponent{Nanoseconds: rejectionDuration.Nanoseconds(), Measured: true},
			Rollback:           SpeculativeCostComponent{Nanoseconds: rollbackDuration.Nanoseconds(), Measured: true},
			Synchronization:    SpeculativeCostComponent{Nanoseconds: syncDuration.Nanoseconds(), Measured: true},
			KnownMemoryBytes:   snap.ResidentBytes(),
			MemoryMeasured:     true,
		},
	}

	return receipt, nil
}

// VerifyMTPBlockBatched evaluates draft tokens against the target session in a single
// batched forward operation, returning a receipt with exact acceptance and latency accounting.
func VerifyMTPBlockBatched(target *Session, draftTokens []int, boundaryLogits []float32, draftTime ...time.Duration) (*Qwen38MTPBatchedReceipt, error) {
	v, err := NewQwen38BatchedVerifier(target)
	if err != nil {
		return nil, err
	}
	dt := time.Duration(0)
	if len(draftTime) > 0 {
		dt = draftTime[0]
	}
	return v.VerifyBlockWithDraftTime(draftTokens, boundaryLogits, dt)
}
