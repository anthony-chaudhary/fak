package model

import (
	"errors"
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen38MTPBatchedReceiptSchema is the canonical schema for batched MTP verification receipts.
const Qwen38MTPBatchedReceiptSchema = "fak/qwen38-mtp-batched-receipt/v1"

// Qwen38MTPBatchedReceipt records verification and commit work separately.
// OneOperation concerns verification, not accepted-prefix replay or end-to-end
// speedup. FullTargetReplaySteps and RecurrentRepairTokens describe commit work.
type Qwen38MTPBatchedReceipt struct {
	SchemaVersion                string                    `json:"schema_version"`
	Engine                       Qwen38MTPEngine           `json:"engine"`
	FallbackEngine               Qwen38MTPEngine           `json:"fallback_engine,omitempty"`
	OneOperation                 bool                      `json:"one_operation"`
	TargetVerificationPath       string                    `json:"target_verification_path"`
	FullTargetReplaySteps        int                       `json:"full_target_replay_steps"`
	RecurrentRepairTokens        int                       `json:"recurrent_repair_tokens"`
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

// VerifyBlockWithDraftTime verifies and commits one accepted prefix through the
// existing target transaction, retaining explicit ordinary native downgrade.
// The caller must own Target exclusively until this synchronous method returns.
// Inclusive wall time includes setup, commit and the supplied measured draft time.
func (v *Qwen38BatchedVerifier) VerifyBlockWithDraftTime(draftTokens []int, boundaryLogits []float32, draftTime time.Duration) (result *Qwen38MTPBatchedReceipt, err error) {
	if v == nil || v.Target == nil || v.Target.M == nil {
		return nil, errors.New("model: qwen3.8 batched verifier requires active target session")
	}
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

	started := time.Now()
	tx, err := beginQwen35MTPTargetTransaction(v.Target, boundaryLogits)
	if err != nil {
		return nil, err
	}
	// Keep a rollback owner beyond tx.finish: a resident release callback can
	// fail after Commit has published accepted state and retired its snapshot.
	// An ordinary error return permits the persistent caller to retry scalar
	// decode, so it must always refer to the exact pre-round target state.
	var rollback *PrefixSnapshot
	committed, returned := false, false
	defer func() {
		retirementStarted := time.Now()
		retireQwen38BatchedRound(tx, rollback, committed, returned, &result, &err)
		if result != nil && err == nil {
			result.Accounting.Synchronization.Nanoseconds += time.Since(retirementStarted).Nanoseconds()
			result.Accounting.Synchronization.Measured = true
			result.SyncTimeNS = time.Duration(result.Accounting.Synchronization.Nanoseconds)
			result.TotalWallTimeNS = draftTime + time.Since(started)
		}
	}()
	setupStarted := time.Now()
	rollback, err = tx.snapshot.Clone()
	if err != nil {
		returned = true
		return nil, err
	}
	tx.receipt.Accounting.Setup.Nanoseconds += time.Since(setupStarted).Nanoseconds()
	tx.receipt.Accounting.KnownMemoryBytes += rollback.ResidentBytes()
	result, err = func() (*Qwen38MTPBatchedReceipt, error) {

		// Preserve the existing ordinary-model verifier. It mutates live state only
		// on its actual batched path; the cacheless path must never earn adoption.
		// A device with no raw-hidden panel capability keeps scalar HAL fallback,
		// which captures raw residuals. An advertised but wrong identity is still
		// refused by the canonical transaction rather than treated as absence.
		legacy := !v.Target.M.Cfg.IsQwen35Hybrid()
		if v.Target.captureTargetHidden && qwen35DevicePanelAdvertised(v.Target) {
			_, rawHiddenDeclared := v.Target.Backend.(compute.Qwen35SequenceRawHiddenBackend)
			legacy = legacy || !rawHiddenDeclared
		}
		if legacy {
			tx.verify = func(draft []int) ([][]float32, TargetVerificationReceipt, error) {
				if !v.Target.M.Cfg.IsQwen35Hybrid() && v.Target.captureTargetHidden {
					// Generic batched verification does not retain raw hidden rows.
					// Ordinary Step does; decline before any panel mutation.
					return nil, TargetVerificationReceipt{
						Schema: targetVerificationReceiptSchema, Engine: targetVerificationEngine,
						Path: targetVerificationDecodePath, DraftTokens: len(draft),
					}, targetVerificationDowngrade("generic batched verification cannot preserve raw target-hidden history")
				}
				// Verify already owns the execution domain. Reentering the public
				// acquiring method here would deadlock an execution-owned backend.
				rows, receipt, verifyErr := v.Target.verifyForwardOneOperation(draft, tx.beforeLogits, nil)
				tx.verifiedLive = verifyErr == nil && receipt.Path == targetVerificationBatchedPath
				return rows, receipt, verifyErr
			}
		}
		rows, err := tx.Verify(draftTokens)
		if err != nil {
			return nil, fmt.Errorf("model: batched MTP target verification: %w", err)
		}

		rejectionStart := time.Now()
		if len(rows) != len(draftTokens) {
			return nil, errors.New("model: batched MTP verification returned an incomplete draft panel")
		}
		for _, row := range rows {
			if len(row) != v.Target.M.Cfg.VocabSize {
				return nil, errors.New("model: batched MTP verification returned malformed logits")
			}
		}
		accepted := 0
		if len(boundaryLogits) > 0 && draftTokens[0] == argmaxF32(boundaryLogits) {
			accepted = 1
			for i := 1; i < len(draftTokens); i++ {
				if draftTokens[i] != argmaxF32(rows[i-1]) {
					break
				}
				accepted++
			}
		}
		rejection := measuredSpeculativeCost(rejectionStart)
		bonusToken := -1
		if accepted == len(draftTokens) {
			bonusToken = argmaxF32(rows[len(rows)-1])
			// The canonical ordinary fallback also advances the complete live
			// target. Adopt only its proven full prefix, avoiding a second scalar
			// replay. Partial/zero acceptance still uses transaction restoration.
			verified := tx.VerificationReceipt()
			if !tx.verifiedLive && !verified.OneOperation && verified.Path == targetVerificationDecodePath && verified.TargetDecodeSteps == len(draftTokens) {
				tx.verifiedLive = true
				tx.lastLogits = append([]float32(nil), rows[len(rows)-1]...)
				tx.receipt.Accounting.KnownMemoryBytes += int64(len(tx.lastLogits)) * 4
			}
		}
		lastLogits, err := tx.Commit(accepted)
		if err != nil {
			return nil, fmt.Errorf("model: batched MTP target commit: %w", err)
		}
		committed = true
		verified := tx.VerificationReceipt()
		accounting := verified.Accounting
		accounting.Drafting = SpeculativeCostComponent{Nanoseconds: draftTime.Nanoseconds(), Measured: draftTime > 0}
		accounting.Rejection = rejection
		// The transaction measured its snapshot and retained result storage before
		// retirement. This is known transaction memory, not process/device peak.
		accounting.MemoryMeasured = true
		fallback, reason := Qwen38MTPEngine(""), Qwen38MTPDowngradeReason("")
		if !verified.OneOperation {
			fallback = Qwen38EngineTargetDecode
			reason = Qwen38MTPDowngradeReason(verified.DowngradeReason)
		}
		return &Qwen38MTPBatchedReceipt{
			SchemaVersion:                Qwen38MTPBatchedReceiptSchema,
			Engine:                       Qwen38EngineMTP,
			FallbackEngine:               fallback,
			OneOperation:                 verified.OneOperation,
			TargetVerificationPath:       verified.Path,
			TargetVerificationOperations: verified.TargetVerificationOperations,
			TargetDecodeSteps:            verified.TargetDecodeSteps,
			FullTargetReplaySteps:        verified.FullTargetReplaySteps,
			RecurrentRepairTokens:        verified.RecurrentRepairTokens,
			DraftTokens:                  append([]int(nil), draftTokens...),
			AcceptedTokens:               append([]int(nil), draftTokens[:accepted]...),
			AcceptedCount:                accepted,
			RejectedCount:                len(draftTokens) - accepted,
			BonusToken:                   bonusToken,
			DowngradeReason:              reason,
			DraftTimeNS:                  draftTime,
			VerifyTimeNS:                 time.Duration(accounting.TargetVerification.Nanoseconds),
			RejectionTimeNS:              time.Duration(rejection.Nanoseconds),
			RollbackTimeNS:               time.Duration(accounting.Rollback.Nanoseconds),
			SyncTimeNS:                   time.Duration(accounting.Synchronization.Nanoseconds),
			TotalWallTimeNS:              draftTime + time.Since(started),
			AcceptedLogits:               lastLogits,
			Accounting:                   accounting,
		}, nil
	}()
	returned = true
	return result, err
}

// qwen38BatchedCleanup contains only a nested cleanup panic. An outer unwind
// remains untouched, and Goexit still runs the caller's registered siblings.
func qwen38BatchedCleanup(action func() error) (err error) {
	completed := false
	defer func() {
		if !completed {
			value := recover()
			if cause, ok := value.(error); ok {
				err = cause
			} else {
				err = fmt.Errorf("model: batched MTP cleanup interrupted: %v", value)
			}
		}
	}()
	err = action()
	completed = true
	return err
}

func retireQwen38BatchedRound(tx *qwen35MTPTargetTransaction, rollback *PrefixSnapshot, committed, returned bool, result **Qwen38MTPBatchedReceipt, err *error) {
	normal := returned
	// Register all independent owners first. Fail-closed panics are reserved for
	// a normal return with no proven retry state; they never replace an unwind.
	defer func() {
		if rollback == nil {
			return
		}
		wasNormal := normal
		normal = false
		closeErr := qwen38BatchedCleanup(func() error { rollback.Close(); return nil })
		normal = wasNormal
		if closeErr != nil && normal {
			*result = nil
			*err = errors.Join(*err, closeErr)
			if committed {
				panic(*err)
			}
		}
	}()
	defer func() {
		if committed || rollback == nil {
			return
		}
		wasNormal := normal
		normal = false
		restoreErr := qwen38BatchedCleanup(func() error {
			if restoreErr := rollback.Restore(tx.target); restoreErr != nil {
				return restoreErr
			}
			tx.target.halStep, tx.target.halLogitsWarm = tx.beforeHALStep, tx.beforeHALWarm
			return nil
		})
		normal = wasNormal
		if restoreErr != nil && normal {
			*result = nil
			*err = errors.Join(*err, restoreErr)
			normal = false
			panic(*err)
		}
	}()
	if !tx.closed {
		wasNormal := normal
		normal = false
		abortErr := qwen38BatchedCleanup(func() error { return abortQwen38BatchedTransaction(tx) })
		normal = wasNormal
		if abortErr != nil && normal {
			*result = nil
			*err = errors.Join(*err, abortErr)
			if rollback == nil {
				normal = false
				panic(*err)
			}
		}
	}
}

// abortQwen38BatchedTransaction terminally detaches all owners even if restoring
// the target panics or exits the goroutine. Restoration failure still fails the
// round; releasing the snapshot does not assert that the target was restored.
func abortQwen38BatchedTransaction(tx *qwen35MTPTargetTransaction) (err error) {
	if tx == nil || tx.closed {
		return nil
	}
	scope := tx.target.beginBackendExecution()
	defer scope.finishError(&err)
	returned := false
	// This cleanup must run before releasing the execution domain. Secondary
	// cleanup panics must not replace a restore panic or an existing error.
	defer func() {
		finished := false
		suppressBackendCleanupPanic(func() {
			tx.finish()
			finished = true
		})
		if !finished && returned {
			err = errors.Join(err, errors.New("model: batched MTP transaction release interrupted"))
		}
	}()
	err = tx.abortOwned()
	returned = true
	return err
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
