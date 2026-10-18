package model

import (
	"context"
	"errors"
	"time"
)

// DFlash2PrepareTiming describes one successful diagnostic PrepareTimed call.
// Proposal includes host BF16 draft work plus native target embedding/head calls.
// Verify covers only the transaction Verify method. Checkpoint/transaction setup,
// prefix selection and cleanup are outside both spans; Commit has separate cost.
// ProposedDraftTokens excludes the exact target anchor and is not an acceptance
// count. No fields are valid unless Complete is true.
type DFlash2PrepareTiming struct {
	Proposal            time.Duration
	Verify              time.Duration
	ProposalRan         bool
	ProposedDraftTokens int
	Complete            bool
	Verification        DFlash2VerificationWork
}

// PrepareTimed uses exactly the same locked implementation as Prepare, adding
// opt-in wall-clock observations only. Ordinary Prepare does not read the clock.
func (q *Qwen38DFlash2Request) PrepareTimed(ctx context.Context, maxTokens int) (*Qwen38DFlash2Round, DFlash2PrepareTiming, error) {
	var timing DFlash2PrepareTiming
	round, err := q.prepare(ctx, maxTokens, &timing)
	if err != nil {
		return round, DFlash2PrepareTiming{}, err
	}
	timing.Complete = true
	return round, timing, nil
}

func (t *DFlash2PrepareTiming) start() time.Time {
	if t == nil {
		return time.Time{}
	}
	return time.Now()
}
func (t *DFlash2PrepareTiming) finishProposal(start time.Time, proposals int) {
	if t != nil {
		t.Proposal, t.ProposalRan, t.ProposedDraftTokens = time.Since(start), true, proposals
	}
}
func (t *DFlash2PrepareTiming) finishVerify(start time.Time) {
	if t != nil {
		t.Verify = time.Since(start)
	}
}

// CommittedPosition observes the shared target/draft boundary without exposing
// either mutable owner. It refuses an unfinished round or a retired request.
func (q *Qwen38DFlash2Request) CommittedPosition(ctx context.Context) (int, error) {
	if q == nil {
		return 0, errors.New("dflash2: nil request")
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if err := q.usable(ctx); err != nil {
		return 0, err
	}
	if !q.bootstrapped || q.pending != nil {
		return 0, errors.New("dflash2: committed boundary requires a bootstrapped request without a pending round")
	}
	if err := q.ref.proposalReady(ctx); err != nil {
		return 0, err
	}
	position, lineage, err := qwen35TargetFeaturePrefix(q.ref.target)
	if err != nil {
		return 0, err
	}
	if position != q.ref.nextPosition || len(lineage.ids) != position || len(q.ref.prefixTokens) != position {
		return 0, errors.New("dflash2: target/draft committed boundary mismatch")
	}
	return position, nil
}

// DFlash2VerificationWork is an owned scalar receipt of completed Verify work.
// Rows includes the anchor; ConfiguredPanelWidth is policy, while Path,
// Operations and DecodeSteps come from the actual verifier. No device-clock
// duration or end-to-end acceleration is inferred from this receipt.
type DFlash2VerificationWork struct {
	Path                 string
	Rows                 int
	ConfiguredPanelWidth int
	Operations           int
	DecodeSteps          int
	OneOperation         bool
}

func (t *DFlash2PrepareTiming) recordVerification(r TargetVerificationReceipt, rows, width int) {
	if t != nil {
		t.Verification = DFlash2VerificationWork{Path: r.Path, Rows: rows, ConfiguredPanelWidth: width,
			Operations: r.TargetVerificationOperations, DecodeSteps: r.TargetDecodeSteps, OneOperation: r.OneOperation}
	}
}
