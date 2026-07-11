package modelroute

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// loopTestClock is a deterministic, movable clock for the fake-clock witness.
type loopTestClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *loopTestClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *loopTestClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// loopTestLease is a fake per-subject lease that reports a chosen set of keys as
// already held, so the tick's colliding-lease refusal is observable.
type loopTestLease struct {
	busy map[string]bool
}

func (l loopTestLease) AcquireSubjectLease(key string) (func(), bool, error) {
	if l.busy[key] {
		return nil, false, nil
	}
	return func() {}, true, nil
}

// loopTestReceipt builds a real, Verify()-valid receipt for one issue through
// the actual cross-audit spine, varying the closing patch per issue so every
// subject gets a distinct subject digest and audit key.
func loopTestReceipt(t *testing.T, issue int, verdict CrossAuditVerdict) IssueAuditReceipt {
	t.Helper()
	manifest := AuthorManifest{
		Schema: CrossAuditAuthorSchema,
		Author: ModelIdentity{
			Harness: "codex", Provider: "openai", Family: "gpt", Model: "gpt-5.4",
			WeightsRevision: "gpt-w54", EndpointClass: "remote", AccountClass: "subscription", ReasoningPosture: "xhigh",
		},
		SourceEvidence: []EvidenceRef{{Kind: "session", Ref: "codex-session:abc"}},
		CommitRange:    "abc123..def456",
	}
	patch := fmt.Sprintf("diff --git a/thing.go b/thing.go\n+fixed issue %d\n", issue)
	ev := crossAuditFixtureEvidence()
	ev.IssueNumber = issue
	ev.IssueURL = fmt.Sprintf("https://github.com/example/repo/issues/%d", issue)
	ev.Diff = patch
	ev.ClosingCommits[0].Patch = patch
	ev.ClosingCommits[0].PatchSHA256 = IssueAuditContentDigest(patch)

	fetcher := IssueAuditFetcherFunc(func(context.Context, int) (IssueAuditEvidence, error) { return ev, nil })
	reviewer := IssueAuditReviewerFunc(func(context.Context, IssueAuditReviewRequest) (IssueAuditReviewResult, error) {
		return IssueAuditReviewResult{Verdict: verdict, Reason: "fixture " + string(verdict)}, nil
	})
	receipt, err := AuditIssue(context.Background(), IssueAuditRequest{
		IssueNumber: issue,
		Author:      manifest,
		Auditor: ModelIdentity{
			Harness: "fak issue audit", Provider: "anthropic", Family: "claude", Model: "claude-review",
			WeightsRevision: "claude-w46", EndpointClass: "hosted", AccountClass: "subscription",
			ReasoningPosture: "high", ProvenanceSource: "session:claude-review",
		},
		IndependencePolicy: crossAuditTestPolicy(),
	}, fetcher, reviewer)
	if err != nil {
		t.Fatalf("build fixture receipt for #%d/%s: %v", issue, verdict, err)
	}
	return receipt
}

func loopTestPaths(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	return filepath.Join(dir, "ledger.jsonl"), filepath.Join(dir, "cursor.json")
}

// countingAuditor records which issues it was asked to audit and returns a
// scripted receipt (or error) per issue.
type countingAuditor struct {
	receipts map[int]IssueAuditReceipt
	errs     map[int]error
	calls    map[int]int
}

func newCountingAuditor() *countingAuditor {
	return &countingAuditor{
		receipts: map[int]IssueAuditReceipt{},
		errs:     map[int]error{},
		calls:    map[int]int{},
	}
}

func (a *countingAuditor) AuditSubject(_ context.Context, s IssueAuditLoopSubject) (IssueAuditReceipt, error) {
	a.calls[s.IssueNumber]++
	if err := a.errs[s.IssueNumber]; err != nil {
		return IssueAuditReceipt{}, err
	}
	return a.receipts[s.IssueNumber], nil
}

func eligible(n int) IssueAuditLoopSubject {
	return IssueAuditLoopSubject{IssueNumber: n, MarkerKey: fmt.Sprintf("k%d", n), Risk: AuditRiskDefault, Eligible: true}
}

// TestIssueAuditLoopAuditsEachEligibleSubjectExactlyOnce is the crash/restart
// witness: two resumed ticks over the same on-disk ledger+cursor prove each
// eligible subject is audited exactly once, an UNAVAILABLE row is retained (not
// lost) and retried to a terminal verdict, and a settled subject is never
// re-audited.
func TestIssueAuditLoopAuditsEachEligibleSubjectExactlyOnce(t *testing.T) {
	ledgerPath, cursorPath := loopTestPaths(t)
	clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}

	auditor := newCountingAuditor()
	auditor.receipts[100] = loopTestReceipt(t, 100, CrossAuditPass)
	auditor.receipts[101] = loopTestReceipt(t, 101, CrossAuditRefute)
	auditor.errs[103] = errors.New("provider connection refused") // UNAVAILABLE

	snapshot := []IssueAuditLoopSubject{
		eligible(100),
		eligible(101),
		{IssueNumber: 102, Eligible: false, IneligibleReason: "not a closed dispatch leaf"}, // DARK
		eligible(103),
	}
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return append([]IssueAuditLoopSubject(nil), snapshot...), nil
	})

	cfg := IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath,
		BatchCap: 10, MaxAttempts: 3, BackoffBase: time.Minute,
		Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
	}

	// Tick 1: audits 100/101/103; 102 is DARK; 103 is unavailable and retained.
	rep, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 1: %v", err)
	}
	if rep.State != IssueAuditLoopAdvancing {
		t.Fatalf("tick 1 state = %s, want ADVANCING", rep.State)
	}
	if rep.Audited != 2 || rep.Passed != 1 || rep.Refuted != 1 || rep.Unavailable != 1 {
		t.Fatalf("tick 1 counts = %+v", rep)
	}
	if len(rep.DarkSubjects) != 1 || rep.DarkSubjects[0] != 102 {
		t.Fatalf("tick 1 dark subjects = %v, want [102]", rep.DarkSubjects)
	}
	if rep.LedgerRows != 2 {
		t.Fatalf("tick 1 ledger rows = %d, want 2", rep.LedgerRows)
	}
	if auditor.calls[102] != 0 {
		t.Fatalf("ineligible #102 was audited %d times, want 0", auditor.calls[102])
	}

	// Tick 2 (resume: reload cursor from disk). 100/101 already settled -> not
	// re-audited. 103 is still inside its retry backoff -> deferred. No progress,
	// but the deferral is transient, so WAIT.
	rep, err = RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 2: %v", err)
	}
	if rep.AlreadySettled != 2 {
		t.Fatalf("tick 2 already_settled = %d, want 2", rep.AlreadySettled)
	}
	if rep.Audited != 0 || rep.State != IssueAuditLoopWait {
		t.Fatalf("tick 2 = state %s audited %d, want WAIT/0", rep.State, rep.Audited)
	}
	if auditor.calls[100] != 1 || auditor.calls[101] != 1 {
		t.Fatalf("settled subjects re-audited: 100=%d 101=%d, want 1/1", auditor.calls[100], auditor.calls[101])
	}
	if rep.LedgerRows != 2 {
		t.Fatalf("tick 2 ledger rows = %d, want 2 (no new appends)", rep.LedgerRows)
	}

	// Advance past 103's backoff and let it succeed: it must be audited exactly
	// once more and reach the ledger — the unavailable row was never lost.
	clock.Advance(2 * time.Minute)
	auditor.errs[103] = nil
	auditor.receipts[103] = loopTestReceipt(t, 103, CrossAuditPass)
	rep, err = RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 3: %v", err)
	}
	if rep.State != IssueAuditLoopAdvancing || rep.Audited != 1 || rep.Passed != 1 {
		t.Fatalf("tick 3 = %+v, want ADVANCING/audited 1", rep)
	}
	if rep.LedgerRows != 3 {
		t.Fatalf("tick 3 ledger rows = %d, want 3", rep.LedgerRows)
	}
	if auditor.calls[103] != 2 {
		t.Fatalf("#103 audited %d times, want 2 (one unavailable + one terminal)", auditor.calls[103])
	}

	// The ledger verifies clean end-to-end with three unique audits.
	v, err := VerifyAuditReceiptLedger(ledgerPath)
	if err != nil {
		t.Fatalf("final ledger verify: %v", err)
	}
	if v.UniqueAudits != 3 {
		t.Fatalf("final unique audits = %d, want 3", v.UniqueAudits)
	}
}

// TestIssueAuditLoopCapsWorkPerTick proves the per-tick batch cap: only BatchCap
// subjects are audited, the rest are cap-deferred (and picked up next tick).
func TestIssueAuditLoopCapsWorkPerTick(t *testing.T) {
	ledgerPath, cursorPath := loopTestPaths(t)
	clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
	auditor := newCountingAuditor()
	for _, n := range []int{100, 101, 102} {
		auditor.receipts[n] = loopTestReceipt(t, n, CrossAuditPass)
	}
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return []IssueAuditLoopSubject{eligible(100), eligible(101), eligible(102)}, nil
	})
	cfg := IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 1,
		Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
	}
	rep, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick: %v", err)
	}
	if rep.Planned != 1 || rep.Audited != 1 || rep.CapDeferred != 2 {
		t.Fatalf("cap tick = planned %d audited %d cap_deferred %d, want 1/1/2", rep.Planned, rep.Audited, rep.CapDeferred)
	}
	total := auditor.calls[100] + auditor.calls[101] + auditor.calls[102]
	if total != 1 {
		t.Fatalf("audited %d subjects this tick, want exactly 1 (cap)", total)
	}
	// Second tick picks up the next capped subject.
	rep, err = RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick 2: %v", err)
	}
	if rep.AlreadySettled != 1 || rep.Audited != 1 {
		t.Fatalf("cap tick 2 = already_settled %d audited %d, want 1/1", rep.AlreadySettled, rep.Audited)
	}
}

// TestIssueAuditLoopRefusesCollidingLease proves a subject whose per-subject
// lease is already held is skipped (never audited) while its siblings proceed.
func TestIssueAuditLoopRefusesCollidingLease(t *testing.T) {
	ledgerPath, cursorPath := loopTestPaths(t)
	clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
	auditor := newCountingAuditor()
	auditor.receipts[100] = loopTestReceipt(t, 100, CrossAuditPass)
	auditor.receipts[101] = loopTestReceipt(t, 101, CrossAuditPass)
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return []IssueAuditLoopSubject{eligible(100), eligible(101)}, nil
	})
	cfg := IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 10,
		Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
		Lease: loopTestLease{busy: map[string]bool{"issue-100": true}},
	}
	rep, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("tick: %v", err)
	}
	if rep.LeaseConflicts != 1 {
		t.Fatalf("lease conflicts = %d, want 1", rep.LeaseConflicts)
	}
	if auditor.calls[100] != 0 {
		t.Fatalf("#100 audited under a held lease %d times, want 0", auditor.calls[100])
	}
	if rep.Audited != 1 || auditor.calls[101] != 1 {
		t.Fatalf("sibling #101 progress = audited %d call %d, want 1/1", rep.Audited, auditor.calls[101])
	}
}

// TestIssueAuditLoopExposesStates checks each typed decision the done condition
// names: DARK (no eligible), STALLED (all pending exhausted), WAIT (caught up).
func TestIssueAuditLoopExposesStates(t *testing.T) {
	t.Run("dark", func(t *testing.T) {
		ledgerPath, cursorPath := loopTestPaths(t)
		clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
		discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
			return []IssueAuditLoopSubject{{IssueNumber: 200, Eligible: false, IneligibleReason: "risk-filtered"}}, nil
		})
		rep, err := RunIssueAuditLoopTick(context.Background(), IssueAuditLoopConfig{
			LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 5,
			Now: clock.Now, Discoverer: discoverer, Auditor: newCountingAuditor(),
		})
		if err != nil {
			t.Fatalf("dark tick: %v", err)
		}
		if rep.State != IssueAuditLoopDark || rep.Eligible != 0 {
			t.Fatalf("dark tick = state %s eligible %d, want DARK/0", rep.State, rep.Eligible)
		}
	})

	t.Run("stalled", func(t *testing.T) {
		ledgerPath, cursorPath := loopTestPaths(t)
		clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
		auditor := newCountingAuditor()
		auditor.errs[300] = errors.New("provider down") // always UNAVAILABLE
		discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
			return []IssueAuditLoopSubject{eligible(300)}, nil
		})
		cfg := IssueAuditLoopConfig{
			LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 5,
			MaxAttempts: 3, BackoffBase: time.Millisecond,
			Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
		}
		// Three failed attempts exhaust the retry budget.
		var rep IssueAuditLoopReport
		var err error
		for i := 0; i < 3; i++ {
			rep, err = RunIssueAuditLoopTick(context.Background(), cfg)
			if err != nil {
				t.Fatalf("stall tick %d: %v", i, err)
			}
			clock.Advance(time.Second)
		}
		// The subject is now dead-lettered; the next observing tick reports STALLED.
		rep, err = RunIssueAuditLoopTick(context.Background(), cfg)
		if err != nil {
			t.Fatalf("stall observe tick: %v", err)
		}
		if rep.State != IssueAuditLoopStalled {
			t.Fatalf("stalled state = %s, want STALLED (rep=%+v)", rep.State, rep)
		}
		if rep.DeadLettered != 1 || len(rep.DeadLetterQueue) != 1 || rep.DeadLetterQueue[0] != 300 {
			t.Fatalf("dead-letter queue = %+v, want [300]", rep.DeadLetterQueue)
		}
	})

	t.Run("wait-caught-up", func(t *testing.T) {
		ledgerPath, cursorPath := loopTestPaths(t)
		clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
		auditor := newCountingAuditor()
		auditor.receipts[400] = loopTestReceipt(t, 400, CrossAuditPass)
		discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
			return []IssueAuditLoopSubject{eligible(400)}, nil
		})
		cfg := IssueAuditLoopConfig{
			LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 5,
			Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
		}
		if _, err := RunIssueAuditLoopTick(context.Background(), cfg); err != nil {
			t.Fatalf("settle tick: %v", err)
		}
		rep, err := RunIssueAuditLoopTick(context.Background(), cfg)
		if err != nil {
			t.Fatalf("caught-up tick: %v", err)
		}
		if rep.State != IssueAuditLoopWait || rep.Audited != 0 || rep.AlreadySettled != 1 {
			t.Fatalf("caught-up = state %s audited %d settled %d, want WAIT/0/1", rep.State, rep.Audited, rep.AlreadySettled)
		}
	})
}

// TestIssueAuditLoopDryRunPlansWithoutSideEffects proves --dry-run plans the
// batch but never calls the auditor, writes the ledger, or writes the cursor.
func TestIssueAuditLoopDryRunPlansWithoutSideEffects(t *testing.T) {
	ledgerPath, cursorPath := loopTestPaths(t)
	clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
	auditor := IssueAuditLoopAuditorFunc(func(context.Context, IssueAuditLoopSubject) (IssueAuditReceipt, error) {
		t.Fatalf("dry-run must not call the auditor")
		return IssueAuditReceipt{}, nil
	})
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return []IssueAuditLoopSubject{eligible(100), eligible(101)}, nil
	})
	rep, err := RunIssueAuditLoopTick(context.Background(), IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 5, DryRun: true,
		Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
	})
	if err != nil {
		t.Fatalf("dry-run tick: %v", err)
	}
	if !rep.DryRun || rep.Planned != 2 || len(rep.PlannedSubjects) != 2 {
		t.Fatalf("dry-run report = %+v, want DryRun/planned 2", rep)
	}
	if _, err := os.Stat(ledgerPath); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote the ledger (err=%v)", err)
	}
	if _, err := os.Stat(cursorPath); !os.IsNotExist(err) {
		t.Fatalf("dry-run wrote the cursor (err=%v)", err)
	}
}

// TestIssueAuditLoopRefusesOverScanCap proves the 500-issue planning ceiling is
// enforced against a misbehaving discoverer rather than silently truncated.
func TestIssueAuditLoopRefusesOverScanCap(t *testing.T) {
	ledgerPath, cursorPath := loopTestPaths(t)
	over := make([]IssueAuditLoopSubject, IssueAuditLoopScanCapMax+1)
	for i := range over {
		over[i] = eligible(1000 + i)
	}
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return over, nil
	})
	_, err := RunIssueAuditLoopTick(context.Background(), IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 5,
		Discoverer: discoverer, Auditor: newCountingAuditor(),
	})
	if err == nil {
		t.Fatalf("over-cap discovery accepted, want refusal")
	}
}

// TestIssueAuditLoopManualReplayIsAtMostOnce proves replaying a settled subject
// re-runs its audit but the deterministic receipt is an idempotent duplicate:
// the ledger does not grow.
func TestIssueAuditLoopManualReplayIsAtMostOnce(t *testing.T) {
	ledgerPath, cursorPath := loopTestPaths(t)
	clock := &loopTestClock{now: time.Unix(1_700_000_000, 0).UTC()}
	auditor := newCountingAuditor()
	auditor.receipts[500] = loopTestReceipt(t, 500, CrossAuditPass)
	discoverer := IssueAuditLoopDiscovererFunc(func(context.Context, int) ([]IssueAuditLoopSubject, error) {
		return []IssueAuditLoopSubject{eligible(500)}, nil
	})
	cfg := IssueAuditLoopConfig{
		LedgerPath: ledgerPath, CursorPath: cursorPath, BatchCap: 5,
		Now: clock.Now, Discoverer: discoverer, Auditor: auditor,
	}
	if _, err := RunIssueAuditLoopTick(context.Background(), cfg); err != nil {
		t.Fatalf("settle tick: %v", err)
	}
	cfg.ReplayIssues = []int{500}
	rep, err := RunIssueAuditLoopTick(context.Background(), cfg)
	if err != nil {
		t.Fatalf("replay tick: %v", err)
	}
	if auditor.calls[500] != 2 {
		t.Fatalf("#500 audited %d times, want 2 (settle + replay)", auditor.calls[500])
	}
	if rep.LedgerRows != 1 {
		t.Fatalf("replay grew the ledger to %d rows, want 1 (idempotent duplicate)", rep.LedgerRows)
	}
	v, err := VerifyAuditReceiptLedger(ledgerPath)
	if err != nil || v.UniqueAudits != 1 {
		t.Fatalf("replay ledger = %d unique audits err=%v, want 1", v.UniqueAudits, err)
	}
}
