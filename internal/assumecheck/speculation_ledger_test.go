package assumecheck

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestSpeculationLedger_RecordAndAssumecheckIntegration tests recording speculative
// invocations and verifying that they integrate as load-bearing assumptions in assumecheck.
func TestSpeculationLedger_RecordAndAssumecheckIntegration(t *testing.T) {
	ledger := NewSpeculationLedger()

	entry := SpeculationEntry{
		ID:                  "spec-1",
		RunID:               "run-100",
		TraceID:             "trace-abc",
		ToolName:            "search_kb",
		CallArguments:       `{"query":"payment refund policy"}`,
		SpeculativeProposer: "streaming_fsm",
		ReadOnlyProof:       "vdso:pure+meta:readOnlyHint",
	}

	if err := ledger.RecordSpeculation(entry); err != nil {
		t.Fatalf("RecordSpeculation failed: %v", err)
	}

	// Verify retrieved entry
	got, ok := ledger.Get("spec-1")
	if !ok {
		t.Fatalf("expected spec-1 to exist in ledger")
	}
	if got.Outcome != OutcomePending {
		t.Errorf("expected outcome %s, got %s", OutcomePending, got.Outcome)
	}
	if got.Timestamp.IsZero() {
		t.Errorf("expected non-zero timestamp")
	}
	if got.ToolName != "search_kb" {
		t.Errorf("expected tool_name search_kb, got %s", got.ToolName)
	}

	// Reject duplicate ID
	if err := ledger.RecordSpeculation(entry); !errors.Is(err, ErrSpeculationDuplicate) {
		t.Fatalf("expected ErrSpeculationDuplicate, got %v", err)
	}

	// Reject empty ID
	if err := ledger.RecordSpeculation(SpeculationEntry{ToolName: "test"}); err == nil {
		t.Fatalf("expected error on empty ID, got nil")
	}

	// Assumecheck projection tests
	assumption := got.AsAssumption()
	if assumption.ID != "speculation:spec-1" {
		t.Errorf("expected assumption ID speculation:spec-1, got %s", assumption.ID)
	}
	if assumption.Owner != "streaming_fsm" {
		t.Errorf("expected owner streaming_fsm, got %s", assumption.Owner)
	}
	if assumption.Level != LevelSubsystem {
		t.Errorf("expected LevelSubsystem, got %s", assumption.Level)
	}
	if assumption.WitnessKind != WitnessLedgerRead {
		t.Errorf("expected WitnessLedgerRead, got %s", assumption.WitnessKind)
	}
	if assumption.ConfidenceClass != "speculative" {
		t.Errorf("expected ConfidenceClass speculative, got %s", assumption.ConfidenceClass)
	}
	if assumption.WitnessStatus != WitnessWired {
		t.Errorf("expected WitnessWired, got %s", assumption.WitnessStatus)
	}

	// In pending state, verdict should be OutcomeUnverifiable
	verdict := got.AsVerdict()
	if verdict.Outcome != OutcomeUnverifiable {
		t.Errorf("expected pending verdict OutcomeUnverifiable, got %s", verdict.Outcome)
	}
}

// TestSpeculationLedger_WitnessCommit tests witnessing a clean speculation commit
// and tracking saved duration and net speedup ratio.
func TestSpeculationLedger_WitnessCommit(t *testing.T) {
	ledger := NewSpeculationLedger()

	entry := SpeculationEntry{
		ID:                  "spec-commit",
		ToolName:            "read_profile",
		CallArguments:       `{"user_id":42}`,
		SpeculativeProposer: "lookahead",
		ReadOnlyProof:       "default-deny-effects-cleared",
	}

	if err := ledger.RecordSpeculation(entry); err != nil {
		t.Fatalf("RecordSpeculation failed: %v", err)
	}

	const saved = 150 * time.Millisecond
	const callDigest = "sha256:abcd1234"
	if err := ledger.WitnessCommit("spec-commit", callDigest, saved); err != nil {
		t.Fatalf("WitnessCommit failed: %v", err)
	}

	got, ok := ledger.Get("spec-commit")
	if !ok {
		t.Fatalf("entry not found")
	}
	if got.Outcome != OutcomeCommitted {
		t.Errorf("expected outcome COMMITTED, got %s", got.Outcome)
	}
	if got.ActualCallDigest != callDigest {
		t.Errorf("expected call digest %s, got %s", callDigest, got.ActualCallDigest)
	}
	if got.CostAccounting.SavedDuration != saved {
		t.Errorf("expected saved duration %v, got %v", saved, got.CostAccounting.SavedDuration)
	}
	if got.CostAccounting.WastedDuration != 0 {
		t.Errorf("expected wasted duration 0, got %v", got.CostAccounting.WastedDuration)
	}
	if got.CostAccounting.NetSpeedupRatio != 2.0 {
		t.Errorf("expected NetSpeedupRatio 2.0, got %f", got.CostAccounting.NetSpeedupRatio)
	}

	// An accepted/committed speculation assumption HOLDS
	verdict := got.AsVerdict()
	if verdict.Outcome != OutcomeHolds {
		t.Errorf("expected verdict OutcomeHolds, got %s", verdict.Outcome)
	}

	// Unknown ID returns not found
	if err := ledger.WitnessCommit("non-existent", "", saved); !errors.Is(err, ErrSpeculationNotFound) {
		t.Errorf("expected ErrSpeculationNotFound, got %v", err)
	}
}

// TestSpeculationLedger_WitnessSquash tests witnessing a squashed misprediction
// carrying SPECULATION_MISPREDICT, wasted tokens, and wasted duration.
func TestSpeculationLedger_WitnessSquash(t *testing.T) {
	ledger := NewSpeculationLedger()

	entry := SpeculationEntry{
		ID:                  "spec-squash",
		ToolName:            "fetch_doc",
		CallArguments:       `{"doc_id":"doc-A"}`,
		SpeculativeProposer: "heuristic",
		ReadOnlyProof:       "meta:readOnlyHint+idempotentHint",
	}

	if err := ledger.RecordSpeculation(entry); err != nil {
		t.Fatalf("RecordSpeculation failed: %v", err)
	}

	const wastedDur = 60 * time.Millisecond
	const wastedTok = int64(240)
	const mispredictReason = "authoritative call requested doc-B; arguments diverged"

	if err := ledger.WitnessSquash("spec-squash", mispredictReason, wastedDur, wastedTok); err != nil {
		t.Fatalf("WitnessSquash failed: %v", err)
	}

	got, ok := ledger.Get("spec-squash")
	if !ok {
		t.Fatalf("entry not found")
	}
	if got.Outcome != OutcomeSquashed {
		t.Errorf("expected outcome SQUASHED, got %s", got.Outcome)
	}
	if got.FailureToken != TokenSpeculationMispredict {
		t.Errorf("expected failure token %s, got %s", TokenSpeculationMispredict, got.FailureToken)
	}
	if got.Reason != mispredictReason {
		t.Errorf("expected reason %q, got %q", mispredictReason, got.Reason)
	}
	if got.CostAccounting.WastedDuration != wastedDur {
		t.Errorf("expected wasted duration %v, got %v", wastedDur, got.CostAccounting.WastedDuration)
	}
	if got.CostAccounting.WastedTokens != wastedTok {
		t.Errorf("expected wasted tokens %d, got %d", wastedTok, got.CostAccounting.WastedTokens)
	}
	if got.CostAccounting.SavedDuration != 0 {
		t.Errorf("expected saved duration 0, got %v", got.CostAccounting.SavedDuration)
	}
	if got.CostAccounting.NetSpeedupRatio != 0.0 {
		t.Errorf("expected NetSpeedupRatio 0.0, got %f", got.CostAccounting.NetSpeedupRatio)
	}

	// Assumption projection should cite SPECULATION_MISPREDICT
	assumption := got.AsAssumption()
	if assumption.RefusalReason != TokenSpeculationMispredict {
		t.Errorf("expected RefusalReason %s, got %s", TokenSpeculationMispredict, assumption.RefusalReason)
	}

	// Verdict should be VIOLATED
	verdict := got.AsVerdict()
	if verdict.Outcome != OutcomeViolated {
		t.Errorf("expected verdict OutcomeViolated, got %s", verdict.Outcome)
	}
	if verdict.Reason != mispredictReason {
		t.Errorf("expected verdict reason %q, got %q", mispredictReason, verdict.Reason)
	}

	// Unknown ID returns not found
	if err := ledger.WitnessSquash("non-existent", "", wastedDur, wastedTok); !errors.Is(err, ErrSpeculationNotFound) {
		t.Errorf("expected ErrSpeculationNotFound, got %v", err)
	}
}

// TestSpeculationLedger_WitnessLeak tests witnessing an effect leak which immediately
// triggers a fail-closed audit refusal with SPECULATIVE_EFFECT_LEAK.
func TestSpeculationLedger_WitnessLeak(t *testing.T) {
	ledger := NewSpeculationLedger()

	entry := SpeculationEntry{
		ID:                  "spec-leak",
		ToolName:            "write_log",
		CallArguments:       `{"msg":"audit trace"}`,
		SpeculativeProposer: "streaming_fsm",
		ReadOnlyProof:       "unstamped",
	}

	if err := ledger.RecordSpeculation(entry); err != nil {
		t.Fatalf("RecordSpeculation failed: %v", err)
	}

	const leakEvidence = "mutation detected in working tree file internal/foo.go prior to promote"
	err := ledger.WitnessLeak("spec-leak", leakEvidence)
	if err == nil {
		t.Fatalf("expected WitnessLeak to return an error, got nil")
	}

	// Must be an AuditRefusalError wrapping ErrSpeculativeEffectLeak
	if !errors.Is(err, ErrSpeculativeEffectLeak) {
		t.Errorf("expected errors.Is(err, ErrSpeculativeEffectLeak) to be true, got %v", err)
	}

	var auditErr *AuditRefusalError
	if !errors.As(err, &auditErr) {
		t.Fatalf("expected error of type *AuditRefusalError, got %T", err)
	}
	if auditErr.Token != TokenSpeculativeEffectLeak {
		t.Errorf("expected token %s, got %s", TokenSpeculativeEffectLeak, auditErr.Token)
	}
	if auditErr.EntryID != "spec-leak" {
		t.Errorf("expected entry ID spec-leak, got %s", auditErr.EntryID)
	}
	if auditErr.Evidence != leakEvidence {
		t.Errorf("expected evidence %q, got %q", leakEvidence, auditErr.Evidence)
	}

	// Check entry updated in ledger
	got, ok := ledger.Get("spec-leak")
	if !ok {
		t.Fatalf("entry not found")
	}
	if got.Outcome != OutcomeLeaked {
		t.Errorf("expected outcome LEAKED, got %s", got.Outcome)
	}
	if got.FailureToken != TokenSpeculativeEffectLeak {
		t.Errorf("expected failure token %s, got %s", TokenSpeculativeEffectLeak, got.FailureToken)
	}

	// Assumption refusal cites SPECULATIVE_EFFECT_LEAK
	assumption := got.AsAssumption()
	if assumption.RefusalReason != TokenSpeculativeEffectLeak {
		t.Errorf("expected RefusalReason %s, got %s", TokenSpeculativeEffectLeak, assumption.RefusalReason)
	}

	// Verdict is VIOLATED
	verdict := got.AsVerdict()
	if verdict.Outcome != OutcomeViolated {
		t.Errorf("expected verdict OutcomeViolated, got %s", verdict.Outcome)
	}

	// Unknown ID returns not found
	if err := ledger.WitnessLeak("non-existent", leakEvidence); !errors.Is(err, ErrSpeculationNotFound) {
		t.Errorf("expected ErrSpeculationNotFound, got %v", err)
	}
}

// TestSpeculationLedger_SummaryAndNetSpeedup tests aggregate summary metrics,
// acceptance rate, net speedup ratio calculations, and JSON export.
func TestSpeculationLedger_SummaryAndNetSpeedup(t *testing.T) {
	ledger := NewSpeculationLedger()

	// 1. Empty ledger baseline
	emptySummary := ledger.Summary()
	if emptySummary.TotalSpeculations != 0 || emptySummary.NetSpeedup != 1.0 || emptySummary.AcceptanceRate != 0.0 {
		t.Errorf("empty ledger summary unexpected: %+v", emptySummary)
	}

	// 2. Add entries
	// spec-1: Committed, saved 200ms
	_ = ledger.RecordSpeculation(SpeculationEntry{ID: "s1", ToolName: "t1"})
	_ = ledger.WitnessCommit("s1", "digest1", 200*time.Millisecond)

	// spec-2: Committed, saved 100ms
	_ = ledger.RecordSpeculation(SpeculationEntry{ID: "s2", ToolName: "t2"})
	_ = ledger.WitnessCommit("s2", "digest2", 100*time.Millisecond)

	// spec-3: Squashed, wasted 100ms, 50 tokens
	_ = ledger.RecordSpeculation(SpeculationEntry{ID: "s3", ToolName: "t3"})
	_ = ledger.WitnessSquash("s3", "mispredict", 100*time.Millisecond, 50)

	// spec-4: Leaked
	_ = ledger.RecordSpeculation(SpeculationEntry{ID: "s4", ToolName: "t4"})
	_ = ledger.WitnessLeak("s4", "state leak")

	// spec-5: Pending
	_ = ledger.RecordSpeculation(SpeculationEntry{ID: "s5", ToolName: "t5"})

	summary := ledger.Summary()
	if summary.TotalSpeculations != 5 {
		t.Errorf("expected TotalSpeculations 5, got %d", summary.TotalSpeculations)
	}
	if summary.CommittedCount != 2 {
		t.Errorf("expected CommittedCount 2, got %d", summary.CommittedCount)
	}
	if summary.SquashedCount != 1 {
		t.Errorf("expected SquashedCount 1, got %d", summary.SquashedCount)
	}
	if summary.LeakedCount != 1 {
		t.Errorf("expected LeakedCount 1, got %d", summary.LeakedCount)
	}
	if summary.PendingCount != 1 {
		t.Errorf("expected PendingCount 1, got %d", summary.PendingCount)
	}
	if summary.TotalSavedMs != 300.0 {
		t.Errorf("expected TotalSavedMs 300.0, got %f", summary.TotalSavedMs)
	}
	if summary.TotalWastedMs != 100.0 {
		t.Errorf("expected TotalWastedMs 100.0, got %f", summary.TotalWastedMs)
	}
	if summary.TotalWastedTokens != 50 {
		t.Errorf("expected TotalWastedTokens 50, got %d", summary.TotalWastedTokens)
	}
	// AcceptanceRate = 2/5 = 0.4
	if summary.AcceptanceRate != 0.4 {
		t.Errorf("expected AcceptanceRate 0.4, got %f", summary.AcceptanceRate)
	}
	// NetSpeedup = 1.0 + (300 - 100) / (300 + 100) = 1.0 + 200/400 = 1.5
	if summary.NetSpeedup != 1.5 {
		t.Errorf("expected NetSpeedup 1.5, got %f", summary.NetSpeedup)
	}

	// 3. Test ExportJSON
	jsonBytes, err := ledger.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}

	var exported SpeculationExport
	if err := json.Unmarshal(jsonBytes, &exported); err != nil {
		t.Fatalf("unmarshal exported JSON failed: %v", err)
	}
	if len(exported.Entries) != 5 {
		t.Errorf("expected 5 exported entries, got %d", len(exported.Entries))
	}
	if exported.Summary.NetSpeedup != 1.5 {
		t.Errorf("expected exported summary NetSpeedup 1.5, got %f", exported.Summary.NetSpeedup)
	}

	// 4. Test break-even speedup
	beLedger := NewSpeculationLedger()
	_ = beLedger.RecordSpeculation(SpeculationEntry{ID: "b1", ToolName: "t1"})
	_ = beLedger.WitnessCommit("b1", "", 100*time.Millisecond)
	_ = beLedger.RecordSpeculation(SpeculationEntry{ID: "b2", ToolName: "t2"})
	_ = beLedger.WitnessSquash("b2", "miss", 100*time.Millisecond, 10)
	beSummary := beLedger.Summary()
	if beSummary.NetSpeedup != 1.0 {
		t.Errorf("expected break-even NetSpeedup 1.0, got %f", beSummary.NetSpeedup)
	}
}

// TestSpeculationLedger_ConcurrentRace tests concurrent reads, writes, commits,
// squashes, and leaks under -race.
func TestSpeculationLedger_ConcurrentRace(t *testing.T) {
	ledger := NewSpeculationLedger()
	const workers = 10
	const opsPerWorker = 50

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		workerID := w
		go func() {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				id := fmt.Sprintf("spec-%d-%d", workerID, i)
				_ = ledger.RecordSpeculation(SpeculationEntry{
					ID:                  id,
					ToolName:            "search_code",
					CallArguments:       `{"pattern":"func main"}`,
					SpeculativeProposer: "streaming_fsm",
					ReadOnlyProof:       "vdso:pure",
				})

				switch i % 4 {
				case 0:
					_ = ledger.WitnessCommit(id, "digest", 20*time.Millisecond)
				case 1:
					_ = ledger.WitnessSquash(id, "mispredict", 10*time.Millisecond, 25)
				case 2:
					_ = ledger.WitnessLeak(id, "file write leak")
				default:
					// remain pending
				}

				_ = ledger.Summary()
				_, _ = ledger.Get(id)
				_ = ledger.Len()
			}
		}()
	}

	wg.Wait()

	if ledger.Len() != workers*opsPerWorker {
		t.Fatalf("expected ledger length %d, got %d", workers*opsPerWorker, ledger.Len())
	}

	summary := ledger.Summary()
	expectedTotal := workers * opsPerWorker
	if summary.TotalSpeculations != expectedTotal {
		t.Fatalf("expected total %d, got %d", expectedTotal, summary.TotalSpeculations)
	}

	// ExportJSON concurrency sanity
	bytes, err := ledger.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON failed: %v", err)
	}
	if len(bytes) == 0 {
		t.Fatalf("expected non-empty JSON export")
	}
}
