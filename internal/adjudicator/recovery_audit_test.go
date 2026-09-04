package adjudicator

import (
	"bytes"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRecoveryAudit_BasicLifecycle(t *testing.T) {
	ledger := NewRecoveryAuditLedger()

	ledger.RecordRefusal("sess-1", 1, "POLICY_BLOCK", "call search_kb")
	if ledger.Len() != 1 {
		t.Fatalf("expected 1 entry, got %d", ledger.Len())
	}
	if ledger.TotalResolved() != 0 {
		t.Fatalf("expected 0 resolved entries, got %d", ledger.TotalResolved())
	}
	if rate := ledger.RecoveryRate(); rate != 0.0 {
		t.Fatalf("expected 0.0 recovery rate, got %f", rate)
	}
	if rate := ledger.AttritionRate(); rate != 0.0 {
		t.Fatalf("expected 0.0 attrition rate, got %f", rate)
	}

	entries := ledger.Entries()
	if entries[0].SessionID != "sess-1" || entries[0].Turn != 1 {
		t.Fatalf("unexpected entry header: %+v", entries[0])
	}
	if entries[0].RefusalReason != "POLICY_BLOCK" || entries[0].SuggestedNextAction != "call search_kb" {
		t.Fatalf("unexpected entry refusal info: %+v", entries[0])
	}
	if entries[0].Recovered || entries[0].Outcome != "" {
		t.Fatalf("expected pending outcome, got %+v", entries[0])
	}

	ledger.RecordOutcome("sess-1", 1, "search_kb", OutcomeRecovered)
	if ledger.TotalResolved() != 1 {
		t.Fatalf("expected 1 resolved entry, got %d", ledger.TotalResolved())
	}
	if rate := ledger.RecoveryRate(); rate != 1.0 {
		t.Fatalf("expected 1.0 recovery rate, got %f", rate)
	}
	if rate := ledger.AttritionRate(); rate != 0.0 {
		t.Fatalf("expected 0.0 attrition rate, got %f", rate)
	}

	updated := ledger.Entries()
	if !updated[0].Recovered || updated[0].Outcome != OutcomeRecovered || updated[0].SubsequentAction != "search_kb" {
		t.Fatalf("unexpected updated entry: %+v", updated[0])
	}
}

func TestRecoveryAudit_RatesAndAttrition(t *testing.T) {
	ledger := NewRecoveryAuditLedger()

	// 1 recovered
	ledger.RecordRefusal("sess-rec", 1, "POLICY_BLOCK", "use read")
	ledger.RecordOutcome("sess-rec", 1, "read", OutcomeRecovered)

	// 1 looped
	ledger.RecordRefusal("sess-loop", 2, "OFF_TRUNK", "merge main")
	ledger.RecordOutcome("sess-loop", 2, "git checkout branch", OutcomeRetried)

	// 1 abandoned
	ledger.RecordRefusal("sess-aban", 3, "OUT_OF_TREE_WRITE", "use local scratch")
	ledger.RecordOutcome("sess-aban", 3, "exit", OutcomeAbandoned)

	// 1 pending (should not affect rates)
	ledger.RecordRefusal("sess-pending", 4, "LANE_DRAINED", "wait")

	if ledger.Len() != 4 {
		t.Fatalf("expected 4 total entries, got %d", ledger.Len())
	}
	if ledger.TotalResolved() != 3 {
		t.Fatalf("expected 3 resolved entries, got %d", ledger.TotalResolved())
	}

	wantRec := 1.0 / 3.0
	wantAtt := 2.0 / 3.0

	if math.Abs(ledger.RecoveryRate()-wantRec) > 1e-9 {
		t.Fatalf("recovery rate got %f, want %f", ledger.RecoveryRate(), wantRec)
	}
	if math.Abs(ledger.AttritionRate()-wantAtt) > 1e-9 {
		t.Fatalf("attrition rate got %f, want %f", ledger.AttritionRate(), wantAtt)
	}
	if math.Abs((ledger.RecoveryRate()+ledger.AttritionRate())-1.0) > 1e-9 {
		t.Fatalf("sum of rates got %f, want 1.0", ledger.RecoveryRate()+ledger.AttritionRate())
	}
}

func TestRecoveryAudit_EmptyAndPendingLedgers(t *testing.T) {
	ledger := NewRecoveryAuditLedger()
	if ledger.RecoveryRate() != 0.0 || ledger.AttritionRate() != 0.0 {
		t.Fatalf("empty ledger must return 0.0 rates")
	}

	ledger.RecordRefusal("sess-1", 1, "R1", "A1")
	ledger.RecordRefusal("sess-2", 2, "R2", "A2")
	if ledger.RecoveryRate() != 0.0 || ledger.AttritionRate() != 0.0 {
		t.Fatalf("pending-only ledger must return 0.0 rates")
	}
}

func TestRecoveryAudit_MultiTurnSession(t *testing.T) {
	ledger := NewRecoveryAuditLedger()

	ledger.RecordRefusal("session-alpha", 1, "REASON_A", "action-1")
	ledger.RecordRefusal("session-alpha", 2, "REASON_B", "action-2")

	ledger.RecordOutcome("session-alpha", 1, "wrong_retry", OutcomeRetried)
	ledger.RecordOutcome("session-alpha", 2, "correct_action", OutcomeRecovered)

	if ledger.Len() != 2 || ledger.TotalResolved() != 2 {
		t.Fatalf("expected 2 resolved entries, got %d entries, %d resolved", ledger.Len(), ledger.TotalResolved())
	}

	entries := ledger.Entries()
	if entries[0].Turn != 1 || entries[0].Outcome != OutcomeRetried || entries[0].Recovered {
		t.Fatalf("unexpected turn 1 entry: %+v", entries[0])
	}
	if entries[1].Turn != 2 || entries[1].Outcome != OutcomeRecovered || !entries[1].Recovered {
		t.Fatalf("unexpected turn 2 entry: %+v", entries[1])
	}

	if ledger.RecoveryRate() != 0.5 || ledger.AttritionRate() != 0.5 {
		t.Fatalf("expected 0.5 / 0.5 rates, got recovery=%f, attrition=%f", ledger.RecoveryRate(), ledger.AttritionRate())
	}
}

func TestRecoveryAudit_JSONLRoundTrip(t *testing.T) {
	original := NewRecoveryAuditLedger()

	fixedTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	original.RecordRefusal("s-1", 1, "POLICY_BLOCK", "try read")
	original.RecordOutcome("s-1", 1, "read", OutcomeRecovered)
	original.entries[0].Timestamp = fixedTime

	original.RecordRefusal("s-2", 2, "OFF_TRUNK", "pull")
	original.RecordOutcome("s-2", 2, "branch", OutcomeRetried)
	original.entries[1].Timestamp = fixedTime

	var buf bytes.Buffer
	if err := original.WriteJSONL(&buf); err != nil {
		t.Fatalf("WriteJSONL failed: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 JSON lines, got %d: %q", len(lines), buf.String())
	}

	restored := NewRecoveryAuditLedger()
	if err := restored.ReadJSONL(&buf); err != nil {
		t.Fatalf("ReadJSONL failed: %v", err)
	}

	if restored.Len() != original.Len() {
		t.Fatalf("restored len %d, want %d", restored.Len(), original.Len())
	}
	if restored.RecoveryRate() != original.RecoveryRate() {
		t.Fatalf("restored recovery rate %f, want %f", restored.RecoveryRate(), original.RecoveryRate())
	}
	if restored.AttritionRate() != original.AttritionRate() {
		t.Fatalf("restored attrition rate %f, want %f", restored.AttritionRate(), original.AttritionRate())
	}

	origEntries := original.Entries()
	restEntries := restored.Entries()
	for i := range origEntries {
		if origEntries[i].SessionID != restEntries[i].SessionID ||
			origEntries[i].Turn != restEntries[i].Turn ||
			origEntries[i].RefusalReason != restEntries[i].RefusalReason ||
			origEntries[i].SuggestedNextAction != restEntries[i].SuggestedNextAction ||
			origEntries[i].SubsequentAction != restEntries[i].SubsequentAction ||
			origEntries[i].Outcome != restEntries[i].Outcome ||
			origEntries[i].Recovered != restEntries[i].Recovered ||
			!origEntries[i].Timestamp.Equal(restEntries[i].Timestamp) {
			t.Fatalf("entry %d mismatch:\norig: %+v\nrest: %+v", i, origEntries[i], restEntries[i])
		}
	}
}

func TestRecoveryAudit_ReadJSONLMalformedAndBlankLines(t *testing.T) {
	ledger := NewRecoveryAuditLedger()

	inputWithBlanks := `
{"session_id":"s-1","turn":1,"refusal_reason":"R1","suggested_next_action":"A1","subsequent_action":"S1","outcome":"recovered","recovered":true,"timestamp":"2026-09-03T12:00:00Z"}

{"session_id":"s-2","turn":2,"refusal_reason":"R2","suggested_next_action":"A2","subsequent_action":"S2","outcome":"abandoned","recovered":false,"timestamp":"2026-09-03T12:01:00Z"}
`
	if err := ledger.ReadJSONL(strings.NewReader(inputWithBlanks)); err != nil {
		t.Fatalf("expected successful read with blank lines, got: %v", err)
	}
	if ledger.Len() != 2 {
		t.Fatalf("expected 2 entries parsed, got %d", ledger.Len())
	}

	badInput := `{"session_id":"s-3", unclosed json`
	if err := ledger.ReadJSONL(strings.NewReader(badInput)); err == nil {
		t.Fatalf("expected error on malformed JSON line, got nil")
	}
}

func TestRecoveryAudit_DirectOutcomeWithoutRefusal(t *testing.T) {
	ledger := NewRecoveryAuditLedger()

	ledger.RecordOutcome("session-orphan", 10, "fallback_command", OutcomeRecovered)
	if ledger.Len() != 1 {
		t.Fatalf("expected 1 entry, got %d", ledger.Len())
	}
	entries := ledger.Entries()
	if entries[0].SessionID != "session-orphan" || entries[0].Turn != 10 || !entries[0].Recovered || entries[0].Outcome != OutcomeRecovered {
		t.Fatalf("unexpected entry: %+v", entries[0])
	}
	if ledger.RecoveryRate() != 1.0 {
		t.Fatalf("expected 1.0 recovery rate, got %f", ledger.RecoveryRate())
	}
}

func TestRecoveryAudit_ConcurrentAccess(t *testing.T) {
	ledger := NewRecoveryAuditLedger()
	var wg sync.WaitGroup

	const goroutines = 20
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			sessID := "concurrent-sess"
			ledger.RecordRefusal(sessID, idx, "CONCURRENT_REASON", "suggested")
			if idx%2 == 0 {
				ledger.RecordOutcome(sessID, idx, "act", OutcomeRecovered)
			} else {
				ledger.RecordOutcome(sessID, idx, "act", OutcomeRetried)
			}
			_ = ledger.RecoveryRate()
			_ = ledger.AttritionRate()
			_ = ledger.TotalResolved()
			_ = ledger.Entries()
		}(i)
	}

	wg.Wait()
	if ledger.Len() != goroutines {
		t.Fatalf("expected %d entries, got %d", goroutines, ledger.Len())
	}
	if ledger.TotalResolved() != goroutines {
		t.Fatalf("expected %d resolved entries, got %d", goroutines, ledger.TotalResolved())
	}
	if ledger.RecoveryRate() != 0.5 || ledger.AttritionRate() != 0.5 {
		t.Fatalf("expected 0.5 rates, got recovery=%f, attrition=%f", ledger.RecoveryRate(), ledger.AttritionRate())
	}
}
