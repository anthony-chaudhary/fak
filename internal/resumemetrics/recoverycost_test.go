package resumemetrics

import "testing"

// TestRecoveryCostRecorders is the #4146 expvar acceptance in miniature: each OBSERVED
// recovery-cost recorder moves its own counter off the known-zero floor, Active() flips, and
// the typed Snapshot surfaces the sums for the /debug/vars fold.
func TestRecoveryCostRecorders(t *testing.T) {
	Reset()
	if Active() {
		t.Fatal("Active() true immediately after Reset")
	}

	RecordRecoverySpend(1500, 900)
	RecordRecoverySpend(500, 100) // accumulates, not last-writer-wins
	RecordRecoverySpend(-9, -9)   // bad reading ignored, never lowers the total
	RecordRecoveryCostHold()
	RecordRecoveryCostHold()

	if !Active() {
		t.Fatal("Active() false after recording recovery spend")
	}
	got := Read()
	if got.RecoveryTokens != 2000 {
		t.Errorf("RecoveryTokens = %d, want 2000", got.RecoveryTokens)
	}
	if got.RecoveryCostMicroUSD != 1000 {
		t.Errorf("RecoveryCostMicroUSD = %d, want 1000", got.RecoveryCostMicroUSD)
	}
	if got.RecoveryCostHolds != 2 {
		t.Errorf("RecoveryCostHolds = %d, want 2", got.RecoveryCostHolds)
	}

	Reset()
	if r := Read(); r.RecoveryTokens != 0 || r.RecoveryCostMicroUSD != 0 || r.RecoveryCostHolds != 0 {
		t.Fatalf("Reset did not clear recovery counters: %+v", r)
	}
}
