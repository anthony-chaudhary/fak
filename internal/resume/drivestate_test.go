package resume

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestDriveCarryFoldPreservesLegacyHold(t *testing.T) {
	legacyJSON := []byte(`{"ts":"2026-07-10T00:00:00Z","session":"sid-a","state":"paused","reason":"operator"}`)
	var legacy DriveStateRow
	if err := json.Unmarshal(legacyJSON, &legacy); err != nil {
		t.Fatalf("decode legacy drive-state row: %v", err)
	}

	legacyStates := FoldDriveStates([]DriveStateRow{legacy})
	if got := legacyStates["sid-a"]; !got.HeldByOperator() {
		t.Fatalf("legacy hold = %+v, want HeldByOperator", got)
	}
	if got := FoldDriveCarry([]DriveStateRow{legacy}); len(got) != 0 {
		t.Fatalf("legacy carry fold = %+v, want empty", got)
	}

	wantCarry := DriveCarry{
		TurnsLeft:           7,
		TokensLeft:          1200,
		ContextTokensLeft:   800,
		SpendMicroCentsLeft: 4500,
		TimeLeftNanos:       30_000_000_000,
		Generation:          2,
		ObjectivePinID:      "pin-1",
		ObjectiveText:       "finish the top issue",
		ObjectiveDigest:     "digest-1",
	}
	rows := []DriveStateRow{
		legacy,
		{TS: "2026-07-10T00:01:00Z", Session: "sid-a", TurnsLeft: ptr64(7), TokensLeft: ptr64(1200), ContextTokensLeft: ptr64(800), SpendMicroCentsLeft: ptr64(4500), TimeLeftNanos: ptr64(30_000_000_000), Generation: ptrInt(2), ObjectivePinID: "pin-1", ObjectiveText: "finish the top issue", ObjectiveDigest: "digest-1"},
	}
	states := FoldDriveStates(rows)
	if got := states["sid-a"]; !got.HeldByOperator() {
		t.Fatalf("hold after carry-only row = %+v, want HeldByOperator", got)
	}
	if got := FoldDriveCarry(rows)["sid-a"]; !reflect.DeepEqual(got, wantCarry) {
		t.Fatalf("carry = %+v, want %+v", got, wantCarry)
	}
}

func TestDriveCarryFoldLastExplicitCarryWins(t *testing.T) {
	first := DriveCarry{TurnsLeft: 8, TokensLeft: 2000}
	latest := DriveCarry{TurnsLeft: 6, TokensLeft: 1500}
	rows := []DriveStateRow{
		{Session: "sid-a", TurnsLeft: ptr64(first.TurnsLeft), TokensLeft: ptr64(first.TokensLeft)},
		{Session: "sid-a", State: string(DrivePaused)},
		{Session: "sid-a", TurnsLeft: ptr64(latest.TurnsLeft), TokensLeft: ptr64(latest.TokensLeft)},
		{Session: " ", TurnsLeft: ptr64(latest.TurnsLeft)},
	}
	got := FoldDriveCarry(rows)
	if len(got) != 1 || !reflect.DeepEqual(got["sid-a"], latest) {
		t.Fatalf("carry fold = %+v, want latest carry for sid-a only", got)
	}
}

// TestDriveCarryMonotoneClampsStaleRegrant proves the through-line the cluster exists to
// fix: a stale later snapshot with HIGHER remaining budget cannot refill a spent-down
// session. Raw FoldDriveCarry takes the last-written (higher) value; ReconcileDriveCarry
// clamps it to the low snapshot.
func TestDriveCarryMonotoneClampsStaleRegrant(t *testing.T) {
	rows := []DriveStateRow{
		{Session: "sid-a", TurnsLeft: ptr64(2), TokensLeft: ptr64(200), ContextTokensLeft: ptr64(150), SpendMicroCentsLeft: ptr64(50), TimeLeftNanos: ptr64(1_000)},
		{Session: "sid-a", State: string(DrivePaused)}, // interleaved hold row: no carry, must not disturb the clamp
		{Session: "sid-a", TurnsLeft: ptr64(10), TokensLeft: ptr64(1000), ContextTokensLeft: ptr64(800), SpendMicroCentsLeft: ptr64(500), TimeLeftNanos: ptr64(9_000)},
	}
	if raw := FoldDriveCarry(rows)["sid-a"]; raw.TurnsLeft != 10 || raw.TokensLeft != 1000 {
		t.Fatalf("raw FoldDriveCarry = %+v, want the last-written higher snapshot (the re-grant we forbid)", raw)
	}
	got := ReconcileDriveCarry(rows)["sid-a"]
	want := DriveCarry{TurnsLeft: 2, TokensLeft: 200, ContextTokensLeft: 150, SpendMicroCentsLeft: 50, TimeLeftNanos: 1_000}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reconciled carry = %+v, want clamped to the low snapshot %+v", got, want)
	}
}

// TestDriveCarryMonotoneDrainedStaysZero proves a spend-drained axis (Left=0) stays 0
// across a later refill attempt — the honest-money rule of Recontinue, on the relaunch path.
func TestDriveCarryMonotoneDrainedStaysZero(t *testing.T) {
	rows := []DriveStateRow{
		{Session: "sid-a", TokensLeft: ptr64(0), SpendMicroCentsLeft: ptr64(0)},
		{Session: "sid-a", TokensLeft: ptr64(5000), SpendMicroCentsLeft: ptr64(9999)},
	}
	got := ReconcileDriveCarry(rows)["sid-a"]
	if got.TokensLeft != 0 || got.SpendMicroCentsLeft != 0 {
		t.Fatalf("drained carry = %+v, want tokens/spend clamped to 0", got)
	}
}

// TestDriveCarryMonotoneReArmRaises proves an explicit operator re-arm is the ONLY thing
// that raises a remaining axis; an ordinary higher row is clamped.
func TestDriveCarryMonotoneReArmRaises(t *testing.T) {
	rows := []DriveStateRow{
		{Session: "sid-a", TurnsLeft: ptr64(1), TokensLeft: ptr64(100)},
		{Session: "sid-a", TurnsLeft: ptr64(9), TokensLeft: ptr64(900)},              // ordinary row: cannot raise
		{Session: "sid-a", TurnsLeft: ptr64(8), TokensLeft: ptr64(800), ReArm: true}, // explicit re-arm: raises
	}
	if mid := ReconcileDriveCarry(rows[:2])["sid-a"]; mid.TurnsLeft != 1 || mid.TokensLeft != 100 {
		t.Fatalf("pre-rearm carry = %+v, want clamped to the low snapshot {1,100}", mid)
	}
	got := ReconcileDriveCarry(rows)["sid-a"]
	if got.TurnsLeft != 8 || got.TokensLeft != 800 {
		t.Fatalf("post-rearm carry = %+v, want the re-armed snapshot {8,800}", got)
	}
}

// TestDriveCarryMonotoneCarriesOmittedAxesAndObjective locks the snapshot semantics: an
// axis a later row omits keeps its prior folded value (not zero), while the objective
// identity fields take the latest value (a re-pin is never clamped away).
func TestDriveCarryMonotoneCarriesOmittedAxesAndObjective(t *testing.T) {
	rows := []DriveStateRow{
		{Session: "sid-a", TurnsLeft: ptr64(5), TokensLeft: ptr64(500), ObjectivePinID: "pin-1", ObjectiveDigest: "dig-1"},
		{Session: "sid-a", TokensLeft: ptr64(300), ObjectivePinID: "pin-1", ObjectiveDigest: "dig-2"}, // omits TurnsLeft; re-pins digest
	}
	got := ReconcileDriveCarry(rows)["sid-a"]
	if got.TurnsLeft != 5 {
		t.Fatalf("TurnsLeft = %d, want prior value 5 carried forward (row omitted it)", got.TurnsLeft)
	}
	if got.TokensLeft != 300 {
		t.Fatalf("TokensLeft = %d, want lowered to 300", got.TokensLeft)
	}
	if got.ObjectivePinID != "pin-1" || got.ObjectiveDigest != "dig-2" {
		t.Fatalf("objective = %q/%q, want pin-1/dig-2 (identity last-wins)", got.ObjectivePinID, got.ObjectiveDigest)
	}
}

func ptr64(v int64) *int64 { return &v }
func ptrInt(v int) *int    { return &v }
