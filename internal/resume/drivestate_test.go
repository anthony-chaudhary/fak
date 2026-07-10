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

func ptr64(v int64) *int64 { return &v }
func ptrInt(v int) *int    { return &v }
