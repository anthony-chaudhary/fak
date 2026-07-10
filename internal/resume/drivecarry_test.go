package resume

import "testing"

func TestDriveCarryRowsFoldLastWins(t *testing.T) {
	rows := []DriveCarryRow{
		{Session: "sid-a", TurnsLeft: 8, TokensLeft: 2000},
		{Session: "sid-b", TurnsLeft: 4, TokensLeft: 500},
		{Session: " sid-a ", TurnsLeft: 6, TokensLeft: 1500, SpendMicroCentsLeft: 9, Generation: 2},
		{Session: " ", TurnsLeft: 99},
	}
	got := FoldDriveCarryRows(rows)
	if len(got) != 2 {
		t.Fatalf("len=%d, want 2: %+v", len(got), got)
	}
	if got["sid-a"].TurnsLeft != 6 || got["sid-a"].TokensLeft != 1500 || got["sid-a"].Generation != 2 {
		t.Fatalf("sid-a=%+v, want latest row", got["sid-a"])
	}
	if got["sid-b"].TurnsLeft != 4 {
		t.Fatalf("sid-b=%+v", got["sid-b"])
	}
	if empty := FoldDriveCarryRows(nil); len(empty) != 0 {
		t.Fatalf("nil fold=%+v", empty)
	}
}
