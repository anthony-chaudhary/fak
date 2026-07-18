package resume

import "testing"

func TestRelaunchCacheAffinityStableAcrossRelaunches(t *testing.T) {
	const uuid = "4b66d1ba-b3dc-4f74-a8f5-d78ef67933c4"
	first := RelaunchCacheAffinityKey(uuid)
	second := RelaunchCacheAffinityKey(uuid)
	if first == "" || first != second {
		t.Fatalf("same transcript routes = %q, %q; want identical non-empty keys", first, second)
	}
	if other := RelaunchCacheAffinityKey("different-transcript"); other == first {
		t.Fatalf("distinct transcript route = %q, want different from %q", other, first)
	}
	if blank := RelaunchCacheAffinityKey("  "); blank != "" {
		t.Fatalf("blank transcript route = %q, want empty", blank)
	}
}

func TestNewRelaunchAffinityRowDerivesKeyAndLeavesTSForShell(t *testing.T) {
	const uuid = "4b66d1ba-b3dc-4f74-a8f5-d78ef67933c4"
	row := NewRelaunchAffinityRow("  " + uuid + "  ")
	if row.Session != uuid {
		t.Fatalf("row.Session = %q, want trimmed %q", row.Session, uuid)
	}
	if row.AffinityKey == "" || row.AffinityKey != RelaunchCacheAffinityKey(uuid) {
		t.Fatalf("row.AffinityKey = %q, want the derived key %q", row.AffinityKey, RelaunchCacheAffinityKey(uuid))
	}
	if row.TS != "" {
		t.Fatalf("row.TS = %q, want \"\" (the shell stamps write time)", row.TS)
	}
	blank := NewRelaunchAffinityRow("   ")
	if blank.Session != "" || blank.AffinityKey != "" {
		t.Fatalf("blank-session row = %+v, want empty session and key (a row the fold skips)", blank)
	}
	if got := FoldRelaunchAffinity([]RelaunchAffinityRow{blank}); len(got) != 0 {
		t.Fatalf("fold of a blank row = %#v, want empty", got)
	}
}

func TestRelaunchCacheAffinityFoldLatestValid(t *testing.T) {
	const session = "session-a"
	derived := RelaunchCacheAffinityKey(session)
	got := FoldRelaunchAffinity([]RelaunchAffinityRow{
		{Session: session, AffinityKey: "old-route"},
		{Session: "", AffinityKey: "ignored"},
		{Session: session, AffinityKey: derived},
		{Session: "session-b", AffinityKey: ""},
	})
	if got[session] != derived || len(got) != 1 {
		t.Fatalf("fold = %#v, want only %q -> %q", got, session, derived)
	}
}
