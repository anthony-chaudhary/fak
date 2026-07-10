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
