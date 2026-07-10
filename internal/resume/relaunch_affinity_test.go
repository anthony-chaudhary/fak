package resume

import "testing"

func TestRelaunchCacheAffinityStableAcrossRelaunches(t *testing.T) {
	const transcriptUUID = "7d9b2146-8772-4dc5-87d6-8f240985d733"

	firstLaunch := RelaunchCacheAffinity(transcriptUUID)
	nextLaunch := RelaunchCacheAffinity(transcriptUUID)
	if firstLaunch == "" {
		t.Fatal("RelaunchCacheAffinity returned an empty key")
	}
	if nextLaunch != firstLaunch {
		t.Fatalf("next relaunch affinity = %q, want preserved %q", nextLaunch, firstLaunch)
	}
}

func TestRelaunchCacheAffinityNormalizesAndSeparatesIdentity(t *testing.T) {
	const transcriptUUID = "7d9b2146-8772-4dc5-87d6-8f240985d733"
	if got, want := RelaunchCacheAffinity("  "+transcriptUUID+"\n"), RelaunchCacheAffinity(transcriptUUID); got != want {
		t.Fatalf("normalized affinity = %q, want %q", got, want)
	}
	if got := RelaunchCacheAffinity("another-transcript"); got == RelaunchCacheAffinity(transcriptUUID) {
		t.Fatalf("distinct transcript UUIDs produced the same affinity %q", got)
	}
	if got := RelaunchCacheAffinity(" \t\n"); got != "" {
		t.Fatalf("empty transcript affinity = %q, want empty", got)
	}
}
