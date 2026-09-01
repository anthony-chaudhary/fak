package optsdefault

import (
	"testing"
	"time"
)

func TestRootNowPassesPinnedValuesThrough(t *testing.T) {
	pinned := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	root, now := RootNow("/tmp/ws", pinned)
	if root != "/tmp/ws" || !now.Equal(pinned) {
		t.Fatalf("pinned values mutated: root=%q now=%v", root, now)
	}
}

func TestRootNowAppliesDefaults(t *testing.T) {
	root, now := RootNow("", time.Time{})
	if root != "." {
		t.Fatalf("empty root: got %q, want %q", root, ".")
	}
	if now.IsZero() || now.Location() != time.UTC {
		t.Fatalf("zero now: got %v, want non-zero UTC", now)
	}
	if d := time.Since(now); d < -time.Minute || d > time.Minute {
		t.Fatalf("default now %v not near the wall clock (%v)", now, d)
	}
}
