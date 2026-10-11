package leaseref

import (
	"testing"
	"time"
)

// TestRedRepairClaimTakeSeeExpire is the 2026-10-10 ctxmmu witness: session A claims the
// repair of a package red at base H; session B — on a DIFFERENT base (a peer's unrelated
// commit moved HEAD) — still sees A's claim and is refused naming A; A's own re-claim renews;
// once A's TTL lapses the package is free and B takes it; release clears it.
func TestRedRepairClaimTakeSeeExpire(t *testing.T) {
	g := newFakeGit()
	s := NewWithRunner(g.run, "")
	t0 := time.Unix(10_000, 0)
	const pkg = "internal/ctxmmu"

	rec, v, err := s.ClaimRedRepair(ctx(), pkg, "aaaa1111", "A", "sess-a", 600, t0)
	if err != nil || !v.OK {
		t.Fatalf("A claim: ok=%v err=%v %+v", v.OK, err, v)
	}
	if gotPkg, gotBase, ok := ParseRedRepairTarget(rec.Target); !ok || gotPkg != pkg || gotBase != "aaaa1111" {
		t.Fatalf("target %q round-trip = (%q,%q,%v)", rec.Target, gotPkg, gotBase, ok)
	}

	// SEE: B's gate lookup at another base still finds A's claim (package-wide).
	seen, err := s.RedRepairClaims(ctx(), []string{pkg, "internal/other"}, t0.Add(time.Minute))
	if err != nil || len(seen) != 1 || seen[0].Holder != "A" || seen[0].SessionID != "sess-a" {
		t.Fatalf("RedRepairClaims = %+v err=%v, want A's one claim", seen, err)
	}
	if none, _ := s.RedRepairClaims(ctx(), []string{"internal/other"}, t0); len(none) != 0 {
		t.Fatalf("unrelated package must see no claim, got %+v", none)
	}

	// B claims the same package at a different base: refused, naming A.
	_, bv, err := s.ClaimRedRepair(ctx(), pkg, "bbbb2222", "B", "sess-b", 600, t0.Add(2*time.Minute))
	if err != nil || bv.OK || bv.Reason != ReasonIntentCollision || bv.Peer == nil || bv.Peer.Holder != "A" {
		t.Fatalf("B duplicate repair must be INTENT_COLLISION naming A, got %+v err=%v", bv, err)
	}

	// A renews its own claim.
	if _, av, err := s.ClaimRedRepair(ctx(), pkg, "aaaa1111", "A", "sess-a", 600, t0.Add(3*time.Minute)); err != nil || !av.OK {
		t.Fatalf("A renew: %+v err=%v", av, err)
	}

	// EXPIRE: past A's renewed TTL the claim is no longer live, and B takes the package.
	late := t0.Add(3*time.Minute + 601*time.Second)
	if live, _ := s.RedRepairClaims(ctx(), []string{pkg}, late); len(live) != 0 {
		t.Fatalf("expired claim must not be surfaced as live: %+v", live)
	}
	if _, bv, err := s.ClaimRedRepair(ctx(), pkg, "bbbb2222", "B", "sess-b", 600, late); err != nil || !bv.OK {
		t.Fatalf("B takeover after A's TTL: %+v err=%v", bv, err)
	}

	// RELEASE on land clears it.
	if err := s.ReleaseRedRepair(ctx(), pkg, "bbbb2222"); err != nil {
		t.Fatalf("release: %v", err)
	}
	if live, _ := s.RedRepairClaims(ctx(), []string{pkg}, late); len(live) != 0 {
		t.Fatalf("released claim still live: %+v", live)
	}
}

func TestParseRedRepairTargetRejectsOtherIntents(t *testing.T) {
	for _, target := range []string{"issue #2155", "red-repair nobase", "red-repair @abc", "red-repair pkg@"} {
		if _, _, ok := ParseRedRepairTarget(target); ok {
			t.Fatalf("%q must not parse as a red-repair target", target)
		}
	}
}
