package accounts

import (
	"testing"
)

// The gate's whole job: permit a same-account token write (default + named dir for ONE
// account is legitimate), refuse the cross-account smear that is Regression A, and stay
// out of the way of a first enrollment into a fresh dir.

func TestGateTokenWrite_RefusesCrossAccountSmear(t *testing.T) {
	root := t.TempDir()
	// gem8's named home, logged in as gem8, carrying gem8's token — the token's true owner.
	mkSeat(t, root, ".claude-gem8-netra", "gem8@netra.test", "uuid-gem8", "GEM8-TOKEN", true)
	// day24's home, logged in as day24, currently no token (the post-quarantine state).
	day24 := mkSeat(t, root, ".claude-day24-netra", "day24@netra.test", "uuid-day24", "", true)

	// Writing gem8's token into day24's dir is the smear — must be refused.
	v := GateTokenWrite(day24.Dir, "GEM8-TOKEN", root)
	if v.Allow {
		t.Fatalf("expected refusal of cross-account write, got allow: %+v", v)
	}
	if v.Reason != "cross-account" {
		t.Fatalf("expected reason cross-account, got %q (%s)", v.Reason, v.Detail)
	}
}

func TestGateTokenWrite_AllowsSameAccountTwoDirs(t *testing.T) {
	root := t.TempDir()
	// gem8's named home owns the token.
	mkSeat(t, root, ".claude-gem8-netra", "gem8@netra.test", "uuid-gem8", "GEM8-TOKEN", true)
	// The default home is ALSO logged into gem8 — writing gem8's token here is correct,
	// not a smear (same account, two dir names).
	def := mkSeat(t, root, ".claude", "gem8@netra.test", "uuid-gem8", "", true)

	v := GateTokenWrite(def.Dir, "GEM8-TOKEN", root)
	if !v.Allow {
		t.Fatalf("expected allow for same-account two-dir write, got refusal: %+v", v)
	}
}

func TestGateTokenWrite_AllowsFreshDirNoLogin(t *testing.T) {
	root := t.TempDir()
	mkSeat(t, root, ".claude-gem8-netra", "gem8@netra.test", "uuid-gem8", "GEM8-TOKEN", true)
	// A brand-new dir with no .claude.json login — nothing to contradict, first enrollment
	// must not be blocked.
	fresh := mkSeat(t, root, ".claude-new", "", "", "", false)

	v := GateTokenWrite(fresh.Dir, "SOME-NEW-TOKEN", root)
	if !v.Allow {
		t.Fatalf("expected allow for fresh dir with no login, got refusal: %+v", v)
	}
}

func TestGateTokenWrite_RefusesEmptyToken(t *testing.T) {
	root := t.TempDir()
	d := mkSeat(t, root, ".claude-x", "x@netra.test", "uuid-x", "", true)
	v := GateTokenWrite(d.Dir, "   \n", root)
	if v.Allow || v.Reason != "empty-token" {
		t.Fatalf("expected empty-token refusal, got %+v", v)
	}
}

// AuditTokenTwins must red on a cross-account smear and stay green when the only shared
// token is one account under two dir names — the exact distinction the live incident hung
// on.

func TestAuditTokenTwins_FlagsCrossAccount(t *testing.T) {
	root := t.TempDir()
	// THE BUG: three homes, three logins, ONE token (the live incident's shape).
	mkSeat(t, root, ".claude", "gem8@netra.test", "uuid-gem8", "SHARED-DEAD-TOKEN", true)
	mkSeat(t, root, ".claude-day24-netra", "day24@netra.test", "uuid-day24", "SHARED-DEAD-TOKEN", true)
	mkSeat(t, root, ".claude-gem8-netra", "gem8@netra.test", "uuid-gem8", "SHARED-DEAD-TOKEN", true)

	findings, err := AuditTokenTwins(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 {
		t.Fatalf("expected exactly 1 cross-account finding, got %d: %+v", len(findings), findings)
	}
	// The finding must name >1 distinct account (the whole reason it's a bug).
	if len(findings[0].Accounts) < 2 {
		t.Fatalf("finding must span >1 account, got %v", findings[0].Accounts)
	}
}

func TestAuditTokenTwins_CleanWhenOneAccountTwoDirs(t *testing.T) {
	root := t.TempDir()
	// gem8 under two dir names, sharing gem8's token — legitimate, must NOT be flagged.
	mkSeat(t, root, ".claude", "gem8@netra.test", "uuid-gem8", "GEM8-TOKEN", true)
	mkSeat(t, root, ".claude-gem8-netra", "gem8@netra.test", "uuid-gem8", "GEM8-TOKEN", true)
	// A second, fully-distinct account with its OWN token — also fine.
	mkSeat(t, root, ".claude-gem7-netra", "gem7@netra.test", "uuid-gem7", "GEM7-TOKEN", true)

	findings, err := AuditTokenTwins(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 0 {
		t.Fatalf("expected clean tree (one-account-two-dirs is legitimate), got %+v", findings)
	}
}
