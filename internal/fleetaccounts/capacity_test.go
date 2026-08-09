package fleetaccounts

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildCapacityPreflightClassifiesClaudeSeats(t *testing.T) {
	home, cfg, regPath := fixture(t)
	reg := LoadRegistry(regPath)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)

	rep := BuildCapacityPreflight(rows, "claude", 1)
	if !rep.OK || rep.Verdict != "OK" {
		t.Fatalf("preflight = %+v, want OK for one required seat", rep)
	}
	if rep.TrueConcurrentCeiling != 4 || rep.FreshSeats != 4 || rep.BlockedSeats != 4 || rep.StaleSeats != 0 || rep.TotalSeats != 8 {
		t.Fatalf("counts = fresh:%d blocked:%d stale:%d total:%d ceiling:%d, want 4/4/0/8/4",
			rep.FreshSeats, rep.BlockedSeats, rep.StaleSeats, rep.TotalSeats, rep.TrueConcurrentCeiling)
	}
	got := byAccount(rep)
	if got[".claude"].State != CapacityFresh {
		t.Fatalf(".claude state = %+v, want fresh", got[".claude"])
	}
	gem8 := got[".claude-gem8-acct"]
	if gem8.State != CapacityBlockedUntil || !strings.HasPrefix(gem8.StateLabel, "blocked-until-") ||
		!strings.Contains(gem8.Reason, "usage limit") {
		t.Fatalf("gem8 state = %+v, want blocked-until usage", gem8)
	}
}

func TestBuildCapacityPreflightSurfacesStaleCredential(t *testing.T) {
	home, cfg, _ := fixture(t)
	acctDir := filepath.Join(home, ".claude-needslogin-acct")
	if err := os.MkdirAll(filepath.Join(acctDir, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(acctDir, ".claude.json"),
		[]byte(`{"oauthAccount":{"accountUuid":"uuid-needs","emailAddress":"needs@example.test"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})
	rep := BuildCapacityPreflight(rows, "claude", 9)
	if rep.OK || rep.Verdict != "UNDER_CAPACITY" || rep.TrueConcurrentCeiling != 8 {
		t.Fatalf("preflight = %+v, want under-capacity for required=9 with eight fresh Claude slots", rep)
	}
	needs := byAccount(rep)[".claude-needslogin-acct"]
	if needs.State != CapacityStale || needs.LoginStatus == nil || *needs.LoginStatus != "needs_login" ||
		!strings.Contains(needs.Reason, "no live credentials") {
		t.Fatalf("needs-login state = %+v, want stale credential reason", needs)
	}
}

func TestBuildCapacityPreflightAllProducts(t *testing.T) {
	home, cfg, regPath := fixture(t)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(regPath))
	rep := BuildCapacityPreflight(rows, "all", 0)
	if rep.TrueConcurrentCeiling != 5 || rep.FreshSeats != 5 || rep.BlockedSeats != 4 {
		t.Fatalf("all-products counts = %+v, want five fresh slots (.claude x4 + opencode-glm x1) and four blocked", rep)
	}
}

// TestBuildCapacityPreflightSplitsRecoverableFromHard pins #3580 at the detection layer: a
// roster carrying both a needs-login seat and a weekly usage-capped seat must not collapse
// them into one "unavailable" bucket. The needs-login seat is recoverable by a `login`; the
// capped seat is hard and can only wait for its reset. The preflight then carries the
// servable-seat gain a reclaim would return to the ceiling.
func TestBuildCapacityPreflightSplitsRecoverableFromHard(t *testing.T) {
	home, cfg, regPath := fixture(t)
	acctDir := filepath.Join(home, ".claude-needslogin-acct")
	if err := os.MkdirAll(filepath.Join(acctDir, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(acctDir, ".claude.json"),
		[]byte(`{"oauthAccount":{"accountUuid":"uuid-needs","emailAddress":"needs@example.test"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(regPath))
	rep := BuildCapacityPreflight(rows, "claude", 0)
	got := byAccount(rep)

	needs := got[".claude-needslogin-acct"]
	if needs.State != CapacityStale || needs.Recovery != CapacityRecoverable || needs.RecoveryAction != capacityActionLogin {
		t.Fatalf("needs-login seat = state:%q recovery:%q action:%q, want stale/recoverable/login",
			needs.State, needs.Recovery, needs.RecoveryAction)
	}
	capped := got[".claude-gem8-acct"]
	if capped.State != CapacityBlockedUntil || capped.Recovery != CapacityHardWalled || capped.RecoveryAction != capacityActionWaitReset {
		t.Fatalf("weekly-capped seat = state:%q recovery:%q action:%q, want blocked-until/hard/wait for reset",
			capped.State, capped.Recovery, capped.RecoveryAction)
	}
	if rep.RecoverableSeats != AccountSessionCap(rowFor(t, rows, ".claude-needslogin-acct")) {
		t.Fatalf("recoverable_seats = %d, want the needs-login seat's session cap", rep.RecoverableSeats)
	}
	if rep.RecoverableSeats <= 0 || rep.HardWalledSeats <= 0 {
		t.Fatalf("split counts = recoverable:%d hard:%d, want both positive", rep.RecoverableSeats, rep.HardWalledSeats)
	}
	if rep.RecoverableSeats+rep.HardWalledSeats+rep.FreshSeats != rep.TotalSeats {
		t.Fatalf("recoverable(%d)+hard(%d)+fresh(%d) != total(%d): every seat must land in exactly one class",
			rep.RecoverableSeats, rep.HardWalledSeats, rep.FreshSeats, rep.TotalSeats)
	}
}

// TestBuildCapacityPreflightOfferableRosterHasNoRecoveryClass pins #3580 acceptance #3 at the
// detection layer: when every counted seat is offerable there is nothing to reclaim, so no
// account carries a recovery class and both aggregate counts stay zero — which, with the
// omitempty tags, leaves the report byte-identical to the pre-split shape.
func TestBuildCapacityPreflightOfferableRosterHasNoRecoveryClass(t *testing.T) {
	home, cfg, _ := fixture(t)
	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})
	rep := BuildCapacityPreflight(rows, "claude", 0)
	if rep.FreshSeats != rep.TotalSeats {
		t.Fatalf("fixture without the registry is not fully offerable: fresh %d of %d", rep.FreshSeats, rep.TotalSeats)
	}
	if rep.RecoverableSeats != 0 || rep.HardWalledSeats != 0 {
		t.Fatalf("offerable roster counts = recoverable:%d hard:%d, want 0/0", rep.RecoverableSeats, rep.HardWalledSeats)
	}
	for _, acct := range rep.Accounts {
		if acct.Recovery != "" || acct.RecoveryAction != "" {
			t.Fatalf("offerable seat %q carries recovery %q/%q, want empty", acct.Account, acct.Recovery, acct.RecoveryAction)
		}
	}
}

func rowFor(t *testing.T, rows []Account, account string) Account {
	t.Helper()
	for _, row := range rows {
		if row.Account == account {
			return row
		}
	}
	t.Fatalf("account %q not on the roster", account)
	return Account{}
}

func byAccount(rep CapacityPreflight) map[string]CapacityAccount {
	out := map[string]CapacityAccount{}
	for _, acct := range rep.Accounts {
		out[acct.Account] = acct
	}
	return out
}
