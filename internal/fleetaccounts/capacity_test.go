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

func byAccount(rep CapacityPreflight) map[string]CapacityAccount {
	out := map[string]CapacityAccount{}
	for _, acct := range rep.Accounts {
		out[acct.Account] = acct
	}
	return out
}
