package accounts

import "testing"

func TestLoginStatusPrimaryStates(t *testing.T) {
	disabled := active("disabled", "u-disabled", "disabled@example.test")
	disabled.Enabled = boolp(false)

	cases := []struct {
		name string
		home Home
		want LoginStatus
	}{
		{"ready", active("ready", "u-ready", "ready@example.test"), LoginReady},
		{"tombstoned", Home{Name: "old", Status: StatusTombstoned, RehomeTo: "ready"}, LoginTombstoned},
		{"disabled", disabled, LoginDisabled},
		{"missing dir", Home{Name: "missing", Dir: "/missing", Identity: Identity{Exists: false}}, LoginMissingDir},
		{"needs login", Home{Name: "needs-login", Dir: "/needs", Identity: Identity{Exists: true, HasCreds: false}}, LoginNeedsLogin},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.home.LoginStatus(); got != tc.want {
				t.Fatalf("LoginStatus = %q, want %q", got, tc.want)
			}
			if got := tc.home.CanServe(); got != (tc.want == LoginReady) {
				t.Fatalf("CanServe = %v for status %q", got, tc.want)
			}
		})
	}
}

func TestLoginReportWarningsAndSummary(t *testing.T) {
	disabled := active("disabled", "u-disabled", "disabled@example.test")
	disabled.Enabled = boolp(false)
	noIdentity := Home{
		Name:     "tokenless",
		Dir:      "/home/tokenless",
		Identity: Identity{Exists: true, HasCreds: true},
	}
	reg := Registry{
		Roles: map[string]string{RoleActive: "alice", RoleAnchor: "alice"},
		Homes: []Home{
			active("alice", "u-alice", "alice@example.test"),
			active("zdup", "u-alice", "alice@example.test"),
			{Name: "gem8", Dir: "/home/gem8", Identity: Identity{Exists: true, HasCreds: true, AccountUUID: "u-gem8", Email: "gem8@example.test", TokenFP: "abc123"}},
			{Name: "day24", Dir: "/home/day24", Identity: Identity{Exists: true, HasCreds: true, AccountUUID: "u-day24", Email: "day24@example.test", TokenFP: "abc123"}},
			noIdentity,
			disabled,
			{Name: "old", Status: StatusTombstoned, RehomeTo: "alice"},
		},
	}

	report := reg.LoginReport()
	if report.Schema != LoginReportSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, LoginReportSchema)
	}
	if report.Summary.Total != len(reg.Homes) {
		t.Fatalf("summary total = %d, want %d", report.Summary.Total, len(reg.Homes))
	}
	if report.Summary.ByStatus[string(LoginReady)] != 5 ||
		report.Summary.ByStatus[string(LoginDisabled)] != 1 ||
		report.Summary.ByStatus[string(LoginTombstoned)] != 1 {
		t.Fatalf("unexpected status counts: %+v", report.Summary.ByStatus)
	}

	byName := map[string]LoginObservation{}
	for _, obs := range report.Seats {
		byName[obs.Name] = obs
	}
	if !hasLoginWarning(byName["zdup"], LoginWarningDuplicateBucket) {
		t.Fatalf("duplicate bucket warning missing: %+v", byName["zdup"])
	}
	if !hasLoginWarning(byName["gem8"], LoginWarningTokenTwin) ||
		!hasLoginWarning(byName["day24"], LoginWarningTokenTwin) {
		t.Fatalf("token twin warning missing: gem8=%+v day24=%+v", byName["gem8"], byName["day24"])
	}
	if !hasLoginWarning(byName["tokenless"], LoginWarningUnverifiedAccount) {
		t.Fatalf("unverified account warning missing: %+v", byName["tokenless"])
	}
	if got := byName["alice"].Roles; len(got) != 2 || got[0] != RoleActive || got[1] != RoleAnchor {
		t.Fatalf("roles = %v, want [active anchor]", got)
	}
	if byName["disabled"].CanServe {
		t.Fatalf("disabled seat must not be can_serve: %+v", byName["disabled"])
	}
}

func TestServeUsesLoginStatusForDisabledSeat(t *testing.T) {
	disabled := active("disabled", "u-disabled", "disabled@example.test")
	disabled.Enabled = boolp(false)
	reg := Registry{
		Roles: map[string]string{RoleAnchor: "anchor"},
		Homes: []Home{
			active("anchor", "u-anchor", "anchor@example.test"),
			disabled,
		},
	}
	got, chain, err := reg.Serve("disabled")
	if err != nil {
		t.Fatalf("Serve disabled: %v", err)
	}
	if got.Name != "anchor" || len(chain) != 1 || chain[0] != "disabled" {
		t.Fatalf("Serve disabled = %q chain=%v, want anchor via [disabled]", got.Name, chain)
	}
}

func hasLoginWarning(obs LoginObservation, want LoginWarning) bool {
	for _, got := range obs.Warnings {
		if got == want {
			return true
		}
	}
	return false
}
