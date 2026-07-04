package fleetaccounts

import "testing"

// TestAccountSessionCapEnvKnob pins the FAK_SESSIONS_PER_ACCOUNT override: it retunes
// the Claude per-account session budget, is ignored for non-Claude products, and
// falls back to the default on a non-positive or unparseable value. Python reads the
// same variable in fleet_accounts._session_cap (see fleet_session_cap_env_test.py) so
// the two stay one-knob-one-way.
func TestAccountSessionCapEnvKnob(t *testing.T) {
	claude := Account{Product: "claude"}

	if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
		t.Fatalf("default claude cap = %d, want %d", got, DefaultClaudeSessionsPerAccount)
	}

	t.Setenv(SessionsPerAccountEnv, "7")
	if got := AccountSessionCap(claude); got != 7 {
		t.Fatalf("FAK_SESSIONS_PER_ACCOUNT=7 claude cap = %d, want 7", got)
	}
	// the knob only widens Claude accounts; non-Claude stays at one session.
	if got := AccountSessionCap(Account{Product: "opencode"}); got != DefaultAccountSessionsPerWorker {
		t.Fatalf("opencode cap = %d, want %d", got, DefaultAccountSessionsPerWorker)
	}

	for _, bad := range []string{"0", "-3", "notanint", "  "} {
		t.Setenv(SessionsPerAccountEnv, bad)
		if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
			t.Fatalf("FAK_SESSIONS_PER_ACCOUNT=%q claude cap = %d, want default %d",
				bad, got, DefaultClaudeSessionsPerAccount)
		}
	}
}
