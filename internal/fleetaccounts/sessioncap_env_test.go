package fleetaccounts

import "testing"

// TestAccountSessionCapEnvKnob pins the hard Claude OAuth identity bound: the
// compatibility knob is parsed but cannot widen concurrency, while non-Claude
// products retain their own default.
func TestAccountSessionCapEnvKnob(t *testing.T) {
	claude := Account{Product: "claude"}

	if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
		t.Fatalf("default claude cap = %d, want %d", got, DefaultClaudeSessionsPerAccount)
	}

	t.Setenv(SessionsPerAccountEnv, "7")
	if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
		t.Fatalf("FAK_SESSIONS_PER_ACCOUNT=7 claude cap = %d, want hard default %d", got, DefaultClaudeSessionsPerAccount)
	}
	// Non-Claude products retain their own one-session default.
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
