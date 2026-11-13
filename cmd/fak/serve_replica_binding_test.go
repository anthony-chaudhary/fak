package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestReplicaBindingConsistency is the H200 half-node witness, table-driven over the
// pure helper. The load-bearing case (b) is the exact defect: replica2's gateway
// launched with the correct --base-url :8010 but replica1's roster binding the local
// account to :8000, so every request forwarded to replica1 while replica2's GPUs
// served nothing and the facade stayed healthy.
func TestReplicaBindingConsistency(t *testing.T) {
	roster := func(accounts ...modelroute.Account) *modelroute.Roster {
		return &modelroute.Roster{Version: modelroute.RosterVersion, Accounts: accounts}
	}
	local := func(port string) modelroute.Account {
		return modelroute.Account{ID: "local", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:" + port + "/v1"}
	}
	remote := modelroute.Account{ID: "openai", Kind: modelroute.KindOpenAI, BaseURL: "https://api.openai.com/v1", CredEnv: "OPENAI_API_KEY"}

	tests := []struct {
		name    string
		baseURL string
		roster  *modelroute.Roster
		wantErr bool
	}{
		{
			// (a) the fixed replica2 case: --base-url and the local account agree.
			name:    "matching loopback binding is legal",
			baseURL: "http://127.0.0.1:8010/v1",
			roster:  roster(local("8010")),
			wantErr: false,
		},
		{
			// (b) the exact witness defect: replica2's --base-url :8010, replica1's roster :8000.
			name:    "reused node-global roster pointing at the other replica is refused",
			baseURL: "http://127.0.0.1:8010/v1",
			roster:  roster(local("8000")),
			wantErr: true,
		},
		{
			// (c) mixed roster: only a remote openai account, no loopback binding.
			name:    "remote-only roster stays legal",
			baseURL: "http://127.0.0.1:8010/v1",
			roster:  roster(remote),
			wantErr: false,
		},
		{
			// (d) no proxy target: empty --base-url, any roster.
			name:    "empty base-url is a no-op",
			baseURL: "",
			roster:  roster(local("8000")),
			wantErr: false,
		},
		{
			// (e) no roster: pre-#2528 behavior, byte-identical.
			name:    "nil roster is a no-op",
			baseURL: "http://127.0.0.1:8010/v1",
			roster:  nil,
			wantErr: false,
		},
		{
			// (f) path suffix ignored: /v1 on --base-url, no path on the account.
			name:    "path suffix does not participate",
			baseURL: "http://127.0.0.1:8010/v1",
			roster: &modelroute.Roster{Version: modelroute.RosterVersion, Accounts: []modelroute.Account{
				{ID: "local", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:8010"},
			}},
			wantErr: false,
		},
		{
			// (g) a mixed roster with BOTH a matching local and a remote account stays legal.
			name:    "mixed local+remote roster with matching local binding is legal",
			baseURL: "http://127.0.0.1:8010/v1",
			roster:  roster(local("8010"), remote),
			wantErr: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := checkReplicaBindingConsistency(tc.baseURL, tc.roster)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("want REPLICA_BINDING_MISMATCH refusal, got nil")
				}
				if !strings.Contains(err.Error(), bailReplicaBindingMismatch) {
					t.Fatalf("error does not name the token: %v", err)
				}
				// The bail must name BOTH values so the operator can act.
				if !strings.Contains(err.Error(), "8010") || !strings.Contains(err.Error(), "8000") {
					t.Fatalf("error must name both host:ports, got: %v", err)
				}
			} else if err != nil {
				t.Fatalf("want nil, got: %v", err)
			}
		})
	}
}

// hostPortOf drops the path suffix and normalizes host:port for the comparison.
func TestHostPortOf(t *testing.T) {
	for in, want := range map[string]string{
		"http://127.0.0.1:8010/v1": "127.0.0.1:8010",
		"http://127.0.0.1:8010":    "127.0.0.1:8010",
		"http://localhost:8000/v1": "localhost:8000",
		"":                         "",
	} {
		if got := hostPortOf(in); got != want {
			t.Errorf("hostPortOf(%q) = %q, want %q", in, got, want)
		}
	}
}
