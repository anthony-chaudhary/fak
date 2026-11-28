package modelroute

import (
	"bytes"
	"strings"
	"testing"
)

func failoverTestAccounts() []Account {
	return []Account{
		{ID: "local-1", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
		{ID: "fleet-1", Kind: KindFleet, BaseURL: "https://fleet.internal.example/v1"},
		{ID: "openai-1", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY"},
		{ID: "anthropic-1", Kind: KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
	}
}

func TestFailoverChainJSONRoundTripAndResolveChain(t *testing.T) {
	r := Roster{
		Version:  RosterVersion,
		Accounts: failoverTestAccounts(),
		Bindings: []Binding{
			{
				Model:          "workhorse",
				Account:        "local-1",
				UpstreamModel:  "qwen3.8-local",
				ResidencyFloor: "any",
				Fallbacks: []FallbackTarget{
					{Account: "fleet-1", UpstreamModel: "qwen3.8-fleet"},
					{Account: "openai-1"},
				},
			},
			{
				Model:   "single",
				Account: "openai-1",
			},
		},
	}

	raw := r.JSON()
	parsed, err := ParseRoster(raw)
	if err != nil {
		t.Fatalf("ParseRoster failed: %v", err)
	}
	if !bytes.Equal(raw, parsed.JSON()) {
		t.Fatalf("JSON round-trip mismatch:\ngot:\n%s\nwant:\n%s", parsed.JSON(), raw)
	}

	chain, err := parsed.ResolveChain("workhorse")
	if err != nil {
		t.Fatalf("ResolveChain(workhorse) failed: %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("ResolveChain(workhorse) len = %d, want 3", len(chain))
	}
	if chain[0].Account != "local-1" || chain[0].UpstreamModel != "qwen3.8-local" || chain[0].ResidencyFloor != "any" {
		t.Fatalf("chain[0] = %+v, want local-1 / qwen3.8-local / any", chain[0])
	}
	if chain[1].Account != "fleet-1" || chain[1].UpstreamModel != "qwen3.8-fleet" || chain[1].ResidencyFloor != "any" {
		t.Fatalf("chain[1] = %+v, want fleet-1 / qwen3.8-fleet / any", chain[1])
	}
	if chain[2].Account != "openai-1" || chain[2].UpstreamModel != "workhorse" || chain[2].ResidencyFloor != "any" {
		t.Fatalf("chain[2] = %+v, want openai-1 / workhorse / any", chain[2])
	}

	singleChain, err := parsed.ResolveChain("single")
	if err != nil {
		t.Fatalf("ResolveChain(single) failed: %v", err)
	}
	singlePrimary, err := parsed.Resolve("single")
	if err != nil {
		t.Fatalf("Resolve(single) failed: %v", err)
	}
	if len(singleChain) != 1 || singleChain[0].Account != singlePrimary.Account || singleChain[0].UpstreamModel != singlePrimary.UpstreamModel {
		t.Fatalf("ResolveChain(single) = %+v, want [%+v]", singleChain, singlePrimary)
	}
}

func TestFailoverChainValidationRejectsUnknownOrDuplicateAccounts(t *testing.T) {
	cases := []struct {
		name      string
		fallbacks []FallbackTarget
		wantErr   string
	}{
		{
			name:      "empty account",
			fallbacks: []FallbackTarget{{Account: ""}},
			wantErr:   "empty account",
		},
		{
			name:      "unknown account",
			fallbacks: []FallbackTarget{{Account: "missing-acct"}},
			wantErr:   "unknown account",
		},
		{
			name:      "duplicates primary",
			fallbacks: []FallbackTarget{{Account: "local-1"}},
			wantErr:   "duplicates account",
		},
		{
			name: "duplicates earlier fallback",
			fallbacks: []FallbackTarget{
				{Account: "openai-1"},
				{Account: "openai-1"},
			},
			wantErr: "duplicates account",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := Roster{
				Version:  RosterVersion,
				Accounts: failoverTestAccounts(),
				Bindings: []Binding{
					{
						Model:     "m",
						Account:   "local-1",
						Fallbacks: tc.fallbacks,
					},
				},
			}
			err := r.Validate()
			if err == nil {
				t.Fatalf("Validate() succeeded, want error containing %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("Validate() err = %q, want substring %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestResidencyFloorValidationAndPropagation(t *testing.T) {
	t.Run("local_only with local primary and fleet fallback passes", func(t *testing.T) {
		r := Roster{
			Version:  RosterVersion,
			Accounts: failoverTestAccounts(),
			Bindings: []Binding{
				{
					Model:          "local-model",
					Account:        "local-1",
					ResidencyFloor: "local_only",
					Fallbacks:      []FallbackTarget{{Account: "fleet-1"}},
				},
			},
		}
		if err := r.Validate(); err != nil {
			t.Fatalf("Validate() failed: %v", err)
		}
		chain, err := r.ResolveChain("local-model")
		if err != nil {
			t.Fatalf("ResolveChain() failed: %v", err)
		}
		for i, tgt := range chain {
			if tgt.ResidencyFloor != "local_only" {
				t.Fatalf("chain[%d].ResidencyFloor = %q, want %q", i, tgt.ResidencyFloor, "local_only")
			}
		}
	})

	t.Run("local_only with remote primary fails", func(t *testing.T) {
		r := Roster{
			Version:  RosterVersion,
			Accounts: failoverTestAccounts(),
			Bindings: []Binding{
				{
					Model:          "bad-primary",
					Account:        "openai-1",
					ResidencyFloor: "local_only",
				},
			},
		}
		if err := r.Validate(); err == nil {
			t.Fatalf("Validate() succeeded for remote primary under local_only, want error")
		}
	})

	t.Run("local_only with remote fallback fails", func(t *testing.T) {
		r := Roster{
			Version:  RosterVersion,
			Accounts: failoverTestAccounts(),
			Bindings: []Binding{
				{
					Model:          "bad-fallback",
					Account:        "local-1",
					ResidencyFloor: "local_only",
					Fallbacks: []FallbackTarget{
						{Account: "fleet-1"},
						{Account: "anthropic-1"},
					},
				},
			},
		}
		if err := r.Validate(); err == nil {
			t.Fatalf("Validate() succeeded for remote fallback under local_only, want error")
		}
	})

	t.Run("unknown residency_floor fails", func(t *testing.T) {
		r := Roster{
			Version:  RosterVersion,
			Accounts: failoverTestAccounts(),
			Bindings: []Binding{
				{
					Model:          "bad-floor",
					Account:        "local-1",
					ResidencyFloor: "eu_only",
				},
			},
		}
		if err := r.Validate(); err == nil {
			t.Fatalf("Validate() succeeded for unknown residency_floor, want error")
		}
	})
}

func TestContextOverflowFallbackOverrideAndFitsContext(t *testing.T) {
	accounts := []Account{
		{ID: "local-1", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1", ContextTokens: 8192},
		{ID: "fleet-1", Kind: KindFleet, BaseURL: "https://fleet.internal.example/v1", ContextTokens: 32768},
		{ID: "openai-1", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY"},
	}
	r := Roster{
		Version:  RosterVersion,
		Accounts: accounts,
		Bindings: []Binding{
			{
				Model:   "workhorse",
				Account: "local-1",
				Fallbacks: []FallbackTarget{
					{Account: "fleet-1", ContextTokens: 65536},
					{Account: "openai-1", ContextTokens: 128000},
				},
			},
		},
	}
	raw := r.JSON()
	parsed, err := ParseRoster(raw)
	if err != nil {
		t.Fatalf("ParseRoster failed: %v", err)
	}
	chain, err := parsed.ResolveChain("workhorse")
	if err != nil {
		t.Fatalf("ResolveChain failed: %v", err)
	}
	if len(chain) != 3 {
		t.Fatalf("chain len = %d, want 3", len(chain))
	}
	if chain[0].ContextTokens != 8192 || chain[1].ContextTokens != 65536 || chain[2].ContextTokens != 128000 {
		t.Fatalf("ContextTokens = [%d, %d, %d], want [8192, 65536, 128000]",
			chain[0].ContextTokens, chain[1].ContextTokens, chain[2].ContextTokens)
	}
	if chain[0].FitsContext(9000) {
		t.Fatalf("chain[0].FitsContext(9000) = true, want false")
	}
	if !chain[1].FitsContext(9000) || !chain[2].FitsContext(128000) {
		t.Fatalf("expected chain[1] and chain[2] to fit 9000/128000 tokens")
	}
	if (Target{ContextTokens: 0}).FitsContext(9000) != true {
		t.Fatalf("unspecified ContextTokens (0) should return true for FitsContext")
	}

	bad := Roster{
		Version:  RosterVersion,
		Accounts: accounts,
		Bindings: []Binding{
			{
				Model:     "bad-ctx",
				Account:   "local-1",
				Fallbacks: []FallbackTarget{{Account: "openai-1", ContextTokens: -1}},
			},
		},
	}
	if err := bad.Validate(); err == nil {
		t.Fatalf("Validate() succeeded for negative fallback context_tokens, want error")
	}
}
