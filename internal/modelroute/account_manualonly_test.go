package modelroute

import (
	"encoding/json"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// ManualOnly — an account that is REACHABLE but never VOLUNTEERED, and the
// AuthScheme override that lets a third-party Anthropic-compatible endpoint be
// reached at all. Both are account-level supply properties.
// ---------------------------------------------------------------------------

// reservedRosterFixture holds a separately-billed vendor endpoint in reserve beside an
// ordinary pooled account. The reserved one is bound (so it can be NAMED) but is not the
// Default (so nothing unbound falls onto it).
func reservedRosterFixture() Roster {
	return Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "oa-personal", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY"},
			{
				ID:         "vendor-gateway",
				Kind:       KindAnthropic,
				BaseURL:    "https://gateway.example.com/serving-endpoints/anthropic",
				CredEnv:    "VENDOR_GATEWAY_TOKEN",
				ManualOnly: true,
				AuthScheme: AuthSchemeBearer,
			},
		},
		Default: "oa-personal",
		Bindings: []Binding{
			{Model: "small", Account: "oa-personal", UpstreamModel: "gpt-5.5"},
			{Model: "reserved-sonnet", Account: "vendor-gateway", UpstreamModel: "vendor-claude-sonnet-5"},
		},
	}
}

func TestReservedAccountStillResolvesWhenNamed(t *testing.T) {
	r := reservedRosterFixture()
	if err := r.Validate(); err != nil {
		t.Fatalf("fixture invalid: %v", err)
	}
	// The whole point of the flag: withheld from automatic pools, NOT disabled.
	tg, err := r.Resolve("reserved-sonnet")
	if err != nil {
		t.Fatalf("resolve reserved-sonnet: %v", err)
	}
	if tg.Account != "vendor-gateway" {
		t.Fatalf("named resolution landed on %q, want vendor-gateway", tg.Account)
	}
	if tg.UpstreamModel != "vendor-claude-sonnet-5" {
		t.Fatalf("upstream model %q not carried", tg.UpstreamModel)
	}
	if !tg.ManualOnly {
		t.Fatal("Target.ManualOnly is false: the dispatch layer cannot record that a reserved credential was spent by explicit request")
	}
	if tg.AuthScheme != AuthSchemeBearer {
		t.Fatalf("Target.AuthScheme = %q, want %q — the planner build would fall back to the shape sniff and 401", tg.AuthScheme, AuthSchemeBearer)
	}
}

func TestUnreservedAccountTargetLeavesFlagsZero(t *testing.T) {
	tg, err := reservedRosterFixture().Resolve("small")
	if err != nil {
		t.Fatalf("resolve small: %v", err)
	}
	if tg.ManualOnly {
		t.Fatal("ordinary pooled account resolved with ManualOnly set")
	}
	if tg.AuthScheme != AuthSchemeDefault {
		t.Fatalf("AuthScheme = %q, want empty (adapter default)", tg.AuthScheme)
	}
}

// TestValidateRefusesReservedDefault is the invariant that keeps the reservation
// honest: Resolve routes EVERY unbound id to the Default, so a reserved account named
// there would quietly serve the widest automatic path in the roster.
func TestValidateRefusesReservedDefault(t *testing.T) {
	r := Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "vendor-gateway", Kind: KindAnthropic, CredEnv: "VENDOR_GATEWAY_TOKEN", ManualOnly: true},
		},
		Default: "vendor-gateway",
	}
	err := r.Validate()
	if err == nil {
		t.Fatal("Validate accepted a manual_only account as the Default; every unbound model id would spend a reserved credential")
	}
	if !strings.Contains(err.Error(), "manual_only") {
		t.Fatalf("error does not name the cause: %v", err)
	}
}

func TestValidateAllowsReservedNonDefault(t *testing.T) {
	if err := reservedRosterFixture().Validate(); err != nil {
		t.Fatalf("a reserved, non-default account must be valid: %v", err)
	}
}

func TestValidateAuthSchemeClosedSet(t *testing.T) {
	cases := map[string]bool{
		AuthSchemeDefault: true,
		AuthSchemeBearer:  true,
		AuthSchemeAPIKey:  true,
		"Bearer":          false, // case matters: a silent fallback would re-introduce the 401
		"token":           false,
		"authorization":   false, // an accepted CLI spelling, but not a roster value
		"x_api_key":       false,
	}
	for scheme, want := range cases {
		r := Roster{
			Version:  RosterVersion,
			Accounts: []Account{{ID: "a", Kind: KindAnthropic, CredEnv: "TOK", AuthScheme: scheme}},
		}
		err := r.Validate()
		if want && err != nil {
			t.Errorf("auth_scheme %q rejected: %v", scheme, err)
		}
		if !want && err == nil {
			t.Errorf("auth_scheme %q accepted; a typo must fail loud, not fall back to the shape sniff", scheme)
		}
	}
}

// TestReservedFieldsOmittedWhenUnset pins the compatibility half: a roster written
// before these fields existed must round-trip byte-identically, so both carry omitempty.
func TestReservedFieldsOmittedWhenUnset(t *testing.T) {
	b, err := json.Marshal(Account{ID: "a", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY"})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{"manual_only", "auth_scheme"} {
		if strings.Contains(string(b), key) {
			t.Errorf("unset field emitted %q: %s", key, b)
		}
	}
}

func TestReservedAccountDecodesFromJSON(t *testing.T) {
	const raw = `{"id":"vendor","kind":"anthropic","cred_env":"TOK","manual_only":true,"auth_scheme":"bearer"}`
	var a Account
	if err := json.Unmarshal([]byte(raw), &a); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !a.ManualOnly || a.AuthScheme != AuthSchemeBearer {
		t.Fatalf("decoded wrong: %+v", a)
	}
}

// TestReadinessSurfacesReservation covers the operator-facing half: a READY account
// that never receives traffic must not read as a broken binding.
func TestReadinessSurfacesReservation(t *testing.T) {
	lookup := func(name string) (string, bool) { return "tok", true }
	rep := reservedRosterFixture().Readiness(lookup)
	var found bool
	for _, row := range rep.Rows {
		if row.ID != "vendor-gateway" {
			if row.ManualOnly {
				t.Errorf("account %q reported as reserved", row.ID)
			}
			continue
		}
		found = true
		if !row.ManualOnly {
			t.Error("reserved account not marked manual_only in the readiness row")
		}
		if row.Status != AccountReady {
			t.Errorf("reserved account status = %q, want ready: a reservation is not a defect", row.Status)
		}
		if !strings.Contains(row.Reason, "manual_only") {
			t.Errorf("reason %q does not explain why nothing lands here", row.Reason)
		}
	}
	if !found {
		t.Fatal("reserved account missing from the readiness report entirely")
	}
}
