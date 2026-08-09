package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// reservedPlacementRoster pairs an ordinary pooled account with a reserved one. Both are
// BOUND — the reserved account is reachable by name — so the only thing that may keep it
// out of the automatic pool is the ManualOnly flag itself.
func reservedPlacementRoster() modelroute.Roster {
	return modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "pooled", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_API_KEY"},
			{
				ID:         "reserved",
				Kind:       modelroute.KindAnthropic,
				BaseURL:    "https://gateway.example.com/serving-endpoints/anthropic",
				CredEnv:    "VENDOR_GATEWAY_TOKEN",
				ManualOnly: true,
				AuthScheme: modelroute.AuthSchemeBearer,
			},
		},
		Default: "pooled",
		Bindings: []modelroute.Binding{
			{Model: "pooled-small", Account: "pooled", UpstreamModel: "gpt-5.5"},
			{Model: "reserved-sonnet", Account: "reserved", UpstreamModel: "vendor-claude-sonnet-5"},
		},
	}
}

// TestPlacementCandidatesExcludeReservedAccounts is the guarantee that makes the flag
// mean anything: this pool is what Place and the escalation walk choose from on their
// own, so a reserved model must never be offered to it.
func TestPlacementCandidatesExcludeReservedAccounts(t *testing.T) {
	r := reservedPlacementRoster()
	if err := r.Validate(); err != nil {
		t.Fatalf("fixture invalid: %v", err)
	}
	got := placementCandidates(r, nil)
	for _, c := range got {
		if c.Model == "reserved-sonnet" {
			t.Fatal("a manual_only account's model entered the automatic placement pool; escalation could spend a reserved credential unasked")
		}
	}
	// And the ordinary account is untouched — the filter must not empty the pool.
	var sawPooled bool
	for _, c := range got {
		if c.Model == "pooled-small" {
			sawPooled = true
		}
	}
	if !sawPooled {
		t.Fatalf("pooled model dropped from the candidate pool: %+v", got)
	}
	if len(got) != 1 {
		t.Fatalf("candidate pool = %+v, want exactly the one pooled model", got)
	}
}

// TestPlacementCandidatesUnchangedWithoutReservation pins that the filter is inert for
// every roster written before the flag existed.
func TestPlacementCandidatesUnchangedWithoutReservation(t *testing.T) {
	r := reservedPlacementRoster()
	for i := range r.Accounts {
		r.Accounts[i].ManualOnly = false
	}
	got := placementCandidates(r, nil)
	if len(got) != 2 {
		t.Fatalf("candidate pool = %+v, want both models when nothing is reserved", got)
	}
}

// TestPlacementCandidatesReservationIsPerAccount checks the flag scopes to the account
// it is set on: two models bound to the SAME reserved account both drop, and a model
// bound elsewhere does not.
func TestPlacementCandidatesReservationIsPerAccount(t *testing.T) {
	r := reservedPlacementRoster()
	r.Bindings = append(r.Bindings,
		modelroute.Binding{Model: "reserved-haiku", Account: "reserved", UpstreamModel: "vendor-claude-haiku"},
		modelroute.Binding{Model: "pooled-large", Account: "pooled", UpstreamModel: "gpt-5.5-pro"},
	)
	if err := r.Validate(); err != nil {
		t.Fatalf("fixture invalid: %v", err)
	}
	got := placementCandidates(r, nil)
	if len(got) != 2 {
		t.Fatalf("candidate pool = %+v, want only the two pooled models", got)
	}
	for _, c := range got {
		if c.Model == "reserved-sonnet" || c.Model == "reserved-haiku" {
			t.Fatalf("reserved model %q offered to the automatic pool", c.Model)
		}
	}
}
