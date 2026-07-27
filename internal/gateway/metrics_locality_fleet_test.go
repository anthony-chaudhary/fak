package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func fleetRoster() *modelroute.Roster {
	return &modelroute.Roster{
		Accounts: []modelroute.Account{
			{ID: "company-glm", Kind: modelroute.KindFleet, BaseURL: "http://glm.internal:8000/v1"},
			{ID: "laptop", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "frontier", Kind: modelroute.KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "glm-5.2", Account: "company-glm"},
			{Model: "qwen3.6-4b", Account: "laptop"},
			{Model: "claude-opus-5", Account: "frontier"},
			// A binding pointing at an account nobody defined. BoundZone refuses it,
			// and so must attribution — a broken roster declares no rung.
			{Model: "kimi-k3", Account: "typo-not-an-account"},
		},
		Default: "frontier",
	}
}

// THE MIDDLE RUNG. A model the company hosts on its own servers is reached by
// PROXYING to it, exactly like a vendor is, and before this the two were
// indistinguishable in the metric: every proxied turn was booked as bought. That
// erased the rung the whole three-zone design exists to create.
func TestAProxiedFleetModelCountsAsSelfHosted(t *testing.T) {
	s := &Server{planner: agent.NewMockPlanner("x"), servedSide: localityVendor, roster: fleetRoster()}
	for _, tc := range []struct {
		name  string
		model string
		want  servingLocality
	}{
		{"a company-hosted server is ours", "glm-5.2", localitySelfHosted},
		{"so is an on-box one", "qwen3.6-4b", localitySelfHosted},
		{"a frontier lab is not", "claude-opus-5", localityVendor},
		// The three ways a rung can fail to be declared. Each stays vendor: credit
		// is only ever given for something an operator actually wrote down.
		{"a model the roster does not bind earns nothing", "some-other-model", localityVendor},
		{"nor does a binding to an account that does not exist", "kimi-k3", localityVendor},
		{"nor does an empty id", "", localityVendor},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := s.servedLocality(tc.model); got != tc.want {
				t.Errorf("servedLocality(%q) = %v, want %v", tc.model, got, tc.want)
			}
		})
	}
}

// The same rule under a dual planner: in-kernel decode is self-hosted because it
// ran here, and a proxied turn is judged by the roster rather than by the fact
// that it was proxied.
func TestTheDualPlannerProxySideAlsoCreditsTheFleet(t *testing.T) {
	dp, err := NewDualPlanner(agent.NewMockPlanner("upstream"), agent.NewMockPlanner("qwen-local"), "qwen-local")
	if err != nil {
		t.Fatalf("NewDualPlanner: %v", err)
	}
	s := &Server{planner: dp, roster: fleetRoster()}
	if got := s.servedLocality("qwen-local"); got != localitySelfHosted {
		t.Errorf("the in-kernel id = %v, want self-hosted", got)
	}
	if got := s.servedLocality("glm-5.2"); got != localitySelfHosted {
		t.Errorf("a proxied company-hosted model = %v, want self-hosted", got)
	}
	if got := s.servedLocality("claude-opus-5"); got != localityVendor {
		t.Errorf("a proxied vendor model = %v, want vendor", got)
	}
}

// Credit requires a declaration. With no roster the gateway knows nothing about
// who owns the upstream, and the honest reading of a proxied turn is that we
// bought it — the direction that under-claims rather than the one that flatters.
func TestWithNoRosterEveryProxiedTurnStaysVendor(t *testing.T) {
	s := &Server{planner: agent.NewMockPlanner("x"), servedSide: localityVendor}
	for _, m := range []string{"glm-5.2", "qwen3.6-4b", "claude-opus-5", ""} {
		if got := s.servedLocality(m); got != localityVendor {
			t.Errorf("servedLocality(%q) with no roster = %v, want vendor", m, got)
		}
	}
	// And the in-kernel-only deployment is unaffected by any of this: it did not
	// proxy, so there is no upstream to ask about.
	inKernel := &Server{planner: agent.NewMockPlanner("x"), servedSide: localitySelfHosted, roster: fleetRoster()}
	if got := inKernel.servedLocality("claude-opus-5"); got != localitySelfHosted {
		t.Errorf("in-kernel-only = %v, want self-hosted regardless of the id asked for", got)
	}
	// An unwired side still classifies nothing rather than guessing.
	unwired := &Server{planner: agent.NewMockPlanner("x"), roster: fleetRoster()}
	if got := unwired.servedLocality("glm-5.2"); got != localityUnknown {
		t.Errorf("an unwired deployment = %v, want unknown", got)
	}
}

// The attribution vocabulary must never be mistaken for a residency one. modelroute
// keeps SelfHosted (who owns the hardware) and OnBox (whether the bytes left the
// machine) apart precisely so that declaring a fleet zone cannot buy a sensitive
// payload a trip off the box, and this test pins that the gateway's counter sits on
// the OWNERSHIP side of that line — a fleet model is credited here while remaining
// off-box, which is exactly the pair the residency floor must keep distinguishing.
func TestSelfHostedAttributionIsNotAResidencyClaim(t *testing.T) {
	if !modelroute.ZoneFleet.SelfHosted() {
		t.Fatal("ZoneFleet stopped being self-hosted; this counter's meaning changed under it")
	}
	if modelroute.ZoneFleet.OnBox() {
		t.Fatal("ZoneFleet became on-box — the residency floor's exemption just widened, " +
			"and this metric must not be the reason anyone believes that is safe")
	}
	s := &Server{planner: agent.NewMockPlanner("x"), servedSide: localityVendor, roster: fleetRoster()}
	if got := s.servedLocality("glm-5.2"); got != localitySelfHosted {
		t.Errorf("an off-box fleet model = %v, want self-hosted (ownership, not residency)", got)
	}
}
