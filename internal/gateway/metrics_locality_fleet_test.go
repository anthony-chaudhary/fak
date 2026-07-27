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

// proxyServer is the shape New builds for a proxy-only deployment: a planner, the
// upstream reading, and optionally the roster.
func proxyServer(upstreamURL string, r *modelroute.Roster) *Server {
	return &Server{
		planner:  agent.NewMockPlanner("x"),
		upstream: upstreamAttribution([]string{upstreamURL}),
		roster:   r,
	}
}

// THE MIDDLE RUNG. A model the company hosts on its own servers is reached by
// PROXYING to it, exactly like a vendor is, and before this the two were
// indistinguishable in the metric: every proxied turn was booked as bought. That
// erased the rung the whole three-zone design exists to create.
func TestAProxiedFleetModelCountsAsSelfHosted(t *testing.T) {
	// The upstream is a vendor API, so an id the roster does not bind is genuinely
	// bought — which is what makes the per-model bindings the only thing that can
	// move a turn off that side.
	s := proxyServer("https://api.anthropic.com", fleetRoster())
	for _, tc := range []struct {
		name  string
		model string
		want  servingLocality
	}{
		{"a company-hosted server is ours", "glm-5.2", localitySelfHosted},
		{"so is an on-box one", "qwen3.6-4b", localitySelfHosted},
		{"a frontier lab is not", "claude-opus-5", localityVendor},
		// The three ways a rung can fail to be declared. Each falls through to the
		// upstream reading rather than to a self-hosted credit: credit is only ever
		// given for something an operator actually wrote down.
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
	s := dualServer(t, "qwen-local")
	s.roster = fleetRoster()
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

// THE UNEARNED ZERO, one level up. `--base-url` is how BOTH self-hosting rungs are
// reached in the common case, so treating the flag itself as proof of a third-party
// API did not merely lose the fleet — it manufactured a CONFIDENT vendor attribution
// for hardware the org owns, which renders downstream as "0% self-hosted" over full
// coverage. That is indistinguishable from a fleet that measured everything and
// bought every token, and it is the opposite conclusion.
func TestAnUndeclaredUpstreamIsUnclassifiedRatherThanBought(t *testing.T) {
	for _, tc := range []struct {
		name string
		url  string
		want servingLocality
	}{
		// This machine, definitionally — the way almost everyone runs ollama or vLLM
		// behind a laptop, and rung one of the ladder.
		{"a loopback upstream is this box", "http://127.0.0.1:11434/v1", localitySelfHosted},
		{"by name too", "http://localhost:8000/v1", localitySelfHosted},
		{"and over a unix socket", "unix:/var/run/vllm.sock", localitySelfHosted},
		// A known vendor endpoint is bought, and saying so costs nothing.
		{"a vendor endpoint is bought", "https://api.anthropic.com", localityVendor},
		{"path and port do not hide it", "https://api.openai.com/v1", localityVendor},
		// The case that used to lie. A private host is NOT read as org-owned — only
		// the roster may say that — but it is not read as a vendor either.
		{"a company server nobody declared is unclassified", "http://glm.internal:8000/v1", localityUnknown},
		{"nor is an unlisted provider guessed at", "https://api.together.xyz/v1", localityUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := proxyServer(tc.url, nil).servedLocality("any-model"); got != tc.want {
				t.Errorf("upstream %q classified %v, want %v", tc.url, got, tc.want)
			}
		})
	}
}

// A replica pool mixing our hardware with a vendor's has no single honest answer:
// per turn we cannot tell which replica served it. Unanimity or nothing.
func TestAMixedUpstreamPoolAbstains(t *testing.T) {
	mixed := &Server{
		planner:  agent.NewMockPlanner("x"),
		upstream: upstreamAttribution([]string{"http://127.0.0.1:8000/v1", "https://api.openai.com/v1"}),
	}
	if got := mixed.servedLocality("m"); got != localityUnknown {
		t.Errorf("a mixed pool classified %v, want unknown", got)
	}
	allOurs := &Server{
		planner:  agent.NewMockPlanner("x"),
		upstream: upstreamAttribution([]string{"http://127.0.0.1:8000/v1", "http://localhost:8001/v1"}),
	}
	if got := allOurs.servedLocality("m"); got != localitySelfHosted {
		t.Errorf("an all-loopback pool classified %v, want self-hosted", got)
	}
	// A deployment that does not proxy has no upstream at all, and that absence is
	// what tells servedLocality to use the deployment side instead of asking.
	if upstreamAttribution(nil) != nil {
		t.Error("a non-proxying deployment must carry no upstream reading")
	}
	if u := upstreamAttribution([]string{"http://glm.internal:8000/v1"}); u == nil || *u != localityUnknown {
		t.Error("a proxying deployment with an unplaceable upstream must still be marked as proxying")
	}
}

// The deployments that do not proxy are untouched by any of this.
func TestTheNonProxyingDeploymentsKeepTheirConstantSide(t *testing.T) {
	inKernel := &Server{planner: agent.NewMockPlanner("x"), servedSide: localitySelfHosted, roster: fleetRoster()}
	if got := inKernel.servedLocality("claude-opus-5"); got != localitySelfHosted {
		t.Errorf("in-kernel-only = %v, want self-hosted regardless of the id asked for", got)
	}
	// The mock serves scripted text. A roster binding must not credit it: the roster
	// is a routing declaration, and nothing was routed.
	mock := &Server{planner: agent.NewMockPlanner("x"), roster: fleetRoster()}
	if got := mock.servedLocality("glm-5.2"); got != localityUnknown {
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
	if got := proxyServer("https://api.anthropic.com", fleetRoster()).servedLocality("glm-5.2"); got != localitySelfHosted {
		t.Errorf("an off-box fleet model = %v, want self-hosted (ownership, not residency)", got)
	}
}
