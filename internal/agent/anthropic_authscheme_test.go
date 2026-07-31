package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// ---------------------------------------------------------------------------
// AnthropicAuthScheme — how an Anthropic-wire credential is presented.
//
// The bug these pin: the scheme used to be inferred from the token's PREFIX, which is
// only a discriminator for first-party Anthropic. A third-party Anthropic-COMPATIBLE
// endpoint authenticates its own tenant token (no sk-ant-* prefix) and accepts it only
// as a bearer, so the sniff sent x-api-key and the call 401'd with a correct base URL,
// model, and body — a failure indistinguishable from a bad credential.
// ---------------------------------------------------------------------------

func TestAnthropicAuthAutoSniffsFirstPartyShapes(t *testing.T) {
	// The historical behavior, unchanged: this is what api.anthropic.com requires.
	t.Run("plain api key rides as x-api-key", func(t *testing.T) {
		h := anthropicAdapter{}.Headers("sk-ant-api03-abc")
		if h["x-api-key"] != "sk-ant-api03-abc" {
			t.Fatalf("x-api-key = %q", h["x-api-key"])
		}
		if _, ok := h["Authorization"]; ok {
			t.Fatal("plain api key must not also ride as a bearer")
		}
		if _, ok := h["anthropic-beta"]; ok {
			t.Fatal("oauth beta set for a non-subscription credential")
		}
	})
	t.Run("subscription token rides as bearer with the oauth beta", func(t *testing.T) {
		h := anthropicAdapter{}.Headers("sk-ant-oat01-abc")
		if h["Authorization"] != "Bearer sk-ant-oat01-abc" {
			t.Fatalf("Authorization = %q", h["Authorization"])
		}
		if h["anthropic-beta"] != AnthropicOAuthBeta {
			t.Fatalf("anthropic-beta = %q, want %q", h["anthropic-beta"], AnthropicOAuthBeta)
		}
		if _, ok := h["x-api-key"]; ok {
			t.Fatal("a subscription token sent as x-api-key 401s with 'invalid x-api-key'")
		}
	})
}

// TestAnthropicAuthBearerCarriesThirdPartyToken is the regression for the reported 401.
func TestAnthropicAuthBearerCarriesThirdPartyToken(t *testing.T) {
	// A third-party gateway PAT: no sk-ant-* prefix, so the sniff would have chosen
	// x-api-key and the endpoint would have refused the credential outright.
	const tok = "vendor-pat-0000"
	h := NewAnthropicTranscriptAdapter(AnthropicAuthBearer).Headers(tok)
	if h["Authorization"] != "Bearer "+tok {
		t.Fatalf("Authorization = %q, want the bearer the endpoint requires", h["Authorization"])
	}
	// Load-bearing: the tenant token must NOT also be copied into a header the endpoint
	// does not expect and may log.
	if _, ok := h["x-api-key"]; ok {
		t.Fatal("bearer scheme also sent x-api-key: the tenant token is duplicated into an unexpected header")
	}
	if _, ok := h["anthropic-beta"]; ok {
		t.Fatal("oauth beta sent for a credential that is not an Anthropic subscription token")
	}
	if h["anthropic-version"] != "2023-06-01" || h["Content-Type"] != "application/json" {
		t.Fatalf("wire headers dropped: %v", h)
	}
}

func TestAnthropicAuthExplicitSchemesOverrideTheSniff(t *testing.T) {
	// The oauth beta is a property of the SUBSCRIPTION CREDENTIAL, not of the scheme, so
	// it rides along however that token was asked to present itself.
	t.Run("bearer keeps the beta for a subscription token", func(t *testing.T) {
		h := NewAnthropicTranscriptAdapter(AnthropicAuthBearer).Headers("sk-ant-oat01-abc")
		if h["Authorization"] == "" || h["anthropic-beta"] != AnthropicOAuthBeta {
			t.Fatalf("headers = %v", h)
		}
	})
	t.Run("x-api-key forced even for a subscription token", func(t *testing.T) {
		h := NewAnthropicTranscriptAdapter(AnthropicAuthAPIKey).Headers("sk-ant-oat01-abc")
		if h["x-api-key"] != "sk-ant-oat01-abc" {
			t.Fatalf("x-api-key = %q; an explicit scheme must win over the sniff", h["x-api-key"])
		}
		if _, ok := h["Authorization"]; ok {
			t.Fatal("explicit x-api-key must not also send a bearer")
		}
	})
}

func TestAnthropicAuthEmptyCredentialSendsNoAuthUnderEveryScheme(t *testing.T) {
	// The loopback dogfood / mock path: no credential means no auth header at all,
	// and an explicit scheme must not invent "Bearer ".
	for _, scheme := range []AnthropicAuthScheme{AnthropicAuthAuto, AnthropicAuthBearer, AnthropicAuthAPIKey} {
		h := NewAnthropicTranscriptAdapter(scheme).Headers("")
		if _, ok := h["Authorization"]; ok {
			t.Errorf("scheme %q invented an Authorization header for an empty credential", scheme)
		}
		if _, ok := h["x-api-key"]; ok {
			t.Errorf("scheme %q invented an x-api-key for an empty credential", scheme)
		}
	}
}

func TestParseAnthropicAuthScheme(t *testing.T) {
	ok := map[string]AnthropicAuthScheme{
		"":              AnthropicAuthAuto,
		"auto":          AnthropicAuthAuto,
		"AUTO":          AnthropicAuthAuto,
		"  bearer  ":    AnthropicAuthBearer,
		"Bearer":        AnthropicAuthBearer,
		"authorization": AnthropicAuthBearer,
		"x-api-key":     AnthropicAuthAPIKey,
		"apikey":        AnthropicAuthAPIKey,
		"api-key":       AnthropicAuthAPIKey,
	}
	for in, want := range ok {
		got, valid := ParseAnthropicAuthScheme(in)
		if !valid || got != want {
			t.Errorf("ParseAnthropicAuthScheme(%q) = %q,%v; want %q,true", in, got, valid, want)
		}
	}
	// A typo must MISS so its call site can fail loud. Falling back to the sniff would
	// re-introduce exactly the 401 the operator set the flag to avoid.
	for _, bad := range []string{"token", "basic", "x-api", "none", "true"} {
		if _, valid := ParseAnthropicAuthScheme(bad); valid {
			t.Errorf("ParseAnthropicAuthScheme(%q) accepted; a typo must not silently mean auto", bad)
		}
	}
}

func TestPlannerAuthSchemeReachesTheAdapter(t *testing.T) {
	// The scheme is only useful if it survives the planner->adapter hop, which is the
	// single place a credential becomes a header.
	p := NewHTTPPlanner("https://gateway.example.com/serving-endpoints/anthropic", "m", "vendor-pat-0000")
	p.Provider = ProviderAnthropic
	p.AnthropicAuthScheme = AnthropicAuthBearer
	ad, err := p.transcriptAdapter()
	if err != nil {
		t.Fatalf("transcriptAdapter: %v", err)
	}
	if got := ad.Headers("vendor-pat-0000")["Authorization"]; got != "Bearer vendor-pat-0000" {
		t.Fatalf("planner scheme did not reach the adapter: Authorization = %q", got)
	}
}

func TestPlannerDefaultSchemeIsUnchanged(t *testing.T) {
	// Every pre-existing caller leaves the field zero and must keep the sniff.
	p := NewHTTPPlanner("https://api.anthropic.com", "m", "sk-ant-api03-abc")
	p.Provider = ProviderAnthropic
	ad, err := p.transcriptAdapter()
	if err != nil {
		t.Fatalf("transcriptAdapter: %v", err)
	}
	h := ad.Headers("sk-ant-api03-abc")
	if h["x-api-key"] != "sk-ant-api03-abc" {
		t.Fatalf("default planner changed the first-party scheme: %v", h)
	}
}

func TestExplicitAdapterStillWinsOverScheme(t *testing.T) {
	// p.Adapter is the pre-existing escape hatch; the new field must not hijack it.
	p := NewHTTPPlanner("https://api.anthropic.com", "m", "k")
	p.Provider = ProviderAnthropic
	p.AnthropicAuthScheme = AnthropicAuthBearer
	p.Adapter = NewAnthropicTranscriptAdapter(AnthropicAuthAPIKey)
	ad, err := p.transcriptAdapter()
	if err != nil {
		t.Fatalf("transcriptAdapter: %v", err)
	}
	if _, ok := ad.Headers("k")["x-api-key"]; !ok {
		t.Fatal("an explicitly supplied Adapter must win over AnthropicAuthScheme")
	}
}

// TestAuthSchemeMirrorsAgent pins modelroute's string spellings against this package's
// typed ones. modelroute deliberately does NOT import internal/agent (agent imports
// modelroute, so it would cycle) and mirrors the values by hand; this test lives here
// because this package can see both, and it fails if either side is renamed alone.
func TestAuthSchemeMirrorsAgent(t *testing.T) {
	pairs := map[string]AnthropicAuthScheme{
		modelroute.AuthSchemeDefault: AnthropicAuthAuto,
		modelroute.AuthSchemeBearer:  AnthropicAuthBearer,
		modelroute.AuthSchemeAPIKey:  AnthropicAuthAPIKey,
	}
	for rosterValue, want := range pairs {
		if rosterValue != string(want) {
			t.Errorf("modelroute %q != agent %q: a roster auth_scheme would no longer bind", rosterValue, want)
		}
		// And the roster's spelling must be one the CLI parser accepts, so a value that
		// validates in the roster cannot fail at the planner build.
		if got, ok := ParseAnthropicAuthScheme(rosterValue); !ok || got != want {
			t.Errorf("roster value %q does not parse to %q (got %q, ok=%v)", rosterValue, want, got, ok)
		}
	}
}
