package modelroute

import (
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Resolve — bind a routed model id to a concrete dispatch target.
// ---------------------------------------------------------------------------

// rosterFixture brings four accounts — a local server, the SAME kind under two
// accounts (the switch), and a remote Anthropic — and binds ids across them.
func rosterFixture() Roster {
	return Roster{
		Version: RosterVersion,
		Accounts: []Account{
			{ID: "local", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "oa-personal", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY"},
			{ID: "oa-work", Kind: KindOpenAI, CredEnv: "OPENAI_WORK_API_KEY"},
			{ID: "claude", Kind: KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
		Default: "oa-personal",
		Bindings: []Binding{
			{Model: "small", Account: "local", UpstreamModel: "llama3.2"},
			{Model: "large", Account: "claude", UpstreamModel: "claude-opus-4-6"},
			{Model: "guard-a", Account: "oa-work", UpstreamModel: "gpt-5.5"},
			{Model: "guard-b", Account: "claude", UpstreamModel: "claude-opus-4-6"},
		},
	}
}

func TestResolveBindingWins(t *testing.T) {
	r := rosterFixture()
	if err := r.Validate(); err != nil {
		t.Fatalf("fixture invalid: %v", err)
	}
	tg, err := r.Resolve("large")
	if err != nil {
		t.Fatalf("resolve large: %v", err)
	}
	if tg.Account != "claude" || tg.Kind != KindAnthropic || tg.UpstreamModel != "claude-opus-4-6" {
		t.Fatalf("large bound wrong: %+v", tg)
	}
	// base_url filled from the Anthropic kind default (account left it empty).
	if tg.BaseURL != KindBaseURL(KindAnthropic) {
		t.Fatalf("base_url not filled from kind default: %q", tg.BaseURL)
	}
	if tg.Local() || !tg.Remote() {
		t.Fatalf("anthropic target should be remote: %+v", tg)
	}
}

func TestResolveLocalAccountIsLocalAndKeyless(t *testing.T) {
	tg, err := rosterFixture().Resolve("small")
	if err != nil {
		t.Fatalf("resolve small: %v", err)
	}
	if !tg.Local() || tg.Remote() {
		t.Fatalf("local-account target must be local: %+v", tg)
	}
	if tg.BaseURL != "http://127.0.0.1:11434/v1" || tg.CredEnv != "" {
		t.Fatalf("local target wrong base/cred: %+v", tg)
	}
	if tg.UpstreamModel != "llama3.2" {
		t.Fatalf("local upstream wrong: %q", tg.UpstreamModel)
	}
}

func TestResolveDefaultAccountForUnboundID(t *testing.T) {
	// "mystery" has no binding; the Default account ("oa-personal") serves it, and
	// with no upstream the routed id is used verbatim on the wire.
	tg, err := rosterFixture().Resolve("mystery")
	if err != nil {
		t.Fatalf("resolve mystery: %v", err)
	}
	if tg.Account != "oa-personal" || tg.UpstreamModel != "mystery" {
		t.Fatalf("default binding wrong: %+v", tg)
	}
}

func TestResolveNoBindingNoDefaultIsFailLoud(t *testing.T) {
	r := rosterFixture()
	r.Default = "" // no fallback
	if _, err := r.Resolve("mystery"); err == nil {
		t.Fatalf("an unbound id with no default must fail loud, not pick an arbitrary account")
	}
}

// ResolvePlan binds an ENSEMBLE whose members live on DIFFERENT accounts, preserving
// member order (the determinism contract Combine relies on), and resolves the Scout.
func TestResolvePlanScoutAndMemberOrderAcrossAccounts(t *testing.T) {
	r := rosterFixture()
	p := Plan{
		Members: []Member{{Model: "guard-a"}, {Model: "guard-b"}},
		Reduce:  ReduceVote,
		Scout:   "small",
	}
	rp, err := r.ResolvePlan(p)
	if err != nil {
		t.Fatalf("resolve plan: %v", err)
	}
	if rp.Scout == nil || !rp.Scout.Local() {
		t.Fatalf("scout aspect must resolve (to the local account here): %+v", rp.Scout)
	}
	if len(rp.Members) != 2 || rp.Members[0].Account != "oa-work" || rp.Members[1].Account != "claude" {
		t.Fatalf("ensemble not split across accounts in member order: %+v", rp.Members)
	}
	// Two members, two providers — the mix-and-match at the ensemble level.
	if rp.Members[0].Kind == rp.Members[1].Kind {
		t.Fatalf("expected members on different provider kinds, got both %q", rp.Members[0].Kind)
	}
}

func TestResolvePlanFailsLoudOnBadMember(t *testing.T) {
	r := rosterFixture()
	r.Default = ""
	p := Plan{Members: []Member{{Model: "small"}, {Model: "nope"}}}
	if _, err := r.ResolvePlan(p); err == nil {
		t.Fatalf("a plan with an unresolvable member must fail loud")
	}
}

func TestResolvePlanFailsLoudOnBadScout(t *testing.T) {
	r := rosterFixture()
	r.Default = ""
	p := Plan{Members: []Member{{Model: "small"}}, Scout: "ghost-scout"}
	if _, err := r.ResolvePlan(p); err == nil {
		t.Fatalf("a plan whose scout cannot resolve must fail loud, not silently drop it")
	}
}

// ResolveDecision is the single entry point the CLI and dispatch share.
func TestResolveDecisionDelegates(t *testing.T) {
	r := rosterFixture()
	m := Manifest{Default: Plan{Members: []Member{{Model: "small"}}}}
	d := m.Route(Subject{Aspect: AspectRequest})
	rp, err := r.ResolveDecision(d)
	if err != nil {
		t.Fatalf("resolve decision: %v", err)
	}
	if len(rp.Members) != 1 || !rp.Members[0].Local() {
		t.Fatalf("decision resolve wrong: %+v", rp)
	}
}

// ---------------------------------------------------------------------------
// EngineRoute — the residency floor reads a DECLARED local/remote prefix.
// ---------------------------------------------------------------------------

// A remote target's route starts with its <kind>: keyword and a local one with
// "local:" — exactly the prefixes internal/engine's residencyGate.remoteRoute keys on
// (the cross-package agreement is pinned in internal/engine/account_residency_test.go).
func TestEngineRouteIsStructurallyHonestAboutLocality(t *testing.T) {
	r := rosterFixture()
	small, _ := r.Resolve("small") // local account
	large, _ := r.Resolve("large") // remote anthropic
	if got := small.EngineRoute(); !strings.HasPrefix(got, "local:") {
		t.Fatalf("local target route must start local:, got %q", got)
	}
	if got := large.EngineRoute(); !strings.HasPrefix(got, string(KindAnthropic)+":") {
		t.Fatalf("remote target route must start with its kind keyword, got %q", got)
	}
}

// Every REMOTE kind's route embeds a floor-recognized keyword, and the LOCAL kind's
// route is local-prefixed — so the floor never fails OPEN on a route it can't classify.
func TestEngineRouteRemoteKindsCarryAFloorKeyword(t *testing.T) {
	keywords := []string{"openai", "anthropic", "gemini", "xai"}
	for _, k := range []ProviderKind{KindOpenAI, KindOpenAIResponses, KindAnthropic, KindGemini, KindXAI} {
		route := Target{Kind: k, Account: "acct", UpstreamModel: "m"}.EngineRoute()
		hit := false
		for _, kw := range keywords {
			if strings.Contains(route, kw) {
				hit = true
			}
		}
		if !hit {
			t.Fatalf("remote kind %q route %q carries no floor keyword (would fail OPEN as not-remote)", k, route)
		}
	}
	if got := (Target{Kind: KindLocal, Account: "a", UpstreamModel: "m"}).EngineRoute(); !strings.HasPrefix(got, "local:") {
		t.Fatalf("local kind must be local-prefixed, got %q", got)
	}
}

// ---------------------------------------------------------------------------
// Validate — fail-loud on every misconfiguration class.
// ---------------------------------------------------------------------------

func TestValidateRejections(t *testing.T) {
	cases := map[string]Roster{
		"no accounts": {Accounts: nil},
		"empty account id": {
			Accounts: []Account{{ID: "", Kind: KindOpenAI, CredEnv: "K"}},
		},
		"account id with delimiter": {
			Accounts: []Account{{ID: "a/b", Kind: KindOpenAI, CredEnv: "K"}},
		},
		"duplicate account id": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}, {ID: "a", Kind: KindAnthropic, CredEnv: "K"}},
		},
		"unknown kind": {
			Accounts: []Account{{ID: "a", Kind: "cohere", CredEnv: "K"}},
		},
		"remote account without cred_env": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI}},
		},
		"local without base_url": {
			Accounts: []Account{{ID: "a", Kind: KindLocal}},
		},
		"local with remote base_url (residency bypass)": {
			Accounts: []Account{{ID: "a", Kind: KindLocal, BaseURL: "https://api.openai.com/v1"}},
		},
		"binding to unknown account": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}},
			Bindings: []Binding{{Model: "x", Account: "ghost"}},
		},
		"duplicate binding model": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}},
			Bindings: []Binding{{Model: "x", Account: "a"}, {Model: "x", Account: "a"}},
		},
		"binding model with delimiter": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}},
			Bindings: []Binding{{Model: "x:y", Account: "a"}},
		},
		"default to unknown account": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}},
			Default:  "ghost",
		},
		"bad version": {
			Version:  "fak-accounts/v2",
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}},
		},
	}
	for name, r := range cases {
		if err := r.Validate(); err == nil {
			t.Fatalf("Validate(%s) should fail", name)
		}
	}
}

// A local account on an explicit loopback host is accepted.
func TestValidateAcceptsLoopbackLocal(t *testing.T) {
	for _, base := range []string{"http://127.0.0.1:11434/v1", "http://localhost:8000/v1", "http://[::1]:1234"} {
		r := Roster{Accounts: []Account{{ID: "l", Kind: KindLocal, BaseURL: base}}}
		if err := r.Validate(); err != nil {
			t.Fatalf("loopback local %q should validate: %v", base, err)
		}
	}
}

// A pasted secret (with '-'/'.') in cred_env is NOT a valid env-var name, so Validate
// rejects it — the footgun guard that keeps real keys out of a committed roster.
func TestValidateRejectsPastedSecretInCredEnv(t *testing.T) {
	r := Roster{Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "sk-ant-oat01-abc.def"}}}
	err := r.Validate()
	if err == nil || !strings.Contains(err.Error(), "env-var name") {
		t.Fatalf("a pasted secret in cred_env must fail loud, got %v", err)
	}
	// A real env-var name passes.
	r.Accounts[0].CredEnv = "OPENAI_API_KEY"
	if err := r.Validate(); err != nil {
		t.Fatalf("a real env-var name should pass: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Round-trip + secret hygiene + the built-in default.
// ---------------------------------------------------------------------------

func TestRosterRoundTrip(t *testing.T) {
	want := DefaultRoster()
	parsed, err := ParseRoster(want.JSON())
	if err != nil {
		t.Fatalf("round-trip parse: %v", err)
	}
	if len(parsed.Accounts) != len(want.Accounts) || len(parsed.Bindings) != len(want.Bindings) {
		t.Fatalf("round-trip lost rows: %+v", parsed)
	}
	if parsed.Default != want.Default {
		t.Fatalf("round-trip lost default: %q", parsed.Default)
	}
}

func TestParseRosterRejectsUnknownField(t *testing.T) {
	bad := `{"version":"fak-accounts/v1","accounts":[{"id":"a","kind":"openai","cred_env":"K","api_key":"sk-oops"}]}`
	if _, err := ParseRoster([]byte(bad)); err == nil {
		t.Fatalf("an unknown field (here a literal api_key secret) must be rejected, not silently kept")
	}
}

// The serialized roster — and every resolved Target — carries env-var NAMES only,
// never a secret value. This is the structural witness behind the "reference, never
// the secret" claim: even if the env var holds "sk-secret", nothing in the manifest
// or the resolved targets contains it.
func TestRosterCarriesNoSecretMaterial(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-this-must-never-appear")
	r := DefaultRoster()
	if strings.Contains(string(r.JSON()), "sk-this-must-never-appear") {
		t.Fatalf("roster JSON leaked a secret value")
	}
	rp, err := r.ResolvePlan(Plan{Members: []Member{{Model: "small"}, {Model: "guard-a"}, {Model: "large"}}, Scout: "medium"})
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	targets := append([]Target{*rp.Scout}, rp.Members...)
	for _, tg := range targets {
		if strings.Contains(tg.CredEnv, "sk-") || strings.Contains(tg.EngineRoute(), "sk-") {
			t.Fatalf("target leaked secret material: %+v route=%q", tg, tg.EngineRoute())
		}
	}
}

// The built-in default is valid and actually demonstrates mix-and-match: its guard
// ensemble members resolve onto two different provider kinds, the cheap aspect goes
// to a local residency-exempt server, and codex binds to the Responses wire.
func TestDefaultRosterIsValidAndMixesProviders(t *testing.T) {
	r := DefaultRoster()
	if err := r.Validate(); err != nil {
		t.Fatalf("DefaultRoster invalid: %v", err)
	}
	a, err := r.Resolve("guard-a")
	if err != nil {
		t.Fatalf("resolve guard-a: %v", err)
	}
	b, err := r.Resolve("guard-b")
	if err != nil {
		t.Fatalf("resolve guard-b: %v", err)
	}
	if a.Kind == b.Kind {
		t.Fatalf("the default guard ensemble should span two providers, got both %q", a.Kind)
	}
	small, _ := r.Resolve("small")
	if !small.Local() {
		t.Fatalf("default should route 'small' to a local server, got %+v", small)
	}
	// The same provider kind under two distinct accounts = the switch.
	codex, ok := r.account("codex")
	if !ok || codex.Kind != KindOpenAIResponses {
		t.Fatalf("codex should bind to the Responses wire: %+v", codex)
	}
}

func TestKindBaseURLDefaults(t *testing.T) {
	if KindBaseURL(KindAnthropic) != "https://api.anthropic.com" {
		t.Fatalf("anthropic default base url wrong: %q", KindBaseURL(KindAnthropic))
	}
	if KindBaseURL(KindOpenAI) != "https://api.openai.com/v1" {
		t.Fatalf("openai default base url wrong: %q", KindBaseURL(KindOpenAI))
	}
	if KindBaseURL(KindOpenAIResponses) != KindBaseURL(KindOpenAI) {
		t.Fatalf("responses should share the openai host")
	}
	if KindBaseURL(KindLocal) != "" {
		t.Fatalf("local kind must have no public default base url, got %q", KindBaseURL(KindLocal))
	}
}
