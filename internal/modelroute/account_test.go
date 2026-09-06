package modelroute

import (
	"encoding/json"
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
	keywords := []string{"openai", "anthropic", "gemini", "xai", "deepseek"}
	for _, k := range []ProviderKind{KindOpenAI, KindOpenAIResponses, KindAnthropic, KindGemini, KindXAI, KindDeepSeek} {
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
		"negative context tokens": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", ContextTokens: -1}},
		},
		"negative max output tokens": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", MaxOutputTokens: -1}},
		},
		"negative requests per minute": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", RequestsPerMinute: -1}},
		},
		"negative requests per day": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", RequestsPerDay: -1}},
		},
		"negative tokens per minute": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", TokensPerMinute: -1}},
		},
		"negative tokens per day": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", TokensPerDay: -1}},
		},
		"requests per minute above requests per day": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", RequestsPerMinute: 1000, RequestsPerDay: 250}},
		},
		"tokens per minute above tokens per day": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K", TokensPerMinute: 400000, TokensPerDay: 200000}},
		},
		"upstream model with colon delimiter": {
			Accounts: []Account{{ID: "a", Kind: KindOpenAI, CredEnv: "K"}},
			Bindings: []Binding{{Model: "x", Account: "a", UpstreamModel: "bad:model"}},
		},
		"deprecated deepseek alias without compatibility marker": {
			Accounts: []Account{{ID: "deepseek", Kind: KindDeepSeek, CredEnv: DeepSeekAPIKeyEnv}},
			Bindings: []Binding{{Model: "legacy-chat", Account: "deepseek", UpstreamModel: "deepseek-chat"}},
		},
		"deprecated deepseek alias missing retirement date": {
			Accounts: []Account{{ID: "deepseek", Kind: KindDeepSeek, CredEnv: DeepSeekAPIKeyEnv}},
			Bindings: []Binding{{Model: "legacy-chat", Account: "deepseek", UpstreamModel: "deepseek-chat", CompatibilityOnly: true, DeprecatedAliasFor: DeepSeekV4FlashModel + " non-thinking mode"}},
		},
		"compatibility marker on non-deprecated alias": {
			Accounts: []Account{{ID: "deepseek", Kind: KindDeepSeek, CredEnv: DeepSeekAPIKeyEnv}},
			Bindings: []Binding{{Model: "deepseek-pro", Account: "deepseek", UpstreamModel: DeepSeekV4ProModel, CompatibilityOnly: true}},
		},
	}
	for name, r := range cases {
		if err := r.Validate(); err == nil {
			t.Fatalf("Validate(%s) should fail", name)
		}
	}
}

func TestValidateAcceptsNamespacedUpstreamModelSlug(t *testing.T) {
	r := Roster{
		Accounts: []Account{{ID: "groq", Kind: KindOpenAI, BaseURL: GroqOpenAIBaseURL, CredEnv: GroqAPIKeyEnv}},
		Bindings: []Binding{{Model: "qwen36-groq", Account: "groq", UpstreamModel: GroqQwen36Model}},
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("provider namespaced upstream slugs such as %q should validate: %v", GroqQwen36Model, err)
	}
	tg, err := r.Resolve("qwen36-groq")
	if err != nil {
		t.Fatalf("resolve Groq Qwen: %v", err)
	}
	if tg.EngineRoute() != "openai:groq/qwen/qwen3.6-27b" {
		t.Fatalf("namespaced upstream should remain visible in EngineRoute, got %q", tg.EngineRoute())
	}
}

func TestValidateAcceptsDeepSeekCompatibilityAliasesWhenMarked(t *testing.T) {
	r := Roster{
		Accounts: []Account{{ID: "deepseek", Kind: KindDeepSeek, CredEnv: DeepSeekAPIKeyEnv}},
		Bindings: []Binding{
			{
				Model:              "deepseek-chat-compat",
				Account:            "deepseek",
				UpstreamModel:      "deepseek-chat",
				CompatibilityOnly:  true,
				DeprecatedAfterUTC: DeepSeekLegacyAliasRetiresUTC,
				DeprecatedAliasFor: DeepSeekV4FlashModel + " non-thinking mode",
			},
			{
				Model:              "deepseek-reasoner-compat",
				Account:            "deepseek",
				UpstreamModel:      "deepseek-reasoner",
				CompatibilityOnly:  true,
				DeprecatedAfterUTC: DeepSeekLegacyAliasRetiresUTC,
				DeprecatedAliasFor: DeepSeekV4FlashModel + " thinking mode",
			},
		},
	}
	if err := r.Validate(); err != nil {
		t.Fatalf("compatibility aliases should validate with explicit retirement metadata: %v", err)
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

func TestRosterReadinessReportsCredentialPresenceWithoutSecrets(t *testing.T) {
	lookup := func(name string) (string, bool) {
		switch name {
		case "OPENAI_API_KEY":
			return "sk-this-must-never-appear", true
		case "OPENAI_WORK_API_KEY":
			return "", true
		default:
			return "", false
		}
	}
	report := rosterFixture().Readiness(lookup)
	if report.Schema != AccountReadinessSchema {
		t.Fatalf("schema = %q, want %q", report.Schema, AccountReadinessSchema)
	}
	if report.Summary.Total != 4 || report.Summary.Ready != 2 || report.Summary.NeedsCredential != 2 {
		t.Fatalf("summary = %+v, want total 4 ready 2 needs_credential 2", report.Summary)
	}
	byID := map[string]AccountReadinessObservation{}
	for _, row := range report.Rows {
		byID[row.ID] = row
	}
	if byID["local"].Credential != CredentialNotRequired || byID["local"].Status != AccountReady {
		t.Fatalf("local readiness wrong: %+v", byID["local"])
	}
	if byID["oa-personal"].Credential != CredentialPresent || byID["oa-personal"].Status != AccountReady {
		t.Fatalf("present remote credential wrong: %+v", byID["oa-personal"])
	}
	if byID["oa-work"].Credential != CredentialMissing || byID["oa-work"].Status != AccountNeedsCredential {
		t.Fatalf("empty env var should be missing: %+v", byID["oa-work"])
	}
	if byID["claude"].Credential != CredentialMissing || byID["claude"].Status != AccountNeedsCredential {
		t.Fatalf("absent env var should be missing: %+v", byID["claude"])
	}
	b, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "sk-this-must-never-appear") {
		t.Fatalf("readiness report leaked the credential value: %s", b)
	}
	if !strings.Contains(string(b), "OPENAI_API_KEY") {
		t.Fatalf("readiness report should carry env-var names for actionability: %s", b)
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
	ds, err := r.Resolve("deepseek-pro")
	if err != nil {
		t.Fatalf("resolve deepseek-pro: %v", err)
	}
	if ds.Kind != KindDeepSeek || ds.Account != "deepseek" || ds.UpstreamModel != DeepSeekV4ProModel || ds.BaseURL != DeepSeekOpenAIBaseURL {
		t.Fatalf("default DeepSeek Pro binding wrong: %+v", ds)
	}
	if ds.ContextTokens != DeepSeekV4ContextTokens || ds.MaxOutputTokens != DeepSeekV4MaxOutputTokens {
		t.Fatalf("default DeepSeek metadata wrong: %+v", ds)
	}
	if ds.Local() || !strings.HasPrefix(ds.EngineRoute(), "deepseek:") {
		t.Fatalf("DeepSeek target should be a remote deepseek route: %+v route=%q", ds, ds.EngineRoute())
	}
	ox, err := r.Resolve(OpenCodeGoOxAlphaModel)
	if err != nil {
		t.Fatalf("resolve Ox Alpha: %v", err)
	}
	if ox.Kind != KindOpenAI || ox.Account != OpenCodeGoProviderKey || ox.BaseURL != OpenCodeGoOpenAIBaseURL ||
		ox.CredEnv != OpenCodeGoAPIKeyEnv || ox.UpstreamModel != OpenCodeGoOxAlphaModel {
		t.Fatalf("default OpenCode Go Ox Alpha binding wrong: %+v", ox)
	}
	if ox.Local() || !strings.HasPrefix(ox.EngineRoute(), "openai:") {
		t.Fatalf("OpenCode Go target should be a remote OpenAI-compatible route: %+v route=%q", ox, ox.EngineRoute())
	}
	groq, err := r.Resolve("qwen36-groq")
	if err != nil {
		t.Fatalf("resolve qwen36-groq: %v", err)
	}
	if groq.Kind != KindOpenAI || groq.Account != "july6netra_groq" || groq.BaseURL != GroqOpenAIBaseURL ||
		groq.CredEnv != GroqAPIKeyEnv || groq.UpstreamModel != GroqQwen36Model ||
		groq.RequestsPerMinute != GroqQwen36RequestsPerMinute || groq.RequestsPerDay != GroqQwen36RequestsPerDay ||
		groq.TokensPerMinute != GroqQwen36TokensPerMinute || groq.TokensPerDay != GroqQwen36TokensPerDay {
		t.Fatalf("default Groq Qwen3.6 binding wrong: %+v", groq)
	}
	if groq.Local() || groq.EngineRoute() != "openai:july6netra_groq/qwen/qwen3.6-27b" {
		t.Fatalf("Groq target should be remote OpenAI-compatible route: %+v route=%q", groq, groq.EngineRoute())
	}
	compound, err := r.Resolve("groq-compound")
	if err != nil {
		t.Fatalf("resolve groq-compound: %v", err)
	}
	if compound.Kind != KindOpenAI || compound.Account != "july6netra_groq_compound" || compound.BaseURL != GroqOpenAIBaseURL ||
		compound.CredEnv != GroqAPIKeyEnv || compound.UpstreamModel != GroqCompoundModel ||
		compound.RequestsPerMinute != GroqCompoundRequestsPerMinute || compound.RequestsPerDay != GroqCompoundRequestsPerDay ||
		compound.TokensPerMinute != 0 || compound.TokensPerDay != 0 {
		t.Fatalf("default Groq Compound binding wrong: %+v", compound)
	}
	if compound.Local() || compound.EngineRoute() != "openai:july6netra_groq_compound/groq/compound" {
		t.Fatalf("Groq Compound target should be remote OpenAI-compatible route: %+v route=%q", compound, compound.EngineRoute())
	}
	dsa, err := r.Resolve("deepseek-pro-anthropic")
	if err != nil {
		t.Fatalf("resolve deepseek-pro-anthropic: %v", err)
	}
	if dsa.Kind != KindAnthropic || dsa.BaseURL != DeepSeekAnthropicBaseURL {
		t.Fatalf("DeepSeek Anthropic profile wrong: %+v", dsa)
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
	if KindBaseURL(KindDeepSeek) != DeepSeekOpenAIBaseURL {
		t.Fatalf("deepseek default base url wrong: %q", KindBaseURL(KindDeepSeek))
	}
}

func TestDefaultRosterAstraBindings(t *testing.T) {
	if GPT6AstraWindowTokens != 1_000_000 {
		t.Fatalf("GPT6AstraWindowTokens = %d, want 1000000", GPT6AstraWindowTokens)
	}
	if GPT6AstraMaxOutputTokens != 131_072 {
		t.Fatalf("GPT6AstraMaxOutputTokens = %d, want 131072", GPT6AstraMaxOutputTokens)
	}
	r := DefaultRoster()
	if err := r.Validate(); err != nil {
		t.Fatalf("DefaultRoster failed Validate: %v", err)
	}
	for _, model := range []string{"astra-gpt-6", "astra gpt 6", "gpt-6 astra"} {
		tg, err := r.Resolve(model)
		if err != nil {
			t.Fatalf("resolve %q: %v", model, err)
		}
		if tg.Account != "openai-personal" {
			t.Errorf("model %q account = %q, want openai-personal", model, tg.Account)
		}
		if tg.UpstreamModel != GPT6AstraModel {
			t.Errorf("model %q upstream = %q, want %q", model, tg.UpstreamModel, GPT6AstraModel)
		}
		if tg.Kind != KindOpenAI {
			t.Errorf("model %q kind = %q, want %q", model, tg.Kind, KindOpenAI)
		}
	}
}
