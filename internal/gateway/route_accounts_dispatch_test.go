package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// route_accounts_dispatch_test.go — the served-gateway witnesses for #2528: a routed
// abstract model id is BOUND through a modelroute.Roster to its account-resolved
// Target.EngineRoute() BEFORE Submit, so the kernel dispatches to the account's engine
// and the residency PDP adjudicates the account-resolved remote/local route. These are
// the gateway half of the epic-#595 spine (fak route --accounts already ships the pure
// resolver + the no-spend api-host probes); this file proves the LIVE dispatch binding.

// acctServer wires an isolated chain with the real residency floor, a routing manifest,
// AND an account roster, registering one engine per account-resolved EngineRoute so a
// resolved dispatch reaches a controllable marker. Mirrors routeServer/ensembleServer but
// for the roster-bound path. Not parallel-safe (mutates the global registry).
func acctServer(t *testing.T, m *modelroute.Manifest, roster *modelroute.Roster, engines map[string]abi.EngineDriver) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{}) // the kernel-default local engine
	for route, e := range engines {
		abi.RegisterEngine(route, e) // keyed on the ACCOUNT-resolved EngineRoute string
	}
	abi.RegisterAdjudicator(0, toolAdj{}) // allow*/deny* by tool-name prefix
	engine.RegisterResidencyGate()        // the REAL residency floor (rank 12)
	srv, err := New(Config{EngineID: "test", Model: "m", VDSO: true, RouteManifest: m, RouteAccounts: roster})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// openaiWorkRoster binds the abstract id "guard-a" to a REMOTE OpenAI account with a
// distinct upstream wire model. Its Target.EngineRoute() is "openai:openai-work/gpt-5.5".
func openaiWorkRoster() *modelroute.Roster {
	return &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "openai-work", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_WORK_API_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "guard-a", Account: "openai-work", UpstreamModel: "gpt-5.5"},
		},
	}
}

// A single-model route resolves THROUGH the roster: buildCall binds the account-resolved
// EngineRoute (not the bare plan member "guard-a") pre-submit, and an allowed non-sensitive
// call dispatches to the engine registered under that route with the account/kind/upstream
// recorded in the result meta (the #2528 observability acceptance, no secret values).
func TestRouteAccountsResolvesSingleModelToEngineRoute(t *testing.T) {
	const route = "openai:openai-work/gpt-5.5"
	s := acctServer(t, pickManifest("allow_run", "guard-a"), openaiWorkRoster(),
		map[string]abi.EngineDriver{route: markerEngine{"acct-openai"}})
	ctx := context.Background()

	tc, err := s.buildCall(ctx, "allow_run", `{"x":1}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != route {
		t.Fatalf("account-resolved route not bound pre-submit: got %q, want %q", tc.Engine, route)
	}

	wv, env, err := s.syscall(ctx, "allow_run", `{"x":1}`, false, "", "")
	if err != nil {
		t.Fatalf("syscall: %v", err)
	}
	if wv.Kind != "ALLOW" {
		t.Fatalf("verdict = %+v, want ALLOW", wv)
	}
	if env == nil || env.Meta["engine"] != "acct-openai" {
		t.Fatalf("call must dispatch to the account-resolved engine; meta = %v", envMeta(env))
	}
	if env.Meta["route_account"] != "openai-work" || env.Meta["route_kind"] != "openai" ||
		env.Meta["route_upstream"] != "gpt-5.5" || env.Meta["route_engine"] != route {
		t.Fatalf("route account observability meta = %v", env.Meta)
	}
	if env.Meta["route_cred_env"] != "OPENAI_WORK_API_KEY" {
		t.Fatalf("route_cred_env should record the env-var NAME, got %v", env.Meta["route_cred_env"])
	}
}

// The observability records the credential env NAME but never the secret VALUE — the
// #2528 "no secret values" acceptance, proven with a live secret set in the environment.
func TestRouteAccountsObservabilityCarriesNoSecret(t *testing.T) {
	t.Setenv("OPENAI_WORK_API_KEY", "sk-super-secret-value")
	const route = "openai:openai-work/gpt-5.5"
	s := acctServer(t, pickManifest("allow_run", "guard-a"), openaiWorkRoster(),
		map[string]abi.EngineDriver{route: markerEngine{"acct-openai"}})

	_, env, err := s.syscall(context.Background(), "allow_run", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("syscall: %v", err)
	}
	if env == nil {
		t.Fatal("nil result envelope")
	}
	if env.Meta["route_cred_env"] != "OPENAI_WORK_API_KEY" {
		t.Fatalf("the credential env NAME is recorded (permitted); got %v", env.Meta["route_cred_env"])
	}
	for k, v := range env.Meta {
		if strings.Contains(v, "sk-super-secret-value") {
			t.Fatalf("secret VALUE leaked into meta[%q] = %q", k, v)
		}
	}
}

// An ensemble resolves EVERY member through the roster independently: member "guard-a"
// binds to a remote OpenAI account, member "guard-b" to a local account, each dispatches
// to its own account-resolved engine, and the fold stays in Plan.Members order.
func TestRouteAccountsEnsembleResolvesEachMember(t *testing.T) {
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "openai-work", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_WORK_API_KEY"},
			{ID: "local-box", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
		},
		Bindings: []modelroute.Binding{
			{Model: "guard-a", Account: "openai-work", UpstreamModel: "up-a"},
			{Model: "guard-b", Account: "local-box", UpstreamModel: "up-b"},
		},
	}
	const routeA = "openai:openai-work/up-a"
	const routeB = "local:local-box/up-b"
	s := acctServer(t, ensembleManifest("allow_ens", modelroute.ReduceConcat, "guard-a", "guard-b"), roster,
		map[string]abi.EngineDriver{
			routeA: outEngine{id: "ea", out: "AAA"},
			routeB: outEngine{id: "eb", out: "BBB"},
		})

	_, env, err := s.syscall(context.Background(), "allow_ens", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("syscall: %v", err)
	}
	if env == nil || env.Content != "AAA\nBBB" {
		t.Fatalf("ensemble concat = %q, want %q (each member resolved through the roster, member order)", envContent(env), "AAA\nBBB")
	}
	if env.Meta["ensemble_members"] != "2" {
		t.Fatalf("ensemble_members = %q, want 2", env.Meta["ensemble_members"])
	}
}

// THE load-bearing residency test: a tenant/sensitive payload routed to a REMOTE account is
// DENIED by the existing residency floor — proving the ACCOUNT-resolved route (not the bare
// plan member) is visible before adjudication. No engine need be registered; the deny lands
// before dispatch.
func TestRouteAccountsSensitiveRemoteDeniedByResidency(t *testing.T) {
	s := acctServer(t, pickManifest("fetch", "guard-a"), openaiWorkRoster(), nil)
	ctx := context.Background()

	tc, err := s.buildCall(ctx, "fetch", `{"id":7}`, true, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "openai:openai-work/gpt-5.5" {
		t.Fatalf("route precondition: Engine=%q, want the account-resolved openai route", tc.Engine)
	}
	tc.Meta["sensitivity"] = "tenant"
	_, v := s.k.Syscall(ctx, tc)
	if v.Kind != abi.VerdictDeny || v.By != "engine-residency" {
		t.Fatalf("sensitive payload -> remote account must be denied by residency, got kind=%v by=%q", v.Kind, v.By)
	}
	if v.Reason != abi.ReasonScopeCrossing {
		t.Fatalf("residency deny reason = %v, want SCOPE_CROSSING", v.Reason)
	}
}

// A routed id with no binding and no default account FAILS LOUD at buildCall — never a
// silent fallback to the default engine. The error names the id and is recoverable.
func TestRouteAccountsUnknownBindingFailsLoud(t *testing.T) {
	roster := &modelroute.Roster{
		Version:  modelroute.RosterVersion,
		Accounts: []modelroute.Account{{ID: "openai-work", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_WORK_API_KEY"}},
		// No binding for "guard-a" and no Default => an unresolvable route.
		Bindings: []modelroute.Binding{{Model: "other", Account: "openai-work"}},
	}
	s := acctServer(t, pickManifest("allow_run", "guard-a"), roster, nil)

	_, err := s.buildCall(context.Background(), "allow_run", `{}`, false, "", "")
	if err == nil {
		t.Fatal("a routed id with no binding and no default must fail loud, got nil error")
	}
	if !strings.Contains(err.Error(), "guard-a") || !strings.Contains(err.Error(), "route accounts") {
		t.Fatalf("fail-loud error should name the id and be recoverable, got: %v", err)
	}
}

// A NATIVE-provider account whose adapter is NOT wired fails LOUD at dispatch ("no engine
// registered for route") rather than silently running through the OpenAI-compatible default
// engine — the #2528 fail-loud-native acceptance. The route is bound and adjudicated (the
// residency-honest anthropic route is visible), then dispatch refuses the missing adapter.
func TestRouteAccountsNativeProviderNoAdapterFailsLoud(t *testing.T) {
	roster := &modelroute.Roster{
		Version:  modelroute.RosterVersion,
		Accounts: []modelroute.Account{{ID: "claude-sub", Kind: modelroute.KindAnthropic, CredEnv: "CLAUDE_CODE_OAUTH_TOKEN"}},
		Bindings: []modelroute.Binding{{Model: "guard-a", Account: "claude-sub", UpstreamModel: "claude-opus-4-6"}},
	}
	// Deliberately register NO engine for the anthropic route.
	s := acctServer(t, pickManifest("allow_run", "guard-a"), roster, nil)
	ctx := context.Background()

	tc, err := s.buildCall(ctx, "allow_run", `{}`, false, "", "")
	if err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if tc.Engine != "anthropic:claude-sub/claude-opus-4-6" {
		t.Fatalf("account route precondition: Engine=%q", tc.Engine)
	}

	r, v := s.k.Syscall(ctx, tc)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("non-sensitive call should ALLOW at adjudication, got %v", v.Kind)
	}
	if r == nil || r.Status != abi.StatusError {
		t.Fatalf("native provider with no wired adapter must fail loud at dispatch (no default fallback), got %+v", r)
	}
	if !strings.Contains(r.Meta["error"], "no engine registered for route") {
		t.Fatalf("dispatch error should name the missing engine route, got meta: %v", r.Meta)
	}
}
