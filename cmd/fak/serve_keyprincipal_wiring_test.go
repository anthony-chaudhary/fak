package main

// serve_keyprincipal_wiring_test.go — proves `fak serve --key-principal` is REACHED, not
// merely registered (#5332).
//
// The flag shipped parse-only: serveFlags carried the field, fs.Var registered it, and
// nothing ever read it — so gateway.Config.KeyPrincipals was always nil, newKeyset always
// returned nil, and every keyset lookup in withAuth was a nil no-op. That is worse than an
// inert flag. With no keyset matched, principalFor (internal/gateway/http_fak_endpoints.go)
// falls through to the CALLER-supplied X-Fak-Principal header and body field, and that value
// is what modelroute Target.Admits adjudicates — so the Account.Principals allowlist was
// satisfiable by any caller willing to assert a header, and the flag's help text promised an
// authentication the binary never performed.
//
// These cases pin the missing link from both ends: the SOURCE (serve.go resolves the specs
// and sets the Config field, so the wiring cannot silently drop out again) and the BEHAVIOR
// (a resolved map really arms auth on a live gateway.New server, while passing no flag
// leaves that server exactly as unauthenticated as it was before).

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// keyPrincipalEnv builds a lookupEnv over a fixed map — the deterministic stand-in for
// os.Getenv, so no case depends on the ambient environment of the machine running it.
func keyPrincipalEnv(m map[string]string) func(string) string {
	return func(name string) string { return m[name] }
}

// TestServeKeyPrincipalsDefaultLeavesSingleBearerPathUnchanged pins the opt-in guarantee:
// an operator who passes no --key-principal gets the pre-#5332 gateway. The flag's spec
// list stays nil, the resolver returns a nil map WITHOUT consulting the environment at all,
// and a server built from that nil map still serves an unauthenticated request.
func TestServeKeyPrincipalsDefaultLeavesSingleBearerPathUnchanged(t *testing.T) {
	fs, sf := newServeFlagSet()
	if !parseFlags(fs, []string{"--require-key-env", "SOME_BEARER_ENV"}) {
		t.Fatal("parsing a --key-principal-free argv must succeed")
	}
	if specs := sf.keyPrincipal.Values(); specs != nil {
		t.Fatalf("no --key-principal must leave the spec list nil, got %v", specs)
	}

	// A resolver that reads ANY env var on the default path would be a behavior change all
	// by itself (a boot that now depends on ambient state), so make a read fatal.
	env := func(name string) string {
		t.Fatalf("the default path must not read the environment for a keyset; read %q", name)
		return ""
	}
	got, ok := serveKeyPrincipals(sf.keyPrincipal.Values(), env, io.Discard)
	if !ok {
		t.Fatal("no --key-principal must resolve cleanly, not refuse to boot")
	}
	if got != nil {
		t.Fatalf("no --key-principal must yield a nil map (newKeyset's only no-keyset answer), got %v", got)
	}

	// End to end: that nil map must leave withAuth's gate closed, i.e. no auth at all, which
	// is what "byte-for-byte the RequireKey-only path" means for an operator with no bearer.
	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "m", VDSO: true, KeyPrincipals: got})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	if code := keyPrincipalProbe(t, srv, ""); code != http.StatusOK {
		t.Fatalf("an unconfigured keyset must leave the gateway unauthenticated; got %d, want %d", code, http.StatusOK)
	}
}

// TestServeKeyPrincipalsArmsTheGatewayKeyset is the positive half: flag-shaped specs go in,
// a real gateway comes out that AUTHENTICATES the bound tenant key and rejects everything
// else. Before the wiring landed this map never reached gateway.New, so the same server
// answered every one of these the same way.
func TestServeKeyPrincipalsArmsTheGatewayKeyset(t *testing.T) {
	fs, sf := newServeFlagSet()
	if !parseFlags(fs, []string{"--key-principal", "acme=ACME_KEY", "--key-principal", "beta=BETA_KEY"}) {
		t.Fatal("parsing two --key-principal specs must succeed")
	}
	env := keyPrincipalEnv(map[string]string{"ACME_KEY": "acme-secret", "BETA_KEY": "beta-secret"})
	got, ok := serveKeyPrincipals(sf.keyPrincipal.Values(), env, io.Discard)
	if !ok {
		t.Fatal("two well-formed specs with both env vars set must resolve")
	}
	if got["acme-secret"] != "acme" || got["beta-secret"] != "beta" {
		t.Fatalf("resolved bindings = %v, want acme-secret->acme and beta-secret->beta", got)
	}

	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "m", VDSO: true, KeyPrincipals: got})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	for _, tc := range []struct {
		name string
		key  string
		code int
	}{
		{"a bound tenant key authenticates", "acme-secret", http.StatusOK},
		{"the second tenant key authenticates too", "beta-secret", http.StatusOK},
		{"no credential is refused once a keyset is armed", "", http.StatusUnauthorized},
		{"the env var NAME is not a credential", "ACME_KEY", http.StatusUnauthorized},
		{"an unknown key is refused", "nope", http.StatusUnauthorized},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if code := keyPrincipalProbe(t, srv, tc.key); code != tc.code {
				t.Fatalf("status = %d, want %d", code, tc.code)
			}
		})
	}
}

// TestServeKeyPrincipalsRefusesToBoot pins the fail-LOUD half. Silently ignoring a bad spec
// is the dangerous answer: the operator believes a tenant is authenticated by key while the
// gateway keeps attributing that tenant from the caller-asserted X-Fak-Principal header.
func TestServeKeyPrincipalsRefusesToBoot(t *testing.T) {
	env := keyPrincipalEnv(map[string]string{"ACME_KEY": "acme-secret", "DUP_KEY": "acme-secret"})
	for _, tc := range []struct {
		name  string
		specs []string
	}{
		{"malformed spec", []string{"acme"}},
		{"unset env var", []string{"acme=MISSING_KEY"}},
		{"two tenants sharing one key", []string{"acme=ACME_KEY", "beta=DUP_KEY"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stderr strings.Builder
			got, ok := serveKeyPrincipals(tc.specs, env, &stderr)
			if ok {
				t.Fatalf("must refuse, got map %v", got)
			}
			if got != nil {
				t.Fatalf("a refused resolve must hand back no map, got %v", got)
			}
			// The refusal has to name the flag and say what it is protecting, or an operator
			// reading a dead serve log cannot tell a typo from a broken gateway.
			for _, want := range []string{"fak serve: --key-principal", "X-Fak-Principal"} {
				if !strings.Contains(stderr.String(), want) {
					t.Fatalf("refusal %q must mention %q", stderr.String(), want)
				}
			}
		})
	}
}

// TestServeSourceWiresKeyPrincipalsIntoGatewayNew is the drift guard on the link itself.
// The behavioral tests above call serveKeyPrincipals directly, so they would still pass if
// buildGateway quietly stopped calling it — the exact regression that produced this bug.
// This one reads the real serve.go and requires the whole chain: flag registered, specs
// resolved from the flag through os.Getenv, and the resolved map assigned into the
// gateway.New(gateway.Config{...}) literal. `fak serve-wiring --check` cannot catch a
// MISSING assignment (its coverage walk only visits fields serve.go already sets), so the
// tree needs this assertion spelled out.
func TestServeSourceWiresKeyPrincipalsIntoGatewayNew(t *testing.T) {
	root := repoRootFromTest(t)
	body, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	src := string(body)

	for _, want := range []string{
		`fs.Var(&sf.keyPrincipal, "key-principal"`,
		"serveKeyPrincipals(sf.keyPrincipal.Values(), os.Getenv, os.Stderr)",
		"gateway.ParseKeyPrincipals(",
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("serve.go must contain %q — --key-principal is parse-only without it", want)
		}
	}
	if !serveConfigAssignments(src)["KeyPrincipals"] {
		t.Fatal("serve.go's gateway.New(gateway.Config{...}) literal must set KeyPrincipals; without it newKeyset gets nil and every --key-principal binding is dropped on the floor")
	}
	if strings.Index(src, "serveKeyPrincipals(sf.keyPrincipal") > strings.Index(src, "srv, err := gateway.New") {
		t.Fatal("the --key-principal specs must be resolved BEFORE gateway.New consumes them")
	}
	// The refusal must be terminal: a resolve failure that fell through would boot a gateway
	// the operator believes is key-authenticated.
	if !strings.Contains(src, "if !keysetOK {\n\t\tos.Exit(2)\n\t}") {
		t.Fatal("a --key-principal resolve failure must exit(2), not fall through into gateway.New")
	}
}

// keyPrincipalProbe drives one authenticated request through the server's real Handler and
// returns its status. /metrics is the cheapest non-exempt route: httptest.NewRequest gives
// the request a documentation-range RemoteAddr rather than a loopback one, so authExempt's
// read-only loopback exemption does not apply and withAuth's credential check really runs.
func keyPrincipalProbe(t *testing.T, srv *gateway.Server, key string) int {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	if key != "" {
		req.Header.Set("X-Api-Key", key)
	}
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	return rec.Code
}
