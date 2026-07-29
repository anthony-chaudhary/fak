package gateway

// keyset_parse_test.go — the operator-facing keyset spec contract (#5332).
//
// ParseKeyPrincipals is what turns the `fak serve --key-principal PRINCIPAL=ENV_VAR`
// flag into Config.KeyPrincipals, so these cases pin the boundary an operator actually
// touches: which specs boot, which REFUSE to boot, and that a spec set which does boot
// really authenticates and attributes each tenant through withAuth.

import (
	"net/http"
	"strings"
	"testing"
)

// envOf builds a lookupEnv over a fixed map — the deterministic stand-in for os.Getenv.
func envOf(m map[string]string) func(string) string {
	return func(name string) string { return m[name] }
}

func TestParseKeyPrincipalsEmptyLeavesSingleKeyPathUnchanged(t *testing.T) {
	// The no-keyset answer must be a nil map (not an empty one): newKeyset returns a nil
	// *keyset for it, which is what keeps the RequireKey-only gate byte-for-byte.
	for name, specs := range map[string][]string{
		"nil":   nil,
		"empty": {},
	} {
		got, err := ParseKeyPrincipals(specs, envOf(nil))
		if err != nil {
			t.Fatalf("%s: unexpected error: %v", name, err)
		}
		if got != nil {
			t.Fatalf("%s: got %v, want a nil map", name, got)
		}
		if ks := newKeyset(got); ks != nil {
			t.Fatalf("%s: must yield a nil keyset, got %+v", name, ks)
		}
	}
}

func TestParseKeyPrincipalsResolvesKeysFromEnv(t *testing.T) {
	env := envOf(map[string]string{
		"ACME_KEY":   "acme-secret",
		"BETA_KEY":   "beta-secret",
		"SPACED_KEY": "  padded-secret  ",
	})
	got, err := ParseKeyPrincipals([]string{
		"acme=ACME_KEY",
		"  beta = BETA_KEY  ", // surrounding whitespace on either side is tolerated
		"gamma=SPACED_KEY",
	}, env)
	if err != nil {
		t.Fatalf("ParseKeyPrincipals: %v", err)
	}
	// The map is keyed by the SECRET and valued by the principal — never the reverse, and
	// never the env var name (that would authenticate the literal string "ACME_KEY").
	want := map[string]string{
		"acme-secret":   "acme",
		"beta-secret":   "beta",
		"padded-secret": "gamma", // trimmed exactly as newKeyset trims
	}
	if len(got) != len(want) {
		t.Fatalf("got %d bindings %v, want %d", len(got), got, len(want))
	}
	for key, principal := range want {
		if got[key] != principal {
			t.Fatalf("binding for %q = %q, want %q (full map %v)", key, got[key], principal, got)
		}
	}
}

func TestParseKeyPrincipalsAllowsRotationOfOnePrincipal(t *testing.T) {
	// Two live keys for ONE tenant is key rotation, not a collision: during the overlap
	// both the retiring and the incoming key must authenticate as the same org.
	env := envOf(map[string]string{"ACME_OLD": "old-secret", "ACME_NEW": "new-secret"})
	got, err := ParseKeyPrincipals([]string{"acme=ACME_OLD", "acme=ACME_NEW"}, env)
	if err != nil {
		t.Fatalf("rotation must be allowed, got error: %v", err)
	}
	if got["old-secret"] != "acme" || got["new-secret"] != "acme" {
		t.Fatalf("both rotation keys must map to acme, got %v", got)
	}
	ks := newKeyset(got)
	for _, key := range []string{"old-secret", "new-secret"} {
		if p, ok := ks.lookup(key); !ok || p != "acme" {
			t.Fatalf("lookup(%q) = (%q,%v), want (acme,true)", key, p, ok)
		}
	}
}

func TestParseKeyPrincipalsFailsClosed(t *testing.T) {
	// Every one of these must REFUSE to produce a keyset. A silently-dropped binding is
	// the failure this guards: the operator believes the tenant's key is armed, the
	// gateway 401s it, and nothing says why.
	env := envOf(map[string]string{
		"ACME_KEY":  "acme-secret",
		"DUP_KEY":   "acme-secret", // byte-identical to ACME_KEY
		"EMPTY_KEY": "",
		"BLANK_KEY": "   ",
	})
	cases := []struct {
		name  string
		specs []string
		want  string // substring the message must carry, so the refusal is actionable
	}{
		{"no separator", []string{"acme"}, "PRINCIPAL=ENV_VAR"},
		{"empty principal", []string{"=ACME_KEY"}, "PRINCIPAL=ENV_VAR"},
		{"empty env name", []string{"acme="}, "PRINCIPAL=ENV_VAR"},
		{"blank both", []string{"   =   "}, "PRINCIPAL=ENV_VAR"},
		{"unset env var", []string{"acme=MISSING_KEY"}, "unset or empty"},
		{"empty env var", []string{"acme=EMPTY_KEY"}, "unset or empty"},
		{"whitespace-only env var", []string{"acme=BLANK_KEY"}, "unset or empty"},
		{"same env var twice", []string{"acme=ACME_KEY", "beta=ACME_KEY"}, "two tenants"},
		{"identical key value", []string{"acme=ACME_KEY", "beta=DUP_KEY"}, "two tenants"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ParseKeyPrincipals(tc.specs, env)
			if err == nil {
				t.Fatalf("must fail closed, got map %v and no error", got)
			}
			if got != nil {
				t.Fatalf("a refused parse must return no map, got %v", got)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q must mention %q so the operator can fix it", err, tc.want)
			}
		})
	}
}

func TestParseKeyPrincipalsFeedsWithAuthAttribution(t *testing.T) {
	// End to end over the seam an operator crosses: flag-shaped specs -> resolved map ->
	// gateway.Config -> withAuth authenticates each tenant AND stamps its principal, while
	// an unknown key still 401s. This is what makes `fak serve` "accept a keyset".
	env := envOf(map[string]string{"ACME_KEY": "acme-secret", "BETA_KEY": "beta-secret"})
	keyPrincipals, err := ParseKeyPrincipals([]string{"acme=ACME_KEY", "beta=BETA_KEY"}, env)
	if err != nil {
		t.Fatalf("ParseKeyPrincipals: %v", err)
	}
	srv, err := New(Config{
		EngineID:      "mock",
		Model:         "m",
		VDSO:          true,
		KeyPrincipals: keyPrincipals,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h := srv.withAuth(principalEchoHandler())

	cases := []struct {
		name      string
		key       string
		code      int
		principal string
	}{
		{"acme key attributes acme", "acme-secret", http.StatusOK, "acme"},
		{"beta key attributes beta", "beta-secret", http.StatusOK, "beta"},
		{"env var name is not a credential", "ACME_KEY", http.StatusUnauthorized, ""},
		{"unknown key", "nope", http.StatusUnauthorized, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := serveAuth(t, h, func(r *http.Request) { r.Header.Set("X-Api-Key", tc.key) })
			if rec.Code != tc.code {
				t.Fatalf("status = %d, want %d (%s)", rec.Code, tc.code, rec.Body.String())
			}
			if tc.code == http.StatusOK && rec.Body.String() != tc.principal {
				t.Fatalf("attributed principal = %q, want %q", rec.Body.String(), tc.principal)
			}
		})
	}
}
