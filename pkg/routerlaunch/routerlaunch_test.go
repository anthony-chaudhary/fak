package routerlaunch

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// routerlaunch_test.go — hermetic tests: no network beyond httptest, no host
// config. Every home-relative read is pinned to a temp dir via both the
// process env (USERPROFILE/HOME) and the userHomeFunc seam; provider names are
// synthetic.

// isolateHome points every home-relative read at dir: the process env
// (USERPROFILE on Windows, HOME elsewhere) and the userHomeFunc seam, so no
// test can reach host config.
func isolateHome(t *testing.T, dir string) {
	t.Helper()
	t.Setenv("USERPROFILE", dir)
	t.Setenv("HOME", dir)
	orig := userHomeFunc
	userHomeFunc = func() (string, error) { return dir, nil }
	t.Cleanup(func() { userHomeFunc = orig })
}

// stubAdvertisedFetch overrides the /v1/models fetch seam so
// PreferredDefaultBackendModel never dials a router.
func stubAdvertisedFetch(t *testing.T, fn func(origin, bearer string, timeout time.Duration) []string) {
	t.Helper()
	orig := advertisedFetch
	advertisedFetch = fn
	t.Cleanup(func() { advertisedFetch = orig })
}

func writeUserFile(t *testing.T, home, rel string, content []byte) {
	t.Helper()
	p := filepath.Join(home, rel)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, content, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestResolveGatewayCredentialOrder(t *testing.T) {
	home := t.TempDir()
	isolateHome(t, home)

	t.Run("first env wins", func(t *testing.T) {
		t.Setenv("RL_TEST_KEY_A", "alpha")
		t.Setenv("RL_TEST_KEY_B", "beta")
		if got := ResolveGatewayCredential("RL_TEST_KEY_A", "RL_TEST_KEY_B"); got != "alpha" {
			t.Fatalf("got %q, want alpha", got)
		}
	})
	t.Run("second env when first empty", func(t *testing.T) {
		t.Setenv("RL_TEST_KEY_A", "")
		t.Setenv("RL_TEST_KEY_B", "beta")
		if got := ResolveGatewayCredential("RL_TEST_KEY_A", "RL_TEST_KEY_B"); got != "beta" {
			t.Fatalf("got %q, want beta", got)
		}
	})
	t.Run("file fallback trims contents", func(t *testing.T) {
		t.Setenv("RL_TEST_KEY_A", "")
		t.Setenv("RL_TEST_KEY_B", "")
		writeUserFile(t, home, UserGatewayKeyFile, []byte("file-key\n"))
		if got := ResolveGatewayCredential("RL_TEST_KEY_A", "RL_TEST_KEY_B"); got != "file-key" {
			t.Fatalf("got %q, want file-key", got)
		}
	})
	t.Run("missing file yields empty", func(t *testing.T) {
		empty := t.TempDir()
		isolateHome(t, empty)
		if got := ResolveGatewayCredential("RL_TEST_KEY_A", "RL_TEST_KEY_B"); got != "" {
			t.Fatalf("got %q, want empty", got)
		}
	})
}

func TestAdvertisedModelsListsIDsInOrderWithBearer(t *testing.T) {
	var sawAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		io.WriteString(w, `{"object":"list","data":[{"id":"testprov.large"},{"id":"testprov.medium"}]}`)
	}))
	defer srv.Close()

	got := AdvertisedModels(srv.URL, "test-bearer", time.Second)
	want := []string{"testprov.large", "testprov.medium"}
	if len(got) != len(want) {
		t.Fatalf("got %d ids %v, want %v", len(got), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ids[%d] = %q, want %q (order must be preserved)", i, got[i], want[i])
		}
	}
	if sawAuth != "Bearer test-bearer" {
		t.Errorf("Authorization = %q, want Bearer test-bearer", sawAuth)
	}
}

func TestAdvertisedModelsOmitsBearerWhenEmpty(t *testing.T) {
	var sawAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		io.WriteString(w, `{"data":[{"id":"testprov.large"}]}`)
	}))
	defer srv.Close()

	if got := AdvertisedModels(srv.URL, "", time.Second); len(got) != 1 || got[0] != "testprov.large" {
		t.Fatalf("got %v, want [testprov.large]", got)
	}
	if sawAuth != "" {
		t.Errorf("Authorization = %q, want absent when bearer is empty", sawAuth)
	}
}

func TestAdvertisedModelsNormalizesOrigin(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, `{"data":[{"id":"testprov.large"}]}`)
	}))
	defer srv.Close()

	if got := AdvertisedModels(srv.URL+"/v1/", "", time.Second); len(got) != 1 || got[0] != "testprov.large" {
		t.Fatalf("trailing /v1/ origin: got %v, want [testprov.large]", got)
	}
}

func TestAdvertisedModelsFailureShapesYieldNil(t *testing.T) {
	t.Run("401", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}))
		defer srv.Close()
		if got := AdvertisedModels(srv.URL, "test-bearer", time.Second); got != nil {
			t.Fatalf("401: got %v, want nil", got)
		}
	})
	t.Run("decode error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			io.WriteString(w, `not json`)
		}))
		defer srv.Close()
		if got := AdvertisedModels(srv.URL, "", time.Second); got != nil {
			t.Fatalf("bad body: got %v, want nil", got)
		}
	})
	t.Run("redirect not followed", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/elsewhere" {
				io.WriteString(w, `{"data":[{"id":"testprov.large"}]}`)
				return
			}
			http.Redirect(w, r, "/elsewhere", http.StatusFound)
		}))
		defer srv.Close()
		if got := AdvertisedModels(srv.URL, "test-bearer", time.Second); got != nil {
			t.Fatalf("redirect: got %v, want nil (no redirects)", got)
		}
	})
	t.Run("unreachable", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
		url := srv.URL
		srv.Close()
		if got := AdvertisedModels(url, "", 500*time.Millisecond); got != nil {
			t.Fatalf("unreachable: got %v, want nil", got)
		}
	})
}

func TestUserRoutingTablePresentAbsentMalformedHome(t *testing.T) {
	t.Run("present", func(t *testing.T) {
		home := t.TempDir()
		isolateHome(t, home)
		content := []byte(`{"default_backend":"testprov"}`)
		writeUserFile(t, home, ".fak/router/routing.json", content)
		got, err := UserRoutingTable()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if string(got) != string(content) {
			t.Fatalf("got %s, want %s", got, content)
		}
	})
	t.Run("absent", func(t *testing.T) {
		isolateHome(t, t.TempDir())
		if _, err := UserRoutingTable(); err == nil {
			t.Fatal("expected error for missing routing.json")
		}
	})
	t.Run("malformed home", func(t *testing.T) {
		orig := userHomeFunc
		userHomeFunc = func() (string, error) { return "", errors.New("no home") }
		t.Cleanup(func() { userHomeFunc = orig })
		if _, err := UserRoutingTable(); err == nil {
			t.Fatal("expected error for failing home lookup")
		}
	})
}

const defaultBackendTable = `{
  "default_backend": "testprov",
  "routes": [
    {"model": "other.model", "backend": "other"},
    {"model": "testprov.large", "aliases": ["big"], "backend": "testprov"},
    {"model": "testprov.small", "backend": "testprov"}
  ]
}`

func TestDefaultBackendModelPrefersDefaultBackendRoute(t *testing.T) {
	got, ok := DefaultBackendModel([]byte(defaultBackendTable), []string{"other.model", "testprov.large", "testprov.small"})
	if !ok || got != "testprov.large" {
		t.Fatalf("got (%q, %v), want (testprov.large, true): the default-backend route must win over earlier advertised ids", got, ok)
	}
}

func TestDefaultBackendModelMatchesAlias(t *testing.T) {
	got, ok := DefaultBackendModel([]byte(defaultBackendTable), []string{"big"})
	if !ok || got != "big" {
		t.Fatalf("got (%q, %v), want (big, true): alias match", got, ok)
	}
}

func TestDefaultBackendModelNoMatchingRoute(t *testing.T) {
	if got, ok := DefaultBackendModel([]byte(defaultBackendTable), []string{"testprov.unknown"}); ok {
		t.Fatalf("got (%q, true), want !ok: no advertised id routes to the default backend", got)
	}
}

func TestDefaultBackendModelNoRouteOnDefaultBackend(t *testing.T) {
	table := `{"default_backend":"testprov","routes":[{"model":"other.model","backend":"other"}]}`
	if got, ok := DefaultBackendModel([]byte(table), []string{"other.model"}); ok {
		t.Fatalf("got (%q, true), want !ok: no route declares the default backend", got)
	}
}

func TestDefaultBackendModelMalformedJSON(t *testing.T) {
	if got, ok := DefaultBackendModel([]byte(`{nope`), []string{"testprov.large"}); ok {
		t.Fatalf("got (%q, true), want !ok: malformed table", got)
	}
}

func TestDefaultBackendModelEmptyAdvertised(t *testing.T) {
	if got, ok := DefaultBackendModel([]byte(defaultBackendTable), nil); ok {
		t.Fatalf("got (%q, true), want !ok: empty advertised list", got)
	}
}

func TestDefaultBackendModelMissingDefaultBackend(t *testing.T) {
	table := `{"routes":[{"model":"testprov.large","backend":"testprov"}]}`
	if got, ok := DefaultBackendModel([]byte(table), []string{"testprov.large"}); ok {
		t.Fatalf("got (%q, true), want !ok: table declares no default_backend", got)
	}
}

func TestPreferredDefaultBackendModelHappyPath(t *testing.T) {
	home := t.TempDir()
	isolateHome(t, home)
	writeUserFile(t, home, ".fak/router/routing.json", []byte(defaultBackendTable))
	writeUserFile(t, home, UserGatewayKeyFile, []byte("file-gw-key\n"))
	t.Setenv("FAK_CLAUDE_ROUTER_KEY", "")
	t.Setenv(EnvRouterGatewayKey, "")

	var gotOrigin, gotBearer string
	stubAdvertisedFetch(t, func(origin, bearer string, _ time.Duration) []string {
		gotOrigin, gotBearer = origin, bearer
		return []string{"testprov.medium", "testprov.large"}
	})

	if got := PreferredDefaultBackendModel("http://127.0.0.1:18101", time.Second); got != "testprov.large" {
		t.Fatalf("got %q, want testprov.large", got)
	}
	if gotOrigin != "http://127.0.0.1:18101" {
		t.Errorf("origin = %q, want pass-through", gotOrigin)
	}
	if gotBearer != "file-gw-key" {
		t.Errorf("bearer = %q, want the key-file credential", gotBearer)
	}
}

func TestPreferredDefaultBackendModelEnvCredentialWins(t *testing.T) {
	home := t.TempDir()
	isolateHome(t, home)
	writeUserFile(t, home, ".fak/router/routing.json", []byte(defaultBackendTable))
	writeUserFile(t, home, UserGatewayKeyFile, []byte("file-gw-key\n"))
	t.Setenv("FAK_CLAUDE_ROUTER_KEY", "router-key")
	t.Setenv(EnvRouterGatewayKey, "gw-key")

	var gotBearer string
	stubAdvertisedFetch(t, func(_, bearer string, _ time.Duration) []string {
		gotBearer = bearer
		return []string{"testprov.large"}
	})

	if got := PreferredDefaultBackendModel("http://127.0.0.1:18101", time.Second); got != "testprov.large" {
		t.Fatalf("got %q, want testprov.large", got)
	}
	if gotBearer != "router-key" {
		t.Errorf("bearer = %q, want FAK_CLAUDE_ROUTER_KEY ahead of %s", gotBearer, EnvRouterGatewayKey)
	}
}

func TestPreferredDefaultBackendModelFailingStepsYieldEmpty(t *testing.T) {
	home := t.TempDir()
	isolateHome(t, home)
	writeUserFile(t, home, ".fak/router/routing.json", []byte(defaultBackendTable))
	t.Setenv("FAK_CLAUDE_ROUTER_KEY", "")
	t.Setenv(EnvRouterGatewayKey, "")

	t.Run("no advertised models", func(t *testing.T) {
		stubAdvertisedFetch(t, func(string, string, time.Duration) []string { return nil })
		if got := PreferredDefaultBackendModel("http://127.0.0.1:18101", time.Second); got != "" {
			t.Fatalf("got %q, want empty", got)
		}
	})
	t.Run("routing table missing", func(t *testing.T) {
		isolateHome(t, t.TempDir())
		stubAdvertisedFetch(t, func(string, string, time.Duration) []string { return []string{"testprov.large"} })
		if got := PreferredDefaultBackendModel("http://127.0.0.1:18101", time.Second); got != "" {
			t.Fatalf("got %q, want empty", got)
		}
	})
	t.Run("routing table malformed", func(t *testing.T) {
		bad := t.TempDir()
		isolateHome(t, bad)
		writeUserFile(t, bad, ".fak/router/routing.json", []byte(`{nope`))
		stubAdvertisedFetch(t, func(string, string, time.Duration) []string { return []string{"testprov.large"} })
		if got := PreferredDefaultBackendModel("http://127.0.0.1:18101", time.Second); got != "" {
			t.Fatalf("got %q, want empty", got)
		}
	})
	t.Run("no advertised id on default backend", func(t *testing.T) {
		stubAdvertisedFetch(t, func(string, string, time.Duration) []string { return []string{"other.model"} })
		if got := PreferredDefaultBackendModel("http://127.0.0.1:18101", time.Second); got != "" {
			t.Fatalf("got %q, want empty", got)
		}
	})
}
