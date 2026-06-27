package main

import (
	"bytes"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// nodeTestRedirectConfig points nodeConfigDir at a fresh temp dir for the duration of a
// test by overriding both the APPDATA (Windows) and HOME (Unix/macOS) env the config
// resolver keys off. Returns the resolved fak config dir so a test can assert on
// node.json directly. t.Setenv restores the prior values on cleanup.
func nodeTestRedirectConfig(t *testing.T) string {
	t.Helper()
	tmp := t.TempDir()
	t.Setenv("APPDATA", tmp)
	t.Setenv("HOME", tmp)
	dir, err := nodeConfigDir()
	if err != nil {
		t.Fatalf("nodeConfigDir after redirect: %v", err)
	}
	return dir
}

// roundTripFunc adapts a function to http.RoundTripper so a test can stand in for the
// network behind nodeHTTPClient with no real socket.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// nodeTestSwapTransport replaces nodeHTTPClient's Transport for the test and restores it
// after, mirroring the package-var swap pattern used in commit_test.go.
func nodeTestSwapTransport(t *testing.T, rt http.RoundTripper) {
	t.Helper()
	prev := nodeHTTPClient.Transport
	nodeHTTPClient.Transport = rt
	t.Cleanup(func() { nodeHTTPClient.Transport = prev })
}

func TestRunNode_dispatchErrors(t *testing.T) {
	t.Run("no args -> 2", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runNode(&out, &errb, nil); code != 2 {
			t.Fatalf("no args should exit 2, got %d (stderr: %s)", code, errb.String())
		}
		if !strings.Contains(errb.String(), "install|status|use|run|forget") {
			t.Fatalf("usage should list subcommands, got: %s", errb.String())
		}
	})
	t.Run("unknown subcommand -> 2", func(t *testing.T) {
		var out, errb bytes.Buffer
		if code := runNode(&out, &errb, []string{"bogus"}); code != 2 {
			t.Fatalf("unknown subcommand should exit 2, got %d", code)
		}
		if !strings.Contains(errb.String(), "unknown subcommand") {
			t.Fatalf("stderr should name the bad subcommand, got: %s", errb.String())
		}
	})
}

func TestNodeUseForgetRoundTrip(t *testing.T) {
	dir := nodeTestRedirectConfig(t)
	var out, errb bytes.Buffer

	// --no-check so we don't probe a real network; key carried through.
	if code := runNode(&out, &errb, []string{"use", "host:9000", "--key", "k", "--no-check"}); code != 0 {
		t.Fatalf("use should exit 0, got %d (stderr: %s)", code, errb.String())
	}
	cfgPath := filepath.Join(dir, "node.json")
	if _, err := os.Stat(cfgPath); err != nil {
		t.Fatalf("node.json should exist at %s: %v", cfgPath, err)
	}
	cfg, err := nodeReadCfg()
	if err != nil {
		t.Fatalf("nodeReadCfg: %v", err)
	}
	if cfg.URL != "http://host:9000" || cfg.Key != "k" {
		t.Fatalf("round-trip mismatch: got %+v, want {http://host:9000 k}", cfg)
	}
	// Both export lines should be printed (bash + PowerShell).
	if !strings.Contains(out.String(), "ANTHROPIC_BASE_URL") || !strings.Contains(out.String(), "ANTHROPIC_API_KEY") {
		t.Fatalf("use should print both export vars, got: %s", out.String())
	}

	// forget removes it; a second forget reports nothing to do.
	out.Reset()
	errb.Reset()
	if code := runNode(&out, &errb, []string{"forget"}); code != 0 {
		t.Fatalf("forget should exit 0, got %d", code)
	}
	if _, err := os.Stat(cfgPath); !os.IsNotExist(err) {
		t.Fatalf("node.json should be gone after forget, stat err = %v", err)
	}
	out.Reset()
	if code := runNode(&out, &errb, []string{"forget"}); code != 0 {
		t.Fatalf("second forget should exit 0, got %d", code)
	}
	if !strings.Contains(out.String(), "no node config") {
		t.Fatalf("second forget should say nothing to forget, got: %s", out.String())
	}
}

func TestNodeUseURLNormalization(t *testing.T) {
	cases := []struct {
		arg  string
		want string
	}{
		{"host", "http://host:8080"},       // bare host -> scheme + default port
		{"host:1234", "http://host:1234"},  // explicit port kept
		{"https://h", "https://h"},         // https scheme preserved, no port appended
		{"https://h:443", "https://h:443"}, // https with explicit port kept
		{"http://h:9", "http://h:9"},       // explicit http kept as-is
	}
	for _, c := range cases {
		t.Run(c.arg, func(t *testing.T) {
			nodeTestRedirectConfig(t)
			var out, errb bytes.Buffer
			if code := runNode(&out, &errb, []string{"use", c.arg, "--no-check"}); code != 0 {
				t.Fatalf("use %q exit %d (stderr: %s)", c.arg, code, errb.String())
			}
			cfg, err := nodeReadCfg()
			if err != nil {
				t.Fatalf("nodeReadCfg: %v", err)
			}
			if cfg.URL != c.want {
				t.Fatalf("use %q -> %q, want %q", c.arg, cfg.URL, c.want)
			}
		})
	}
}

func TestNodeUseFlagOrder(t *testing.T) {
	// Regression: Go's flag package stops at the first non-flag token, so flags AFTER the
	// positional HOST must still be parsed. Both orders must record the key.
	for _, argv := range [][]string{
		{"use", "host:9000", "--key", "k", "--no-check"},
		{"use", "--key", "k", "--no-check", "host:9000"},
		{"use", "--key", "k", "host:9000", "--no-check"},
	} {
		t.Run(strings.Join(argv, "_"), func(t *testing.T) {
			nodeTestRedirectConfig(t)
			var out, errb bytes.Buffer
			if code := runNode(&out, &errb, argv); code != 0 {
				t.Fatalf("use %v exit %d (stderr: %s)", argv, code, errb.String())
			}
			cfg, err := nodeReadCfg()
			if err != nil {
				t.Fatalf("nodeReadCfg: %v", err)
			}
			if cfg.URL != "http://host:9000" || cfg.Key != "k" {
				t.Fatalf("use %v -> %+v, want {http://host:9000 k}", argv, cfg)
			}
		})
	}
}

func TestNodeChildEnv(t *testing.T) {
	withKey := nodeChildEnv(nodeCfg{URL: "http://h:8080", Key: "secret"})
	if len(withKey) != 2 {
		t.Fatalf("with key: want 2 pairs, got %d (%v)", len(withKey), withKey)
	}
	if withKey[0] != [2]string{"ANTHROPIC_BASE_URL", "http://h:8080"} {
		t.Fatalf("with key: base-url pair wrong: %v", withKey[0])
	}
	if withKey[1] != [2]string{"ANTHROPIC_API_KEY", "secret"} {
		t.Fatalf("with key: api-key pair wrong: %v", withKey[1])
	}

	noKey := nodeChildEnv(nodeCfg{URL: "http://h:8080"})
	if len(noKey) != 1 {
		t.Fatalf("no key: want 1 pair, got %d (%v)", len(noKey), noKey)
	}
	if noKey[0] != [2]string{"ANTHROPIC_BASE_URL", "http://h:8080"} {
		t.Fatalf("no key: base-url pair wrong: %v", noKey[0])
	}
}

func TestNodeRun_noConfig(t *testing.T) {
	nodeTestRedirectConfig(t) // empty temp dir => no node.json
	var out, errb bytes.Buffer
	if code := runNode(&out, &errb, []string{"run", "--", "true"}); code != 2 {
		t.Fatalf("run without config should exit 2, got %d (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "fak node use") {
		t.Fatalf("run-without-config should point at `fak node use`, got: %s", errb.String())
	}
}

func TestNodeRun_noCommand(t *testing.T) {
	dir := nodeTestRedirectConfig(t)
	// Seed a config so the missing-command path (not the missing-config path) is hit.
	if err := nodeWriteCfg(nodeCfg{URL: "http://h:8080"}); err != nil {
		t.Fatalf("seed config: %v", err)
	}
	_ = dir
	var out, errb bytes.Buffer
	if code := runNode(&out, &errb, []string{"run", "--"}); code != 2 {
		t.Fatalf("run with bare -- and no command should exit 2, got %d", code)
	}
	if !strings.Contains(errb.String(), "no command given") {
		t.Fatalf("should report missing command, got: %s", errb.String())
	}
}

func TestNodeProbeHealth(t *testing.T) {
	t.Run("2xx -> ok", func(t *testing.T) {
		nodeTestSwapTransport(t, roundTripFunc(func(r *http.Request) (*http.Response, error) {
			if !strings.HasSuffix(r.URL.Path, "/healthz") {
				t.Fatalf("probe should hit /healthz, got %s", r.URL.Path)
			}
			return &http.Response{
				StatusCode: 200,
				Status:     "200 OK",
				Body:       http.NoBody,
				Header:     make(http.Header),
			}, nil
		}))
		status, ok := nodeProbeHealth("http://node:8080")
		if !ok || !strings.Contains(status, "200") {
			t.Fatalf("2xx should be ok, got ok=%v status=%q", ok, status)
		}
	})
	t.Run("dial error -> not ok", func(t *testing.T) {
		nodeTestSwapTransport(t, roundTripFunc(func(r *http.Request) (*http.Response, error) {
			return nil, &nodeDialErr{}
		}))
		status, ok := nodeProbeHealth("http://node:8080")
		if ok {
			t.Fatalf("dial error should not be ok, got ok=true status=%q", status)
		}
		if status == "" {
			t.Fatalf("dial error should surface the error text as status")
		}
	})
}

// nodeDialErr is a stand-in connection error for the probe's failure path.
type nodeDialErr struct{}

func (*nodeDialErr) Error() string { return "dial tcp: connection refused" }
