package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

func TestGuardHostFromBase(t *testing.T) {
	cases := []struct{ in, want string }{
		{"https://api.anthropic.com", "api.anthropic.com"},
		{"https://api.openai.com/v1", "api.openai.com"},
		{"http://dgx1:8080/v1", "dgx1:8080"},
		{"dgx1:8080/v1", "dgx1:8080"},
		{"127.0.0.1:11434", "127.0.0.1:11434"},
		{"", ""},
	}
	for _, c := range cases {
		if got := guardHostFromBase(c.in); got != c.want {
			t.Errorf("guardHostFromBase(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestGuardIsLoopbackHost(t *testing.T) {
	for _, h := range []string{"127.0.0.1:11434", "localhost:1234", "127.0.0.1", "::1", "[::1]:8080"} {
		if !guardIsLoopbackHost(h) {
			t.Errorf("guardIsLoopbackHost(%q) = false, want true", h)
		}
	}
	for _, h := range []string{"api.anthropic.com", "dgx1:8080", "10.0.0.5:8080"} {
		if guardIsLoopbackHost(h) {
			t.Errorf("guardIsLoopbackHost(%q) = true, want false", h)
		}
	}
}

func TestGuardResolveServingNodes(t *testing.T) {
	// Default proxy: one serving node, the provider host, kind proxy.
	got := guardResolveServingNodes(guardEndpointNodes{provider: "anthropic", resolvedBase: "https://api.anthropic.com"})
	if len(got) != 1 || got[0].ID != "api.anthropic.com" || got[0].Kind != "proxy" || got[0].Role != "serving" {
		t.Fatalf("proxy nodes = %+v, want one anthropic proxy serving node", got)
	}

	// --remote-serve: the lab box, kind remote-serve.
	got = guardResolveServingNodes(guardEndpointNodes{provider: "openai", resolvedBase: "http://dgx1:8080/v1", remoteServe: true})
	if len(got) != 1 || got[0].ID != "dgx1:8080" || got[0].Kind != "remote-serve" {
		t.Fatalf("remote-serve nodes = %+v, want one dgx1:8080 remote-serve node", got)
	}

	// Detected local server: loopback host, kind local-server.
	got = guardResolveServingNodes(guardEndpointNodes{provider: "openai", resolvedBase: "http://127.0.0.1:11434/v1"})
	if len(got) != 1 || got[0].Kind != "local-server" {
		t.Fatalf("local nodes = %+v, want one local-server node", got)
	}

	// Pure in-kernel (--gguf): the box itself, no proxy host.
	got = guardResolveServingNodes(guardEndpointNodes{localModel: true, localAlias: "qwen2.5:7b"})
	if len(got) != 1 || got[0].ID != "in-kernel" || got[0].Kind != "in-kernel" {
		t.Fatalf("in-kernel nodes = %+v, want one in-kernel node", got)
	}

	// --gguf --alongside: both the API host and the in-kernel node.
	got = guardResolveServingNodes(guardEndpointNodes{provider: "anthropic", resolvedBase: "https://api.anthropic.com", localModel: true, localAlong: true, localAlias: "local"})
	if len(got) != 2 || got[0].Kind != "proxy" || got[1].Kind != "in-kernel" {
		t.Fatalf("alongside nodes = %+v, want a proxy + an in-kernel node", got)
	}
}

// TestNewGuardEndpointsProviderKernelNode proves the provider always yields at least the
// kernel node (this host) plus the serving node — "multiple nodes" by construction — and
// no accounts when no seat is in use (activeDirFn nil).
func TestNewGuardEndpointsProviderKernelNode(t *testing.T) {
	p := newGuardEndpointsProvider(nil, nil, guardEndpointNodes{provider: "anthropic", resolvedBase: "https://api.anthropic.com"})
	ep := p()
	if len(ep.Accounts) != 0 {
		t.Fatalf("accounts with nil activeDirFn = %+v, want none", ep.Accounts)
	}
	if len(ep.Nodes) != 2 || ep.Nodes[0].Role != "kernel" || ep.Nodes[1].Role != "serving" {
		t.Fatalf("nodes = %+v, want a kernel + a serving node", ep.Nodes)
	}
	if ep.Nodes[0].ID == "" {
		t.Fatal("kernel node id is empty, want a hostname or placeholder")
	}
}

// TestGuardResolveAccountsMarksActiveAndWalled builds a fake home with two Claude seats,
// makes one active and the other walled, and checks the roster the status area renders.
func TestGuardResolveAccountsMarksActiveAndWalled(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)        // POSIX UserHomeDir
	t.Setenv("USERPROFILE", home) // Windows UserHomeDir
	seatA := filepath.Join(home, ".claude")
	seatB := filepath.Join(home, ".claude-work")
	writeFakeSeat(t, seatA, "uuid-a", "a@x")
	writeFakeSeat(t, seatB, "uuid-b", "b@x")

	activeDir := func() string { return seatA }
	walled := func() map[string]bool { return map[string]bool{"uuid:uuid-b": true} }

	got := guardResolveAccounts(activeDir, walled)
	if len(got) != 2 {
		t.Fatalf("accounts = %+v, want 2 seats", got)
	}
	byName := map[string]gateway.SessionAccount{}
	for _, a := range got {
		byName[a.Name] = a
	}
	if a := byName["default"]; !a.Active || a.Walled {
		t.Errorf("seat 'default' = %+v, want active, not walled", a)
	}
	if a := byName["work"]; a.Active || !a.Walled {
		t.Errorf("seat 'work' = %+v, want walled, not active", a)
	}

	// Nil activeDirFn (non-subscription session) → no accounts.
	if got := guardResolveAccounts(nil, nil); got != nil {
		t.Errorf("guardResolveAccounts(nil,nil) = %+v, want nil", got)
	}
}

func TestGuardHarnessToSession(t *testing.T) {
	snap := harnessres.Snapshot{Samples: 5, GoroutinesPeak: 12}
	snap.Kernel.HaveRSS = true
	snap.Kernel.RSSBytes = 8192
	snap.Kernel.HaveNet = true
	snap.Kernel.NetRxBytes = 100
	snap.Kernel.NetTxBytes = 200
	got := guardHarnessToSession(snap)
	if got.Samples != 5 || got.KernelRSSBytes != 8192 || got.NetRxBytes != 100 || got.NetTxBytes != 200 || got.GoroutinesPeak != 12 {
		t.Fatalf("guardHarnessToSession = %+v, want the sampled axes", got)
	}
}

// writeFakeSeat writes the minimal .claude.json + .credentials.json a Discover/CanServe
// treat as a live, logged-in seat, so guardResolveAccounts sees a ready roster.
func writeFakeSeat(t *testing.T, dir, uuid, email string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	cfg := map[string]any{"oauthAccount": map[string]any{"emailAddress": email, "accountUuid": uuid}}
	b, _ := json.Marshal(cfg)
	if err := os.WriteFile(filepath.Join(dir, ".claude.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"), []byte(`{"claudeAiOauth":{"accessToken":"x","expiresAt":1}}`), 0o600); err != nil {
		t.Fatal(err)
	}
}
