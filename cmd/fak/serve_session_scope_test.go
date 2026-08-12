package main

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// seedHostRegistry writes descriptors into a registry file the way a live serve on a
// shared box would, and returns its path — the stand-in for the per-user
// <UserConfigDir>/fak/session-registry.json that every serve on the host shares.
func seedHostRegistry(t *testing.T, dir string, ids ...string) string {
	t.Helper()
	path := filepath.Join(dir, "session-registry.json")
	reg := session.NewRegistry(session.NewFileStore(path))
	now := time.Now()
	for _, id := range ids {
		if _, err := reg.RegisterWithMeta(id, "host", session.State{TraceID: id, Run: session.Running},
			session.DefaultDescriptorTTL, now, session.DescriptorMeta{}); err != nil {
			t.Fatalf("seed %s: %v", id, err)
		}
	}
	return path
}

// The #5825 witness. A serve hydrates its session table from an on-disk registry, and
// that table is what a fanned lifecycle op writes through — `--op pause --all` walks
// exactly these entries. --fleet-bus-dir scopes the BUS and nothing about this, so before
// this leaf there was no way to start a serve that could not reach the host's other
// sessions: the demo serve in the issue reported `affected=12` against peers' work.
//
// The three arms are the whole contract. The default must keep adopting the host's
// sessions (that reach is correct for a real fleet and this leaf must not change it); an
// explicit path must adopt only its own; "off" must adopt nothing at all.
func TestServeSessionRegistryScopesWhichSessionsAFanCanReach(t *testing.T) {
	peers := []string{"peer-alpha", "peer-bravo", "peer-charlie"}

	t.Run("default still adopts the host registry", func(t *testing.T) {
		shared := seedHostRegistry(t, t.TempDir(), peers...)
		t.Setenv(sessionRegistryEnv, shared)
		t.Cleanup(func() { serveSessionDurability = nil; serveSessionRegistryPath = "" })

		tbl := session.NewTable()
		var stderr bytes.Buffer
		if err := configureServeSessionDurability(tbl, "", &stderr); err != nil {
			t.Fatalf("configureServeSessionDurability() error = %v", err)
		}
		if got := len(tbl.Snapshot()); got != len(peers) {
			t.Fatalf("default hydrated %d session(s), want %d — this leaf must not change the default fleet's reach", got, len(peers))
		}
		if serveSessionRegistryPath != shared {
			t.Errorf("scope path = %q, want the resolved default %q", serveSessionRegistryPath, shared)
		}
	})

	t.Run("an explicit registry adopts only its own", func(t *testing.T) {
		shared := seedHostRegistry(t, t.TempDir(), peers...)
		t.Setenv(sessionRegistryEnv, shared)
		private := filepath.Join(t.TempDir(), "private-registry.json")
		t.Cleanup(func() { serveSessionDurability = nil; serveSessionRegistryPath = "" })

		tbl := session.NewTable()
		var stderr bytes.Buffer
		if err := configureServeSessionDurability(tbl, private, &stderr); err != nil {
			t.Fatalf("configureServeSessionDurability() error = %v", err)
		}
		if got := tbl.Snapshot(); len(got) != 0 {
			t.Fatalf("a serve scoped to its own registry adopted %d host session(s): %v — a fanned --all would write to peers", len(got), got)
		}
		if serveSessionRegistryPath != private {
			t.Errorf("scope path = %q, want %q", serveSessionRegistryPath, private)
		}
		if serveSessionDurability == nil {
			t.Error("an explicit path must still MIRROR (only 'off' disables durability)")
		}
	})

	t.Run("off adopts nothing and persists nothing", func(t *testing.T) {
		shared := seedHostRegistry(t, t.TempDir(), peers...)
		t.Setenv(sessionRegistryEnv, shared)
		t.Cleanup(func() { serveSessionDurability = nil; serveSessionRegistryPath = "" })

		tbl := session.NewTable()
		var stderr bytes.Buffer
		if err := configureServeSessionDurability(tbl, "off", &stderr); err != nil {
			t.Fatalf("configureServeSessionDurability() error = %v", err)
		}
		if got := tbl.Snapshot(); len(got) != 0 {
			t.Fatalf("--session-registry off adopted %d session(s): %v", len(got), got)
		}
		if serveSessionDurability != nil {
			t.Error("'off' must leave a pure in-memory table with no mirror")
		}
		if got := serveSessionRegistryScopeLabel(); !strings.Contains(got, "in-memory") {
			t.Errorf("scope label = %q, want it to say the table is in-memory only", got)
		}
	})
}

// The flag has to exist on the serve surface and default to empty, or the plumbing above
// stays unreachable from a command line — which is the actual #5825 defect (the path
// argument was hard-coded to "" at the single call site).
func TestServeFlagSetDefinesSessionRegistry(t *testing.T) {
	fs, sf := newServeFlagSet()

	f := fs.Lookup("session-registry")
	if f == nil {
		t.Fatal("fak serve defines no --session-registry flag: the session table's scope is unreachable from the CLI")
	}
	if f.DefValue != "" {
		t.Errorf("--session-registry default = %q, want empty so an unflagged serve keeps today's shared-registry behaviour", f.DefValue)
	}
	if sf.sessionRegistry == nil {
		t.Fatal("serveFlags.sessionRegistry is nil: the flag is defined but never read")
	}

	// The trap this leaf exists to close is that --fleet-bus-dir READS like a sandbox.
	// If the help for the flag that actually scopes sessions does not say so, an operator
	// reaching for isolation still reaches for the wrong knob.
	if !strings.Contains(f.Usage, "--fleet-bus-dir") {
		t.Error("--session-registry help does not mention --fleet-bus-dir, so it never corrects the confusion that caused #5825")
	}
	if !strings.Contains(f.Usage, "off") {
		t.Error("--session-registry help does not document the 'off' value the plumbing already accepts")
	}
}

// The defect in #5825 was never that the plumbing could not scope a session table — it
// already honoured a path and "off" before this leaf. It was that the ONE call site passed
// a hard-coded "", so no flag could reach it. Every other test here would stay green if
// that "" came back, which is exactly the hole this closes.
//
// This reads source rather than calling resolveSessionPlane because that function bails
// through os.Exit on a dozen config paths and reads process env for keys — invoking it
// under test would end the test binary, not return an error.
func TestServeSessionPlanePassesTheFlagAndNotAHardCodedPath(t *testing.T) {
	raw, err := os.ReadFile(filepath.FromSlash("serve_stages.go"))
	if err != nil {
		t.Fatalf("read serve_stages.go: %v", err)
	}
	call := regexp.MustCompile(`configureServeSessionDurability\(\s*serveSessions\s*,\s*([^,]+?)\s*,`)
	m := call.FindAllStringSubmatch(string(raw), -1)
	if len(m) == 0 {
		t.Fatal("no configureServeSessionDurability(serveSessions, ...) call in serve_stages.go — the session plane no longer configures durability at all")
	}
	for _, got := range m {
		arg := strings.TrimSpace(got[1])
		if arg != "*sf.sessionRegistry" {
			t.Errorf("serve passes %s as the session registry path, want *sf.sessionRegistry — a literal here makes --session-registry unreachable and re-opens #5825", arg)
		}
	}
}

// The scope label is operator-facing: an unconfigured scope must not render as an empty
// path, which would read like "no registry" when the truth is "not configured yet".
func TestServeSessionRegistryScopeLabelNeverReadsAsNoRegistry(t *testing.T) {
	t.Cleanup(func() { serveSessionRegistryPath = "" })
	for _, tc := range []struct{ path, want string }{
		{"", "(not configured)"},
		{"off", "off (in-memory only)"},
		{filepath.Join("tmp", "r.json"), filepath.Join("tmp", "r.json")},
	} {
		serveSessionRegistryPath = tc.path
		if got := serveSessionRegistryScopeLabel(); got != tc.want {
			t.Errorf("scope label for %q = %q, want %q", tc.path, got, tc.want)
		}
	}
}
