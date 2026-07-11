package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestRouteAccountsCoverComplete checks the happy path: the starter roster
// `--accounts-dump` emits fully covers the built-in manifest, so `--accounts-cover`
// exits 0 and reports COMPLETE.
func TestRouteAccountsCoverComplete(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "roster.json")
	if err := os.WriteFile(path, modelroute.DefaultRoster().JSON(), 0o644); err != nil {
		t.Fatalf("write roster: %v", err)
	}
	var out, errb bytes.Buffer
	if code := runRoute(&out, &errb, []string{"--accounts-cover", path}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), "COMPLETE") {
		t.Fatalf("expected a COMPLETE verdict:\n%s", out.String())
	}
}

// TestRouteAccountsCoverUnbound checks the fail-closed path: a roster with one
// binding and NO default leaves the manifest's other routed ids unbound, so
// `--accounts-cover` exits 1 and names the UNBOUND holes.
func TestRouteAccountsCoverUnbound(t *testing.T) {
	roster := modelroute.Roster{
		Version:  modelroute.RosterVersion,
		Accounts: []modelroute.Account{{ID: "a", Kind: modelroute.KindOpenAI, CredEnv: "OPENAI_API_KEY"}},
		Bindings: []modelroute.Binding{{Model: "small", Account: "a"}},
		// no Default -> every non-'small' routed id is a hole
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "sparse.json")
	if err := os.WriteFile(path, roster.JSON(), 0o644); err != nil {
		t.Fatalf("write roster: %v", err)
	}
	var out, errb bytes.Buffer
	if code := runRoute(&out, &errb, []string{"--accounts-cover", path}); code != 1 {
		t.Fatalf("exit = %d, want 1 for an incomplete roster (stderr: %s)\n%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "UNBOUND") || !strings.Contains(out.String(), "INCOMPLETE") {
		t.Fatalf("expected UNBOUND rows and an INCOMPLETE verdict:\n%s", out.String())
	}
}

// TestRouteAccountsCoverJSON checks the machine surface round-trips the tallies.
func TestRouteAccountsCoverJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "roster.json")
	if err := os.WriteFile(path, modelroute.DefaultRoster().JSON(), 0o644); err != nil {
		t.Fatalf("write roster: %v", err)
	}
	var out, errb bytes.Buffer
	if code := runRoute(&out, &errb, []string{"--accounts-cover", path, "--json"}); code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), `"unbound": 0`) {
		t.Fatalf("json must report unbound=0 for a complete roster:\n%s", out.String())
	}
}
