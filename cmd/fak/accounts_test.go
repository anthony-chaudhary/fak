package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// mkHome creates a fake Claude config home under root with a logged-in identity.
func mkHome(t *testing.T, root, dir, email string, creds bool) string {
	t.Helper()
	full := filepath.Join(root, dir)
	if err := os.MkdirAll(filepath.Join(full, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	body := `{"oauthAccount":{"emailAddress":"` + email + `","accountUuid":"u-` + email + `"}}`
	if err := os.WriteFile(filepath.Join(full, ".claude.json"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	if creds {
		if err := os.WriteFile(filepath.Join(full, ".credentials.json"), []byte(`{}`), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return full
}

func TestRunAccountsDiscoverAndList(t *testing.T) {
	home := t.TempDir()
	mkHome(t, home, ".claude-gem8-seat", "gem8@example.test", true)
	mkHome(t, home, ".claude-q-seat", "gem8@example.test", true) // the lie: named q, is gem8

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"discover", "--home", home}); rc != 0 {
		t.Fatalf("discover rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), `"q-seat"`) || !strings.Contains(out.String(), `"gem8-seat"`) {
		t.Fatalf("discover output missing homes:\n%s", out.String())
	}

	out.Reset()
	errb.Reset()
	// Point --registry at a nonexistent path so list falls back to discovery.
	miss := filepath.Join(home, "no-registry.json")
	if rc := runAccounts(&out, &errb, []string{"list", "--home", home, "--registry", miss}); rc != 0 {
		t.Fatalf("list rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "gem8@example.test") {
		t.Fatalf("list missing identity:\n%s", got)
	}
	if !strings.Contains(got, "WARN name<>identity") {
		t.Fatalf("list should flag the q-seat name-lie:\n%s", got)
	}
}

func TestRunAccountsResolveRehome(t *testing.T) {
	home := t.TempDir()
	gem8 := mkHome(t, home, ".claude-gem8-seat", "gem8@example.test", true)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"gem8-seat","dir":"` + jsonPath(gem8) + `","default":true},` +
		`{"name":"q","status":"tombstoned","rehome_to":"gem8-seat"}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"resolve", "q", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("resolve rc=%d stderr=%s", rc, errb.String())
	}
	if strings.TrimSpace(out.String()) != gem8 {
		t.Fatalf("resolve q = %q, want rehomed dir %q", strings.TrimSpace(out.String()), gem8)
	}
	if !strings.Contains(errb.String(), "rehoming") {
		t.Fatalf("resolve should warn about the rehome, stderr=%s", errb.String())
	}

	// --env form prints the export line.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"resolve", "q", "--registry", regPath, "--home", home, "--env"}); rc != 0 {
		t.Fatalf("resolve --env rc=%d", rc)
	}
	if !strings.Contains(out.String(), "CLAUDE_CONFIG_DIR="+gem8) {
		t.Fatalf("resolve --env = %q", out.String())
	}
}

func TestRunAccountsPull(t *testing.T) {
	home := t.TempDir()
	gem8 := mkHome(t, home, ".claude-gem8-seat", "gem8@example.test", true)

	// A shared-store bundle for tombstoned "q": a session under its project slug.
	store := filepath.Join(home, ".claude-shared-history")
	qbundle := filepath.Join(store, "q", "projects", "C--work-demo")
	if err := os.MkdirAll(qbundle, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(qbundle, "sess.jsonl"), []byte(`{"type":"mode"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	reg := `{"version":"fak-config-homes/v1","shared_history":"` + jsonPath(store) + `","homes":[` +
		`{"name":"gem8-seat","dir":"` + jsonPath(gem8) + `","default":true},` +
		`{"name":"q","status":"tombstoned","rehome_to":"gem8-seat"}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	// dry-run: announces, copies nothing.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"pull", "q", "--registry", regPath, "--home", home, "--dry-run"}); rc != 0 {
		t.Fatalf("pull dry-run rc=%d err=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "would pull") {
		t.Fatalf("dry-run should announce: %s", out.String())
	}
	if _, err := os.Stat(filepath.Join(gem8, "projects", "C--work-demo", "sess.jsonl")); err == nil {
		t.Fatalf("dry-run must not copy")
	}

	// real pull: the bundle lands in gem8's config home.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"pull", "q", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("pull rc=%d err=%s", rc, errb.String())
	}
	pulled := filepath.Join(gem8, "projects", "C--work-demo", "sess.jsonl")
	if _, err := os.Stat(pulled); err != nil {
		t.Fatalf("pull should have copied the session into gem8: %v\noutput=%s", err, out.String())
	}

	// pulling an active seat is a no-op.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"pull", "gem8-seat", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("pull active rc=%d", rc)
	}
	if !strings.Contains(out.String(), "nothing to pull") {
		t.Fatalf("active pull should be a no-op: %s", out.String())
	}
}

func TestRunAccountsValidateBadRegistry(t *testing.T) {
	home := t.TempDir()
	regPath := filepath.Join(home, "registry.json")
	// tombstone with no rehome_to -> Validate must reject.
	if err := os.WriteFile(regPath, []byte(`{"homes":[{"name":"q","status":"tombstoned"}]}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"validate", "--registry", regPath}); rc == 0 {
		t.Fatalf("validate of a bad registry should be non-zero, stdout=%s", out.String())
	}
}

func TestRunAccountsUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, nil); rc != 2 {
		t.Fatalf("no args rc=%d, want 2", rc)
	}
	if rc := runAccounts(&out, &errb, []string{"bogus"}); rc != 2 {
		t.Fatalf("bogus subcommand rc=%d, want 2", rc)
	}
}

func TestRunAccountsVersion(t *testing.T) {
	// The version surface must name the build, the registry schema it supports, and the verb
	// set — the three facts that let an operator see a stale binary instead of hitting a raw
	// "flag provided but not defined" on a verb the binary predates.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"version"}); rc != 0 {
		t.Fatalf("version rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"fak ", "fak-config-homes/v1", "remove", "version"} {
		if !strings.Contains(got, want) {
			t.Fatalf("version output missing %q:\n%s", want, got)
		}
	}

	// --json emits a machine-readable object carrying the same facts.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"version", "--json"}); rc != 0 {
		t.Fatalf("version --json rc=%d stderr=%s", rc, errb.String())
	}
	if j := out.String(); !strings.Contains(j, `"registry_version"`) || !strings.Contains(j, `"verbs"`) {
		t.Fatalf("version --json missing keys:\n%s", j)
	}
}

// jsonPath escapes a Windows path's backslashes for embedding in a JSON string literal.
func jsonPath(p string) string { return strings.ReplaceAll(p, `\`, `\\`) }
