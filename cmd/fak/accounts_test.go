package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
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

func TestRunAccountsRemoveArchive(t *testing.T) {
	// `remove --archive` must do the WHOLE retirement in one command: tombstone the registry,
	// rename the config dir to the .DELETED-<date> form, and repoint the registry entry — the
	// manual dance this used to take. (This is the "super easy to remove" guarantee.)
	//
	// HERMETIC ROSTER ISOLATION: `remove` regenerates the dos + job roster VIEWS as its final
	// step, and their default paths come from process-global state (os.UserHomeDir for the dos
	// view, FAK_JOB_ROSTER for the job view). Left unpinned, this test once overwrote a live
	// operator's real ~/.claude/accounts.yaml and job roster with its temp-dir `anchor-seat`,
	// breaking the `(u)` account switcher until the views were re-synced from the registry.
	// Clear the env redirect and rely on the --home redirect (re-derives the dos view under the
	// temp home) so this test can only ever touch files inside t.TempDir().
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	seat := mkHome(t, home, ".claude-old-seat", "old@example.test", true)
	anchorName := "anchor-seat-" + strings.ReplaceAll(t.Name(), "/", "-")
	anchor := mkHome(t, home, ".claude-"+anchorName, "anchor@example.test", true)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"old-seat","dir":"` + jsonPath(seat) + `"},` +
		`{"name":"` + anchorName + `","dir":"` + jsonPath(anchor) + `","default":true}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"remove", "--name", "old-seat", "--archive",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("remove --archive rc=%d stderr=%s", rc, errb.String())
	}

	// The original dir is gone; exactly one .DELETED-* archive exists.
	if _, err := os.Stat(seat); err == nil {
		t.Fatalf("original dir should have been renamed away: %s", seat)
	}
	archived, _ := filepath.Glob(filepath.Join(home, ".claude-old-seat.DELETED-*"))
	if len(archived) != 1 {
		t.Fatalf("want exactly one archived dir, got %v\noutput=%s", archived, out.String())
	}

	// The registry entry is renamed, tombstoned, and its dir repointed at the archive.
	reg2, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should still validate after archive: %v", err)
	}
	var found bool
	for _, h := range reg2.Homes {
		if strings.HasPrefix(h.Name, "old-seat.DELETED-") {
			found = true
			if h.Active() {
				t.Fatalf("archived seat must be tombstoned: %+v", h)
			}
			if !strings.Contains(h.Dir, ".DELETED-") {
				t.Fatalf("archived seat dir not repointed: %q", h.Dir)
			}
		}
	}
	if !found {
		t.Fatalf("no archived registry entry found:\n%s", reg2.JSON())
	}

	// Hermeticity witness: the regenerated dos roster must live UNDER the temp home, never the
	// real ~/.claude/accounts.yaml. If --home failed to redirect the view, this test would have
	// clobbered the operator's live switcher roster — assert it landed in the sandbox instead.
	dosView := filepath.Join(home, ".claude", "accounts.yaml")
	if _, err := os.Stat(dosView); err != nil {
		t.Fatalf("dos roster view should have been regenerated under the temp home at %s: %v", dosView, err)
	}
	realHome, _ := os.UserHomeDir()
	if realHome != "" {
		realDosView := filepath.Join(realHome, ".claude", "accounts.yaml")
		if rel, err := filepath.Rel(home, realDosView); err == nil && !strings.HasPrefix(rel, "..") {
			t.Skipf("real home is inside the temp dir (unexpected); skipping leak assertion")
		}
		// The regenerated view names this test's unique anchor; a real roster must NOT.
		if data, err := os.ReadFile(realDosView); err == nil && strings.Contains(string(data), anchorName) {
			t.Fatalf("test leaked its temp-dir roster into the REAL dos view %s (contains %q)", realDosView, anchorName)
		}
	}
}

// jsonPath escapes a Windows path's backslashes for embedding in a JSON string literal.
func jsonPath(p string) string { return strings.ReplaceAll(p, `\`, `\\`) }
