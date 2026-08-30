package main

import (
	"bytes"
	"encoding/json"
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
	if !strings.Contains(got, "LOGIN") || !strings.Contains(got, "ready") {
		t.Fatalf("list should expose the login status column:\n%s", got)
	}
	if !strings.Contains(got, "WARN name<>identity") {
		t.Fatalf("list should flag the q-seat name-lie:\n%s", got)
	}
}

// TestAccountsTombstonedHiddenByDefault pins the presentation boundary: retired seats
// do not appear in default human or JSON rosters or their summaries. --all remains the
// explicit forensic escape hatch over the canonical registry.
func TestAccountsTombstonedHiddenByDefault(t *testing.T) {
	home := t.TempDir()
	// Codex discovery is intentionally independent of --home. Pin its user-home
	// seam so this fixture cannot absorb the operator's real ~/.codex seat.
	t.Setenv("FLEET_USER_HOME", home)
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	needs := mkHome(t, home, ".claude-needs-seat", "needs@example.test", false)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"},` +
		`{"name":"needs-seat","dir":"` + jsonPath(needs) + `"},` +
		`{"name":"old-1","status":"tombstoned","rehome_to":"ready-seat"},` +
		`{"name":"old-2","status":"tombstoned","rehome_to":"ready-seat"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	run := func(t *testing.T, args ...string) string {
		t.Helper()
		var out, errb bytes.Buffer
		full := append(args, "--registry", regPath, "--home", home)
		if rc := runAccounts(&out, &errb, full); rc != 0 {
			t.Fatalf("%v rc=%d stderr=%s", args, rc, errb.String())
		}
		return out.String()
	}

	for _, sub := range []string{"list", "status"} {
		t.Run(sub+" default hides tombstones", func(t *testing.T) {
			got := run(t, sub)
			if strings.Contains(got, "old-1") || strings.Contains(got, "old-2") {
				t.Fatalf("%s default must not print tombstoned seat rows:\n%s", sub, got)
			}
			if strings.Contains(got, "tombston") {
				t.Fatalf("%s default must not disclose tombstones, even as a count:\n%s", sub, got)
			}
			if !strings.Contains(got, "1/2 active seat(s) can serve") {
				t.Fatalf("%s summary should describe only the two visible seats:\n%s", sub, got)
			}
			if !strings.Contains(got, "ready-seat") {
				t.Fatalf("%s must still list the live seats:\n%s", sub, got)
			}
		})
		t.Run(sub+" --all reveals tombstones", func(t *testing.T) {
			got := run(t, sub, "--all")
			if !strings.Contains(got, "old-1") || !strings.Contains(got, "old-2") {
				t.Fatalf("%s --all should show every tombstoned seat:\n%s", sub, got)
			}
			if strings.Contains(got, "hidden — ") {
				t.Fatalf("%s --all should print no collapse note:\n%s", sub, got)
			}
		})
	}

	for _, sub := range []string{"list", "status"} {
		t.Run(sub+" default JSON hides tombstones", func(t *testing.T) {
			got := run(t, sub, "--json")
			if strings.Contains(got, "old-1") || strings.Contains(got, "old-2") || strings.Contains(got, `"tombstoned"`) {
				t.Fatalf("%s default JSON leaked retired seats: %s", sub, got)
			}
			if !strings.Contains(got, `"total": 2`) {
				t.Fatalf("%s default JSON summary should cover visible seats only: %s", sub, got)
			}
		})
		t.Run(sub+" --all JSON reveals tombstones", func(t *testing.T) {
			got := run(t, sub, "--json", "--all")
			if !strings.Contains(got, "old-1") || !strings.Contains(got, "old-2") || !strings.Contains(got, `"tombstoned"`) {
				t.Fatalf("%s --all JSON should expose retired seats: %s", sub, got)
			}
		})
	}

}

// TestRunAccountsListJSONRoster pins #4593: `list --json` must emit the per-seat
// LoginReport roster (schema+summary+seats[]) — the same machine shape `status --json`
// produces — NOT the raw registry persistence wrapper whose seats hide under .homes.
// A consumer that iterates the top level must reach real per-seat can_serve records.
func TestRunAccountsListJSONRoster(t *testing.T) {
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	needs := mkHome(t, home, ".claude-needs-seat", "needs@example.test", false)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"},` +
		`{"name":"needs-seat","dir":"` + jsonPath(needs) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"list", "--json", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("list --json rc=%d stderr=%s", rc, errb.String())
	}

	// The top level must decode as the LoginReport roster, not the registry wrapper:
	// a real seats[] array (never nested under .homes) carrying per-seat can_serve.
	var report accounts.LoginReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatalf("list --json must decode as a LoginReport: %v\n%s", err, out.String())
	}
	if report.Schema != "fak.accounts.login.v1" {
		t.Fatalf("list --json schema=%q, want fak.accounts.login.v1:\n%s", report.Schema, out.String())
	}
	if len(report.Seats) == 0 {
		t.Fatalf("list --json must carry a top-level seats[] roster, got none:\n%s", out.String())
	}
	var sawReady bool
	for _, s := range report.Seats {
		if s.Name == "ready-seat" && s.CanServe {
			sawReady = true
		}
	}
	if !sawReady {
		t.Fatalf("list --json seats[] must carry per-seat can_serve (ready-seat can_serve=true):\n%s", out.String())
	}
	// Guard against a regression back to the registry wrapper: it would surface the
	// seats under a top-level ".homes" key rather than the "seats" roster.
	if strings.Contains(out.String(), `"homes"`) {
		t.Fatalf("list --json must not emit the raw registry wrapper (top-level .homes):\n%s", out.String())
	}
}

func TestRunAccountsStatusReport(t *testing.T) {
	home := t.TempDir()
	// Keep automatic Codex discovery inside the fixture rather than the host home.
	t.Setenv("FLEET_USER_HOME", home)
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	needsLogin := mkHome(t, home, ".claude-needs-seat", "needs@example.test", false)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"},` +
		`{"name":"needs-seat","dir":"` + jsonPath(needsLogin) + `"},` +
		`{"name":"old","status":"tombstoned","rehome_to":"ready-seat"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"status", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("status rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{"fak.accounts.login.v1", "ready-seat", "needs_login", "summary:"} {
		if !strings.Contains(got, want) {
			t.Fatalf("status output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "old") || strings.Contains(got, "tombston") {
		t.Fatalf("default status leaked retired seat:\n%s", got)
	}

	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"status", "--json", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("status --json rc=%d stderr=%s", rc, errb.String())
	}
	j := out.String()
	for _, want := range []string{`"schema": "fak.accounts.login.v1"`, `"status": "needs_login"`, `"can_serve": 1`} {
		if !strings.Contains(j, want) {
			t.Fatalf("status --json missing %q:\n%s", want, j)
		}
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

	// #3213: `resolve --name q` resolves identically to the positional `resolve q`,
	// so an operator who just ran `remove --name q` can reach for the same flag here.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"resolve", "--name", "q", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("resolve --name rc=%d stderr=%s", rc, errb.String())
	}
	if strings.TrimSpace(out.String()) != gem8 {
		t.Fatalf("resolve --name q = %q, want rehomed dir %q (must match positional form)", strings.TrimSpace(out.String()), gem8)
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

	// #3213: `pull --name gem8-seat` routes through the same shared helper as the
	// positional form, so the --name fallback works for pull too.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"pull", "--name", "gem8-seat", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("pull --name rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "nothing to pull") {
		t.Fatalf("pull --name should match positional (no-op on active seat): %s", out.String())
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
	for _, want := range []string{"fak ", "fak-config-homes/v1", "remove", "restore", "status", "version"} {
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

func TestRunAccountsRestoreArchive(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	seat := mkHome(t, home, ".claude-old-seat", "old@example.test", true)
	anchor := mkHome(t, home, ".claude-anchor-seat", "anchor@example.test", true)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"old-seat","dir":"` + jsonPath(seat) + `"},` +
		`{"name":"follower","status":"tombstoned","rehome_to":"old-seat"},` +
		`{"name":"anchor-seat","dir":"` + jsonPath(anchor) + `","default":true}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"remove", "--name", "old-seat", "--archive", "--rehome-to", "anchor-seat",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("remove --archive rc=%d stderr=%s", rc, errb.String())
	}
	if _, err := os.Stat(seat); err == nil {
		t.Fatalf("remove should rename original dir away: %s", seat)
	}
	archived, _ := filepath.Glob(filepath.Join(home, ".claude-old-seat.DELETED-*"))
	if len(archived) != 1 {
		t.Fatalf("want one archived dir, got %v", archived)
	}

	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{
		"restore", "--name", "old-seat",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("restore rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if _, err := os.Stat(seat); err != nil {
		t.Fatalf("restore should recreate original dir %s: %v", seat, err)
	}
	if _, err := os.Stat(archived[0]); err == nil {
		t.Fatalf("restore should rename archive away: %s", archived[0])
	}

	got, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should validate after restore: %v", err)
	}
	var restored, follower accounts.Home
	for _, h := range got.Homes {
		switch h.Name {
		case "old-seat":
			restored = h
		case "follower":
			follower = h
		}
		if strings.HasPrefix(h.Name, "old-seat.DELETED-") {
			t.Fatalf("restore left archived registry handle behind: %+v", h)
		}
	}
	if restored.Name == "" || !restored.Active() || restored.Dir != seat || !restored.EnabledOrDefault() {
		t.Fatalf("old-seat not restored active: %+v", restored)
	}
	if follower.RehomeTo != "old-seat" {
		t.Fatalf("follower rehome_to = %q, want old-seat", follower.RehomeTo)
	}
	dosView := filepath.Join(home, ".claude", "accounts.yaml")
	dos, err := os.ReadFile(dosView)
	if err != nil {
		t.Fatalf("dos view should be regenerated under temp home: %v", err)
	}
	if !strings.Contains(string(dos), "name: old-seat") || strings.Contains(string(dos), "old-seat.DELETED-") {
		t.Fatalf("generated dos view did not reflect restored active seat:\n%s", dos)
	}
}

func TestRunAccountsRestoreInPlace(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	seat := mkHome(t, home, ".claude-plain-seat", "plain@example.test", true)
	anchor := mkHome(t, home, ".claude-anchor-seat", "anchor@example.test", true)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"plain-seat","dir":"` + jsonPath(seat) + `"},` +
		`{"name":"anchor-seat","dir":"` + jsonPath(anchor) + `","default":true}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	// remove WITHOUT --archive: tombstones in place, dir left untouched.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{
		"remove", "--name", "plain-seat", "--rehome-to", "anchor-seat",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("remove rc=%d stderr=%s", rc, errb.String())
	}
	if _, err := os.Stat(seat); err != nil {
		t.Fatalf("plain remove must leave the dir in place: %s: %v", seat, err)
	}
	if archived, _ := filepath.Glob(filepath.Join(home, ".claude-plain-seat.DELETED-*")); len(archived) != 0 {
		t.Fatalf("plain remove must not archive the dir, got %v", archived)
	}

	// restore: no archive exists, so it must take the in-place un-tombstone path.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{
		"restore", "--name", "plain-seat",
		"--registry", regPath, "--home", home,
	}); rc != 0 {
		t.Fatalf("in-place restore rc=%d stderr=%s stdout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "in place") {
		t.Fatalf("expected in-place restore message, got:\n%s", out.String())
	}
	if _, err := os.Stat(seat); err != nil {
		t.Fatalf("in-place restore must not move the dir: %s: %v", seat, err)
	}

	got, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should validate after in-place restore: %v", err)
	}
	var restored accounts.Home
	for _, h := range got.Homes {
		if h.Name == "plain-seat" {
			restored = h
		}
	}
	if restored.Name == "" || !restored.Active() || restored.Dir != seat || !restored.EnabledOrDefault() {
		t.Fatalf("plain-seat not restored active in place: %+v", restored)
	}
	if restored.RehomeTo != "" || restored.TombstonedAt != "" || restored.TombstoneReason != "" {
		t.Fatalf("in-place restore left tombstone fields set: %+v", restored)
	}
	dos, err := os.ReadFile(filepath.Join(home, ".claude", "accounts.yaml"))
	if err != nil {
		t.Fatalf("dos view should be regenerated: %v", err)
	}
	if !strings.Contains(string(dos), "name: plain-seat") {
		t.Fatalf("generated dos view missing restored seat:\n%s", dos)
	}
}

func TestRunAccountsRemoveMovesRolesOffTombstone(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	home := t.TempDir()
	day27 := mkHome(t, home, ".claude-day27-netra", "day27@example.test", true)
	anchor := mkHome(t, home, ".claude-default", "anchor@example.test", true)

	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"day27-netra","dir":"` + jsonPath(day27) + `"},` +
		`{"name":"default","dir":"` + jsonPath(anchor) + `"}` +
		`],"roles":{"active":"day27-netra","anchor":"default"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"remove", "--name", "day27-netra", "--rehome-to", "default",
		"--reason", "retired test account",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("remove role-holder rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "registry: role active -> default (was day27-netra)") {
		t.Fatalf("remove should report the role move, got:\n%s", out.String())
	}

	got, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry should validate after tombstoning role-holder: %v", err)
	}
	if got.Roles[accounts.RoleActive] != "default" {
		t.Fatalf("active role = %q, want default", got.Roles[accounts.RoleActive])
	}
	if got.Roles[accounts.RoleAnchor] != "default" {
		t.Fatalf("anchor role = %q, want default", got.Roles[accounts.RoleAnchor])
	}
	var tomb accounts.Home
	for _, h := range got.Homes {
		if h.Name == "day27-netra" {
			tomb = h
			break
		}
	}
	if tomb.Name == "" || tomb.Active() || tomb.RehomeTo != "default" || tomb.EnabledOrDefault() {
		t.Fatalf("day27-netra tombstone not recorded correctly: %+v", tomb)
	}

	dosView := filepath.Join(home, ".claude", "accounts.yaml")
	dos, err := os.ReadFile(dosView)
	if err != nil {
		t.Fatalf("dos view should be regenerated under temp home: %v", err)
	}
	if strings.Contains(string(dos), "name: day27-netra") || strings.Contains(string(dos), "active_default: day27-netra") {
		t.Fatalf("tombstoned active account leaked into generated dos view:\n%s", dos)
	}
	if !strings.Contains(string(dos), "active_default: default") {
		t.Fatalf("generated dos view should move active_default to default:\n%s", dos)
	}
}

// TestAccountsCheckQuantifiesDriftAndFlagsHandEdit covers #3214: `check` reports a +N/-M
// magnitude for any drift, warns only when the on-disk view carries hand-authored content
// `sync` would clobber, and prints an inline diff under --diff.
func TestAccountsCheckQuantifiesDriftAndFlagsHandEdit(t *testing.T) {
	home := t.TempDir()
	seat := mkHome(t, home, ".claude-a-seat", "a@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"a-seat","dir":"` + jsonPath(seat) + `","default":true}` +
		`]}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	jobView := filepath.Join(home, "claude_accounts.yaml")

	// Generate the canonical projection first, so we can perturb it in known ways.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"sync", "--registry", regPath, "--home", home, "--dos-view", "", "--job-view", jobView}); rc != 0 {
		t.Fatalf("sync rc=%d stderr=%s", rc, errb.String())
	}
	projection, err := os.ReadFile(jobView)
	if err != nil {
		t.Fatal(err)
	}

	// (1) Stale-but-clean drift: drop the last two projection lines. `check` reports a
	// magnitude and must NOT warn about a hand-edit (the on-disk file is a strict subset of
	// what the generator emits — every line is generator-shaped).
	lines := strings.Split(strings.TrimRight(string(projection), "\n"), "\n")
	if len(lines) < 3 {
		t.Fatalf("projection unexpectedly short (%d lines):\n%s", len(lines), projection)
	}
	stale := strings.Join(lines[:len(lines)-2], "\n") + "\n"
	if err := os.WriteFile(jobView, []byte(stale), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"check", "--registry", regPath, "--home", home, "--dos-view", "", "--job-view", jobView}); rc != 1 {
		t.Fatalf("stale check rc=%d want drift exit 1; stdout=%s", rc, out.String())
	}
	got := out.String()
	if !strings.Contains(got, "DRIFT job:") || !strings.Contains(got, "(+") || !strings.Contains(got, "-0)") {
		t.Fatalf("stale check should carry a +N/-0 magnitude:\n%s", got)
	}
	if strings.Contains(got, "hand-edited") {
		t.Fatalf("stale (generator-shaped) drift must NOT be flagged as hand-edited:\n%s", got)
	}

	// (2) Hand-edited drift: prepend a doc header of comment lines the generator never emits.
	// `check` must additionally warn that `sync` will overwrite the edits, and --diff must
	// show the removed header lines.
	handEdited := "# ==== HAND-AUTHORED DOC HEADER ====\n# rationale line two\n# rationale line three\n" + string(projection)
	if err := os.WriteFile(jobView, []byte(handEdited), 0o644); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"check", "--registry", regPath, "--home", home, "--dos-view", "", "--job-view", jobView, "--diff"}); rc != 1 {
		t.Fatalf("hand-edited check rc=%d want drift exit 1; stdout=%s", rc, out.String())
	}
	got = out.String()
	for _, want := range []string{
		"DRIFT job:",
		"appears hand-edited",
		"`sync` will overwrite those edits",
		"- # ==== HAND-AUTHORED DOC HEADER ====", // --diff shows the removed header line
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("hand-edited check missing %q:\n%s", want, got)
		}
	}
}

// jsonPath escapes a Windows path's backslashes for embedding in a JSON string literal.
func jsonPath(p string) string { return strings.ReplaceAll(p, `\`, `\\`) }
