package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

func modverFixtureReport() modver.Report {
	score := 8.5
	return modver.Report{
		Head:       "deadbee1",
		AppVersion: "0.37.0",
		Modules: []modver.Module{
			{Name: "cmd/fak", Kind: "cmd", Rev: 12, LastCommit: "aaa11111", LastDate: "2026-07-02T10:00:00Z"},
			{Name: "internal/gateway", Kind: "internal", Rev: 5, LastCommit: "bbb22222", LastDate: "2026-07-01T09:00:00Z", Score: &score},
		},
	}
}

func TestRenderModuleReport(t *testing.T) {
	var sb strings.Builder
	renderModuleReport(&sb, modverFixtureReport())
	out := sb.String()
	for _, want := range []string{
		"head deadbee1", "app 0.37.0", "2 modules",
		"r12+gaaa11111", "2026-07-02  cmd/fak",
		"r5+gbbb22222", "internal/gateway  score 8.5",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q in:\n%s", want, out)
		}
	}
}

// TestRenderModuleReportNFiltered is the #2490 witness at the CLI render seam:
// a filtered/truncated view (fewer rows than the repo total) must announce
// "showing N of M" so a --only/--top view is never mistaken for the whole repo,
// while an unfiltered view keeps the plain "M modules" header.
func TestRenderModuleReportNFiltered(t *testing.T) {
	rep := modverFixtureReport() // 2 modules
	var full strings.Builder
	renderModuleReportN(&full, rep, len(rep.Modules))
	if !strings.Contains(full.String(), "2 modules") || strings.Contains(full.String(), "showing") {
		t.Errorf("unfiltered header wrong:\n%s", full.String())
	}

	view, err := rep.View("cmd/", "name", 0) // keep only cmd/fak
	if err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	renderModuleReportN(&sb, view, len(rep.Modules))
	out := sb.String()
	if !strings.Contains(out, "showing 1 of 2 modules") {
		t.Errorf("filtered header should say 'showing 1 of 2 modules':\n%s", out)
	}
	if !strings.Contains(out, "cmd/fak") || strings.Contains(out, "internal/gateway") {
		t.Errorf("filtered view should contain only cmd/fak:\n%s", out)
	}
}

func TestStampModverLedgerRoundtrip(t *testing.T) {
	root := t.TempDir()
	rep := modverFixtureReport()
	var out, errOut strings.Builder

	if code := stampModverLedger(&out, &errOut, root, "sub/ledger.jsonl", rep); code != 0 {
		t.Fatalf("first stamp exit %d: %s", code, errOut.String())
	}
	path := filepath.Join(root, "sub", "ledger.jsonl")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if lines := strings.Count(string(b), "\n"); lines != 2 {
		t.Fatalf("first stamp wrote %d rows, want 2:\n%s", lines, b)
	}
	if !strings.Contains(string(b), `"schema":"fak-module-versions/1"`) {
		t.Errorf("ledger rows missing schema tag:\n%s", b)
	}

	out.Reset()
	if code := stampModverLedger(&out, &errOut, root, "sub/ledger.jsonl", rep); code != 0 {
		t.Fatalf("second stamp exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "0 of 2 modules moved") {
		t.Errorf("second stamp should be a converged no-op, got: %s", out.String())
	}
}

// TestRenderGhostReport is the #2477 render witness: the tombstone table shows
// each deleted module's final version (r<rev>+g<deletion sha>), the date it died,
// and its name, with a header counting the ghosts.
func TestRenderGhostReport(t *testing.T) {
	var sb strings.Builder
	renderGhostReport(&sb, []modver.Ghost{
		{Name: "internal/deleted", Kind: "internal", Rev: 4, DeletedCommit: "dead1234", DeletedDate: "2026-07-01T09:00:00Z"},
		{Name: "cmd/oldtool", Kind: "cmd", Rev: 1, DeletedCommit: "beef5678", DeletedDate: "2026-06-15T12:00:00Z"},
	})
	out := sb.String()
	for _, want := range []string{
		"2 ghost modules",
		"r4+gdead1234", "2026-07-01  internal/deleted",
		"r1+gbeef5678", "2026-06-15  cmd/oldtool",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("ghost render missing %q in:\n%s", want, out)
		}
	}
}

// TestRunVersionModulesGhostsCLI is the #2477 done-condition witness at the CLI
// seam: `fak version modules --ghosts` (and --ghosts --json) rendered over a REAL
// repo lists a deleted module, while the live report never would.
func TestRunVersionModulesGhostsCLI(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	repo := ghostCLIRepo(t)
	// Create then fully delete a module so the real repo has a ghost to render.
	ghostCLIWrite(t, filepath.Join(repo, "internal", "gone", "x.go"), "package gone\n")
	ghostCLIGit(t, repo, "add", "internal/gone/x.go")
	ghostCLIGit(t, repo, "commit", "-q", "-m", "add gone")
	ghostCLIGit(t, repo, "rm", "-q", "internal/gone/x.go")
	ghostCLIGit(t, repo, "commit", "-q", "-m", "rm gone")

	var out, errOut strings.Builder
	if code := runVersionModules(&out, &errOut, []string{"--ghosts", "--dir", repo}); code != 0 {
		t.Fatalf("--ghosts exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "internal/gone") {
		t.Errorf("ghost listing missing internal/gone over real repo:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "ghost modules") {
		t.Errorf("ghost header missing:\n%s", out.String())
	}
	// The live report over the same repo must NOT list the deleted module.
	var live, liveErr strings.Builder
	if code := runVersionModules(&live, &liveErr, []string{"--dir", repo}); code != 0 {
		t.Fatalf("live report exit %d: %s", code, liveErr.String())
	}
	if strings.Contains(live.String(), "internal/gone") {
		t.Errorf("deleted module leaked into the live report:\n%s", live.String())
	}

	// --ghosts --json emits machine-readable rows carrying the deletion commit.
	var jout, jerr strings.Builder
	if code := runVersionModules(&jout, &jerr, []string{"--ghosts", "--json", "--dir", repo}); code != 0 {
		t.Fatalf("--ghosts --json exit %d: %s", code, jerr.String())
	}
	for _, want := range []string{`"module": "internal/gone"`, `"deleted_commit"`} {
		if !strings.Contains(jout.String(), want) {
			t.Errorf("ghost JSON missing %q in:\n%s", want, jout.String())
		}
	}
}

func ghostCLIRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	ghostCLIGit(t, repo, "init", "-q", "-b", "main")
	ghostCLIGit(t, repo, "config", "core.autocrlf", "false")
	ghostCLIGit(t, repo, "config", "user.name", "test")
	ghostCLIGit(t, repo, "config", "user.email", "test@example.com")
	ghostCLIWrite(t, filepath.Join(repo, "cmd", "fak", "main.go"), "package main\nfunc main() {}\n")
	ghostCLIGit(t, repo, "add", "cmd/fak/main.go")
	ghostCLIGit(t, repo, "commit", "-q", "-m", "base")
	return repo
}

func ghostCLIGit(t *testing.T, cwd string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = cwd
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v in %s: %v\n%s", args, cwd, err, out)
	}
}

func ghostCLIWrite(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
