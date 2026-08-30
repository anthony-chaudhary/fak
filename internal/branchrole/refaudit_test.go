package branchrole

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestClassifyHardcodedRef(t *testing.T) {
	cases := []struct {
		path string
		line string
		want string
	}{
		{".github/workflows/ci.yml", "branches: [main, master]", RefClassWorkflowCovered},
		{"tools/extend_preflight.py", `branch == "master"`, RefClassDevelopmentSource},
		{"tools/fleet_control_pane.py", `DEFAULT_WORKTREE_MASTER_REF = "origin/master"`, RefClassDevelopmentSource},
		{"tools/dispatch_worker.py", "fresh origin/main worktree", RefClassDevelopmentSource},
		{"internal/corelockgate/corelockgate.go", "HEAD and origin/main move", RefClassDevelopmentSource},
		{"tools/issue_resolve_progress.py", `"origin/main"`, RefClassDevelopmentSource},
		{"cmd/fak/selfupdate_install.go", "pristine origin/main checkout", RefClassPublicFrontDoor},
		{"docs/stable-releases/2026-06-stable.md", "committed to `master`", RefClassHistorical},
		{"tools/bench_migrate.py", `"branch": "master"`, RefClassFixture},
		{"tools/demo_robustness_scorecard.py", `@(latest|main|master)`, RefClassPublicGuard},
		{"cmd/fak/new.go", `git fetch origin main`, RefClassUnclassified},
	}
	for _, tc := range cases {
		if got := ClassifyHardcodedRef(tc.path, tc.line); got != tc.want {
			t.Fatalf("ClassifyHardcodedRef(%q, %q) = %q, want %q", tc.path, tc.line, got, tc.want)
		}
	}
}

func TestScanHardcodedRefFileHandlesLongLines(t *testing.T) {
	dir := t.TempDir()
	// A single data line far larger than bufio.Scanner's 64 KiB token cap, with a
	// hard-coded ref embedded mid-line. The pre-fix scanner aborted this file with
	// "bufio.Scanner: token too long", which reds the whole audit gate on any tree
	// carrying a generated .json/.jsonl/.txt with a long line.
	huge := strings.Repeat("x", 70*1024) + " origin/main " + strings.Repeat("y", 70*1024)
	path := filepath.Join(dir, "big.json")
	content := "first line\n" + huge + "\n" + "git switch master\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	rows, err := scanHardcodedRefFile(path, "experiments/big.json")
	if err != nil {
		t.Fatalf("scanHardcodedRefFile on a >64 KiB line: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("want 2 findings (long line + trailing line), got %d: %+v", len(rows), rows)
	}
	if rows[0].Line != 2 {
		t.Fatalf("embedded ref line number = %d, want 2", rows[0].Line)
	}
	if rows[1].Line != 3 || rows[1].Text != "git switch master" {
		t.Fatalf("line after the long line misread: %+v", rows[1])
	}
}

// TestAuditHardcodedRefsScansOnlyWhatGitTracks pins the boundary of the live-tree
// gate: it audits what is committed or staged, and nothing else.
//
// Before the fix the walk pruned only gitignored paths, so an UNTRACKED file reddened
// the audit. On this shared multi-session trunk checkout that is not a governance
// signal but a false one -- #4334's live red came from a peer's uncommitted
// cmd/fak/stallscan_skew.go, a file this session could neither classify honestly (it
// may never land, or land renamed) nor commit (it is not this session's to commit).
//
// The staged arm is the other half, and the reason this is a boundary rather than a
// blanket exemption: dropping untracked content must NOT let the worker who is
// actually landing a fresh unclassified ref slip past the ratchet. Staging is what
// git commit does, so that worker is still gated at exactly the moment it matters.
func TestAuditHardcodedRefsScansOnlyWhatGitTracks(t *testing.T) {
	root := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	write := func(rel, body string) {
		t.Helper()
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	git("init")

	// Identical unclassified content in all three; only git's view of each differs.
	// The slash is load-bearing: the audit matches `origin/main`, so prose like
	// "git fetch origin main" would sail past hardcodedRefLine and prove nothing.
	const ref = "// resolved against origin/main at read time\n"
	write("cmd/fak/staged_ref.go", ref)
	write("cmd/fak/untracked_ref.go", ref)
	write("cmd/fak/ignored_ref.go", ref)
	write(".gitignore", "ignored_ref.go\n")

	// Staging is the entire distinction under test: `ls-files --others` is defined
	// against the INDEX, so `git add` is what flips a file from other to tracked.
	// This test deliberately makes no commit -- a commit here would either run the
	// host's real hooks against a throwaway repo or have to skip them, and skipping
	// commit guards is exactly the thing this repo refuses. The already-committed
	// arm needs no synthetic fixture anyway: TestHardcodedRefAuditCurrentTreeClassified
	// below audits this repo's own trunk, which is nothing but committed content.
	git("add", "--", "cmd/fak/staged_ref.go")

	findings, err := AuditHardcodedRefs(root)
	if err != nil {
		t.Fatalf("AuditHardcodedRefs: %v", err)
	}
	seen := map[string]bool{}
	for _, finding := range findings {
		seen[finding.Path] = true
	}
	if !seen["cmd/fak/staged_ref.go"] {
		t.Errorf("cmd/fak/staged_ref.go is in git's index and must still be audited; got %+v", findings)
	}
	for _, unwanted := range []string{"cmd/fak/untracked_ref.go", "cmd/fak/ignored_ref.go"} {
		if seen[unwanted] {
			t.Errorf("%s is not tracked and must not red another session's gate; got %+v", unwanted, findings)
		}
	}
}

func TestCurrentOperationalAndFixtureRefsAreClassified(t *testing.T) {
	want := map[string]string{
		"cmd/fak/codex_freshness.go":                         RefClassDevelopmentSource,
		"cmd/fak/stallscan_skew.go":                          RefClassDevelopmentSource,
		"cmd/fak/sweep_parked.go":                            RefClassDevelopmentSource,
		"cmd/microcontextdemo/natural_multitool.go":          RefClassFixture,
		"internal/devcmd/ci_preflight.go":                    RefClassDevelopmentSource,
		"internal/selfinstall/roles.go":                      RefClassPublicFrontDoor,
		"internal/workdelivery/testdata/e2e/happy-path.json": RefClassFixture,
	}
	for path, class := range want {
		if got := ClassifyHardcodedRef(path, "origin/main"); got != class {
			t.Errorf("ClassifyHardcodedRef(%q)=%q, want %q", path, got, class)
		}
	}
}

func TestHardcodedRefAuditCurrentTreeClassified(t *testing.T) {
	root := repoRootForRefAudit(t)
	findings, err := AuditHardcodedRefs(root)
	if err != nil {
		t.Fatalf("AuditHardcodedRefs: %v", err)
	}
	var unclassified []string
	classes := map[string]int{}
	for _, finding := range findings {
		classes[finding.Class]++
		if finding.Class == RefClassUnclassified {
			unclassified = append(unclassified, finding.Path+":"+itoa(finding.Line)+" "+finding.Text)
		}
	}
	if len(unclassified) > 0 {
		t.Fatalf("unclassified hard-coded branch refs:\n%s", strings.Join(unclassified, "\n"))
	}
	for _, want := range []string{RefClassDevelopmentSource, RefClassWorkflowCovered, RefClassAuditDoc} {
		if classes[want] == 0 {
			t.Fatalf("audit saw no %s rows; classes=%v", want, classes)
		}
	}
}

func repoRootForRefAudit(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "dos.toml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("dos.toml not found from %s", dir)
		}
		dir = parent
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
