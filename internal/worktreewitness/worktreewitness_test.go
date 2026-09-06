package worktreewitness

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeGit records git invocations and provides scripted responses for tests.
type fakeGit struct {
	sha         string
	top         string
	statusOut   string
	removeFails bool
	calls       []string
}

func (g *fakeGit) run(dir, name string, args ...string) (string, int, error) {
	g.calls = append(g.calls, strings.Join(args, " "))
	sub := args[2:]
	switch {
	case len(sub) >= 2 && sub[0] == "rev-parse" && sub[1] == "--show-toplevel":
		return g.top + "\n", 0, nil
	case len(sub) >= 1 && sub[0] == "rev-parse":
		return g.sha + "\n", 0, nil
	case len(sub) >= 2 && sub[0] == "worktree" && sub[1] == "add":
		return "", 0, nil
	case len(sub) >= 2 && sub[0] == "worktree" && sub[1] == "remove":
		if g.removeFails {
			return "locked", 1, nil
		}
		return "", 0, nil
	case len(sub) >= 2 && sub[0] == "worktree" && sub[1] == "prune":
		return "", 0, nil
	case len(sub) >= 1 && sub[0] == "status":
		return g.statusOut, 0, nil
	case len(sub) >= 1 && sub[0] == "diff":
		return "the-diff", 0, nil
	case len(sub) >= 1 && sub[0] == "fetch":
		return "", 0, nil
	}
	return "", 0, nil
}

// fakeCmd records command execution arguments and provides scripted outputs.
type fakeCmd struct {
	code     int
	out      string
	startErr error
	ranInDir string
}

func (c *fakeCmd) run(dir, name string, args ...string) (string, int, error) {
	c.ranInDir = dir
	if c.startErr != nil {
		return "", 0, c.startErr
	}
	return c.out, c.code, nil
}

// repoAbs resolves the absolute path of the mock repo across platforms.
func repoAbs(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs("/repo")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	return abs
}

func baseCfg() Config {
	return Config{Repo: "/repo", Command: []string{"go", "test", "./cmd/fak"}}
}

func TestClassifyGreen(t *testing.T) {
	cases := []struct {
		code  int
		ranOK bool
		want  bool
	}{
		{0, true, true},
		{1, true, false},
		{0, false, false},
		{2, true, false},
	}
	for _, tc := range cases {
		if got := classifyGreen(tc.code, tc.ranOK); got != tc.want {
			t.Errorf("classifyGreen(%d,%v)=%v want %v", tc.code, tc.ranOK, got, tc.want)
		}
	}
}

func TestRun_GreenCleanTree(t *testing.T) {
	git := &fakeGit{sha: "abcdef0123456789", top: repoAbs(t)}
	cmd := &fakeCmd{code: 0, out: "ok"}
	res, err := Run(baseCfg(), git.run, cmd.run)
	if err != nil {
		t.Fatalf("Run err: %v", err)
	}
	if !res.Green {
		t.Errorf("want green")
	}
	if res.ExitCode != 0 {
		t.Errorf("ExitCode=%d want 0", res.ExitCode)
	}
	if res.SHA != "abcdef0123456789" || res.ShortSHA != "abcdef012345" {
		t.Errorf("SHA=%q ShortSHA=%q", res.SHA, res.ShortSHA)
	}
	if res.Ref != DefaultRef {
		t.Errorf("Ref=%q want %q", res.Ref, DefaultRef)
	}
	if res.Archived != "" {
		t.Errorf("clean tree archived %q", res.Archived)
	}
	if !containsCall(git.calls, "worktree remove --force") {
		t.Errorf("worktree not reaped; calls=%v", git.calls)
	}
}

func TestRun_RedCommandStillReaps(t *testing.T) {
	git := &fakeGit{sha: "deadbeefcafe0000", top: repoAbs(t)}
	cmd := &fakeCmd{code: 1, out: "FAIL"}
	res, err := Run(baseCfg(), git.run, cmd.run)
	if err != nil {
		t.Fatalf("a red command is not a Run error: %v", err)
	}
	if res.Green {
		t.Errorf("want red")
	}
	if res.ExitCode != 1 {
		t.Errorf("ExitCode=%d want 1", res.ExitCode)
	}
	if !containsCall(git.calls, "worktree remove --force") {
		t.Errorf("red command must still reap; calls=%v", git.calls)
	}
}

func TestRun_DirtyWorktreeArchivesBeforeReap(t *testing.T) {
	archiveDir := t.TempDir()
	git := &fakeGit{sha: "0011223344556677", top: repoAbs(t), statusOut: " M some/file.go\n?? scratch.tmp\n"}
	cmd := &fakeCmd{code: 0, out: "ok"}
	cfg := baseCfg()
	cfg.ArchiveDir = archiveDir
	res, err := Run(cfg, git.run, cmd.run)
	if err != nil {
		t.Fatalf("Run err: %v", err)
	}
	if res.Archived == "" {
		t.Fatalf("dirty tree should have archived")
	}
	patch, rerr := os.ReadFile(filepath.Join(res.Archived, "diff.patch"))
	if rerr != nil {
		t.Fatalf("read archived patch: %v", rerr)
	}
	if string(patch) != "the-diff" {
		t.Errorf("archived patch=%q want the-diff", patch)
	}
	ai, ri := callIndex(git.calls, "diff HEAD"), callIndex(git.calls, "worktree remove --force")
	if ai < 0 || ri < 0 || ai > ri {
		t.Errorf("archive(diff) must precede reap(remove); calls=%v", git.calls)
	}
}

func TestRun_RemoveFailureFallsBackToPrune(t *testing.T) {
	git := &fakeGit{sha: "aaaabbbbccccdddd", top: repoAbs(t), removeFails: true}
	cmd := &fakeCmd{code: 0, out: "ok"}
	res, err := Run(baseCfg(), git.run, cmd.run)
	if err != nil {
		t.Fatalf("Run err: %v", err)
	}
	if !containsCall(git.calls, "worktree prune") {
		t.Errorf("remove failure must fall back to prune; calls=%v", git.calls)
	}
	if res.ReapNote == "" {
		t.Errorf("want a ReapNote recording the prune fallback")
	}
}

func TestRun_NoCommandIsError(t *testing.T) {
	git := &fakeGit{sha: "x", top: repoAbs(t)}
	cmd := &fakeCmd{}
	cfg := baseCfg()
	cfg.Command = nil
	if _, err := Run(cfg, git.run, cmd.run); err == nil {
		t.Errorf("empty command must error")
	}
}

func TestRun_CommandStartFailureIsError(t *testing.T) {
	git := &fakeGit{sha: "1122334455667788", top: repoAbs(t)}
	cmd := &fakeCmd{startErr: fmt.Errorf("exec: \"nope\": not found")}
	res, err := Run(baseCfg(), git.run, cmd.run)
	if err == nil {
		t.Errorf("a command that cannot start must be a Run error, not a silent red")
	}
	if res.Green {
		t.Errorf("a non-started command is not green")
	}
	if !containsCall(git.calls, "worktree remove --force") {
		t.Errorf("start failure must still reap; calls=%v", git.calls)
	}
}

func TestRun_RunsCommandInModuleDir(t *testing.T) {
	git := &fakeGit{sha: "1234123412341234", top: repoAbs(t)}
	cmd := &fakeCmd{code: 0}
	if _, err := Run(baseCfg(), git.run, cmd.run); err != nil {
		t.Fatalf("Run err: %v", err)
	}
	if !strings.Contains(cmd.ranInDir, "fak-witness-") {
		t.Errorf("command ran in %q, want a fak-witness- worktree", cmd.ranInDir)
	}
	if strings.Contains(cmd.ranInDir, "repo") && !strings.Contains(cmd.ranInDir, "fak-witness-") {
		t.Errorf("command must not run in the caller's repo: %q", cmd.ranInDir)
	}
}

func TestSplitRef(t *testing.T) {
	cases := map[string][2]string{
		"origin/main":     {"origin", "main"},
		"upstream/master": {"upstream", "master"},
		"main":            {"origin", "main"},
	}
	for ref, want := range cases {
		r, s := splitRef(ref)
		if r != want[0] || s != want[1] {
			t.Errorf("splitRef(%q)=(%q,%q) want (%q,%q)", ref, r, s, want[0], want[1])
		}
	}
}

func TestShortSHA(t *testing.T) {
	if ShortSHA("abcdef0123456789") != "abcdef012345" {
		t.Errorf("long SHA not truncated to 12")
	}
	if ShortSHA("short") != "short" {
		t.Errorf("short SHA changed")
	}
}

func containsCall(calls []string, want string) bool { return callIndex(calls, want) >= 0 }

func callIndex(calls []string, want string) int {
	for i, c := range calls {
		if strings.Contains(c, want) {
			return i
		}
	}
	return -1
}
