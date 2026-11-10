package childenv

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv("FAK_CHILDENV_PROBE") == "1" {
		// Probe mode: report the PWD this re-executed test binary actually sees.
		fmt.Println(os.Getenv("PWD"))
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// cleanTempDir returns a fresh absolute directory in the form ForDir names it.
// Under Git Bash, GOTMPDIR can give t.TempDir() mixed separators, and ForDir
// always names the absolute (and so cleaned) path.
func cleanTempDir(t *testing.T) string {
	t.Helper()
	return filepath.Clean(t.TempDir())
}

// TestDirVar pins the shared predicate every launcher uses to drop inherited
// PWD/OLDPWD: exact names in any letter case (Windows treats variable names
// case-insensitively), never look-alikes that merely contain them.
func TestDirVar(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"PWD", true},
		{"pwd", true},
		{"Pwd", true},
		{"OLDPWD", true},
		{"oldpwd", true},
		{"OldPwd", true},
		{"PWD_X", false},
		{"XPWD", false},
		{"OLDPWDX", false},
		{"MYOLDPWD", false},
		{"OLD", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := DirVar(tc.name); got != tc.want {
			t.Errorf("DirVar(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestForDirReplacesInheritedPWD pins the core rule: every inherited PWD and
// OLDPWD, in any letter case, is dropped, look-alike names survive in order,
// and exactly one PWD naming dir is appended last. A stale launcher PWD makes
// opencode boot a second instance for the wrong directory and hang after its
// turn; the input slice must stay untouched because callers reuse it.
func TestForDirReplacesInheritedPWD(t *testing.T) {
	dir := cleanTempDir(t)
	cases := []struct {
		name string
		env  []string
		kept []string
	}{
		{
			name: "upper-case PWD and OLDPWD",
			env:  []string{"PATH=/bin", "PWD=/launcher", "HOME=/h", "OLDPWD=/before"},
			kept: []string{"PATH=/bin", "HOME=/h"},
		},
		{
			name: "mixed-case variants as Windows names them",
			env:  []string{"Pwd=/a", "PATH=/bin", "pwd=/b", "oldpwd=/c", "OldPwd=/d"},
			kept: []string{"PATH=/bin"},
		},
		{
			name: "look-alike names are kept in order",
			env:  []string{"PWD_X=keep", "XPWD=keep", "PWD=/stale", "OLDPWD_Y=keep", "MYOLDPWD=keep", `=C:=C:\work`},
			kept: []string{"PWD_X=keep", "XPWD=keep", "OLDPWD_Y=keep", "MYOLDPWD=keep", `=C:=C:\work`},
		},
		{
			name: "empty inherited PWD value",
			env:  []string{"PWD=", "A=1", "OLDPWD="},
			kept: []string{"A=1"},
		},
		{
			name: "no inherited PWD",
			env:  []string{"A=1", "B=2"},
			kept: []string{"A=1", "B=2"},
		},
		{
			name: "nil env",
			env:  nil,
			kept: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := slices.Clone(tc.env)
			got := ForDir(tc.env, dir)
			want := append(slices.Clone(tc.kept), "PWD="+dir)
			if !slices.Equal(got, want) {
				t.Fatalf("ForDir(%q, dir) =\n  %q\nwant\n  %q", tc.env, got, want)
			}
			if !slices.Equal(tc.env, before) {
				t.Fatalf("ForDir modified its input: %q, want %q", tc.env, before)
			}
		})
	}
}

// TestForDirMakesRelativeDirAbsolute pins that a relative exec.Cmd.Dir still
// yields an absolute PWD: a relative PWD is as wrong to opencode as a stale one.
func TestForDirMakesRelativeDirAbsolute(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	rel := filepath.Join("sub", "worktree")
	want := filepath.Join(wd, rel)
	got := ForDir([]string{"PWD=/stale", "A=1"}, rel)
	if !slices.Equal(got, []string{"A=1", "PWD=" + want}) {
		t.Fatalf("ForDir(relative) = %q, want [A=1 PWD=%s]", got, want)
	}
}

// TestForDirEmptyDirNamesWorkingDirectory pins the inherit-cwd case (no
// exec.Cmd.Dir): the child runs in this process's directory, so PWD must name
// it even when this process's own PWD is stale.
func TestForDirEmptyDirNamesWorkingDirectory(t *testing.T) {
	stale := t.TempDir()
	t.Setenv("PWD", stale)
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	got := ForDir([]string{"PWD=" + stale, "A=1", "OLDPWD=" + stale}, "")
	if !slices.Equal(got, []string{"A=1", "PWD=" + wd}) {
		t.Fatalf("ForDir(\"\") = %q, want [A=1 PWD=%s]", got, wd)
	}
	pwdInfo, err := os.Stat(wd)
	if err != nil {
		t.Fatal(err)
	}
	dotInfo, err := os.Stat(".")
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(pwdInfo, dotInfo) {
		t.Fatalf("PWD %q does not name the working directory", wd)
	}
}

// TestForDirUnreadableWorkingDirectoryOnlyDropsPWD pins the fallback when the
// working directory cannot be read: the stale PWD is still dropped and no
// empty or guessed PWD is invented.
func TestForDirUnreadableWorkingDirectoryOnlyDropsPWD(t *testing.T) {
	gone := filepath.Join(t.TempDir(), "gone")
	if err := os.Mkdir(gone, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(gone)
	if err := os.Remove(gone); err != nil {
		t.Skipf("cannot remove the working directory on this platform: %v", err)
	}
	if wd, err := os.Getwd(); err == nil {
		t.Skipf("platform still resolves a removed working directory (%q)", wd)
	}
	got := ForDir([]string{"PWD=/stale", "A=1", "OLDPWD=/old"}, "")
	if !slices.Equal(got, []string{"A=1"}) {
		t.Fatalf("ForDir(\"\") with unreadable cwd = %q, want [A=1]", got)
	}
}

// TestApplyNilEnvInheritsCorrectedEnviron pins that a nil cmd.Env ("inherit
// the launcher's environment") becomes a copy of os.Environ() whose PWD names
// cmd.Dir, so a launcher that never set Env cannot leak its stale PWD/OLDPWD.
func TestApplyNilEnvInheritsCorrectedEnviron(t *testing.T) {
	dir := cleanTempDir(t)
	t.Setenv("PWD", t.TempDir())
	t.Setenv("OLDPWD", t.TempDir())
	t.Setenv("FAK_CHILDENV_TEST_MARK", "inherited")

	cmd := &exec.Cmd{Dir: dir}
	Apply(cmd)

	// Never print cmd.Env on failure: it is this process's whole environment.
	if !slices.Contains(cmd.Env, "FAK_CHILDENV_TEST_MARK=inherited") {
		t.Fatalf("Apply(nil Env) did not inherit os.Environ() (%d entries, marker missing)", len(cmd.Env))
	}
	var pwds []string
	for _, kv := range cmd.Env {
		name, value, _ := strings.Cut(kv, "=")
		if strings.EqualFold(name, "OLDPWD") {
			t.Fatalf("Apply kept launcher %q", kv)
		}
		if strings.EqualFold(name, "PWD") {
			pwds = append(pwds, value)
		}
	}
	if !slices.Equal(pwds, []string{dir}) {
		t.Fatalf("PWD entries = %q, want exactly [%s]", pwds, dir)
	}
}

// TestApplyExplicitEnvNamesDir pins that an explicit cmd.Env is corrected
// rather than replaced by os.Environ(), including an explicitly empty
// environment, and that its PWD names cmd.Dir.
func TestApplyExplicitEnvNamesDir(t *testing.T) {
	dir := cleanTempDir(t)
	t.Setenv("FAK_CHILDENV_TEST_MARK", "launcher-only")
	cases := []struct {
		name string
		env  []string
		want []string
	}{
		{"stale PWD in explicit env", []string{"A=1", "PWD=/stale", "oldpwd=/old", "B=2"}, []string{"A=1", "B=2", "PWD=" + dir}},
		{"explicit empty env", []string{}, []string{"PWD=" + dir}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cmd := &exec.Cmd{Dir: dir, Env: tc.env}
			Apply(cmd)
			if !slices.Equal(cmd.Env, tc.want) {
				t.Fatalf("Apply(Env=%q) = %q, want %q", tc.env, cmd.Env, tc.want)
			}
		})
	}
}

// TestApplyNamesChildPWDInRealProcess pins the package's reason to exist end
// to end: a real child launched through Apply sees a PWD naming its own
// working directory, not the launcher's stale value.
// fak-test:runtime fast est=1s
func TestApplyNamesChildPWDInRealProcess(t *testing.T) {
	dir := cleanTempDir(t)
	stale := t.TempDir()
	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append([]string{"PWD=" + stale, "OLDPWD=" + stale, "FAK_CHILDENV_PROBE=1"}, os.Environ()...)
	cmd.Dir = dir
	Apply(cmd)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("probe child failed: %v", err)
	}
	if got := strings.TrimSpace(string(out)); got != dir {
		t.Fatalf("child PWD = %q, want %q", got, dir)
	}
}

// BenchmarkForDirLargeEnv measures ForDir over a ~512-entry environment with
// stale PWD/OLDPWD, the per-launch cost paid by every harness launcher.
func BenchmarkForDirLargeEnv(b *testing.B) {
	env := make([]string, 0, 514)
	for i := 0; i < 512; i++ {
		env = append(env, fmt.Sprintf("VAR_%03d=value-%d", i, i))
	}
	env = append(env, "PWD=/stale-launcher-dir", "OLDPWD=/older")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ForDir(env, "C:\\fixturedir")
	}
}
