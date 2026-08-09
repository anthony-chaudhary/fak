package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestTestShFastMirrorRetriesRsyncExit23AndForwardsVerbose(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skip("bash unavailable")
	}
	root := repoRootForWrapperTest(t)
	dir := t.TempDir()
	bin := filepath.Join(dir, "bin")
	if err := os.Mkdir(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	countFile := filepath.Join(dir, "rsync-count")
	argsFile := filepath.Join(dir, "go-args")
	writeExecutable(t, filepath.Join(bin, "rsync"), `#!/usr/bin/env bash
set -euo pipefail
n=0
if [ -f "$FAK_FAKE_RSYNC_COUNT" ]; then n=$(cat "$FAK_FAKE_RSYNC_COUNT"); fi
n=$((n + 1))
printf '%s' "$n" > "$FAK_FAKE_RSYNC_COUNT"
dest="${@: -1}"
mkdir -p "$dest"
if [ "$n" -eq 1 ]; then exit 23; fi
exit 0
`)
	writeExecutable(t, filepath.Join(bin, "go"), `#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$FAK_FAKE_GO_ARGS"
exit 0
`)

	cmd := exec.Command(bash, filepath.Join(root, "test.sh"), "-v", "./internal/gateway")
	cmd.Env = append(os.Environ(),
		"PATH="+bin+string(os.PathListSeparator)+os.Getenv("PATH"),
		"FAK_FAST=1",
		"FAK_FAST_DIR="+filepath.Join(dir, "scratch"),
		"FAK_FAKE_RSYNC_COUNT="+countFile,
		"FAK_FAKE_GO_ARGS="+argsFile,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("test.sh failed: %v\n%s", err, out)
	}
	count, err := os.ReadFile(countFile)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(count)) != "2" {
		t.Fatalf("rsync calls = %q, want 2; output:\n%s", count, out)
	}
	rawArgs, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Fields(string(rawArgs))
	want := []string{"test", "-v", "./internal/gateway"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("go args = %v, want %v; output:\n%s", got, want, out)
	}
	if !strings.Contains(string(out), "exit 23") {
		t.Fatalf("retry should explain transient rsync exit 23; output:\n%s", out)
	}
}

func TestTestPs1ForwardsDashVToRest(t *testing.T) {
	powershell, ok := lookPathAny("powershell.exe", "powershell", "pwsh")
	if !ok {
		t.Skip("PowerShell unavailable")
	}
	root := repoRootForWrapperTest(t)
	script := filepath.Join(root, "test.ps1")
	if strings.HasSuffix(strings.ToLower(filepath.Base(powershell)), ".exe") || runtime.GOOS == "windows" {
		script = windowsPathForPowerShell(script)
	}
	command := "$env:FAK_TEST_PS1_ECHO_ARGS='1'; & " + powerShellSingleQuoted(script) + " -v ./internal/gateway"
	args := []string{"-NoProfile"}
	if strings.Contains(strings.ToLower(filepath.Base(powershell)), "powershell") {
		args = append(args, "-ExecutionPolicy", "Bypass")
	}
	args = append(args, "-Command", command)
	cmd := exec.Command(powershell, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("test.ps1 echo failed: %v\n%s", err, out)
	}
	var got []string
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("bad echoed args JSON: %v\n%s", err, out)
	}
	want := []string{"-v", "./internal/gateway"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("test.ps1 args = %v, want %v", got, want)
	}
}

// TestTestPs1PreservesShellMetacharsThroughWSL is the regression guard for #2248:
// a `go test` arg with shell metacharacters (a `-run 'A|B'` alternation or a
// `-run 'Test.*(A|B)'` group) must survive the Windows -> wsl.exe -> bash bridge as
// a single verbatim argument. Before the fix, test.ps1 let wsl.exe reparse the
// joined command through /bin/sh -c, so the pipe became a shell pipe and the parens
// a subshell; the run died with "command not found" or a syntax error. It drives
// the REAL test.ps1 with test.sh's FAK_TEST_SH_ECHO_ARGS hatch, so it observes
// exactly what reached bash's "$@".
func TestTestPs1PreservesShellMetacharsThroughWSL(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("the corruption is specific to the Windows -> wsl.exe command boundary")
	}
	powershell, ok := lookPathAny("powershell.exe", "powershell", "pwsh")
	if !ok {
		t.Skip("PowerShell unavailable")
	}
	if _, err := exec.LookPath("wsl.exe"); err != nil {
		t.Skip("wsl.exe unavailable")
	}
	root := repoRootForWrapperTest(t)
	// wsl.exe merely EXISTING does not mean WSL can SEE this checkout. When the drvfs
	// /mnt/<drive> automount is broken -- `wsl.exe -e ls /mnt/c` answers "Input/output
	// error" -- test.ps1's `wsl.exe -e bash /mnt/c/.../test.sh` cannot start at all,
	// and this test hard-fails on a box-level fault that has nothing to do with the
	// argument passing it guards. Probe the exact file test.ps1 will hand to bash and
	// skip when WSL cannot read it. This narrows only WHEN the test runs; it never
	// weakens WHAT it asserts, so a real #2248 regression on a healthy box still fails.
	if reason := wslCannotReachWrapperScript(filepath.Join(root, "test.sh")); reason != "" {
		t.Skip(reason)
	}
	script := filepath.Join(root, "test.ps1")

	// Each pattern carries a metacharacter a wrapping shell would reparse. Passed
	// as a single argv element, each must come back out of bash's "$@" unchanged.
	patterns := []string{
		"TestA|TestB",
		"Test.*(Compact|Overview)",
		"TestX|TestY|TestZ",
	}
	for _, pat := range patterns {
		pat := pat
		t.Run(pat, func(t *testing.T) {
			cmd := exec.Command(powershell, "-NoProfile", "-ExecutionPolicy", "Bypass",
				"-File", script, "./cmd/fak", "-run", pat)
			// FAK_TEST_SH_ECHO_ARGS makes test.sh echo its args and exit; WSLENV
			// carries it across into WSL (the /u flag = forward Windows -> WSL, the
			// same convention test.ps1 uses for FAK_FAST). FAK_FAST=0 keeps the run
			// off the ext4 mirror path (the hatch exits before it regardless).
			cmd.Env = wrapperTestEnv("FAK_TEST_SH_ECHO_ARGS=1", "WSLENV=FAK_TEST_SH_ECHO_ARGS/u", "FAK_FAST=0")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("test.ps1 echo run failed: %v\n%s", err, out)
			}
			got := splitNonEmptyLines(string(out))
			want := []string{"./cmd/fak", "-run", pat}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("args through the Windows -> WSL bridge = %#v, want %#v\nraw output:\n%s", got, want, out)
			}
		})
	}
}

// wslCannotReachWrapperScript reports, in one sentence fit for t.Skip, why WSL cannot
// read winPath from inside the distro -- or "" when it can, which is the only case in
// which a wsl.exe-bridged wrapper test is meaningful. It mirrors test.ps1's own path
// translation (lowercased drive letter under the default /mnt automount; test.ps1
// deliberately does not shell out to wslpath, because PowerShell mangles a `C:\...`
// argument on the way in) and honours FAK_WSL_DISTRO the same way, so the probe targets
// exactly the file test.ps1 would exec. A non-empty return always means a broken or
// absent /mnt mount on this box, never a regression in the code under test.
func wslCannotReachWrapperScript(winPath string) string {
	mount, ok := wslMountPathForWrapperTest(winPath)
	if !ok {
		return "cannot map " + winPath + " onto a WSL /mnt path"
	}
	args := []string{}
	if distro := os.Getenv("FAK_WSL_DISTRO"); distro != "" {
		args = append(args, "-d", distro)
	}
	// `test -r` is the cheapest question that fails for every flavour of unreachable:
	// automount off, drvfs I/O error, or the checkout genuinely absent inside WSL.
	args = append(args, "-e", "test", "-r", mount)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "wsl.exe", args...).CombinedOutput()
	if err == nil {
		return ""
	}
	msg := "WSL cannot read " + mount + ", so test.ps1 cannot reach this checkout" +
		" (the /mnt drvfs automount is broken or disabled on this box): " + err.Error()
	if detail := strings.TrimSpace(string(out)); detail != "" {
		msg += ": " + detail
	}
	return msg
}

// wslMountPathForWrapperTest translates `C:\a\b` to `/mnt/c/a/b`, the same inline
// translation test.ps1 performs. It reports false for anything that is not a
// drive-letter absolute path, since no /mnt mapping exists for those.
func wslMountPathForWrapperTest(winPath string) (string, bool) {
	if len(winPath) < 3 || winPath[1] != ':' || (winPath[2] != '\\' && winPath[2] != '/') {
		return "", false
	}
	drive := winPath[0]
	if drive >= 'A' && drive <= 'Z' {
		drive += 'a' - 'A'
	}
	if drive < 'a' || drive > 'z' {
		return "", false
	}
	return "/mnt/" + string(drive) + strings.ReplaceAll(winPath[2:], "\\", "/"), true
}

// wrapperTestEnv returns the current environment with any keys named in overrides
// removed, then overrides appended — so a caller-set WSLENV/FAK_FAST wins over an
// inherited one instead of leaving a duplicate the child shell resolves ambiguously.
func wrapperTestEnv(overrides ...string) []string {
	drop := map[string]bool{}
	for _, kv := range overrides {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			drop[kv[:i]] = true
		}
	}
	var env []string
	for _, kv := range os.Environ() {
		if i := strings.IndexByte(kv, '='); i >= 0 && drop[kv[:i]] {
			continue
		}
		env = append(env, kv)
	}
	return append(env, overrides...)
}

// splitNonEmptyLines splits on newlines, trims a trailing CR, and drops blank
// lines — the shape test.sh's echo hatch emits (one argument per line).
func splitNonEmptyLines(s string) []string {
	var out []string
	for _, ln := range strings.Split(s, "\n") {
		ln = strings.TrimRight(ln, "\r")
		if strings.TrimSpace(ln) != "" {
			out = append(out, ln)
		}
	}
	return out
}

func repoRootForWrapperTest(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func writeExecutable(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

func lookPathAny(names ...string) (string, bool) {
	for _, name := range names {
		if path, err := exec.LookPath(name); err == nil {
			return path, true
		}
	}
	return "", false
}

func windowsPathForPowerShell(path string) string {
	path = filepath.Clean(path)
	slash := filepath.ToSlash(path)
	parts := strings.SplitN(strings.TrimPrefix(slash, "/mnt/"), "/", 2)
	if len(parts) == 2 && len(parts[0]) == 1 {
		return strings.ToUpper(parts[0]) + `:\` + strings.ReplaceAll(parts[1], "/", `\`)
	}
	return path
}

func powerShellSingleQuoted(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}
