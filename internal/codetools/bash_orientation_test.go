package codetools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFocusedCommandDeniesFlagSmuggling(t *testing.T) {
	for _, command := range []string{
		"go build --overlay=ov.json ./...",
		"go build --toolexec ./x ./...",
		"go vet --vettool ./x ./...",
		"go build --modfile=m.mod ./...",
		"go build --modfile m.mod ./...",
		"go test -toolexec=./x ./...",
		"go test -exec ./x ./...",
		"go test -c -o C:/x.exe ./x",
		"go build -C /tmp ./...",
		"go build -mod=mod ./...",
		"go build -mod mod ./...",
		"go test -modcacherw ./...",
		"go list -m -json exfil/data@v1",
		"go build ./internal/... @",
		"git diff --output=C:/x",
		"git diff --output C:/x",
		"git diff --outp=C:/x",
		"git log --ext-diff",
		"git log --textconv",
		"git diff --no-index a b",
		"git status --output=C:/x",
		"git diff -OC:/Windows/win.ini",
		"git log -O../../x",
		"git diff --orderfile=x",
	} {
		if focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=true, want denied", command)
		}
	}
}

func TestFocusedCommandKeepsReadOnlyFormsAllowed(t *testing.T) {
	for _, command := range []string{
		"go test ./internal/codetools -run X -count=1",
		"go build ./...",
		"go vet ./...",
		"go list ./...",
		"git diff",
		"git diff --stat",
		"git log --oneline -3",
		"git status --short",
		"ls",
		"ls -la cmd",
		"pwd",
		"go version",
		"gofmt -l .",
	} {
		if !focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=false, want allowed", command)
		}
	}
}

func TestBashNativeOrientationLsAndPwd(t *testing.T) {
	ts, root := newTestToolset(t)
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "a.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".hidden"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	res, ok := ts.nativeOrientation("ls", root)
	if !ok || res.Stdout != "a.txt\nsub\n" || res.ExitCode != 0 {
		t.Fatalf("ls: ok=%v stdout=%q exit=%d", ok, res.Stdout, res.ExitCode)
	}
	res, ok = ts.nativeOrientation("ls -a", root)
	if !ok || !strings.Contains(res.Stdout, ".hidden\n") || !strings.Contains(res.Stdout, "..\n") {
		t.Fatalf("ls -a: ok=%v stdout=%q", ok, res.Stdout)
	}
	res, ok = ts.nativeOrientation("ls -la sub", root)
	if !ok || res.ExitCode != 0 || res.Stderr != "" {
		t.Fatalf("ls -la sub: ok=%v exit=%d stderr=%q", ok, res.ExitCode, res.Stderr)
	}
	res, ok = ts.nativeOrientation("ls /etc", root)
	if !ok || res.ExitCode == 0 || res.Stderr == "" || strings.Contains(res.Stdout, "passwd") {
		t.Fatalf("ls /etc must be refused outside the workspace: ok=%v exit=%d stdout=%q stderr=%q", ok, res.ExitCode, res.Stdout, res.Stderr)
	}
	res, ok = ts.nativeOrientation("ls --color=auto", root)
	if !ok || res.ExitCode != 2 || !strings.Contains(res.Stderr, "unsupported option") {
		t.Fatalf("ls unsupported option: ok=%v exit=%d stderr=%q", ok, res.ExitCode, res.Stderr)
	}

	res, ok = ts.nativeOrientation("pwd", root)
	if !ok || res.Stdout != ".\n" {
		t.Fatalf("pwd at root: ok=%v stdout=%q", ok, res.Stdout)
	}
	res, ok = ts.nativeOrientation("pwd", filepath.Join(root, "sub"))
	if !ok || res.Stdout != "sub\n" {
		t.Fatalf("pwd in sub: ok=%v stdout=%q", ok, res.Stdout)
	}
	if _, ok := ts.nativeOrientation("go version", root); ok {
		t.Fatal("only ls and pwd are served in-process")
	}
}

func TestBashWindowsLsAndPwdRunWithoutCmdExe(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("in-process ls and pwd replace cmd.exe only on Windows")
	}
	ts, root := newTestToolset(t)
	if err := os.WriteFile(filepath.Join(root, "marker.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "ls"}))
	if bad {
		t.Fatalf("ls refused: %s", out)
	}
	if got := decodeResult(t, out)["stdout"].(string); !strings.Contains(got, "marker.txt") {
		t.Fatalf("ls stdout=%q, want marker.txt", got)
	}
	out, bad = ts.bash(context.Background(), argsOf(t, BashArgs{Command: "pwd"}))
	if bad {
		t.Fatalf("pwd refused: %s", out)
	}
	if got := decodeResult(t, out)["stdout"].(string); got != ".\n" {
		t.Fatalf("pwd stdout=%q, want workspace-relative .", got)
	}
}

func TestBashWindowsCwdExecutableHijackBlocked(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("cmd.exe searches the working directory for executables only on Windows")
	}
	ts, root := newTestToolset(t)
	const marker = "HIJACKED-GO-BAT"
	if err := os.WriteFile(filepath.Join(root, "go.bat"), []byte("@echo off\r\necho "+marker+"\r\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: "go version"}))
	if bad {
		t.Fatalf("go version refused: %s", out)
	}
	stdout := decodeResult(t, out)["stdout"].(string)
	if strings.Contains(stdout, marker) {
		t.Fatalf("workspace go.bat ran in place of the toolchain: %q", stdout)
	}
	if !strings.Contains(stdout, "go version") {
		t.Fatalf("real go did not run: %q", stdout)
	}
}

func TestFocusedGoAllowlistRefusesFlagLeaks(t *testing.T) {
	for _, command := range []string{
		"go test -outputdir=C:/x ./...",
		"go test -cpuprofile ../x ./...",
		"go test -test.cpuprofile=C:/x ./...",
		"go test -args -test.outputdir=/tmp ./...",
		"go test -fuzzcachedir=C:/x ./...",
		"go test -memprofile=C:/x ./...",
		"go test -ldflags=-linkmode=external -ldflags=-extld=./x ./...",
		"go test -extldflags=-wrapper,./x ./...",
		"go build -compiler=gccgo ./...",
		"go test -trace=C:/x ./...",
		"go test -coverprofile=C:/x ./...",
		"go build -gcflags=-S ./...",
		"go build -asmflags=-S ./...",
		"go build -buildmode=plugin ./...",
		"go build -pkgdir=x ./...",
		"go build -x ./...",
		"go test -c ./x",
		"go test -exec=./x ./...",
		"go test -e -o ./x",
		"go test -- ./...",
		"go test -run 'TestA|TestB' -v ./x",
		"go test -run",
		"go test -count=abc ./...",
		"go test -tags ../x ./...",
		"go test -covermode=foo ./...",
		"go test -vet=../x ./...",
		"go test -shuffle=../x ./...",
		"go test -p=../x ./...",
		"go test -timeout 1s;rm ./...",
		"go list -f {{.Dir}} ./...",
		"go list -format=x ./...",
		"go list -m all",
		"go list -export ./...",
		"go build ../x",
		"go build C:/x",
		"go build /abs/x",
		"go build x@v1",
		"go build -test.v ./...",
		"go build -modcacherw ./...",
		"go build -overlay=ov.json ./...",
		"go test -run=X -vet=off ./x -args -test.v",
		"git diff -- ../secret",
		"git diff -- /etc/passwd",
		"git diff -- C:/x",
		"git diff --relative=../x",
		"git log -- ../x",
		"git status --output=C:/x",
	} {
		if focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=true, want denied", command)
		}
	}
}

func TestFocusedCommandAllowlistKeepsWorkflowForms(t *testing.T) {
	for _, command := range []string{
		"go test ./internal/codetools -run X -count=1",
		"go test ./... -race -short -timeout 120s",
		"go test -bench . -benchmem -benchtime 2x ./x",
		"go test -count 1 -v -cover -covermode=atomic -cpu 1,2 -parallel 4 -skip Slow -list Test ./x",
		"go test -shuffle on -failfast -json ./x",
		"go build ./...",
		"go build -tags cuda ./...",
		"go build -p 4 -trimpath -race ./x",
		"go vet ./...",
		"go list ./...",
		"go list -deps ./x",
		"go list -e -find ./x",
		"go version",
		"gofmt -l .",
		"gofmt -s -l internal",
		"gofmt -d -e x.go",
		"git diff",
		"git diff --stat",
		"git diff -- cmd/x.go",
		"git log --oneline -3",
		"git status --short",
		"ls",
		"ls -la",
		"ls -la cmd",
		"pwd",
	} {
		if !focusedCommandAllowed(command) {
			t.Errorf("focusedCommandAllowed(%q)=false, want allowed", command)
		}
	}
}

func TestGitInvocationHardening(t *testing.T) {
	if got := hardenGitArgs("git log --oneline -3"); got != "git log --no-ext-diff --no-textconv --oneline -3" {
		t.Errorf("hardenGitArgs(log)=%q", got)
	}
	if got := hardenGitArgs("git status --short"); got != "git status --short" {
		t.Errorf("hardenGitArgs(status)=%q, want unchanged", got)
	}
	env := gitHardenedEnv([]string{"PATH=/bin", "GIT_EXTERNAL_DIFF=evil", "git_pager=less", "GIT_CONFIG_COUNT=9", "GIT_CONFIG_KEY_7=core.sshCommand"})
	joined := strings.Join(env, "\n")
	for _, want := range []string{"PATH=/bin", "GIT_PAGER=cat", "GIT_CONFIG_COUNT=3", "GIT_CONFIG_KEY_1=core.fsmonitor", "GIT_CONFIG_VALUE_1=false", "GIT_CONFIG_VALUE_2=cat"} {
		if !strings.Contains(joined, want) {
			t.Errorf("git env missing %q in %q", want, joined)
		}
	}
	for _, drop := range []string{"GIT_EXTERNAL_DIFF=", "GIT_CONFIG_COUNT=9", "GIT_CONFIG_KEY_7="} {
		if strings.Contains(joined, drop) {
			t.Errorf("git env kept inherited %q", drop)
		}
	}
}
