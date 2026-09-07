package doshook

import (
	"bytes"
	"errors"
	"io"
	"path/filepath"
	"strings"
	"testing"
)

type reply struct {
	exitCode int
	stdout   []byte
	err      error
}

type callRecord struct {
	argv  []string
	stdin []byte
}

type fakeRunner struct {
	replies []reply
	calls   []callRecord
}

func newFakeRunner(replies ...reply) *fakeRunner {
	return &fakeRunner{
		replies: replies,
	}
}

func (f *fakeRunner) Run(argv []string, stdin []byte) (int, []byte, error) {
	argvCopy := make([]string, len(argv))
	copy(argvCopy, argv)
	stdinCopy := make([]byte, len(stdin))
	copy(stdinCopy, stdin)
	f.calls = append(f.calls, callRecord{argv: argvCopy, stdin: stdinCopy})

	if len(f.replies) == 0 {
		return 0, nil, nil
	}
	idx := len(f.calls) - 1
	if idx >= len(f.replies) {
		idx = len(f.replies) - 1
	}
	r := f.replies[idx]
	return r.exitCode, r.stdout, r.err
}

var (
	denyJSON    = []byte(`{"hookSpecificOutput":{"permissionDecision":"deny"}}`)
	testPayload = []byte(`{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}`)
)

func TestNativeOwnedReturnsNativeStdout(t *testing.T) {
	native := filepath.Join("tools", ".bin", "dos-hook")
	runner := newFakeRunner(reply{exitCode: 0, stdout: denyJSON})
	out, err := RunHook("pretool", ".", testPayload, native, runner.Run)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(out, denyJSON) {
		t.Fatalf("got %s, want %s", string(out), string(denyJSON))
	}
	if len(runner.calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(runner.calls))
	}
	if !strings.Contains(runner.calls[0].argv[0], "dos-hook") {
		t.Fatalf("expected native binary in argv[0], got %v", runner.calls[0].argv)
	}
}

func TestNativeDelegateFallsBackToPython(t *testing.T) {
	native := filepath.Join("tools", ".bin", "dos-hook")
	runner := newFakeRunner(
		reply{exitCode: 3, stdout: nil},
		reply{exitCode: 0, stdout: denyJSON},
	)
	out, err := RunHook("posttool", ".", testPayload, native, runner.Run)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(out, denyJSON) {
		t.Fatalf("got %s, want %s", string(out), string(denyJSON))
	}
	if len(runner.calls) != 2 {
		t.Fatalf("expected 2 calls, got %d", len(runner.calls))
	}
	if !bytes.Equal(runner.calls[0].stdin, testPayload) {
		t.Fatalf("first call stdin mismatch")
	}
	if !bytes.Equal(runner.calls[1].stdin, testPayload) {
		t.Fatalf("second call stdin mismatch")
	}
	cmdLine := strings.Join(runner.calls[1].argv, " ")
	if !strings.Contains(cmdLine, "dos.cli") {
		t.Fatalf("expected dos.cli in fallback argv, got %s", cmdLine)
	}
}

func TestNativeAbsentFallsBackToPython(t *testing.T) {
	runner := newFakeRunner(reply{exitCode: 0, stdout: []byte("ok")})
	out, err := RunHook("pretool", ".", testPayload, "", runner.Run)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(out) != "ok" {
		t.Fatalf("got %s, want ok", string(out))
	}
	if len(runner.calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(runner.calls))
	}
	cmdLine := strings.Join(runner.calls[0].argv, " ")
	if !strings.Contains(cmdLine, "dos.cli") {
		t.Fatalf("expected dos.cli in argv, got %s", cmdLine)
	}
}

func TestPythonExitCodeCoercedToZero(t *testing.T) {
	runner := newFakeRunner(reply{exitCode: 2, stdout: nil})
	out, err := RunHook("pretool", ".", testPayload, "", runner.Run)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("expected empty stdout, got %s", string(out))
	}

	var stdout bytes.Buffer
	rc := Run(
		[]string{"pretool", "--workspace", "."},
		bytes.NewReader(testPayload),
		&stdout,
		io.Discard,
		map[string]string{"CLAUDE_PROJECT_DIR": "/no/such/root"},
		runner.Run,
	)
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d", rc)
	}
	if stdout.Len() != 0 {
		t.Fatalf("expected empty stdout, got %s", stdout.String())
	}
}

func TestDenyPreservedFromNative(t *testing.T) {
	native := filepath.Join("tools", ".bin", "dos-hook")
	runner := newFakeRunner(reply{exitCode: 0, stdout: denyJSON})
	out, err := RunHook("pretool", ".", testPayload, native, runner.Run)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(out, denyJSON) {
		t.Fatalf("got %s, want %s", string(out), string(denyJSON))
	}
}

func TestDenyPreservedFromPython(t *testing.T) {
	runner := newFakeRunner(reply{exitCode: 2, stdout: denyJSON})
	out, err := RunHook("pretool", ".", testPayload, "", runner.Run)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !bytes.Equal(out, denyJSON) {
		t.Fatalf("got %s, want %s", string(out), string(denyJSON))
	}
}

func TestMainSucceedsEvenOnPanicOrError(t *testing.T) {
	boomRunner := func(argv []string, stdin []byte) (int, []byte, error) {
		panic("backend blew up")
	}
	rc := Run(
		[]string{"pretool", "--workspace", "."},
		bytes.NewReader(testPayload),
		io.Discard,
		io.Discard,
		map[string]string{"CLAUDE_PROJECT_DIR": "/no/such/root"},
		boomRunner,
	)
	if rc != 0 {
		t.Fatalf("expected rc 0 on panic, got %d", rc)
	}

	errRunner := func(argv []string, stdin []byte) (int, []byte, error) {
		return 1, nil, errors.New("backend error")
	}
	rc = Main(
		[]string{"pretool", "--workspace", "."},
		bytes.NewReader(testPayload),
		io.Discard,
		io.Discard,
		map[string]string{"CLAUDE_PROJECT_DIR": "/no/such/root"},
		errRunner,
	)
	if rc != 0 {
		t.Fatalf("expected rc 0 on error, got %d", rc)
	}
}

func TestWorkspaceArgParsed(t *testing.T) {
	runner := newFakeRunner(reply{exitCode: 0, stdout: []byte("ok")})
	var stdout bytes.Buffer
	rc := Run(
		[]string{"pretool", "--workspace", "custom/project/dir"},
		bytes.NewReader(testPayload),
		&stdout,
		io.Discard,
		map[string]string{"CLAUDE_PROJECT_DIR": "/no/such/root"},
		runner.Run,
	)
	if rc != 0 {
		t.Fatalf("expected rc 0, got %d", rc)
	}
	if len(runner.calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(runner.calls))
	}
	cmdLine := strings.Join(runner.calls[0].argv, " ")
	if !strings.Contains(cmdLine, "--workspace custom/project/dir") {
		t.Fatalf("expected custom workspace in argv, got %s", cmdLine)
	}

	runnerDefault := newFakeRunner(reply{exitCode: 0, stdout: []byte("ok")})
	Run(
		[]string{"pretool"},
		bytes.NewReader(testPayload),
		&stdout,
		io.Discard,
		map[string]string{"CLAUDE_PROJECT_DIR": "/no/such/root"},
		runnerDefault.Run,
	)
	if len(runnerDefault.calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(runnerDefault.calls))
	}
	cmdLineDefault := strings.Join(runnerDefault.calls[0].argv, " ")
	if !strings.Contains(cmdLineDefault, "--workspace .") {
		t.Fatalf("expected default '.' workspace in argv, got %s", cmdLineDefault)
	}
}

func TestRepoRootPrefersClaudeProjectDir(t *testing.T) {
	root := RepoRoot(map[string]string{"CLAUDE_PROJECT_DIR": "/some/ws"})
	if filepath.ToSlash(root) != "/some/ws" {
		t.Fatalf("got %s, want /some/ws", root)
	}
}

func TestNativeBinaryResolutionPrefersProjectDir(t *testing.T) {
	bin := NativeBinary("/no/such/root")
	if bin != "" {
		t.Fatalf("expected empty string for nonexistent root, got %s", bin)
	}
}
