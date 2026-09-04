package childproc

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	helperEnv     = "CHILDPROC_TEST_HELPER"
	helperCodeEnv = "CHILDPROC_TEST_CODE"
)

// TestHelperProcess is the subprocess re-exec target for hermetic childproc tests.
// Invariant: inert pass when helperEnv is not set in environment.
func TestHelperProcess(t *testing.T) {
	mode := os.Getenv(helperEnv)
	if mode == "" {
		return
	}
	switch mode {
	case "echo":
		_, _ = os.Stdout.WriteString("stdout-marker-data")
		_, _ = os.Stderr.WriteString("stderr-marker-data")
		os.Exit(0)
	case "exit":
		codeStr := os.Getenv(helperCodeEnv)
		code, err := strconv.Atoi(codeStr)
		if err != nil {
			code = 1
		}
		os.Exit(code)
	case "sleep":
		time.Sleep(15 * time.Second)
		os.Exit(0)
	case "env":
		_, _ = os.Stdout.WriteString(os.Getenv("CHILDPROC_CUSTOM_ENV"))
		os.Exit(0)
	case "pwd":
		dir, _ := os.Getwd()
		_, _ = os.Stdout.WriteString(dir)
		os.Exit(0)
	case "stdin":
		b, _ := io.ReadAll(os.Stdin)
		_, _ = os.Stdout.Write(b)
		os.Exit(0)
	case "spam":
		chunk := bytes.Repeat([]byte("0123456789abcdef"), 1024) // 16 KB
		for i := 0; i < 32; i++ {
			_, _ = os.Stdout.Write(chunk)
		}
		os.Exit(0)
	case "grandchild":
		gc := exec.Command(os.Args[0], "-test.run=^TestHelperProcess$")
		gc.Env = append(os.Environ(), helperEnv+"=sleep")
		_ = gc.Start()
		time.Sleep(15 * time.Second)
		os.Exit(0)
	default:
		os.Exit(0)
	}
}

func newTestHelperCommand(mode string, extraEnv ...string) *Command {
	cmd := NewCommand(os.Args[0], "-test.run=^TestHelperProcess$")
	cmd.Env = append(os.Environ(), helperEnv+"="+mode)
	cmd.Env = append(cmd.Env, extraEnv...)
	return cmd
}

func TestRunBasicCommand(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("echo")

	res, err := cmd.Run(ctx)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	if res == nil {
		t.Fatal("expected non-nil Result")
	}
	if res.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", res.ExitCode)
	}
	if !res.Success() {
		t.Errorf("Success() = false, want true")
	}
	if res.TimedOut {
		t.Errorf("TimedOut = true, want false")
	}
	if res.Duration <= 0 {
		t.Errorf("Duration = %v, want > 0", res.Duration)
	}
	if res.Err() != nil {
		t.Errorf("res.Err() = %v, want nil", res.Err())
	}
}

func TestOutputCapture(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("echo")

	res, err := cmd.Run(ctx)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	wantOut := "stdout-marker-data"
	if string(res.Stdout) != wantOut {
		t.Errorf("Stdout = %q, want %q", string(res.Stdout), wantOut)
	}
	if res.StdoutString() != wantOut {
		t.Errorf("StdoutString() = %q, want %q", res.StdoutString(), wantOut)
	}
}

func TestErrorCapture(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("echo")

	res, err := cmd.Run(ctx)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}
	wantErr := "stderr-marker-data"
	if string(res.Stderr) != wantErr {
		t.Errorf("Stderr = %q, want %q", string(res.Stderr), wantErr)
	}
	if res.StderrString() != wantErr {
		t.Errorf("StderrString() = %q, want %q", res.StderrString(), wantErr)
	}
}

func TestLaunchFailure(t *testing.T) {
	ctx := context.Background()
	cmd := NewCommand("non_existent_executable_12345_xyz_impossible")

	res, err := cmd.Run(ctx)
	if err == nil {
		t.Fatal("expected error for non-existent binary, got nil")
	}
	if res == nil {
		t.Fatal("expected non-nil Result on launch failure")
	}
	// Guard: fail-closed exit code propagation
	if res.ExitCode != ExitCodeUnknown {
		t.Errorf("ExitCode = %d, want %d", res.ExitCode, ExitCodeUnknown)
	}
	if res.Success() {
		t.Errorf("Success() = true, want false")
	}
	if res.TimedOut {
		t.Errorf("TimedOut = true, want false")
	}
}

func TestEnvironmentAndWorkingDir(t *testing.T) {
	ctx := context.Background()

	// Test custom environment
	cmdEnv := newTestHelperCommand("env", "CHILDPROC_CUSTOM_ENV=verified-env-token")
	resEnv, err := cmdEnv.Run(ctx)
	if err != nil {
		t.Fatalf("Run env failed: %v", err)
	}
	if resEnv.StdoutString() != "verified-env-token" {
		t.Errorf("StdoutString() = %q, want verified-env-token", resEnv.StdoutString())
	}

	// Test custom directory
	tempDir := t.TempDir()
	cmdDir := newTestHelperCommand("pwd")
	cmdDir.Dir = tempDir
	resDir, err := cmdDir.Run(ctx)
	if err != nil {
		t.Fatalf("Run pwd failed: %v", err)
	}
	gotDir := strings.TrimSpace(resDir.StdoutString())
	normGotDir, _ := filepath.EvalSymlinks(gotDir)
	normTempDir, _ := filepath.EvalSymlinks(tempDir)
	if !strings.EqualFold(gotDir, tempDir) && !strings.EqualFold(normGotDir, normTempDir) {
		t.Errorf("Dir = %q, want %q", gotDir, tempDir)
	}
}

func TestStdinCapture(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("stdin")
	cmd.Stdin = bytes.NewReader([]byte("sample-stream-input-bytes"))

	res, err := cmd.Run(ctx)
	if err != nil {
		t.Fatalf("Run stdin failed: %v", err)
	}
	if res.StdoutString() != "sample-stream-input-bytes" {
		t.Errorf("StdoutString() = %q, want sample-stream-input-bytes", res.StdoutString())
	}
}

func TestBoundedBufferTruncation(t *testing.T) {
	ctx := context.Background()
	cmd := newTestHelperCommand("spam")
	cmd.MaxOutputBytes = 1024

	res, err := cmd.Run(ctx)
	if err != nil {
		t.Fatalf("Run spam failed: %v", err)
	}
	if int64(len(res.Stdout)) != 1024 {
		t.Errorf("len(Stdout) = %d, want 1024", len(res.Stdout))
	}
}

func TestNilAndEmptyCommandGuards(t *testing.T) {
	ctx := context.Background()

	var nilCmd *Command
	if _, err := nilCmd.Run(ctx); err == nil {
		t.Error("nil Command.Run should return error")
	}

	emptyCmd := &Command{}
	if _, err := emptyCmd.Run(ctx); err == nil {
		t.Error("empty Command.Run should return error")
	}

	var nilRes *Result
	if nilRes.StdoutString() != "" {
		t.Errorf("nil Result.StdoutString() = %q, want empty", nilRes.StdoutString())
	}
	if nilRes.StderrString() != "" {
		t.Errorf("nil Result.StderrString() = %q, want empty", nilRes.StderrString())
	}
	if nilRes.Success() {
		t.Errorf("nil Result.Success() = true, want false")
	}
	if nilRes.Err() == nil {
		t.Errorf("nil Result.Err() = nil, want error")
	}
}
