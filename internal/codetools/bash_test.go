package codetools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func bashEcho(text string) string {
	if runtime.GOOS == "windows" {
		return "echo " + text
	}
	return "printf " + text
}
func bashExit(code int) string {
	if runtime.GOOS == "windows" {
		return "exit /b " + fmt.Sprint(code)
	}
	return "exit " + fmt.Sprint(code)
}
func bashSleep() string {
	if runtime.GOOS == "windows" {
		return "ping -n 6 127.0.0.1 >nul"
	}
	return "sleep 5"
}
func bashWriteFile(name, body string) string {
	if runtime.GOOS == "windows" {
		return "echo " + body + ">" + name
	}
	return "printf " + body + ">" + name
}

func TestBashAllowedCommandCwdAndExitStatus(t *testing.T) {
	ts, root := newTestToolset(t)
	if err := os.Mkdir(filepath.Join(root, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashEcho("hello"), Cwd: "sub"}))
	if bad {
		t.Fatalf("bash: %s", out)
	}
	got := decodeResult(t, out)
	if !strings.Contains(got["stdout"].(string), "hello") || int(got["exit_code"].(float64)) != 0 {
		t.Fatalf("result=%v", got)
	}
	out, bad = ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashExit(7)}))
	if bad {
		t.Fatalf("exit: %s", out)
	}
	got = decodeResult(t, out)
	if int(got["exit_code"].(float64)) != 7 {
		t.Fatalf("exit=%v", got)
	}
}

func TestBashConfinementDefaultDenyAndCacheScope(t *testing.T) {
	ts, _ := newTestToolset(t)
	body := argsOf(t, BashArgs{Command: bashEcho("x"), Cwd: "../escape"})
	c := &abi.ToolCall{Tool: ToolBash, Args: abi.Ref{Kind: abi.RefInline, Inline: body}, Meta: CallMeta(ToolBash, "tenant")}
	v := ts.Adjudicate(context.Background(), c)
	if v.Kind != abi.VerdictDeny || v.Meta["code"] != CodePathEscape {
		t.Fatalf("escape=%+v", v)
	}
	denied, _ := New(Config{Root: t.TempDir(), Policy: Policy{Allow: map[string]bool{ToolRead: true}}})
	c.Args.Inline = argsOf(t, BashArgs{Command: bashEcho("x")})
	v = denied.Adjudicate(context.Background(), c)
	if v.Meta["code"] != CodeDefaultDeny {
		t.Fatalf("default deny=%+v", v)
	}
	c.Meta = map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
	v = ts.Adjudicate(context.Background(), c)
	if v.Meta["code"] != CodeCacheScope {
		t.Fatalf("cache scope=%+v", v)
	}
}

func TestBashTimeoutCancellationAndBoundedOutput(t *testing.T) {
	ts, _ := New(Config{Root: t.TempDir(), Limits: Limits{MaxOutputBytes: 4, MaxCommandTime: 100 * time.Millisecond}})
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashEcho("abcdefgh")}))
	if bad {
		t.Fatalf("output: %s", out)
	}
	got := decodeResult(t, out)
	if got["stdout_truncated"] != true || len(got["stdout"].(string)) != 4 {
		t.Fatalf("bounded=%v", got)
	}
	out, bad = ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashSleep()}))
	if bad {
		t.Fatalf("timeout: %s", out)
	}
	got = decodeResult(t, out)
	if got["timed_out"] != true {
		t.Fatalf("timeout=%v", got)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, bad = ts.bash(ctx, argsOf(t, BashArgs{Command: bashEcho("never")}))
	if !bad || errCode(t, out) != CodeCanceled {
		t.Fatalf("canceled=%s", out)
	}
}

func TestBashScratchProcessWitness(t *testing.T) {
	ts, root := newTestToolset(t)
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashWriteFile("witness.txt", "kernel"), Cwd: "."}))
	if bad {
		t.Fatalf("bash: %s", out)
	}
	b, err := os.ReadFile(filepath.Join(root, "witness.txt"))
	if err != nil || !strings.Contains(string(b), "kernel") {
		t.Fatalf("witness=%q err=%v", b, err)
	}
}

func TestBashNonExistentCwdFallsBackToRootWithWarning(t *testing.T) {
	ts, _ := newTestToolset(t)
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashEcho("recovered"), Cwd: "vanished_worktree"}))
	if bad {
		t.Fatalf("bash with missing cwd should gracefully fallback: %s", out)
	}
	got := decodeResult(t, out)
	if !strings.Contains(got["stdout"].(string), "recovered") {
		t.Fatalf("stdout=%v, want recovered", got["stdout"])
	}
	if !strings.Contains(got["stderr"].(string), "does not exist; executed in") {
		t.Fatalf("stderr=%v, want fallback warning", got["stderr"])
	}
}

func TestBashMissingRootRefusesWithDirectoryNotFound(t *testing.T) {
	temp := t.TempDir()
	deletedRoot := filepath.Join(temp, "deleted_root")
	ts, _ := New(Config{Root: deletedRoot})
	out, bad := ts.bash(context.Background(), argsOf(t, BashArgs{Command: bashEcho("test")}))
	if !bad {
		t.Fatalf("expected refusal when root does not exist, got %s", out)
	}
	if !strings.Contains(string(out), CodeNotFound) || !strings.Contains(string(out), "DIRECTORY_NOT_FOUND") {
		t.Fatalf("expected DIRECTORY_NOT_FOUND refusal, got %s", out)
	}
}
