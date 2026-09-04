package codetools

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type bashEngine struct{ t *Toolset }

func (bashEngine) Caps() []abi.Capability { return nil }
func (bashEngine) WeightBearing() bool    { return false }
func (e bashEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, bad := e.t.bash(ctx, in)
	return result(ctx, c, in, out, bad, EngineBash), nil
}

type BashResult struct {
	Stdout          string `json:"stdout"`
	Stderr          string `json:"stderr"`
	ExitCode        int    `json:"exit_code"`
	TimedOut        bool   `json:"timed_out,omitempty"`
	StdoutTruncated bool   `json:"stdout_truncated,omitempty"`
	StderrTruncated bool   `json:"stderr_truncated,omitempty"`
}

type boundedBuffer struct {
	b         bytes.Buffer
	max       int
	truncated bool
}

func (w *boundedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	room := w.max - w.b.Len()
	if room > 0 {
		if room > len(p) {
			room = len(p)
		}
		_, _ = w.b.Write(p[:room])
	}
	if n > room {
		w.truncated = true
	}
	return n, nil
}
func (w *boundedBuffer) String() string { return w.b.String() }

func (t *Toolset) bash(ctx context.Context, body []byte) ([]byte, bool) {
	if r := canceled(ctx); r != nil {
		return r.JSON(), true
	}
	var a BashArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if t.focusedCommands && !focusedCommandAllowed(a.Command) {
		return refuse(CodeCommandDeny, "command is outside the focused coding allowlist").JSON(), true
	}
	cwd := t.root
	var cwdWarning string
	if a.Cwd != "" {
		resolved, r := t.resolve(a.Cwd)
		if r != nil {
			return r.JSON(), true
		}
		cwd = filepath.Clean(resolved.Abs)
		if fi, err := os.Stat(cwd); err != nil || !fi.IsDir() {
			if rootFi, rootErr := os.Stat(t.root); rootErr == nil && rootFi.IsDir() {
				cwdWarning = fmt.Sprintf("working directory %s does not exist; executed in %s\n", cwd, t.root)
				cwd = t.root
			} else {
				return refuse(CodeNotFound, fmt.Sprintf("DIRECTORY_NOT_FOUND: working directory %q does not exist", cwd)).JSON(), true
			}
		}
	} else if fi, err := os.Stat(cwd); err != nil || !fi.IsDir() {
		return refuse(CodeNotFound, fmt.Sprintf("DIRECTORY_NOT_FOUND: working directory %q does not exist", cwd)).JSON(), true
	}
	timeout := t.limits.MaxCommandTime
	if a.TimeoutMS > 0 && time.Duration(a.TimeoutMS)*time.Millisecond < timeout {
		timeout = time.Duration(a.TimeoutMS) * time.Millisecond
	}
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	gotmpDir := t.ensureGoTmpDir()
	command := t.rewriteCommandForContainment(a.Command, cwd, gotmpDir)
	name, args := shellCommand(command)
	cmd := exec.CommandContext(runCtx, name, args...)
	cmd.Dir = cwd
	cmd.Env = enforceGoTmpEnv(os.Environ(), gotmpDir)
	var stdout, stderr boundedBuffer
	stdout.max = t.limits.MaxOutputBytes
	stderr.max = t.limits.MaxOutputBytes
	if cwdWarning != "" {
		_, _ = stderr.Write([]byte(cwdWarning))
	}
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	configureProcessTree(cmd)
	err := cmd.Start()
	if err == nil {
		err = cmd.Wait()
	}
	t.containmentCleanup(gotmpDir)
	timedOut := errors.Is(runCtx.Err(), context.DeadlineExceeded)
	exitCode := 0
	if err != nil {
		exitCode = -1
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			exitCode = ee.ExitCode()
		}
	}
	out := okJSON(BashResult{Stdout: stdout.String(), Stderr: stderr.String(), ExitCode: exitCode, TimedOut: timedOut, StdoutTruncated: stdout.truncated, StderrTruncated: stderr.truncated})
	if runCtx.Err() != nil && !timedOut && errors.Is(runCtx.Err(), context.Canceled) {
		return refuse(CodeCanceled, runCtx.Err().Error()).JSON(), true
	}
	return out, false
}

func shellCommand(command string) (string, []string) {
	if runtime.GOOS == "windows" {
		return "cmd.exe", []string{"/d", "/s", "/c", command}
	}
	return "/bin/sh", []string{"-c", command}
}

func focusedCommandAllowed(command string) bool {
	if strings.ContainsAny(command, "\r\n;&|<>`$%") {
		return false
	}
	fields := strings.Fields(command)
	if len(fields) < 2 {
		return false
	}
	for _, field := range fields {
		if strings.Trim(field, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./:=,@+-") != "" {
			return false
		}
	}
	switch fields[0] {
	case "go":
		return fields[1] == "test"
	case "git":
		return fields[1] == "diff" || (fields[1] == "status" && len(fields) == 3 && fields[2] == "--short")
	default:
		return false
	}
}
