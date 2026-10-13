package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type execCommandAllowAdjudicator struct{}

func (execCommandAllowAdjudicator) Caps() []abi.Capability { return nil }
func (execCommandAllowAdjudicator) Adjudicate(_ context.Context, call *abi.ToolCall) abi.Verdict {
	if call.Tool == "exec_command" || call.Tool == codetools.ToolBash {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "exec-command-test"}
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "exec-command-test"}
}

type execCommandVerdictAdjudicator struct {
	original abi.VerdictKind
	adapted  abi.VerdictKind
}

func (execCommandVerdictAdjudicator) Caps() []abi.Capability { return nil }
func (a execCommandVerdictAdjudicator) Adjudicate(ctx context.Context, call *abi.ToolCall) abi.Verdict {
	kind := abi.VerdictDefer
	switch call.Tool {
	case "exec_command":
		kind = a.original
	case codetools.ToolBash:
		kind = a.adapted
	}
	verdict := abi.Verdict{Kind: kind, By: "exec-command-verdict-test"}
	if kind == abi.VerdictTransform {
		verdict.Payload = abi.TransformPayload{NewArgs: call.Args}
	}
	return verdict
}

// fak-test:runtime fast est=100ms lane=default
func TestFakSyscallExecCommandRefusesUnclaimedCommandRoute(t *testing.T) {
	wrong := &generatedResultEngine{}
	srv := newTestServer(t)
	srv.workspaceAdmissionPermissive = true
	abi.RegisterEngine("test", wrong)
	abi.RegisterAdjudicator(5, execCommandAllowAdjudicator{})

	resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "exec_command",
		"arguments": map[string]any{"cmd": "echo must-not-run"},
		"trace_id":  "exec-command-unclaimed",
	})
	if resp.Verdict.Kind != "DENY" || resp.Verdict.By != "exec-command-route" || resp.Result != nil {
		t.Fatalf("response = %+v, want typed unclaimed-route denial", resp)
	}
	if got := wrong.calls.Load(); got != 0 {
		t.Fatalf("default/model engine executed %d time(s), want 0", got)
	}
}

type generatedResultEngine struct{ calls atomic.Int64 }

func (*generatedResultEngine) Caps() []abi.Capability { return nil }
func (e *generatedResultEngine) Complete(context.Context, *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	payload := []byte(`{"generated_tokens":17,"finish_reason":"stop"}`)
	return &abi.Result{
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))},
		Meta:    map[string]string{"engine": "wrong-model"},
	}, nil
}

type commandResultEngine struct {
	calls   atomic.Int64
	payload string
}

func (*commandResultEngine) Caps() []abi.Capability { return nil }
func (e *commandResultEngine) Complete(context.Context, *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	payload := []byte(e.payload)
	return &abi.Result{
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: payload, Len: int64(len(payload))},
		Meta:    map[string]string{"engine": codetools.EngineBash},
	}, nil
}

func execCommandFixture(t *testing.T, wrong *generatedResultEngine) (*Server, string, string) {
	t.Helper()
	srv := newTestServer(t)
	srv.workspaceAdmissionPermissive = true
	abi.RegisterEngine("test", wrong)
	abi.RegisterAdjudicator(5, execCommandAllowAdjudicator{})
	command := "printf 'sentinel-out'; printf 'sentinel-err' >&2; exit 7"
	if runtime.GOOS == "windows" {
		command = "echo sentinel-out & echo sentinel-err 1>&2 & exit /b 7"
	}
	root := t.TempDir()
	if _, err := codetools.Register(codetools.Config{
		Root:                 root,
		ExactCommandsOnly:    true,
		ExactAllowedCommands: []string{command},
	}, 20); err != nil {
		t.Fatal(err)
	}
	return srv, root, command
}

// fak-test:runtime fast est=100ms lane=default
func TestFakSyscallExecCommandPreservesAdmittedCommandResult(t *testing.T) {
	wrong := &generatedResultEngine{}
	srv, root, command := execCommandFixture(t, wrong)

	resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool": "exec_command",
		"arguments": map[string]any{
			"cmd":     command,
			"workdir": root,
		},
		"read_only": true,
		"trace_id":  "exec-command-sentinel",
	})
	if resp.Verdict.Kind != "ALLOW" || resp.Result == nil || resp.Result.Status != "OK" {
		t.Fatalf("response = %+v, want admitted command result", resp)
	}
	var result codetools.BashResult
	if err := json.Unmarshal([]byte(resp.Result.Content), &result); err != nil {
		t.Fatalf("command result schema: %v; content=%q", err, resp.Result.Content)
	}
	if !strings.Contains(result.Stdout, "sentinel-out") || !strings.Contains(result.Stderr, "sentinel-err") || result.ExitCode != 7 {
		t.Fatalf("command result = %+v, want exact sentinels and exit 7", result)
	}
	if got := wrong.calls.Load(); got != 0 {
		t.Fatalf("default/model engine executed %d time(s), want 0", got)
	}
	for key, want := range map[string]string{
		"requested_tool":       "exec_command",
		"requested_route":      "exec_command",
		"executed_route":       codetools.EngineBash,
		"admitted_result_type": "codetools.bash-result/1",
	} {
		if got := resp.Result.Meta[key]; got != want {
			t.Errorf("result meta %s = %q, want %q", key, got, want)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestFakSyscallExecCommandRejectsMismatchedEngineResult(t *testing.T) {
	wrongDefault := &generatedResultEngine{}
	srv, root, command := execCommandFixture(t, wrongDefault)
	wrongCommand := &generatedResultEngine{}
	abi.RegisterEngine(codetools.EngineBash, wrongCommand)

	resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "exec_command",
		"arguments": map[string]any{"cmd": command, "workdir": root},
		"trace_id":  "exec-command-mismatch",
	})
	if resp.Verdict.Kind != "DENY" || resp.Verdict.By != "exec-command-result" || resp.Result != nil {
		t.Fatalf("response = %+v, want typed result-contract denial", resp)
	}
	if got := wrongDefault.calls.Load(); got != 0 {
		t.Fatalf("default/model engine executed %d time(s), want 0", got)
	}
	if got := wrongCommand.calls.Load(); got != 1 {
		t.Fatalf("mismatched command engine executed %d time(s), want 1", got)
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestFakSyscallExecCommandRejectsDispatchingNonAllowVerdicts(t *testing.T) {
	for _, tc := range []struct {
		name              string
		original, adapted abi.VerdictKind
	}{
		{name: "original transform", original: abi.VerdictTransform, adapted: abi.VerdictAllow},
		{name: "original defer", original: abi.VerdictDefer, adapted: abi.VerdictAllow},
		{name: "adapted transform", original: abi.VerdictAllow, adapted: abi.VerdictTransform},
		{name: "adapted defer", original: abi.VerdictAllow, adapted: abi.VerdictDefer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wrongDefault := &generatedResultEngine{}
			srv := newTestServer(t)
			srv.workspaceAdmissionPermissive = true
			abi.RegisterEngine("test", wrongDefault)
			abi.RegisterAdjudicator(5, execCommandVerdictAdjudicator{original: tc.original, adapted: tc.adapted})
			command := "echo must-not-run"
			root := t.TempDir()
			wrongCommand := &generatedResultEngine{}
			abi.RegisterEngine(codetools.EngineBash, wrongCommand)

			resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
				"tool":      "exec_command",
				"arguments": map[string]any{"cmd": command, "workdir": root},
				"trace_id":  "exec-command-non-allow",
			})
			if resp.Verdict.Kind != "DENY" || resp.Result != nil {
				t.Fatalf("response = %+v, want typed policy denial", resp)
			}
			if got := wrongDefault.calls.Load(); got != 0 {
				t.Fatalf("default/model engine executed %d time(s), want 0", got)
			}
			if got := wrongCommand.calls.Load(); got != 0 {
				t.Fatalf("command engine executed %d time(s), want 0", got)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestFakSyscallExecCommandPolicyDenyCannotExecute(t *testing.T) {
	wrongDefault := &generatedResultEngine{}
	srv := newTestServer(t)
	srv.workspaceAdmissionPermissive = true
	abi.RegisterEngine("test", wrongDefault)
	abi.RegisterAdjudicator(5, execCommandVerdictAdjudicator{original: abi.VerdictDeny, adapted: abi.VerdictAllow})
	command := "echo must-not-run"
	if _, err := codetools.Register(codetools.Config{
		Root:                 t.TempDir(),
		ExactCommandsOnly:    true,
		ExactAllowedCommands: []string{command},
	}, 20); err != nil {
		t.Fatal(err)
	}
	wrongCommand := &generatedResultEngine{}
	abi.RegisterEngine(codetools.EngineBash, wrongCommand)

	resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "exec_command",
		"arguments": map[string]any{"cmd": command},
		"trace_id":  "exec-command-policy-deny",
	})
	if resp.Verdict.Kind != "DENY" || resp.Verdict.By != "exec-command-verdict-test" || resp.Result != nil {
		t.Fatalf("response = %+v, want original policy denial", resp)
	}
	if got := wrongDefault.calls.Load(); got != 0 {
		t.Fatalf("default/model engine executed %d time(s), want 0", got)
	}
	if got := wrongCommand.calls.Load(); got != 0 {
		t.Fatalf("command engine executed %d time(s), want 0", got)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestFakSyscallExecCommandRejectsWorkdirOutsideRoot(t *testing.T) {
	wrongDefault := &generatedResultEngine{}
	srv, _, command := execCommandFixture(t, wrongDefault)
	wrongCommand := &generatedResultEngine{}
	abi.RegisterEngine(codetools.EngineBash, wrongCommand)

	resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "exec_command",
		"arguments": map[string]any{"cmd": command, "workdir": t.TempDir()},
		"trace_id":  "exec-command-workdir-escape",
	})
	if resp.Verdict.Kind != "DENY" || resp.Result != nil {
		t.Fatalf("response = %+v, want workdir confinement denial", resp)
	}
	if got := wrongDefault.calls.Load(); got != 0 {
		t.Fatalf("default/model engine executed %d time(s), want 0", got)
	}
	if got := wrongCommand.calls.Load(); got != 0 {
		t.Fatalf("command engine executed %d time(s), want 0", got)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestFakSyscallExecCommandRejectsInvalidResultScalars(t *testing.T) {
	for _, tc := range []struct {
		name, payload string
	}{
		{name: "null stdout", payload: `{"stdout":null,"stderr":"","exit_code":0}`},
		{name: "null stderr", payload: `{"stdout":"","stderr":null,"exit_code":0}`},
		{name: "null exit code", payload: `{"stdout":"","stderr":"","exit_code":null}`},
		{name: "numeric stdout", payload: `{"stdout":1,"stderr":"","exit_code":0}`},
		{name: "boolean stderr", payload: `{"stdout":"","stderr":false,"exit_code":0}`},
		{name: "string exit code", payload: `{"stdout":"","stderr":"","exit_code":"0"}`},
		{name: "null optional flag", payload: `{"stdout":"","stderr":"","exit_code":0,"timed_out":null}`},
		{name: "string optional flag", payload: `{"stdout":"","stderr":"","exit_code":0,"stdout_truncated":"false"}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wrongDefault := &generatedResultEngine{}
			srv, root, command := execCommandFixture(t, wrongDefault)
			invalid := &commandResultEngine{payload: tc.payload}
			abi.RegisterEngine(codetools.EngineBash, invalid)

			resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
				"tool":      "exec_command",
				"arguments": map[string]any{"cmd": command, "workdir": root},
				"trace_id":  "exec-command-invalid-result",
			})
			if resp.Verdict.Kind != "DENY" || resp.Verdict.By != "exec-command-result" || resp.Result != nil {
				t.Fatalf("response = %+v, want typed result-contract denial", resp)
			}
			if got := wrongDefault.calls.Load(); got != 0 {
				t.Fatalf("default/model engine executed %d time(s), want 0", got)
			}
			if got := invalid.calls.Load(); got != 1 {
				t.Fatalf("invalid command engine executed %d time(s), want 1", got)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestFakSyscallExecCommandAliasRecordsRouteTelemetry(t *testing.T) {
	wrongDefault := &generatedResultEngine{}
	srv, root, command := execCommandFixture(t, wrongDefault)
	var logs []string
	srv.logf = func(format string, args ...any) {
		logs = append(logs, fmt.Sprintf(format, args...))
	}

	resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "functions.exec_command",
		"arguments": map[string]any{"command": command, "cwd": root},
		"trace_id":  "exec-command-alias",
	})
	if resp.Verdict.Kind != "ALLOW" || resp.Result == nil || resp.Result.Status != "OK" {
		t.Fatalf("response = %+v, want admitted aliased command result", resp)
	}
	if got := resp.Result.Meta["requested_route"]; got != "functions.exec_command" {
		t.Fatalf("requested route = %q, want functions.exec_command", got)
	}

	var routeEvent map[string]any
	for _, line := range logs {
		var event map[string]any
		if json.Unmarshal([]byte(line), &event) == nil && event["event"] == "gateway_syscall_route" {
			routeEvent = event
			if strings.Contains(line, command) {
				t.Fatalf("route telemetry exposed command arguments: %s", line)
			}
		}
	}
	if routeEvent == nil {
		t.Fatalf("route telemetry missing from logs: %v", logs)
	}
	for key, want := range map[string]string{
		"requested_route":      "functions.exec_command",
		"executed_route":       codetools.EngineBash,
		"admitted_result_type": execCommandResultType,
	} {
		if got := routeEvent[key]; got != want {
			t.Errorf("telemetry %s = %v, want %q", key, got, want)
		}
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestFakSyscallExecCommandRejectsInvalidArguments(t *testing.T) {
	for _, tc := range []struct {
		name string
		args any
	}{
		{name: "null object", args: nil},
		{name: "null command", args: map[string]any{"cmd": nil}},
		{name: "null workdir", args: map[string]any{"cmd": "echo must-not-run", "workdir": nil}},
		{name: "null timeout", args: map[string]any{"cmd": "echo must-not-run", "timeout_ms": nil}},
		{name: "mismatched command aliases", args: map[string]any{"cmd": "echo one", "command": "echo two"}},
		{name: "mismatched workdir aliases", args: map[string]any{"cmd": "echo must-not-run", "workdir": "one", "cwd": "two"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wrongDefault := &generatedResultEngine{}
			srv := newTestServer(t)
			srv.workspaceAdmissionPermissive = true
			abi.RegisterEngine("test", wrongDefault)
			abi.RegisterAdjudicator(5, execCommandAllowAdjudicator{})

			resp := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
				"tool":      "functions.exec_command",
				"arguments": tc.args,
				"trace_id":  "exec-command-invalid-arguments",
			})
			if resp.Verdict.Kind != "DENY" || resp.Verdict.By != "exec-command-arguments" || resp.Result != nil {
				t.Fatalf("response = %+v, want typed argument denial", resp)
			}
			if got := wrongDefault.calls.Load(); got != 0 {
				t.Fatalf("default/model engine executed %d time(s), want 0", got)
			}
		})
	}
}
