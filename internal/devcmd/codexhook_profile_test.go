package devcmd

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// fak-test:justify why=contract when=changed:internal/devcmd/codexhook_profile.go
// fak-test:runtime fast est=3s lane=default
func TestInspectCodexHookProfileContextForwardsCallerContext(t *testing.T) {
	oldCommand := codexHookAppServerCommand
	defer func() { codexHookAppServerCommand = oldCommand }()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var commandContext context.Context
	codexHookAppServerCommand = func(got context.Context, _ string, _ ...string) *exec.Cmd {
		commandContext = got
		return exec.CommandContext(got, os.Args[0], "-test.run=^TestCodexHookProfileAppServerHelper$", "-test.count=1", "--", "reply")
	}
	_, _ = inspectCodexHookProfileContext(ctx, t.TempDir(), t.TempDir(), os.Args[0])
	if commandContext != ctx {
		t.Fatal("profile inspection replaced the caller's cancellation context")
	}
}

// fak-test:justify why=regression when=changed:internal/devcmd/codexhook_profile.go
// fak-test:runtime fast est=3s lane=default
func TestQueryCodexHooksContextReapsDescendantHeldPipe(t *testing.T) {
	dir := t.TempDir()
	rootPIDPath := filepath.Join(dir, "root-pid")
	descendantPIDPath := filepath.Join(dir, "descendant-pid")
	oldCommand := codexHookAppServerCommand
	defer func() { codexHookAppServerCommand = oldCommand }()
	codexHookAppServerCommand = func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		return exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCodexHookProfileAppServerHelper$", "-test.count=1", "--", "parent", rootPIDPath, descendantPIDPath)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	var queryErr error
	descendantPID := 0
	home, workspace := t.TempDir(), t.TempDir()
	readPID := func(path string) int {
		pidBytes, err := os.ReadFile(path)
		if err != nil {
			return 0
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(pidBytes)))
		if err != nil || pid <= 0 {
			return 0
		}
		return pid
	}
	t.Cleanup(func() {
		cancel()
		for _, pid := range []int{readPID(rootPIDPath), readPID(descendantPIDPath)} {
			if pid > 0 && processalive.Check(pid) {
				_, _ = procguard.KillPID(pid)
			}
		}
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("query goroutine did not join during cleanup")
		}
	})
	go func() {
		_, queryErr = queryCodexHooksContext(ctx, home, workspace, os.Args[0])
		close(done)
	}()

	deadline := time.Now().Add(5 * time.Second)
	for {
		pidBytes, err := os.ReadFile(descendantPIDPath)
		if err == nil {
			descendantPID, err = strconv.Atoi(strings.TrimSpace(string(pidBytes)))
			if err != nil || descendantPID <= 0 {
				t.Fatalf("invalid descendant pid %q: %v", pidBytes, err)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("descendant did not reach cancellation handshake")
		}
		time.Sleep(10 * time.Millisecond)
	}
	cancel()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("query did not return after cancellation")
	}
	if !errors.Is(queryErr, context.Canceled) {
		t.Fatalf("query error=%v, want context cancellation", queryErr)
	}
	deathDeadline := time.Now().Add(5 * time.Second)
	for processalive.Check(descendantPID) {
		if time.Now().After(deathDeadline) {
			t.Fatalf("descendant pid %d remained alive after query reaped", descendantPID)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestCodexHookProfileAppServerHelper(t *testing.T) {
	separator := slices.Index(os.Args, "--")
	if separator < 0 || separator+1 >= len(os.Args) {
		return
	}
	switch os.Args[separator+1] {
	case "reply":
		_, _ = os.Stdout.WriteString("{\"id\":2,\"result\":{\"data\":[]}}\n")
	case "parent":
		rootPIDPath, descendantPIDPath := os.Args[separator+2], os.Args[separator+3]
		if err := os.WriteFile(rootPIDPath, []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			os.Exit(91)
		}
		child := exec.Command(os.Args[0], "-test.run=^TestCodexHookProfileAppServerHelper$", "-test.count=1", "--", "child", descendantPIDPath)
		child.Stdout = os.Stdout
		child.Stderr = os.Stderr
		if err := child.Start(); err != nil {
			os.Exit(92)
		}
		_ = child.Wait()
	case "child":
		if err := os.WriteFile(os.Args[separator+2], []byte(strconv.Itoa(os.Getpid())), 0o600); err != nil {
			os.Exit(93)
		}
		time.Sleep(30 * time.Second)
	}
}

func TestClassifyEffectiveHookStates(t *testing.T) {
	root := t.TempDir()
	source := filepath.Join(root, "hooks", "hooks.json")
	if err := os.MkdirAll(filepath.Join(root, "bin"), 0755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(root, "bin", "dos-hook"), []byte("x"), 0755)
	_ = os.WriteFile(filepath.Join(root, "bin", "dos-hook-"+runtimeGOOS()+"-"+runtimeGOARCH()+exeSuffix()), []byte("x"), 0755)
	tests := []struct {
		name        string
		enabled     bool
		trust, want string
	}{{"ok", true, "trusted", "effective"}, {"disabled", false, "trusted", "disabled"}, {"changed", true, "modified", "stale_hash"}, {"untrusted", true, "untrusted", "untrusted"}, {"unknown", true, "mystery", "unknown"}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := effectiveHook{Enabled: tt.enabled, TrustStatus: tt.trust, PluginID: "p", SourcePath: source, Command: "$root/bin/dos-hook"}
			classifyEffectiveHook(&h, root)
			if h.State != tt.want {
				t.Fatalf("state=%s want %s", h.State, tt.want)
			}
		})
	}
}
func runtimeGOOS() string   { return runtime.GOOS }
func runtimeGOARCH() string { return runtime.GOARCH }
func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}
func TestNormalizeHookEvent(t *testing.T) {
	for in, want := range map[string]string{"PreToolUse": "pre_tool_use", "post_tool_use": "post_tool_use", "SubagentStop": "subagent_stop", "StopFailure": "stop_failure"} {
		if got := normalizeHookEvent(in); got != want {
			t.Fatalf("%s => %s", in, got)
		}
	}
}

func TestObservedHookEventsIncludeStopFamily(t *testing.T) {
	for _, event := range []string{"pre_tool_use", "post_tool_use", "stop", "subagent_stop", "stop_failure"} {
		if !isObservedHookEvent(event) {
			t.Fatalf("%s was not observed", event)
		}
	}
	if isObservedHookEvent("session_start") {
		t.Fatal("unrelated lifecycle event entered tool/Stop profile")
	}
}

func TestHookCommandPlatformCompatibility(t *testing.T) {
	tests := []struct {
		name, command, goos string
		want                bool
	}{
		{
			name:    "windows rejects selected POSIX stop command",
			command: `root="${CODEX_PLUGIN_ROOT:-}"; command -p sh "$root/bin/dos-hook" stop 2>/dev/null || python3 -m dos.cli hook stop`,
			goos:    "windows",
			want:    true,
		},
		{
			name:    "windows accepts native adapter",
			command: `& (Join-Path $env:PLUGIN_ROOT 'bin\dos-hook-codex.ps1') stop --workspace .; exit $LASTEXITCODE`,
			goos:    "windows",
		},
		{
			name:    "linux accepts POSIX command",
			command: `root="${CODEX_PLUGIN_ROOT:-}"; command -p sh "$root/bin/dos-hook" stop`,
			goos:    "linux",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := hookCommandPlatformIncompatible(tt.command, tt.goos); got != tt.want {
				t.Fatalf("incompatible=%v want %v", got, tt.want)
			}
		})
	}
}

func TestClassifyEffectiveHookRejectsIncompatibleWindowsCommand(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("classification follows the current host platform")
	}
	h := effectiveHook{
		Enabled:     true,
		TrustStatus: "trusted",
		Command:     `root="${CODEX_PLUGIN_ROOT:-}"; command -p sh "$root/bin/dos-hook" stop`,
	}
	classifyEffectiveHook(&h, t.TempDir())
	if h.State != "platform_incompatible" {
		t.Fatalf("state=%s want platform_incompatible", h.State)
	}
	if h.Remediation == "" {
		t.Fatal("platform incompatibility needs actionable remediation")
	}
}

func TestHookProfileAcceptsZeroPostToolUseWithMandatoryEnforcementHooks(t *testing.T) {
	r := mandatoryOnlyHookProfile()
	if r.Verdict != "HEALTHY" || len(r.Errors) != 0 {
		t.Fatalf("profile=%+v", r)
	}
	post := findHookEventContract(t, r, "post_tool_use")
	if post.Requirement != "optional" || post.Status != "intentionally_disabled" || !strings.Contains(post.Detail, "foreground spam") {
		t.Fatalf("post contract=%+v", post)
	}
	for _, event := range []string{"pre_tool_use", "stop", "subagent_stop"} {
		contract := findHookEventContract(t, r, event)
		if contract.Requirement != "mandatory" || contract.Status != "effective" {
			t.Fatalf("%s contract=%+v", event, contract)
		}
	}
	var out bytes.Buffer
	writeHookProfile(&out, r)
	if !strings.Contains(out.String(), "post_tool_use requirement=optional status=intentionally_disabled") {
		t.Fatalf("human profile did not explain optional PostToolUse:\n%s", out.String())
	}
}

func TestHookProfileRequiresEveryMandatoryEnforcementHook(t *testing.T) {
	for _, missing := range []string{"pre_tool_use", "stop", "subagent_stop"} {
		t.Run(missing, func(t *testing.T) {
			r := hookProfileReport{}
			for _, h := range mandatoryEffectiveHooks() {
				if h.EventName != missing {
					r.Hooks = append(r.Hooks, h)
				}
			}
			applyCodexHookContract(&r)
			if r.Verdict != "UNHEALTHY" || !slices.Contains(r.Errors, "missing effective "+missing+" hook") {
				t.Fatalf("profile=%+v", r)
			}
		})
	}
}

func TestApplyCodexHookContractRebuildsGeneratedErrors(t *testing.T) {
	r := hookProfileReport{Errors: []string{"hooks/list source error", "missing effective post_tool_use hook"}}
	applyCodexHookContract(&r)
	if r.Verdict != "UNHEALTHY" || len(r.Errors) != 4 {
		t.Fatalf("first profile=%+v", r)
	}
	r.Hooks = mandatoryEffectiveHooks()
	applyCodexHookContract(&r)
	if r.Verdict != "UNHEALTHY" || !slices.Equal(r.Errors, []string{"hooks/list source error"}) {
		t.Fatalf("second profile=%+v", r)
	}
	r.Errors = nil
	applyCodexHookContract(&r)
	if r.Verdict != "HEALTHY" || len(r.Errors) != 0 {
		t.Fatalf("third profile=%+v", r)
	}
}

func TestHookProfileDiagnosesPresentUnhealthyPostToolUse(t *testing.T) {
	for _, state := range []string{"disabled", "stale_hash", "untrusted", "platform_incompatible", "missing_executable"} {
		t.Run(state, func(t *testing.T) {
			r := hookProfileReport{Hooks: append(mandatoryEffectiveHooks(), effectiveHook{EventName: "post_tool_use", State: state})}
			applyCodexHookContract(&r)
			if r.Verdict != "UNHEALTHY" {
				t.Fatalf("profile=%+v", r)
			}
			post := findHookEventContract(t, r, "post_tool_use")
			if post.Status != "unhealthy" {
				t.Fatalf("post contract=%+v", post)
			}
		})
	}
}

func mandatoryEffectiveHooks() []effectiveHook {
	return []effectiveHook{
		{EventName: "pre_tool_use", State: "effective", Enabled: true},
		{EventName: "stop", State: "effective", Enabled: true},
		{EventName: "subagent_stop", State: "effective", Enabled: true},
	}
}

func mandatoryOnlyHookProfile() hookProfileReport {
	r := hookProfileReport{Hooks: mandatoryEffectiveHooks()}
	applyCodexHookContract(&r)
	return r
}

func effectivePostToolProfile() hookProfileReport {
	r := mandatoryOnlyHookProfile()
	r.Hooks = append(r.Hooks, effectiveHook{EventName: "post_tool_use", State: "effective", Enabled: true})
	applyCodexHookContract(&r)
	return r
}

func findHookEventContract(t *testing.T, r hookProfileReport, event string) hookEventContract {
	t.Helper()
	for _, contract := range r.EventContracts {
		if contract.EventName == event {
			return contract
		}
	}
	t.Fatalf("missing %s event contract in %+v", event, r.EventContracts)
	return hookEventContract{}
}
