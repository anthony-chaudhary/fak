package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
)

func TestBuildLaunchArgv(t *testing.T) {
	const fakBin = "/usr/local/bin/fak"
	cases := []struct {
		name string
		opts launchOpts
		want []string
	}{
		{
			name: "guard on, skip-perms on (the default)",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions"},
		},
		{
			name: "guard on, skip-perms on, with passthrough",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, passthrough: []string{"--resume", "abc"}},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions", "--resume", "abc"},
		},
		{
			name: "guard on, skip-perms off (Claude prompts)",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: false},
			want: []string{fakBin, "guard", "--", "claude"},
		},
		{
			name: "guard off, skip-perms on (direct, no kernel hop)",
			opts: launchOpts{command: "claude", useGuard: false, skipPermissions: true},
			want: []string{"claude", "--dangerously-skip-permissions"},
		},
		{
			name: "guard off, skip-perms off",
			opts: launchOpts{command: "claude", useGuard: false, skipPermissions: false},
			want: []string{"claude"},
		},
		{
			// Codex gets ITS bypass flag, not Claude's --dangerously-skip-permissions (which
			// Codex rejects as an unexpected argument). This is the bug that made
			// `fak accounts launch --command codex` fail before the agent ever started.
			name: "codex guard on, skip-perms on -> codex bypass flag, not the claude flag",
			opts: launchOpts{command: "codex", useGuard: true, skipPermissions: true},
			want: []string{fakBin, "guard", "--", "codex", "--dangerously-bypass-approvals-and-sandbox"},
		},
		{
			name: "codex with passthrough keeps order after the bypass flag",
			opts: launchOpts{command: "codex", useGuard: true, skipPermissions: true, passthrough: []string{"exec", "do x"}},
			want: []string{fakBin, "guard", "--", "codex", "--dangerously-bypass-approvals-and-sandbox", "exec", "do x"},
		},
		{
			name: "codex skip-perms off gets no bypass flag (codex prompts)",
			opts: launchOpts{command: "codex", useGuard: true, skipPermissions: false},
			want: []string{fakBin, "guard", "--", "codex"},
		},
		{
			// An agent fak has no known bypass flag for must NOT be handed the claude flag;
			// the kernel floor under guard still adjudicates every call.
			name: "unknown agent skip-perms on gets no flag, not the claude flag",
			opts: launchOpts{command: "opencode", useGuard: true, skipPermissions: true},
			want: []string{fakBin, "guard", "--", "opencode"},
		},
		{
			// Ultracode injects the session-only --settings for Claude, after the bypass flag
			// and before any passthrough — parity with the `f` shortcut's workflow-on default.
			name: "claude ultracode on adds --settings after the bypass flag",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, ultracode: true},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions", "--settings", `{"ultracode":true}`},
		},
		{
			name: "claude ultracode on with passthrough keeps --settings before passthrough",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, ultracode: true, passthrough: []string{"-p", "hi"}},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions", "--settings", `{"ultracode":true}`, "-p", "hi"},
		},
		{
			// Ultracode is Claude-specific; --settings is never handed to a non-Claude agent.
			name: "codex ultracode on gets no --settings",
			opts: launchOpts{command: "codex", useGuard: true, skipPermissions: true, ultracode: true},
			want: []string{fakBin, "guard", "--", "codex", "--dangerously-bypass-approvals-and-sandbox"},
		},
		{
			// The default model is pinned via --model for a Claude launch, after the
			// bypass flag and before ultracode's --settings — so a switched seat starts on
			// the configured default regardless of its own saved default.
			name: "claude default model adds --model after the bypass flag",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, model: defaultLaunchModel},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions", "--model", defaultLaunchModel},
		},
		{
			// --model precedes --settings, and both precede any passthrough — so a caller's own
			// `-- --model x` still comes later.
			name: "claude model + ultracode order: --model then --settings then passthrough",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, ultracode: true, model: defaultLaunchModel, passthrough: []string{"--model", "sonnet"}},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions", "--model", defaultLaunchModel, "--settings", `{"ultracode":true}`, "--model", "sonnet"},
		},
		{
			// An empty model opts out: the seat's own saved default stands (no --model emitted).
			name: "claude empty model omits --model",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, model: ""},
			want: []string{fakBin, "guard", "--", "claude", "--dangerously-skip-permissions"},
		},
		{
			// --model is Claude-specific: a Claude model id is never handed to a non-Claude agent.
			name: "codex model gets no --model",
			opts: launchOpts{command: "codex", useGuard: true, skipPermissions: true, model: defaultLaunchModel},
			want: []string{fakBin, "guard", "--", "codex", "--dangerously-bypass-approvals-and-sandbox"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := buildLaunchArgv(fakBin, tc.opts)
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("buildLaunchArgv = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// TestLaunchSkipPermsFlag pins the per-agent mapping of the "kernel is the permission system"
// flag — the fix for the launcher feeding every agent Claude's flag. The codex value mirrors
// the flag the repo's own codex dispatch (tools/issue_resolve_dispatch.py) uses; an unknown
// agent yields "" so it is never handed a wrong flag. Matching normalizes paths/suffixes/case
// via guardAgentBaseName.
func TestLaunchSkipPermsFlag(t *testing.T) {
	cases := []struct {
		command string
		want    string
	}{
		{"claude", "--dangerously-skip-permissions"},
		{"claude-code", "--dangerously-skip-permissions"},
		{"/usr/local/bin/claude", "--dangerously-skip-permissions"}, // absolute path normalized
		{"codex", "--dangerously-bypass-approvals-and-sandbox"},
		{"Codex", "--dangerously-bypass-approvals-and-sandbox"},     // case-insensitive
		{"codex.exe", "--dangerously-bypass-approvals-and-sandbox"}, // Windows launcher suffix
		{`C:\tools\codex.exe`, "--dangerously-bypass-approvals-and-sandbox"},
		{"opencode", ""}, // known agent, but no bypass-flag mapping -> none, not the claude flag
		{"aider", ""},
		{"", ""},
	}
	for _, tc := range cases {
		if got := launchSkipPermsFlag(tc.command); got != tc.want {
			t.Errorf("launchSkipPermsFlag(%q) = %q, want %q", tc.command, got, tc.want)
		}
	}
}

// launchRegistry writes a registry with one active seat pointed at by the active role and
// returns (registryPath, seatDir).
func launchRegistry(t *testing.T, home string) (string, string) {
	t.Helper()
	seat := mkHome(t, home, ".claude-gem8-seat", "gem8@example.test", true)
	reg := `{"version":"fak-config-homes/v1",` +
		`"homes":[{"name":"gem8-seat","dir":"` + jsonPath(seat) + `"}],` +
		`"roles":{"active":"gem8-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return regPath, seat
}

func TestRunAccountsLaunchDryRun(t *testing.T) {
	home := t.TempDir()
	regPath, seat := launchRegistry(t, home)

	var out, errb bytes.Buffer
	// No --name => defaults to the active-role seat. --dry-run prints the plan, no exec.
	rc := runAccounts(&out, &errb, []string{"launch", "--dry-run", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --dry-run rc=%d stderr=%s", rc, errb.String())
	}
	gotErr := errb.String()
	for _, want := range []string{
		`seat "gem8-seat"`,
		"CLAUDE_CONFIG_DIR = <account-dir>",
		"login             = ready (can_serve=true)",
		"guard             = on",
		"model             = " + defaultLaunchModel,
		"--dangerously-skip-permissions",
		"--model " + defaultLaunchModel,
		"dry-run",
	} {
		if !strings.Contains(gotErr, want) {
			t.Fatalf("dry-run plan missing %q:\n%s", want, gotErr)
		}
	}
	// stdout echoes the scriptable command: it must be the guard wrap.
	gotOut := strings.TrimSpace(out.String())
	if !strings.Contains(gotOut, "guard -- claude --dangerously-skip-permissions") {
		t.Fatalf("dry-run stdout command = %q", gotOut)
	}
	if strings.Contains(gotErr, seat) {
		t.Fatalf("dry-run stderr leaked raw account dir %q:\n%s", seat, gotErr)
	}
}

// TestRunAccountsLaunchModelOptOut pins the opt-out: `--model ""` launches with the seat's own
// saved default (no --model handed to Claude), while the default pins the configured model.
func TestRunAccountsLaunchModelOptOut(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var gotArgv []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		gotArgv = argv
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--model", "", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --model '' rc=%d stderr=%s", rc, errb.String())
	}
	if joined := strings.Join(gotArgv, " "); strings.Contains(joined, "--model") {
		t.Fatalf("--model '' should omit --model, got argv %q", joined)
	}
	if !strings.Contains(errb.String(), "model             = seat default") {
		t.Fatalf("launch plan should note the seat-default model:\n%s", errb.String())
	}
}

func TestRunAccountsLaunchExecSeam(t *testing.T) {
	home := t.TempDir()
	regPath, seat := launchRegistry(t, home)

	var gotArgv, gotEnv []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, env []string) launchRunResult {
		gotArgv, gotEnv = argv, env
		return launchRunResult{Code: 7}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home, "--", "--resume", "xyz"})
	if rc != 7 {
		t.Fatalf("launch rc=%d (want the seam's 7); stderr=%s", rc, errb.String())
	}
	// The guard wrap must be present, ending in the passthrough args.
	if len(gotArgv) < 4 || gotArgv[1] != "guard" || gotArgv[2] != "--" {
		t.Fatalf("argv not a guard wrap: %#v", gotArgv)
	}
	joined := strings.Join(gotArgv, " ")
	wantTail := "claude --dangerously-skip-permissions --model " + defaultLaunchModel + " --settings " + ultracodeSettingsArg + " --resume xyz"
	if !strings.HasSuffix(joined, wantTail) {
		t.Fatalf("argv tail wrong: %q", joined)
	}
	// The seat's CLAUDE_CONFIG_DIR must be injected into the child env.
	found := false
	for _, kv := range gotEnv {
		if kv == "CLAUDE_CONFIG_DIR="+seat {
			found = true
		}
	}
	if !found {
		t.Fatalf("env missing CLAUDE_CONFIG_DIR=%s", seat)
	}
}

func TestAccountsLaunchBrokerDenyDoesNotStartWorker(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)
	t.Setenv("ANTHROPIC_API_KEY", "sk-env-secret")

	oldBroker := launchSpawnBroker
	oldRun := accountsLaunchRun
	var attempt launchBrokerAttempt
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		attempt = a
		return denyLaunchBrokerGrant(a, "unit-test-deny")
	}
	called := false
	accountsLaunchRun = func(_, _ io.Writer, _ []string, _ []string) launchRunResult {
		called = true
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		accountsLaunchRun = oldRun
	})

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("broker-denied launch rc=%d, want 1; stderr=%s", rc, errb.String())
	}
	if called {
		t.Fatal("accounts launch runner was called after broker denial")
	}
	if attempt.Surface != "accounts_launch" || attempt.Metadata.AgentRunID == "" ||
		!strings.HasPrefix(attempt.Metadata.PolicyDigest, "policy-sha256:") {
		t.Fatalf("broker attempt = %+v, want accounts launch AgentRun/PolicyDigest metadata", attempt)
	}
	for _, leak := range []string{"sk-env-secret", attempt.Env["CLAUDE_CONFIG_DIR"]} {
		if leak != "" && strings.Contains(errb.String(), leak) {
			t.Fatalf("broker-denied stderr leaked %q:\n%s", leak, errb.String())
		}
	}
	if !strings.Contains(errb.String(), "spawn broker denied launch: unit-test-deny") {
		t.Fatalf("stderr missing broker denial:\n%s", errb.String())
	}
}

func TestAccountsLaunchBrokerAllowCarriesMetadata(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	oldBroker := launchSpawnBroker
	oldRun := accountsLaunchRun
	var attempt launchBrokerAttempt
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		attempt = a
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	accountsLaunchRun = func(_, _ io.Writer, _ []string, _ []string) launchRunResult {
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		accountsLaunchRun = oldRun
	})

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("broker-allowed launch rc=%d; stderr=%s", rc, errb.String())
	}
	if attempt.Metadata.AgentRunID == "" || attempt.Metadata.PolicyDigest == "" {
		t.Fatalf("broker attempt metadata = %+v, want AgentRun/PolicyDigest", attempt.Metadata)
	}
	for _, want := range []string{"agent_run", attempt.Metadata.AgentRunID, attempt.Metadata.PolicyDigest, "broker=unit-test-allow"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("allowed stderr missing %q:\n%s", want, errb.String())
		}
	}
	if strings.Contains(errb.String(), attempt.Env["CLAUDE_CONFIG_DIR"]) {
		t.Fatalf("allowed stderr leaked raw account dir:\n%s", errb.String())
	}
}

func TestRunAccountsLaunchFallsBackToOpus48WhenDefaultFableUnavailable(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var calls [][]string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		calls = append(calls, append([]string(nil), argv...))
		if len(calls) == 1 {
			return launchRunResult{Code: 1, Stderr: `error: model "fable" is not available for this account`}
		}
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch fallback rc=%d stderr=%s", rc, errb.String())
	}
	if len(calls) != 2 {
		t.Fatalf("launch attempts = %d, want primary + fallback; calls=%#v", len(calls), calls)
	}
	first, second := strings.Join(calls[0], " "), strings.Join(calls[1], " ")
	if !strings.Contains(first, "--model "+defaultLaunchModel) {
		t.Fatalf("primary launch did not use default Fable model: %q", first)
	}
	for _, want := range []string{"--model " + defaultLaunchFallbackModel, "--settings " + ultracodeSettingsArg} {
		if !strings.Contains(second, want) {
			t.Fatalf("fallback launch missing %q:\n%s", want, second)
		}
	}
	for _, want := range []string{"retrying once", defaultLaunchFallbackModel, "fallback command"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("fallback stderr missing %q:\n%s", want, errb.String())
		}
	}
}

func TestRunAccountsLaunchDoesNotFallbackWhenModelExplicit(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var calls int
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, _ []string, _ []string) launchRunResult {
		calls++
		return launchRunResult{Code: 1, Stderr: `error: model "fable" is not available for this account`}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"launch", "--name", "gem8-seat", "--model", defaultLaunchModel,
		"--registry", regPath, "--home", home,
	})
	if rc != 1 {
		t.Fatalf("explicit-model launch rc=%d, want primary failure; stderr=%s", rc, errb.String())
	}
	if calls != 1 {
		t.Fatalf("explicit --model should not auto-fallback; attempts=%d", calls)
	}
	if strings.Contains(errb.String(), "retrying once") {
		t.Fatalf("explicit --model emitted fallback retry:\n%s", errb.String())
	}
}

func TestLaunchModelUnavailableClassifier(t *testing.T) {
	cases := []struct {
		stderr string
		want   bool
	}{
		{`error: model "fable" is not available for this account`, true},
		{`invalid model: fable`, true},
		{`model_not_found: fable`, true},
		{`network unavailable while contacting provider`, false},
		{`model claude-opus-4-8 is not available`, false},
		{`permission denied`, false},
	}
	for _, tc := range cases {
		if got := launchModelUnavailable(tc.stderr, defaultLaunchModel); got != tc.want {
			t.Errorf("launchModelUnavailable(%q) = %v, want %v", tc.stderr, got, tc.want)
		}
	}
}

func TestRunAccountsLaunchWarnsOnStaleBinary(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	const oldRev = "1111111111111111111111111111111111111111"
	const headRev = "2222222222222222222222222222222222222222"
	origRun := accountsLaunchRun
	origStamp := accountsLaunchStamp
	origHead := accountsLaunchHeadRev
	t.Cleanup(func() {
		accountsLaunchRun = origRun
		accountsLaunchStamp = origStamp
		accountsLaunchHeadRev = origHead
	})

	launched := false
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		launched = true
		if len(argv) < 3 || argv[1] != "guard" {
			t.Fatalf("stale warning should not disable the default guard launch: %#v", argv)
		}
		return launchRunResult{Code: 0}
	}
	accountsLaunchStamp = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: oldRev, HasVCS: true}
	}
	accountsLaunchHeadRev = func() string { return headRev }

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch rc=%d stderr=%s", rc, errb.String())
	}
	if !launched {
		t.Fatal("launch exec seam was not called")
	}
	got := errb.String()
	for _, want := range []string{
		"WARNING: running fak binary",
		"built from 111111111111",
		"checkout is at 222222222222",
		"fak self-update",
		"guard will re-exec the same stale file",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("stale launch warning missing %q:\n%s", want, got)
		}
	}
}

func TestRunAccountsLaunchDirectNoGuard(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var gotArgv []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		gotArgv = argv
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--guard=false", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --guard=false rc=%d stderr=%s", rc, errb.String())
	}
	want := []string{"claude", "--dangerously-skip-permissions", "--model", defaultLaunchModel, "--settings", ultracodeSettingsArg}
	if !reflect.DeepEqual(gotArgv, want) {
		t.Fatalf("direct launch argv = %#v, want %#v", gotArgv, want)
	}
}

func TestActiveLaunchSeatName(t *testing.T) {
	// Active role wins.
	reg := accounts.Registry{
		Homes: []accounts.Home{{Name: "a", Dir: "/a"}, {Name: "b", Dir: "/b"}},
		Roles: map[string]string{accounts.RoleActive: "b"},
	}
	if got, ok := activeLaunchSeatName(reg); !ok || got != "b" {
		t.Fatalf("active-role pick = %q,%v want b,true", got, ok)
	}
	// No role, a "default" seat wins.
	reg = accounts.Registry{Homes: []accounts.Home{{Name: "x", Dir: "/x"}, {Name: "default", Dir: "/d"}}}
	if got, ok := activeLaunchSeatName(reg); !ok || got != "default" {
		t.Fatalf("default pick = %q,%v want default,true", got, ok)
	}
	// No role, no default, exactly one active seat wins.
	reg = accounts.Registry{Homes: []accounts.Home{{Name: "solo", Dir: "/s"}}}
	if got, ok := activeLaunchSeatName(reg); !ok || got != "solo" {
		t.Fatalf("solo pick = %q,%v want solo,true", got, ok)
	}
	// No role, no default, multiple active seats => ambiguous.
	reg = accounts.Registry{Homes: []accounts.Home{{Name: "p", Dir: "/p"}, {Name: "q", Dir: "/q"}}}
	if got, ok := activeLaunchSeatName(reg); ok {
		t.Fatalf("ambiguous pick should fail, got %q,%v", got, ok)
	}
}
