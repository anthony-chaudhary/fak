package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
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
		{
			// The managed-cache posture flags ride the guard argv, spliced between `guard` and
			// `--` so guard parses them and the agent never sees them.
			name: "claude managed-cache posture rides the guard argv before --",
			opts: launchOpts{command: "claude", useGuard: true, skipPermissions: true, guardCacheArgs: []string{"--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on"}},
			want: []string{fakBin, "guard", "--api-key-env", "ANTHROPIC_API_KEY", "--managed-cache", "on", "--", "claude", "--dangerously-skip-permissions"},
		},
		{
			// With guard OFF there is no guard process to carry the posture, so the flags are
			// dropped — the posture is a guard-session concept, not an agent flag.
			name: "guard off drops the managed-cache posture",
			opts: launchOpts{command: "claude", useGuard: false, skipPermissions: true, guardCacheArgs: []string{"--managed-cache", "on"}},
			want: []string{"claude", "--dangerously-skip-permissions"},
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

func TestAccountsLaunchSkipPermissionsOptionResolution(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want bool
	}{
		{name: "omitted Codex keeps native approvals and sandbox", args: []string{"--command", "codex"}, want: false},
		{name: "explicit Codex bypass", args: []string{"--command", "codex", "--skip-permissions"}, want: true},
		{name: "explicit false Codex", args: []string{"--command", "codex", "--skip-permissions=false"}, want: false},
		{name: "omitted Claude preserves historical bypass", args: []string{"--command", "claude"}, want: true},
		{name: "explicit false Claude", args: []string{"--command", "claude", "--skip-permissions=false"}, want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cmd, code := parseAccountsCmd(io.Discard, "launch", tc.args)
			if code != 0 {
				t.Fatalf("parseAccountsCmd code=%d", code)
			}
			got := resolveAccountsLaunchSkipPermissions(*cmd.launchCommand, *cmd.launchSkipPerms, flagSet(cmd.fs, "skip-permissions"))
			if got != tc.want {
				t.Fatalf("resolved skip-permissions=%v, want %v", got, tc.want)
			}
		})
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

	// Pin the managed-cache knobs so the test exercises the UNSET default (on) deterministically,
	// regardless of the developer's ambient FAK_MANAGED_CACHE / FAK_GUARD_API_KEY_ENV.
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")

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
		// The launcher's default posture is ON, and the plan says so in the operator's own
		// words — a bare launch is born in ultracode with no /effort typed into it.
		"ultracode         = on (--ultracode=on;",
		// The unconfigured launch now defaults managed cache to on (operator policy).
		"on (forces the stable-prefix 1h-TTL cache upgrade regardless of billing)",
		"--dangerously-skip-permissions",
		"--model " + defaultLaunchModel,
		"dry-run",
	} {
		if !strings.Contains(gotErr, want) {
			t.Fatalf("dry-run plan missing %q:\n%s", want, gotErr)
		}
	}
	// stdout echoes the scriptable command: it must be the guard wrap, carrying the on-by-default
	// managed-cache posture before `--`.
	gotOut := strings.TrimSpace(out.String())
	if !strings.Contains(gotOut, "guard --managed-cache on -- claude --dangerously-skip-permissions") {
		t.Fatalf("dry-run stdout command = %q", gotOut)
	}
	if strings.Contains(gotErr, seat) {
		t.Fatalf("dry-run stderr leaked raw account dir %q:\n%s", seat, gotErr)
	}
}

func TestRunAccountsCodexLaunchDryRunPermissions(t *testing.T) {
	root := t.TempDir()
	userHome := filepath.Join(root, "home")
	codexHome := filepath.Join(userHome, ".codex-blue")
	if err := os.MkdirAll(codexHome, 0o755); err != nil {
		t.Fatal(err)
	}
	const token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJhY2N0LWJsdWUifQ.sig"
	auth := `{"auth_mode":"chatgpt","tokens":{"access_token":"` + token + `","account_id":"acct-blue"}}`
	if err := os.WriteFile(filepath.Join(codexHome, "auth.json"), []byte(auth), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_USER_HOME", userHome)
	t.Setenv("FLEET_CONFIG_HOME", filepath.Join(root, "config"))
	t.Setenv("FLEET_REG_DIR", filepath.Join(root, "fleet-registry"))
	t.Setenv("FLEET_POLICY_DIR", filepath.Join(root, "policy"))
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")

	for _, tc := range []struct {
		name       string
		extra      []string
		wantBypass bool
		wantBanner string
	}{
		{name: "bare keeps native Codex layer", wantBanner: "Codex native approvals + sandbox (default); fak gates remain active"},
		{name: "explicit flag selects full bypass", extra: []string{"--skip-permissions"}, wantBypass: true, wantBanner: "Codex full approval/sandbox bypass explicitly requested"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := []string{"launch", "--dry-run", "--registry", filepath.Join(root, "accounts.json"), "--home", filepath.Join(root, "claude-home"), "--name", "blue", "--command", "codex", "--managed-cache", "auto", "--ultracode", "off"}
			args = append(args, tc.extra...)
			var out, errb bytes.Buffer
			if rc := runAccounts(&out, &errb, args); rc != 0 {
				t.Fatalf("Codex accounts dry-run rc=%d stderr=%s", rc, errb.String())
			}
			combined := out.String() + "\n" + errb.String()
			if has := strings.Contains(combined, "--dangerously-bypass-approvals-and-sandbox"); has != tc.wantBypass {
				t.Fatalf("accounts dry-run bypass present=%v, want %v:\n%s", has, tc.wantBypass, combined)
			}
			if !strings.Contains(errb.String(), tc.wantBanner) || !strings.Contains(errb.String(), "fak gates remain active") {
				t.Fatalf("accounts dry-run banner missing permissions posture:\n%s", errb.String())
			}
		})
	}
}

func TestAccountsLaunchSkipPermissionsHelpDistinguishesCodexFromFakGates(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"launch", "--help"}); rc != 2 {
		t.Fatalf("accounts launch --help rc=%d, want 2", rc)
	}
	for _, want := range []string{"keeps Codex native approvals + sandbox", "does not bypass fak routing, capacity, policy, hook, or loop gates"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("accounts launch help missing %q:\n%s", want, errb.String())
		}
	}
}

func TestPrintAccountsLaunchPlanShowsVersion(t *testing.T) {
	var errb bytes.Buffer
	p := launchParams{}
	home := accounts.Home{Name: "test-seat"}
	id := accounts.Identity{Email: "test@example.com"}
	printAccountsLaunchPlan(&errb, p, "claude", home, id, launchBrokerGrant{}, "auto", false)

	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	want := fmt.Sprintf("fak %s · accounts launch — seat %q", ver, "test-seat")
	if !strings.Contains(errb.String(), want) {
		t.Fatalf("printAccountsLaunchPlan output missing %q, got:\n%s", want, errb.String())
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

	// Pin the managed-cache knobs so the seam exercises the UNSET default (on) deterministically.
	t.Setenv(fleetManagedCacheEnv, "")
	t.Setenv(fleetGuardAPIKeyEnvEnv, "")

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
	// The guard wrap must be present, carrying the on-by-default managed-cache posture before `--`
	// and ending in the passthrough args.
	if len(gotArgv) < 4 || gotArgv[1] != "guard" {
		t.Fatalf("argv not a guard wrap: %#v", gotArgv)
	}
	joined := strings.Join(gotArgv, " ")
	if !strings.Contains(joined, "guard --managed-cache on --") {
		t.Fatalf("argv missing on-by-default managed-cache posture before --: %#v", gotArgv)
	}
	// The default posture is --ultracode=on, so a bare launch is born in ultracode: the
	// --settings payload rides between the pinned --model and the caller's passthrough.
	wantTail := "claude --dangerously-skip-permissions --model " + defaultLaunchModel +
		" --settings " + ultracodeSettingsArg + " --resume xyz"
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

// TestRunAccountsLaunchFallsBackWhenDefaultOpusUnavailable covers the UNKNOWN-MODEL wall: a CLI
// build or a seat that does not know the default Opus 5 id refuses it at startup, and the launch
// hops to the FIRST fallback rung — the previous Opus generation — so the degrade stays inside the
// Opus class instead of dropping a tier. The capped-bucket wall (which walks on to Fable 5) is the
// sibling test below.
func TestRunAccountsLaunchFallsBackWhenDefaultOpusUnavailable(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var calls [][]string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		calls = append(calls, append([]string(nil), argv...))
		if len(calls) == 1 {
			return launchRunResult{Code: 1, Stderr: `error: model "` + defaultLaunchModel + `" is not available for this account`}
		}
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	// --ultracode=on is explicit here: the point of this test is that the posture RIDES the
	// fallback hop, and the default (auto -> off for an unclassified launch, #5016) would emit no
	// --settings to check.
	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--ultracode=on", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch fallback rc=%d stderr=%s", rc, errb.String())
	}
	if len(calls) != 2 {
		t.Fatalf("launch attempts = %d, want primary + fallback; calls=%#v", len(calls), calls)
	}
	first, second := strings.Join(calls[0], " "), strings.Join(calls[1], " ")
	if !strings.Contains(first, "--model "+defaultLaunchModel) {
		t.Fatalf("primary launch did not use default Opus model: %q", first)
	}
	for _, want := range []string{"--model " + defaultLaunchFallbackFirst(), "--settings " + ultracodeSettingsArg} {
		if !strings.Contains(second, want) {
			t.Fatalf("fallback launch missing %q:\n%s", want, second)
		}
	}
	for _, want := range []string{"falling back to", "unknown-model", defaultLaunchFallbackFirst(), "fallback command"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("fallback stderr missing %q:\n%s", want, errb.String())
		}
	}
}

// TestRunAccountsLaunchFallsBackWhenDefaultOpusHitsWeeklyLimit is the goal case: a usage/weekly
// limit is NOT an "unknown model" error, so the old unknown-model-only detector would have let the
// walled Opus startup fail without ever trying the fallback. The broadened classifier recognizes
// the cap and the chain falls forward to Fable 5.
//
// A weekly cap is ALLOCATION-BUCKET scoped, so it walls the whole Opus class: the default and the
// previous-generation rung both refuse, and only Fable 5 — which bills against its own bucket —
// starts. That walk-through is the point of ordering the chain the way it is, so the test drives
// every rung rather than stopping at the first hop.
func TestRunAccountsLaunchFallsBackWhenDefaultOpusHitsWeeklyLimit(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var calls [][]string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		calls = append(calls, append([]string(nil), argv...))
		if isOpusLaunchModel(modelArg(argv)) {
			// No "model" / "fable" token at all — a bucket-scoped weekly cap, exactly what the
			// old unknown-model gate misses. Every Opus rung shares the capped bucket.
			return launchRunResult{Code: 1, Stderr: "Claude usage limit reached — your weekly limit resets at 2026-07-10T00:00:00Z"}
		}
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("weekly-limit fallback rc=%d stderr=%s", rc, errb.String())
	}
	if len(calls) < 2 {
		t.Fatalf("launch attempts = %d, want primary + fallback; calls=%#v", len(calls), calls)
	}
	last := strings.Join(calls[len(calls)-1], " ")
	if !strings.Contains(last, "--model "+defaultLaunchFallbackNonOpus()) {
		t.Fatalf("weekly-limit fallback did not walk the capped Opus bucket through to Fable: %q", last)
	}
	if !strings.Contains(errb.String(), "usage-limit") {
		t.Fatalf("fallback plan should name the usage-limit kind:\n%s", errb.String())
	}
}

// TestRunAccountsLaunchWalksMultiModelFallbackChain proves --fallback-model is an ordered CHAIN,
// not a single retry: the default and the first fallback are both unavailable, and the launch walks
// on to the second fallback, which starts.
func TestRunAccountsLaunchWalksMultiModelFallbackChain(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	var models []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		models = append(models, modelArg(argv))
		if len(models) < 3 {
			// A transient throttle is bucket-scoped (no model-name gate), so each hop fires and
			// the chain walks opus -> fable -> sonnet regardless of which id the error names.
			return launchRunResult{Code: 1, Stderr: `Error 429: too many requests`}
		}
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"launch", "--name", "gem8-seat",
		"--fallback-model", "fable,claude-sonnet-5",
		"--registry", regPath, "--home", home,
	})
	if rc != 0 {
		t.Fatalf("chain launch rc=%d stderr=%s", rc, errb.String())
	}
	want := []string{defaultLaunchModel, "fable", "claude-sonnet-5"}
	if !reflect.DeepEqual(models, want) {
		t.Fatalf("fallback chain models = %#v, want %#v", models, want)
	}
}

// modelArg returns the value passed after --model in a launch argv, or "" if none.
func modelArg(argv []string) string {
	for i, a := range argv {
		if a == "--model" && i+1 < len(argv) {
			return argv[i+1]
		}
	}
	return ""
}

// isOpusLaunchModel reports whether a launch --model value names an Opus-class rung. A weekly cap
// is allocation-bucket scoped, so a test simulating one must wall EVERY Opus rung, not just the id
// the chain happens to start on.
func isOpusLaunchModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "opus")
}

// defaultLaunchFallbackFirst is the FIRST rung of the compiled-in fallback chain — the model a
// refused default launch hops to. defaultLaunchFallbackModel is a multi-rung CHAIN (previous-Opus
// for an unknown-model wall, then Fable 5 for a capped Opus bucket), so a test asserting on "the
// fallback model" must name a rung rather than the whole constant.
func defaultLaunchFallbackFirst() string { return splitModelChain(defaultLaunchFallbackModel)[0] }

// defaultLaunchFallbackNonOpus is the first rung of the compiled-in chain that bills against a
// DIFFERENT allocation bucket than Opus — the rung a capped-Opus walk has to reach to start.
func defaultLaunchFallbackNonOpus() string {
	for _, m := range splitModelChain(defaultLaunchFallbackModel) {
		if !isOpusLaunchModel(m) {
			return m
		}
	}
	return ""
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
		want   launchModelUnavailKind
	}{
		// Unknown / invalid / unentitled model — must name the model dimension AND the tried id.
		// The id is interpolated from defaultLaunchModel (the id these cases are classified
		// AGAINST, below), so a fleet model bump moves the fixture with the default instead of
		// silently reclassifying every unknown-model case as `available`.
		{`error: model "` + defaultLaunchModel + `" is not available for this account`, launchModelUnknown},
		{`invalid model: ` + defaultLaunchModel, launchModelUnknown},
		{`model_not_found: ` + defaultLaunchModel, launchModelUnknown},
		{`your account does not have access to model ` + defaultLaunchModel, launchModelUnknown},
		// Usage / weekly / session caps — the class the old unknown-model-only detector MISSED.
		// They are bucket-scoped and need not name the model id at all.
		{`Claude usage limit reached — your weekly limit resets at 2026-07-10`, launchModelUsageLimit},
		{`session limit reached; try again in 3 hours`, launchModelUsageLimit},
		{`You've hit your 5-hour limit`, launchModelUsageLimit},
		// Transient throttle / overload — a different model's pool may be clear.
		{`Error 429: too many requests`, launchModelRateLimit},
		{`overloaded_error: the server is overloaded`, launchModelRateLimit},
		// Not a model-unavailability the chain should act on.
		{`network unavailable while contacting provider`, launchModelAvailable},
		{`model fable is not available`, launchModelAvailable}, // names a DIFFERENT model than the default (opus)
		{`permission denied`, launchModelAvailable},
		{`Error: Not logged in. Run /login`, launchModelAvailable}, // auth wall — a model switch cannot fix it
	}
	for _, tc := range cases {
		if got := classifyLaunchModelUnavailable(tc.stderr, defaultLaunchModel); got != tc.want {
			t.Errorf("classifyLaunchModelUnavailable(%q) = %v, want %v", tc.stderr, got, tc.want)
		}
		wantBool := tc.want != launchModelAvailable
		if got := launchModelUnavailable(tc.stderr, defaultLaunchModel); got != wantBool {
			t.Errorf("launchModelUnavailable(%q) = %v, want %v", tc.stderr, got, wantBool)
		}
	}
}

func TestRunAccountsLaunchWarnsOnSkewedBinary(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	const oldRev = "1111111111111111111111111111111111111111"
	const tipRev = "2222222222222222222222222222222222222222"
	origRun := accountsLaunchRun
	origAssess := accountsLaunchAssess
	t.Cleanup(func() {
		accountsLaunchRun = origRun
		accountsLaunchAssess = origAssess
	})

	launched := false
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		launched = true
		if len(argv) < 3 || argv[1] != "guard" {
			t.Fatalf("skew warning should not disable the default guard launch: %#v", argv)
		}
		return launchRunResult{Code: 0}
	}
	// A stamped launcher provably BEHIND origin/main (a strict ancestor of the tip).
	accountsLaunchAssess = func() versionskew.Assessment {
		return versionskew.Assessment{Verdict: versionskew.Skewed, Running: oldRev, TrunkTip: tipRev, Relation: versionskew.RelBehind}
	}

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
		"origin/main is at 222222222222",
		"provably BEHIND",
		"fak self-update",
		"guard will re-exec the same stale file",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("skewed launch warning missing %q:\n%s", want, got)
		}
	}
}

// TestRunAccountsLaunchWarnsOnDivergedBinary: a launcher OFF the trunk line (neither behind nor
// ahead of origin/main) was invisible to the old binstamp Stale/Unstamped surface — binstamp has
// no "diverged" concept. versionskew makes it its own refusable token, so launch now names it.
func TestRunAccountsLaunchWarnsOnDivergedBinary(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	origRun := accountsLaunchRun
	origAssess := accountsLaunchAssess
	t.Cleanup(func() {
		accountsLaunchRun = origRun
		accountsLaunchAssess = origAssess
	})

	launched := false
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		launched = true
		return launchRunResult{Code: 0}
	}
	accountsLaunchAssess = func() versionskew.Assessment {
		return versionskew.Assessment{Verdict: versionskew.Diverged, Running: "3333333333333333333333333333333333333333", TrunkTip: "4444444444444444444444444444444444444444", Relation: versionskew.RelDiverged}
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch rc=%d stderr=%s", rc, errb.String())
	}
	if !launched {
		t.Fatal("launch exec seam was not called")
	}
	got := errb.String()
	for _, want := range []string{"WARNING: running fak binary", "OFF the trunk line", "origin/main is at 444444444444"} {
		if !strings.Contains(got, want) {
			t.Fatalf("diverged launch warning missing %q:\n%s", want, got)
		}
	}
}

// TestRunAccountsLaunchQuietOnAheadBinary: an AHEAD launcher (a fresh local build not yet pushed)
// read as binstamp.Stale under the old equality check and would have emitted a FALSE stale
// warning. versionskew keeps Ahead distinct and NON-refusable, so launch now stays silent.
func TestRunAccountsLaunchQuietOnAheadBinary(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	origRun := accountsLaunchRun
	origAssess := accountsLaunchAssess
	t.Cleanup(func() {
		accountsLaunchRun = origRun
		accountsLaunchAssess = origAssess
	})

	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult { return launchRunResult{Code: 0} }
	accountsLaunchAssess = func() versionskew.Assessment {
		return versionskew.Assessment{Verdict: versionskew.Ahead, Running: "5555555555555555555555555555555555555555", TrunkTip: "6666666666666666666666666666666666666666", Relation: versionskew.RelAhead}
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch rc=%d stderr=%s", rc, errb.String())
	}
	if strings.Contains(errb.String(), "WARNING: running fak binary") {
		t.Fatalf("AHEAD launcher must NOT emit a stale-binary warning:\n%s", errb.String())
	}
}

// TestRunAccountsLaunchWarnsOnUnstampedBinary: a launcher that carries no VCS stamp cannot be
// called Stale (binstamp yields Unknown/CauseUnstamped), so the old Stale-only guard passed it
// silently. It must now warn — an unstamped guard is the #3306 blind spot and the default path
// re-execs this same file.
func TestRunAccountsLaunchWarnsOnUnstampedBinary(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	origRun := accountsLaunchRun
	origAssess := accountsLaunchAssess
	t.Cleanup(func() {
		accountsLaunchRun = origRun
		accountsLaunchAssess = origAssess
	})

	launched := false
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
		launched = true
		if len(argv) < 3 || argv[1] != "guard" {
			t.Fatalf("unstamped warning should not disable the default guard launch: %#v", argv)
		}
		return launchRunResult{Code: 0}
	}
	// An unstamped binary: no VCS revision at all -> versionskew.Unstamped.
	accountsLaunchAssess = func() versionskew.Assessment {
		return versionskew.Assessment{Verdict: versionskew.Unstamped}
	}

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
		"NO VCS stamp",
		"UNVERIFIABLE",
		"fak self-update --force",
		"guard will re-exec the same stale file",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("unstamped launch warning missing %q:\n%s", want, got)
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
	// Default --ultracode=on => the ultracode --settings rides even an unguarded launch.
	want := []string{"claude", "--dangerously-skip-permissions", "--model", defaultLaunchModel, "--settings", ultracodeSettingsArg}
	if !reflect.DeepEqual(gotArgv, want) {
		t.Fatalf("direct launch argv = %#v, want %#v", gotArgv, want)
	}
}

// TestResolveUltracodePosture pins the posture x work-class table #5016 documents: an explicit
// on/off always wins, and `auto` earns ultracode ONLY for rigor-class work — grind and the
// unclassified/interactive case stay OFF for latency. The table is unchanged by the flag default
// moving to `on`; that default only decides which ROW an unflagged launch lands on, which
// TestRunAccountsLaunchUltracodePosture below pins separately.
func TestResolveUltracodePosture(t *testing.T) {
	cases := []struct {
		posture string
		kind    ultracodeWorkKind
		want    bool
		wantErr bool
	}{
		// auto routes per work class.
		{"auto", ultracodeKindRigor, true, false},
		{"auto", ultracodeKindGrind, false, false},
		{"auto", ultracodeKindUnknown, false, false},
		{"", ultracodeKindRigor, true, false}, // empty normalizes to auto
		{"", ultracodeKindUnknown, false, false},
		// An explicit posture wins over the work class in BOTH directions.
		{"on", ultracodeKindUnknown, true, false},
		{"on", ultracodeKindGrind, true, false},
		{"off", ultracodeKindRigor, false, false},
		// Case and surrounding space are normalized.
		{"OFF", ultracodeKindRigor, false, false},
		{"  on  ", ultracodeKindGrind, true, false},
		{"Auto", ultracodeKindRigor, true, false},
		// The old bool flag's values stay accepted as aliases, so an existing
		// `--ultracode=false` script keeps working.
		{"true", ultracodeKindUnknown, true, false},
		{"false", ultracodeKindRigor, false, false},
		// A typo fails loud rather than silently picking a posture.
		{"maybe", ultracodeKindRigor, false, true},
		{"1", ultracodeKindRigor, false, true},
	}
	for _, tc := range cases {
		got, err := resolveUltracodePosture(tc.posture, tc.kind)
		if (err != nil) != tc.wantErr {
			t.Errorf("resolveUltracodePosture(%q, %v) err = %v, wantErr = %v", tc.posture, tc.kind, err, tc.wantErr)
			continue
		}
		if got != tc.want {
			t.Errorf("resolveUltracodePosture(%q, %v) = %v, want %v", tc.posture, tc.kind, got, tc.want)
		}
	}
}

// TestRunAccountsLaunchUltracodePosture pins the launcher default: --ultracode is a three-value
// posture auto|on|off defaulting to ON, so a BARE launch is born in ultracode and no operator has
// to type /effort ultracode into a fresh session. The #5016 work-class table is still one flag
// away — an explicit `--ultracode=auto` on an unclassified launch resolves to OFF — and
// `--ultracode=off` is the direct opt-out.
func TestRunAccountsLaunchUltracodePosture(t *testing.T) {
	cases := []struct {
		name    string
		args    []string
		wantArg bool
	}{
		{"default (on) launches in ultracode", nil, true},
		{"explicit auto + unclassified launch omits ultracode", []string{"--ultracode=auto"}, false},
		{"explicit on forces ultracode", []string{"--ultracode=on"}, true},
		{"explicit off omits ultracode", []string{"--ultracode=off"}, false},
		{"legacy bool true still forces ultracode", []string{"--ultracode=true"}, true},
		{"legacy bool false still omits ultracode", []string{"--ultracode=false"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			home := t.TempDir()
			regPath, _ := launchRegistry(t, home)

			var gotArgv []string
			orig := accountsLaunchRun
			accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) launchRunResult {
				gotArgv = argv
				return launchRunResult{Code: 0}
			}
			t.Cleanup(func() { accountsLaunchRun = orig })

			args := append([]string{"launch", "--name", "gem8-seat"}, tc.args...)
			args = append(args, "--registry", regPath, "--home", home)

			var out, errb bytes.Buffer
			if rc := runAccounts(&out, &errb, args); rc != 0 {
				t.Fatalf("launch rc=%d stderr=%s", rc, errb.String())
			}
			joined := strings.Join(gotArgv, " ")
			if got := strings.Contains(joined, ultracodeSettingsArg); got != tc.wantArg {
				t.Fatalf("ultracode --settings present = %v, want %v\nargv: %s", got, tc.wantArg, joined)
			}
		})
	}
}

// TestRunAccountsLaunchRejectsInvalidUltracodePosture: a typo must fail loud rather than silently
// picking a posture — the same fail-on-bad-mode discipline the managed-cache knob uses.
func TestRunAccountsLaunchRejectsInvalidUltracodePosture(t *testing.T) {
	home := t.TempDir()
	regPath, _ := launchRegistry(t, home)

	launched := false
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, _, _ []string) launchRunResult {
		launched = true
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"launch", "--name", "gem8-seat", "--ultracode", "sometimes",
		"--registry", regPath, "--home", home,
	})
	if rc == 0 {
		t.Fatalf("invalid --ultracode should not succeed; stderr=%s", errb.String())
	}
	if launched {
		t.Fatal("invalid --ultracode must not launch the agent")
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

func TestActiveLaunchSeatFallsForwardWhenActiveIsWalled(t *testing.T) {
	now := time.Date(2026, 7, 11, 15, 0, 0, 0, time.UTC)
	reg := accounts.Registry{
		Homes: []accounts.Home{
			{Name: "walled", Dir: "/walled", Enabled: boolPtr(false)},
			{Name: "room", Dir: "/room"},
		},
		Roles: map[string]string{accounts.RoleActive: "walled", accounts.RoleAnchor: "walled"},
	}
	got, ok, fellForward := activeLaunchSeatNameAt(reg, now)
	if !ok || !fellForward || got != "room" {
		t.Fatalf("pick = %q,%v,%v want room,true,true", got, ok, fellForward)
	}
	anchor, _ := reg.Role(accounts.RoleAnchor)
	if anchor.Name != "walled" {
		t.Fatalf("anchor moved to %q; launch fallback must not alter anchor", anchor.Name)
	}
}

func TestActiveLaunchSeatStaysPutWhenActiveHasRoom(t *testing.T) {
	now := time.Date(2026, 7, 11, 15, 0, 0, 0, time.UTC)
	reg := accounts.Registry{
		Homes: []accounts.Home{{Name: "active", Dir: "/active"}, {Name: "other", Dir: "/other"}},
		Roles: map[string]string{accounts.RoleActive: "active"},
	}
	got, ok, fellForward := activeLaunchSeatNameAt(reg, now)
	if !ok || fellForward || got != "active" {
		t.Fatalf("pick = %q,%v,%v want active,true,false", got, ok, fellForward)
	}
}

func TestRunAccountsLaunchBareReportsActiveFallback(t *testing.T) {
	home := t.TempDir()
	regPath := filepath.Join(home, "accounts.json")
	walledDir := mkHome(t, home, "walled", "walled@example.test", true)
	roomDir := mkHome(t, home, "room", "room@example.test", true)
	reg := accounts.Registry{
		Homes: []accounts.Home{
			{Name: "walled", Dir: walledDir, Enabled: boolPtr(false)},
			{Name: "room", Dir: roomDir},
		},
		Roles: map[string]string{accounts.RoleActive: "walled", accounts.RoleAnchor: "room"},
	}
	if err := accounts.SaveRegistry(regPath, reg); err != nil {
		t.Fatal(err)
	}

	orig := accountsLaunchRun
	var gotEnv []string
	accountsLaunchRun = func(_, _ io.Writer, _ []string, env []string) launchRunResult {
		gotEnv = env
		return launchRunResult{}
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--guard=false", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "active walled is walled; launching room with room") {
		t.Fatalf("missing fallback note: %q", errb.String())
	}
	if got := testEnvValue(gotEnv, "CLAUDE_CONFIG_DIR"); got != roomDir {
		t.Fatalf("CLAUDE_CONFIG_DIR=%q want room dir", got)
	}
}

func testEnvValue(env []string, key string) string {
	prefix := key + "="
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			return strings.TrimPrefix(item, prefix)
		}
	}
	return ""
}
