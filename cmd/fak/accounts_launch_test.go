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
		"CLAUDE_CONFIG_DIR = " + seat,
		"login             = ready (can_serve=true)",
		"guard             = on",
		"--dangerously-skip-permissions",
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
}

func TestRunAccountsLaunchExecSeam(t *testing.T) {
	home := t.TempDir()
	regPath, seat := launchRegistry(t, home)

	var gotArgv, gotEnv []string
	orig := accountsLaunchRun
	accountsLaunchRun = func(_, _ io.Writer, argv, env []string) int {
		gotArgv, gotEnv = argv, env
		return 7
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
	wantTail := "claude --dangerously-skip-permissions --settings " + ultracodeSettingsArg + " --resume xyz"
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
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
		launched = true
		if len(argv) < 3 || argv[1] != "guard" {
			t.Fatalf("stale warning should not disable the default guard launch: %#v", argv)
		}
		return 0
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
	accountsLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
		gotArgv = argv
		return 0
	}
	t.Cleanup(func() { accountsLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--name", "gem8-seat", "--guard=false", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --guard=false rc=%d stderr=%s", rc, errb.String())
	}
	want := []string{"claude", "--dangerously-skip-permissions", "--settings", ultracodeSettingsArg}
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
