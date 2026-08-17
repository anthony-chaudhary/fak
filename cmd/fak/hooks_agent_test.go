package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// cmd/fak/hooks_agent_test.go — the #5607 contract: an agent-lifecycle hook that COULD NOT RUN
// must say so instead of exiting 0.
//
// The defect these pin lives in .claude/settings.json, where every entry is
// `python -c "...; subprocess.call(argv); sys.exit(0)"`. subprocess.call returns the child's exit
// code, the value is dropped, and sys.exit(0) overrides — so a missing script, a crashed
// interpreter, and a clean pass are byte-identical to the harness. That is failclosed-audit.md
// FINDING 2, and it is the epic #5601 shape: a check whose absence renders like a check that
// passed.

// TestAgentHookOutcome_ContractTable is the core of the issue, driven directly because the
// classification is pure. The load-bearing rows are the two that used to collapse into 0.
func TestAgentHookOutcome_ContractTable(t *testing.T) {
	cases := []struct {
		name       string
		ran        bool
		rc         int
		wantStatus string
		wantExit   int
		why        string
	}{
		{"never ran", false, -1, "could-not-run", 1,
			"THE defect: the delegate never executed, so it cannot have passed"},
		{"ran clean", true, 0, "ran", 0,
			"the only case that may exit 0"},
		{"deliberate block", true, 2, "blocked", 2,
			"exit 2 is the harness BLOCK signal and must be forwarded untouched"},
		{"ran and failed", true, 1, "failed", 1,
			"it executed but errored — visible, and still not a pass"},
		{"ran and failed oddly", true, 77, "failed", 1,
			"any non-0 non-2 child code is a non-blocking failure, never silently 0"},
	}
	if len(cases) == 0 {
		t.Fatal("empty contract table — this test would be vacuous")
	}
	for _, c := range cases {
		status, exit := agentHookOutcome(c.ran, c.rc)
		if status != c.wantStatus || exit != c.wantExit {
			t.Errorf("agentHookOutcome(ran=%v, rc=%d) = (%q, %d), want (%q, %d) — %s",
				c.ran, c.rc, status, exit, c.wantStatus, c.wantExit, c.why)
		}
	}
}

// TestAgentHookOutcome_CouldNotRunIsNeverTheBlockCode pins the asymmetry with
// `fak hooks pre-commit`, where could-not-run IS 2. Here 2 is the harness's block signal: if a
// missing script exited 2, one absent file would refuse every tool call fleet-wide. That risk is
// exactly why the Python wrappers coerce everything to 0, so the fix must not reintroduce it.
func TestAgentHookOutcome_CouldNotRunIsNeverTheBlockCode(t *testing.T) {
	status, exit := agentHookOutcome(false, -1)
	if exit == 2 {
		t.Fatalf("could-not-run exited 2 (status %q) — that is the harness BLOCK code; a missing "+
			"delegate would refuse every tool call in the fleet", status)
	}
	if exit == 0 {
		t.Fatalf("could-not-run exited 0 (status %q) — indistinguishable from a clean pass, which "+
			"is the whole defect #5607 reports", status)
	}
}

// TestAgentHookRegistry_NonEmptyFloor keeps the registry honest: an empty registry would make
// every other test here pass vacuously while the verb checked nothing.
func TestAgentHookRegistry_NonEmptyFloor(t *testing.T) {
	reg := agentHookRegistry()
	if len(reg) == 0 {
		t.Fatal("agentHookRegistry() is empty — every delegate test below would be vacuous")
	}
	seenEvent := map[string]bool{}
	for _, d := range reg {
		if strings.TrimSpace(d.Name) == "" {
			t.Errorf("delegate for event %q has no name", d.Event)
		}
		if d.Argv == nil {
			t.Errorf("delegate %s/%s has a nil Argv — it could never run", d.Event, d.Name)
		}
		known := false
		for _, e := range agentHookEvents {
			if e == d.Event {
				known = true
				break
			}
		}
		if !known {
			t.Errorf("delegate %s registered against unknown event %q (valid: %v)", d.Name, d.Event, agentHookEvents)
		}
		seenEvent[d.Event] = true
	}
	for _, e := range agentHookEvents {
		if !seenEvent[e] {
			t.Errorf("event %q has no registered delegate — `fak hooks agent %s` could only ever refuse", e, e)
		}
	}
}

// TestAgentHookRegistry_MirrorsSettingsJSON pins the cutover. Every lifecycle event wired in the
// live .claude/settings.json must have a compiled delegate, or moving that entry over would drop
// a check on the floor — silently, which is the failure mode this issue is about.
func TestAgentHookRegistry_MirrorsSettingsJSON(t *testing.T) {
	root := resolveRoot("")
	if root == "" {
		t.Skip("no repo root resolvable")
	}
	raw, err := os.ReadFile(filepath.Join(root, ".claude", "settings.json"))
	if err != nil {
		t.Skipf("no .claude/settings.json to mirror: %v", err)
	}
	var cfg struct {
		Hooks map[string][]struct {
			Hooks []struct {
				Command string `json:"command"`
			} `json:"hooks"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("settings.json is not parseable: %v", err)
	}
	// Harness event name -> this verb's event token.
	want := map[string]string{"PreToolUse": "pretool", "PostToolUse": "posttool", "Stop": "stop"}
	wired := 0
	for harnessEvent, token := range want {
		entries, ok := cfg.Hooks[harnessEvent]
		if !ok || len(entries) == 0 {
			continue
		}
		wired++
		if _, err := agentHookPick(token, ""); err != nil {
			// Ambiguity is fine (it means several delegates exist); absence is not.
			if strings.Contains(err.Error(), "no registered delegate") || strings.Contains(err.Error(), "unknown event") {
				t.Errorf("%s is wired in settings.json but `fak hooks agent %s` has no delegate: %v",
					harnessEvent, token, err)
			}
		}
	}
	if wired == 0 {
		t.Fatal("settings.json declared none of PreToolUse/PostToolUse/Stop — the mirror test would be vacuous")
	}
}

// TestProjectSettingsUsesCrossPlatformRepoGuardDelegate pins #7053: the Windows hook host
// has no `sh`, so project admission must travel through FAK's native hook registry.
func TestProjectSettingsUsesCrossPlatformRepoGuardDelegate(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	settingsPath := filepath.Join(root, ".claude", "settings.json")
	raw, err := os.ReadFile(settingsPath)
	if err != nil {
		t.Fatal(err)
	}
	var settings struct {
		Hooks map[string][]struct {
			Matcher string `json:"matcher"`
			Hooks   []struct {
				Command string   `json:"command"`
				Args    []string `json:"args"`
			} `json:"hooks"`
		} `json:"hooks"`
	}
	if err := json.Unmarshal(raw, &settings); err != nil {
		t.Fatalf("parse %s: %v", settingsPath, err)
	}
	pretool := settings.Hooks["PreToolUse"]
	if len(pretool) != 1 || len(pretool[0].Hooks) != 1 {
		t.Fatalf("PreToolUse must have exactly one project repo-guard hook: %+v", pretool)
	}
	if pretool[0].Matcher != "Bash|Read|Write|Edit|MultiEdit|NotebookEdit" {
		t.Fatalf("repo-guard matcher changed: %q", pretool[0].Matcher)
	}
	hook := pretool[0].Hooks[0]
	wantArgs := []string{"hooks", "agent", "pretool", "-delegate", "repoguard"}
	if hook.Command != "fak" || !slices.Equal(hook.Args, wantArgs) {
		t.Fatalf("repo-guard must use the cross-platform agent-hook registry, got command=%q args=%q", hook.Command, hook.Args)
	}
	if strings.Contains(strings.ToLower(hook.Command), "sh") {
		t.Fatalf("repo-guard transport must not require a POSIX shell on Windows: %q", hook.Command)
	}
}

// TestProjectSettingsDoNotDuplicateDOSPlugin pins the #2702 boundary: Claude workers load
// the enabled user-scope DOS plugin, so project settings retain only FAK-owned hooks. Registering
// dos_hook.py here makes every headless tool result traverse DOS more than once.
func TestProjectSettingsDoNotDuplicateDOSPlugin(t *testing.T) {
	root := resolveRoot("")
	if root == "" {
		t.Skip("no repo root resolvable")
	}
	raw, err := os.ReadFile(filepath.Join(root, ".claude", "settings.json"))
	if err != nil {
		t.Fatalf("read settings.json: %v", err)
	}
	if bytes.Contains(raw, []byte("dos_hook.py")) || bytes.Contains(raw, []byte("dos.cli hook")) {
		t.Fatal("project settings duplicate the user-scope DOS plugin hook path")
	}
}

// TestAgentHookPick_UnknownEventRefuses: an unrecognized event must refuse and name the valid
// set, never select nothing and let the caller report a pass.
func TestAgentHookPick_UnknownEventRefuses(t *testing.T) {
	for _, ev := range []string{"NOSUCHEVENT", "pre-tool", "PreToolUse", "", "  "} {
		d, err := agentHookPick(ev, "")
		if err == nil {
			t.Errorf("agentHookPick(%q, \"\") selected %s/%s — an unknown event must refuse", ev, d.Event, d.Name)
			continue
		}
		if !strings.Contains(err.Error(), "pretool") {
			t.Errorf("agentHookPick(%q) error %q does not name the valid event set", ev, err)
		}
	}
}

// TestAgentHookPick_AmbiguousEventRefuses: pretool carries more than one delegate, and two JSON
// decision payloads concatenated on one stdout parse as neither. Refuse rather than guess.
func TestAgentHookPick_AmbiguousEventRefuses(t *testing.T) {
	n := 0
	for _, d := range agentHookRegistry() {
		if d.Event == "pretool" {
			n++
		}
	}
	if n < 2 {
		t.Skipf("pretool has %d delegate(s); ambiguity is only reachable with 2+", n)
	}
	if d, err := agentHookPick("pretool", ""); err == nil {
		t.Fatalf("ambiguous event selected %s without --delegate", d.Name)
	}
	// ...and naming one resolves it.
	if _, err := agentHookPick("pretool", "repoguard"); err != nil {
		t.Errorf("naming a registered delegate still refused: %v", err)
	}
}

// TestAgentHookPick_UnknownDelegateRefuses is the #5604 lesson applied here: a typo'd selector
// must refuse, not resolve to an empty selection that runs nothing and reports success.
func TestAgentHookPick_UnknownDelegateRefuses(t *testing.T) {
	d, err := agentHookPick("stop", "NOSUCHDELEGATE")
	if err == nil {
		t.Fatalf("agentHookPick(stop, NOSUCHDELEGATE) selected %s — a typo must refuse", d.Name)
	}
	if !strings.Contains(err.Error(), "valid:") {
		t.Errorf("refusal %q does not list the valid delegates", err)
	}
}

// TestRunHooksAgent_MissingDelegateReportsCouldNotRun is the end-to-end witness for the reported
// bug. Pointed at a root with no tools/ at all, today's wrapper exits 0; this must exit non-zero,
// name the delegate, and NOT use the block code.
func TestRunHooksAgent_MissingDelegateReportsCouldNotRun(t *testing.T) {
	empty := t.TempDir()
	var out, errb bytes.Buffer
	rc := runHooksAgent(&out, &errb, strings.NewReader("{}"),
		[]string{"stop", "--root", empty, "--json"})

	if rc == 0 {
		t.Fatalf("a delegate that does not exist reported success (exit 0)\nstdout: %s\nstderr: %s",
			out.String(), errb.String())
	}
	if rc == 2 {
		t.Errorf("could-not-run exited 2 — that is the harness BLOCK code and would refuse every tool call")
	}
	var got struct {
		Status       string   `json:"status"`
		Skipped      []string `json:"skipped"`
		SkippedCount int      `json:"skipped_count"`
		ExitCode     int      `json:"exit_code"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("--json did not emit valid JSON: %v\n%s", err, out.String())
	}
	if got.Status != "could-not-run" {
		t.Errorf("status = %q, want could-not-run", got.Status)
	}
	if got.SkippedCount != 1 || len(got.Skipped) != 1 {
		t.Errorf("skipped ledger = %v (count %d), want exactly the one delegate that did not run",
			got.Skipped, got.SkippedCount)
	}
}

// TestRunHooksAgent_HumanRunNamesTheGap: without --json the degraded run still has to be legible
// on stderr — a silent non-zero is only marginally better than a silent zero.
func TestRunHooksAgent_HumanRunNamesTheGap(t *testing.T) {
	empty := t.TempDir()
	var out, errb bytes.Buffer
	if rc := runHooksAgent(&out, &errb, strings.NewReader("{}"), []string{"stop", "--root", empty}); rc == 0 {
		t.Fatalf("expected a non-zero could-not-run, got 0")
	}
	s := errb.String()
	for _, want := range []string{"could-not-run", "stop", "NOT checked"} {
		if !strings.Contains(s, want) {
			t.Errorf("degraded stderr missing %q; got:\n%s", want, s)
		}
	}
}

// TestRunHooksAgent_UnknownEventExitsNonBlocking drives the CLI surface: a usage refusal is
// visible but must never be the block code, or a typo in settings.json would wedge the fleet.
func TestRunHooksAgent_UnknownEventExitsNonBlocking(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runHooksAgent(&out, &errb, strings.NewReader(""), []string{"NOSUCHEVENT"})
	if rc == 0 {
		t.Fatalf("unknown event exited 0 having run nothing; stderr: %s", errb.String())
	}
	if rc == 2 {
		t.Errorf("unknown event exited 2 (the harness BLOCK code) — a settings typo must not refuse every tool call")
	}
	if !strings.Contains(errb.String(), "NOSUCHEVENT") {
		t.Errorf("refusal does not name the offending event; stderr: %s", errb.String())
	}
}

// TestRunHooksAgent_ForwardsChildExitAndStdout proves the two properties the wrapper broke: the
// child's exit code survives, and its stdout (the JSON decision payload for PreToolUse) is
// forwarded byte-for-byte rather than rewritten by the wrapper.
func TestRunHooksAgent_ForwardsChildExitAndStdout(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	// A stand-in dos_hook.py that answers on stdout and refuses with the block code.
	script := "import sys; sys.stdout.write('{\"decision\":\"block\"}'); sys.exit(2)\n"
	if err := os.WriteFile(filepath.Join(root, "tools", "dos_hook.py"), []byte(script), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	rc := runHooksAgent(&out, &errb, strings.NewReader("{}"), []string{"stop", "--root", root})
	if rc != 2 {
		t.Fatalf("child exited 2 (a deliberate block) but the verb returned %d — the refusal was swallowed\nstderr: %s",
			rc, errb.String())
	}
	if !strings.Contains(out.String(), `"decision":"block"`) {
		t.Errorf("child stdout was not forwarded verbatim; got: %q", out.String())
	}
}

// TestRunHooksAgent_CleanChildExitsZeroQuietly keeps the working path working: a delegate that
// runs and allows must exit 0 and stay quiet, because this fires on every single tool call.
func TestRunHooksAgent_CleanChildExitsZeroQuietly(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "tools", "dos_hook.py"), []byte("import sys; sys.exit(0)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runHooksAgent(&out, &errb, strings.NewReader("{}"), []string{"stop", "--root", root}); rc != 0 {
		t.Fatalf("a clean delegate returned %d, want 0\nstdout: %s\nstderr: %s", rc, out.String(), errb.String())
	}
	if errb.Len() != 0 {
		t.Errorf("clean run wrote to stderr on a per-tool-call hook: %q", errb.String())
	}
}
