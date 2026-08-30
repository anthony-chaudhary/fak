package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestBuildCodexLaunchArgvDefault(t *testing.T) {
	got := buildCodexLaunchArgv("/bin/fak", codexLaunchOptions{
		splitMode:     "auto",
		splitWhere:    "bottom",
		splitInterval: 2 * time.Second,
		codexConfig:   true,
	})
	want := []string{
		"/bin/fak", "guard",
		"--split", "auto",
		"--split-where", "bottom",
		"--split-interval", "2s",
		"--",
		"codex",
		"-c", "model_auto_compact_token_limit=96000",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCodexLaunchArgv default = %#v\nwant %#v", got, want)
	}
}

func TestBuildCodexLaunchArgvAdvancedFlagsAndPassthrough(t *testing.T) {
	got := buildCodexLaunchArgv("fak.exe", codexLaunchOptions{
		skipPermissions: true,
		splitMode:       "on",
		splitWhere:      "right",
		splitInterval:   500 * time.Millisecond,
		policyPath:      "policy.json",
		apiKeyEnv:       "ALT_OPENAI_KEY",
		baseURL:         "https://api.example.test/v1",
		model:           "gpt-test",
		auditPath:       "audit.jsonl",
		noAudit:         true,
		quiet:           true,
		localAuto:       true,
		ggufPath:        "qwen.gguf",
		gpuBackend:      "cuda",
		tokenizerPath:   "tokenizer.json",
		codexConfig:     false,
		codexHome:       "codex-home",
		passthrough:     []string{"exec", "--json", "summarize AGENTS.md"},
	})
	want := []string{
		"fak.exe", "guard",
		"--split", "on",
		"--split-where", "right",
		"--split-interval", "500ms",
		"--policy", "policy.json",
		"--api-key-env", "ALT_OPENAI_KEY",
		"--base-url", "https://api.example.test/v1",
		"--model", "gpt-test",
		"--audit", "audit.jsonl",
		"--audit", "off",
		"--quiet",
		"--local",
		"--gguf", "qwen.gguf",
		"--backend", "cuda",
		"--tokenizer", "tokenizer.json",
		"--codex-config=false",
		"--codex-home", "codex-home",
		"--",
		"codex",
		"-c", "model_auto_compact_token_limit=96000",
		"--dangerously-bypass-approvals-and-sandbox",
		"exec", "--json", "summarize AGENTS.md",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCodexLaunchArgv advanced = %#v\nwant %#v", got, want)
	}
}

func TestBuildCodexLaunchArgvCompactionLimitAllowsExplicitOverride(t *testing.T) {
	got := buildCodexLaunchArgv("fak", codexLaunchOptions{
		splitInterval: time.Second,
		codexConfig:   true,
		passthrough: []string{
			"-c", "model_auto_compact_token_limit=120000", "exec", "work",
		},
	})
	joined := strings.Join(got, " ")
	want := "codex -c model_auto_compact_token_limit=96000 -c model_auto_compact_token_limit=120000 exec work"
	if !strings.Contains(joined, want) {
		t.Fatalf("Codex compaction defaults/override order = %q, want substring %q", joined, want)
	}
}

func TestBuildCodexLaunchArgvManagedCache(t *testing.T) {
	// A non-default posture is forwarded to guard as --managed-cache <mode>...
	got := buildCodexLaunchArgv("fak", codexLaunchOptions{
		skipPermissions: true,
		splitMode:       "auto",
		splitWhere:      "bottom",
		splitInterval:   time.Second,
		managedCache:    "on",
		codexConfig:     true,
	})
	want := []string{
		"fak", "guard",
		"--split", "auto",
		"--split-where", "bottom",
		"--split-interval", "1s",
		"--managed-cache", "on",
		"--",
		"codex",
		"-c", "model_auto_compact_token_limit=96000",
		"--dangerously-bypass-approvals-and-sandbox",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCodexLaunchArgv managed-cache = %#v\nwant %#v", got, want)
	}

	// ...but auto (guard's own default) emits nothing, so an unconfigured argv is unchanged.
	for _, mode := range []string{"", "auto"} {
		got := buildCodexLaunchArgv("fak", codexLaunchOptions{
			skipPermissions: true,
			splitMode:       "auto",
			splitWhere:      "bottom",
			splitInterval:   time.Second,
			managedCache:    mode,
			codexConfig:     true,
		})
		if strings.Contains(strings.Join(got, " "), "--managed-cache") {
			t.Fatalf("managedCache=%q must emit no --managed-cache, got %#v", mode, got)
		}
	}
}

func TestBuildCodexLaunchArgvSkipPermissionsOff(t *testing.T) {
	got := buildCodexLaunchArgv("fak", codexLaunchOptions{
		skipPermissions: false,
		splitMode:       "off",
		splitWhere:      "bottom",
		splitInterval:   time.Second,
		codexConfig:     true,
		passthrough:     []string{"exec", "do x"},
	})
	want := []string{"fak", "guard", "--split", "off", "--split-where", "bottom", "--split-interval", "1s", "--", "codex", "-c", "model_auto_compact_token_limit=96000", "exec", "do x"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("skip-permissions off argv = %#v\nwant %#v", got, want)
	}
}

func TestRunCodexDryRun(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--dry-run",
		"--split", "off",
		"--policy", "floor.json",
		"--api-key-env", "MY_OPENAI_KEY",
		"--",
		"exec", "--json", "check the repo",
	})
	if rc != 0 {
		t.Fatalf("runCodex dry-run rc=%d stderr=%s", rc, errb.String())
	}
	gotOut := out.String()
	for _, want := range []string{
		"guard --split off",
		"--policy floor.json",
		"--api-key-env MY_OPENAI_KEY",
		"codex -c model_auto_compact_token_limit=96000 --dangerously-bypass-approvals-and-sandbox exec --json check the repo",
	} {
		if !strings.Contains(gotOut, want) {
			t.Fatalf("dry-run stdout missing %q:\n%s", want, gotOut)
		}
	}
	gotErr := errb.String()
	for _, want := range []string{"agent 80% / fak info 20%", "approval/sandbox bypass (managed default)", "Codex subagents inherit this mode", "fak gates remain active", "dry-run"} {
		if !strings.Contains(gotErr, want) {
			t.Fatalf("dry-run stderr missing %q:\n%s", want, gotErr)
		}
	}
}

func TestRunCodexDryRunExplicitSkipPermissions(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--dry-run",
		"--split", "off",
		"--skip-permissions",
		"--", "exec", "check the repo",
	})
	if rc != 0 {
		t.Fatalf("explicit bypass dry-run rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "codex -c model_auto_compact_token_limit=96000 --dangerously-bypass-approvals-and-sandbox exec check the repo") {
		t.Fatalf("explicit bypass dry-run omitted Codex flag:\n%s", out.String())
	}
	for _, want := range []string{"approval/sandbox bypass (managed default)", "fak gates remain active"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("explicit bypass banner missing %q:\n%s", want, errb.String())
		}
	}
}

func TestRunCodexPermissionHelpNamesManagedDefaultAndNativeOptOut(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runCodex(&out, &errb, []string{"--help"}); rc != 2 {
		t.Fatalf("runCodex --help rc=%d, want 2", rc)
	}
	for _, want := range []string{
		"default true for managed launches",
		"restore Codex's native approval prompts and sandbox",
		"Codex subagents inherit this parent permission mode",
		"fak routing, capacity, policy, hook, and loop gates still apply",
	} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("Codex help missing %q:\n%s", want, errb.String())
		}
	}
}

func TestCodexDryRunSubprocessPermissions(t *testing.T) {
	repoRoot := filepath.Clean(filepath.Join("..", ".."))
	built := filepath.Join(t.TempDir(), "fak-codex-permissions-test")
	if runtime.GOOS == "windows" {
		built += ".exe"
	}
	build := exec.Command("go", "build", "-buildvcs=false", "-o", built, "./cmd/fak")
	build.Dir = repoRoot
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fak subprocess witness: %v\n%s", err, output)
	}

	for _, tc := range []struct {
		name       string
		extra      []string
		wantBypass bool
		wantBanner string
	}{
		{name: "bare uses managed bypass default", wantBypass: true, wantBanner: "approval/sandbox bypass (managed default)"},
		{name: "native opt-out restores Codex layer", extra: []string{"--native-permissions"}, wantBanner: "native approvals + sandbox explicitly restored"},
		{name: "legacy explicit flag still selects bypass", extra: []string{"--skip-permissions"}, wantBypass: true, wantBanner: "approval/sandbox bypass (managed default)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			args := []string{"codex", "--freshness-gate", "off", "--dry-run", "--split", "off"}
			args = append(args, tc.extra...)
			args = append(args, "--", "exec", "check the repo")
			cmd := exec.Command(built, args...)
			cmd.Dir = repoRoot
			cmd.Env = append(os.Environ(), "FLEET_CODEX_LOOP_GATE=off")
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("fak codex subprocess: %v\n%s", err, output)
			}
			got := string(output)
			if has := strings.Contains(got, "--dangerously-bypass-approvals-and-sandbox"); has != tc.wantBypass {
				t.Fatalf("subprocess bypass present=%v, want %v:\n%s", has, tc.wantBypass, got)
			}
			for _, want := range []string{tc.wantBanner, "Codex subagents inherit this mode", "fak gates remain active"} {
				if !strings.Contains(got, want) {
					t.Fatalf("subprocess output missing %q:\n%s", want, got)
				}
			}
		})
	}
}

func TestRunCodexExecSeam(t *testing.T) {
	orig := codexLaunchRun
	var gotArgv, gotEnv []string
	codexLaunchRun = func(_, _ io.Writer, argv, env []string) int {
		gotArgv = append([]string{}, argv...)
		gotEnv = append([]string{}, env...)
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{"--split", "off", "--loop-gate", "off", "--", "exec", "do x"})
	if rc != 17 {
		t.Fatalf("runCodex rc=%d, want seam rc 17; stderr=%s", rc, errb.String())
	}
	if len(gotArgv) == 0 || gotArgv[1] != "guard" {
		t.Fatalf("argv was not a guard launch: %#v", gotArgv)
	}
	if !strings.Contains(strings.Join(gotArgv, " "), "--dangerously-bypass-approvals-and-sandbox") {
		t.Fatalf("bare managed Codex exec omitted bypass flag: %#v", gotArgv)
	}
	if !strings.HasSuffix(strings.Join(gotArgv, " "), "-- codex -c model_auto_compact_token_limit=96000 --dangerously-bypass-approvals-and-sandbox exec do x") {
		t.Fatalf("argv tail wrong: %#v", gotArgv)
	}
	if len(gotEnv) == 0 {
		t.Fatal("expected environment to be forwarded to child")
	}
}

func TestRunCodexLoopGateRefusesBeforeSpawn(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := codexLauncherLoopFixtureForProvider(t, "fak")
	if err := writeCodexGuardWitness(home, "loop-session"); err != nil {
		t.Fatal(err)
	}
	orig := codexLaunchRun
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		t.Fatal("codex child spawned despite loop gate refusal")
		return 99
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--loop-gate-limit", "2",
		"--",
		"exec", "do x",
	})
	if rc != 1 {
		t.Fatalf("runCodex loop gate rc=%d, want 1; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	got := errb.String()
	for _, want := range []string{
		"loop gate REFUSE fail-on=loop verdict=LOOP",
		"fak sessions codex-loop --recent",
		"update_plan",
		"Plan updated",
		"loop-session verdict=LOOP",
		"operator override: rerun as `fak codex --loop-gate off`",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("loop-gate stderr missing %q:\n%s", want, got)
		}
	}
}

func TestRunCodexLoopGateDefaultOffSkipsAudit(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	t.Setenv("FLEET_CODEX_LOOP_GATE", "")
	home := codexLauncherLoopFixtureForProvider(t, "fak")
	if err := writeCodexGuardWitness(home, "loop-session"); err != nil {
		t.Fatal(err)
	}
	orig := codexLaunchRun
	spawned := false
	codexLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
		spawned = true
		if !strings.Contains(strings.Join(argv, " "), "--codex-loop-gate off") {
			t.Fatalf("default-off outer launcher did not suppress the nested audit: %#v", argv)
		}
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--", "exec", "do x",
	})
	if rc != 17 || !spawned {
		t.Fatalf("default-off Codex launch rc=%d spawned=%v, want guarded child rc=17; stderr=%s", rc, spawned, errb.String())
	}
	if strings.Contains(errb.String(), "loop gate REFUSE") || strings.Contains(errb.String(), "loop gate allow") {
		t.Fatalf("default-off Codex launch evaluated the loop gate:\n%s", errb.String())
	}
}

func TestRunCodexLoopGateEnvironmentOptInRefusesBeforeSpawn(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	t.Setenv("FLEET_CODEX_LOOP_GATE", "loop")
	home := codexLauncherLoopFixtureForProvider(t, "fak")
	if err := writeCodexGuardWitness(home, "loop-session"); err != nil {
		t.Fatal(err)
	}
	orig := codexLaunchRun
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		t.Fatal("Codex child spawned despite environment-opted-in loop gate refusal")
		return 99
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate-since-hours", "0",
		"--", "exec", "do x",
	})
	if rc != 1 || !strings.Contains(errb.String(), "loop gate REFUSE fail-on=loop verdict=LOOP") {
		t.Fatalf("environment-opted-in loop gate rc=%d, want witnessed refusal; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
}

func TestRunCodexLoopGateHelpSaysOptInDefaultOff(t *testing.T) {
	var out, errb bytes.Buffer
	if rc := runCodex(&out, &errb, []string{"--help"}); rc != 2 {
		t.Fatalf("runCodex --help rc=%d, want 2", rc)
	}
	for _, want := range []string{"opt-in pre-launch audit", "else off", "loop|action"} {
		if !strings.Contains(errb.String(), want) {
			t.Fatalf("Codex help missing %q:\n%s", want, errb.String())
		}
	}
}

func TestRunCodexLoopGateIgnoresNewestLoopFromDifferentDirectory(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "16")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(name, sessionID, cwd string, loop bool) {
		lines := []string{fmt.Sprintf(`{"timestamp":"2026-07-16T10:00:00Z","type":"session_meta","payload":{"session_id":%q,"cwd":%q,"model_provider":"fak"}}`, sessionID, cwd)}
		if loop {
			for i := 1; i <= 3; i++ {
				lines = append(lines,
					fmt.Sprintf(`{"timestamp":"2026-07-16T10:00:0%dZ","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{}","call_id":"c%d"}}`, i*2-1, i),
					fmt.Sprintf(`{"timestamp":"2026-07-16T10:00:0%dZ","type":"response_item","payload":{"type":"function_call_output","call_id":"c%d","output":"same failure"}}`, i*2, i))
			}
		}
		writeCodexLoopFixture(t, filepath.Join(sessionsDir, name), lines)
	}
	launchDir := filepath.Join(t.TempDir(), "repo-a")
	otherDir := filepath.Join(t.TempDir(), "repo-b")
	if err := os.MkdirAll(launchDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(otherDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Older same-repository rollout is clean; globally newest rollout is a loop
	// from another checkout and must not poison this launch.
	write("rollout-2026-07-16T09-00-00-same.jsonl", "same", launchDir, false)
	write("rollout-2026-07-16T10-00-00-other.jsonl", "other", otherDir, true)
	if err := writeCodexGuardWitness(home, "other"); err != nil {
		t.Fatal(err)
	}

	var errb bytes.Buffer
	rc := runCodexLoopGate(&errb, codexLoopGateConfig{
		Threshold: "loop", CodexHome: home, SinceHours: 0, WorkingDir: launchDir,
	})
	if rc != 0 {
		t.Fatalf("unrelated repository loop poisoned launch: rc=%d stderr=%s", rc, errb.String())
	}
}

func TestRunCodexLoopGateStillRefusesLoopInSameDirectory(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := codexLauncherLoopFixtureForProvider(t, "fak")
	loopPath := filepath.Join(home, "sessions", "2026", "07", "06", "rollout-2026-07-06T02-25-00-loop.jsonl")
	data, err := os.ReadFile(loopPath)
	if err != nil {
		t.Fatal(err)
	}
	cwd := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(cwd, 0o755); err != nil {
		t.Fatal(err)
	}
	data = bytes.Replace(data, []byte(`"model_provider":"fak"`), []byte(fmt.Sprintf(`"cwd":%q,"model_provider":"fak"`, cwd)), 1)
	if err := os.WriteFile(loopPath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := writeCodexGuardWitness(home, "loop-session"); err != nil {
		t.Fatal(err)
	}
	var errb bytes.Buffer
	rc := runCodexLoopGate(&errb, codexLoopGateConfig{Threshold: "loop", CodexHome: home, SinceHours: 0, WorkingDir: cwd})
	if rc != 1 || !strings.Contains(errb.String(), "loop gate REFUSE") {
		t.Fatalf("same-repository loop was not refused: rc=%d stderr=%s", rc, errb.String())
	}
}

func TestCodexLoopWorkingDirsOverlapSubdirectories(t *testing.T) {
	root := filepath.Join(t.TempDir(), "repo")
	if !codexWorkingDirsOverlap(root, filepath.Join(root, "subdir")) {
		t.Fatal("repo root and subdirectory should share launch-loop scope")
	}
	if codexWorkingDirsOverlap(root, filepath.Join(t.TempDir(), "other")) {
		t.Fatal("unrelated repositories should not share launch-loop scope")
	}
}

func TestRunCodexLoopGateAllowsForwardProgressPlanTraffic(t *testing.T) {
	// Regression: a guarded (fak-provider) session that called update_plan many
	// times, each with a DISTINCT plan, is forward planning progress — not a
	// no-progress loop. It must not poison the next `fak codex` launch. This is
	// the false positive reported for Epic #4277 (update_plan count=44
	// args_digests=44 refusing a fresh guarded launch).
	t.Setenv("CODEX_THREAD_ID", "")
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "11")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(sessionsDir, "rollout-2026-07-11T08-00-00-progress.jsonl")
	lines := []string{
		`{"timestamp":"2026-07-11T15:00:00.000Z","type":"session_meta","payload":{"session_id":"progress-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"fak","git":{"commit_hash":"abc1234","branch":"main"}}}`,
	}
	for i, step := range []string{"one", "two", "three", "four", "five", "six"} {
		call := fmt.Sprintf("plan_%d", i+1)
		lines = append(lines,
			fmt.Sprintf(`{"timestamp":"2026-07-11T15:0%d:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"%s\",\"status\":\"in_progress\"}]}","call_id":"%s"}}`, i, step, call),
			fmt.Sprintf(`{"timestamp":"2026-07-11T15:0%d:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"%s","output":"Plan updated"}}`, i, call),
		)
	}
	writeCodexLoopFixture(t, path, lines)

	orig := codexLaunchRun
	spawned := false
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		spawned = true
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--", "exec", "keep planning",
	})
	if rc != 17 || !spawned {
		t.Fatalf("forward-progress plan traffic blocked launch: rc=%d spawned=%v stderr=%s", rc, spawned, errb.String())
	}
	if strings.Contains(errb.String(), "loop gate REFUSE") {
		t.Fatalf("distinct-plan update_plan traffic poisoned the guarded relaunch:\n%s", errb.String())
	}
}

func TestRunCodexLoopGateAllowsNewestAbruptCrash(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := codexLauncherLoopFixtureForProvider(t, "fak")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "06")
	older := filepath.Join(sessionsDir, "rollout-2026-07-06T02-25-00-loop.jsonl")
	crash := filepath.Join(sessionsDir, "rollout-2026-07-06T03-25-00-crash.jsonl")
	writeCodexLoopFixture(t, crash, []string{
		`{"timestamp":"2026-07-06T03:25:00.000Z","type":"session_meta","payload":{"session_id":"crash-session","originator":"codex-tui","model_provider":"fak"}}`,
		`{"timestamp":"2026-07-06T03:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"plan_1"}}`,
		`{"timestamp":"2026-07-06T03:25:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T03:25:13.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"plan_2"}}`,
		`{"timestamp":"2026-07-06T03:25:14.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T03:25:23.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"plan_3"}}`,
		`{"timestamp":"2026-07-06T03:25:24.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_3","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T03:25:33.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{}","call_id":"shell_pending"}}`,
	})
	now := time.Now()
	if err := os.Chtimes(older, now.Add(-time.Minute), now.Add(-time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(crash, now, now); err != nil {
		t.Fatal(err)
	}

	orig := codexLaunchRun
	spawned := false
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		spawned = true
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--", "exec", "resume after crash",
	})
	if rc != 17 || !spawned {
		t.Fatalf("crash-then-relaunch rc=%d spawned=%v, want guarded child rc=17; stderr=%s", rc, spawned, errb.String())
	}
	if strings.Contains(errb.String(), "loop gate REFUSE") {
		t.Fatalf("abrupt latest rollout poisoned relaunch:\n%s", errb.String())
	}
}

func TestNewestCodexLoopLaunchScanIsByteBounded(t *testing.T) {
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "06")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(sessionsDir, "rollout-large.jsonl")
	fh, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	lines := []string{
		`{"timestamp":"2026-07-06T02:25:00.000Z","type":"session_meta","payload":{"session_id":"large-loop","model_provider":"fak"}}`,
		strings.Repeat("x", int(codexLoopLaunchMaxBytes)+1024),
		`{"timestamp":"2026-07-06T02:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"plan_1"}}`,
		`{"timestamp":"2026-07-06T02:25:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:13.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"plan_2"}}`,
		`{"timestamp":"2026-07-06T02:25:14.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:23.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{}","call_id":"plan_3"}}`,
		`{"timestamp":"2026-07-06T02:25:24.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_3","output":"Plan updated"}}`,
	}
	if _, err := io.WriteString(fh, strings.Join(lines, "\n")+"\n"); err != nil {
		fh.Close()
		t.Fatal(err)
	}
	if err := fh.Close(); err != nil {
		t.Fatal(err)
	}

	rep, scan, err := diagnoseNewestCodexLoopForLaunch(home, 0, "")
	if err != nil {
		t.Fatal(err)
	}
	if !scan.Truncated || scan.BytesRead > codexLoopLaunchMaxBytes {
		t.Fatalf("launch scan = %+v, want truncated and <= %d bytes", scan, codexLoopLaunchMaxBytes)
	}
	if rep.Scanned != 1 || rep.Verdict != "LOOP" {
		t.Fatalf("bounded launch report scanned=%d verdict=%s, want 1/LOOP", rep.Scanned, rep.Verdict)
	}
}

func TestRunCodexLoopGateAllowsGuardedRemediationForDirectLoops(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := codexLauncherLoopFixture(t)
	orig := codexLaunchRun
	var spawned bool
	codexLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
		spawned = true
		if !strings.Contains(strings.Join(argv, " "), " guard ") {
			t.Fatalf("remediation child was not routed through fak guard: %#v", argv)
		}
		if !strings.Contains(strings.Join(argv, " "), "--codex-loop-gate off") {
			t.Fatalf("outer gate did not suppress the duplicate inner verdict: %#v", argv)
		}
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--",
		"exec", "do x",
	})
	if rc != 17 || !spawned {
		t.Fatalf("direct-loop remediation rc=%d spawned=%v, want guarded child rc=17; stderr=%s", rc, spawned, errb.String())
	}
	if !strings.Contains(errb.String(), "remediation allow") || strings.Contains(errb.String(), "pass --loop-gate off") {
		t.Fatalf("remediation guidance is still contradictory:\n%s", errb.String())
	}
}

func TestRunCodexLoopGateRefusesCurrentDirectThread(t *testing.T) {
	home, threadID := codexLauncherCurrentThreadFixture(t)
	t.Setenv("CODEX_THREAD_ID", threadID)
	orig := codexLaunchRun
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		t.Fatal("codex child spawned despite current-thread direct-provider refusal")
		return 99
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--",
		"exec", "do x",
	})
	if rc != 1 {
		t.Fatalf("runCodex current-thread gate rc=%d, want 1; stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
	got := errb.String()
	for _, want := range []string{
		"current-thread gate REFUSE fail-on=unguarded verdict=OK reason=codex_session_bypassed_fak_guard",
		"session        : " + threadID + " provider=openai",
		"next action    : launch future Codex sessions through `fak codex`",
		"operator override: rerun as `fak codex --loop-gate off`",
		"the flag belongs after the fak verb",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("current-thread gate stderr missing %q:\n%s", want, got)
		}
	}
}

func TestRunCodexLoopGateAllowsCurrentGuardWitnessedThread(t *testing.T) {
	home, threadID := codexLauncherCurrentThreadFixture(t)
	t.Setenv("CODEX_THREAD_ID", threadID)
	if err := writeCodexGuardWitness(home, threadID); err != nil {
		t.Fatal(err)
	}
	orig := codexLaunchRun
	spawned := false
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		spawned = true
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "loop",
		"--loop-gate-since-hours", "0",
		"--",
		"exec", "do x",
	})
	if rc != 17 || !spawned {
		t.Fatalf("guard-witnessed current-thread gate rc=%d spawned=%v, want child rc=17; stdout=%s stderr=%s", rc, spawned, out.String(), errb.String())
	}
	if strings.Contains(errb.String(), "current-thread gate REFUSE") {
		t.Fatalf("durably witnessed current thread was misclassified as unguarded:\n%s", errb.String())
	}
}
func TestRunCodexLoopGateOffAllowsSpawn(t *testing.T) {
	home := codexLauncherLoopFixture(t)
	orig := codexLaunchRun
	var spawned bool
	codexLaunchRun = func(_, _ io.Writer, argv, _ []string) int {
		spawned = true
		if !strings.Contains(strings.Join(argv, " "), "--codex-home "+home) {
			t.Fatalf("codex-home not forwarded to guard argv: %#v", argv)
		}
		return 17
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{
		"--split", "off",
		"--codex-home", home,
		"--loop-gate", "off",
		"--",
		"exec", "do x",
	})
	if rc != 17 || !spawned {
		t.Fatalf("runCodex loop gate off rc=%d spawned=%v stderr=%s", rc, spawned, errb.String())
	}
}

func TestRunCodexLoopGateInvalidThreshold(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	orig := codexLaunchRun
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int {
		t.Fatal("codex child spawned despite invalid loop gate")
		return 99
	}
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{"--loop-gate", "urgent"})
	if rc != 2 || !strings.Contains(errb.String(), "invalid --loop-gate") {
		t.Fatalf("invalid loop gate rc=%d stdout=%s stderr=%s", rc, out.String(), errb.String())
	}
}

func TestRunCodexInvalidSplitFlags(t *testing.T) {
	for _, argv := range [][]string{
		{"--split", "sideways"},
		{"--split-where", "diagonal"},
	} {
		var out, errb bytes.Buffer
		if rc := runCodex(&out, &errb, argv); rc != 2 {
			t.Fatalf("runCodex(%v) rc=%d stderr=%s", argv, rc, errb.String())
		}
	}
}

func codexLauncherLoopFixture(t *testing.T) string {
	return codexLauncherLoopFixtureForProvider(t, "openai")
}

func codexLauncherLoopFixtureForProvider(t *testing.T, provider string) string {
	t.Helper()
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "06")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	loopPath := filepath.Join(sessionsDir, "rollout-2026-07-06T02-25-00-loop.jsonl")
	writeCodexLoopFixture(t, loopPath, []string{
		`{"timestamp":"2026-07-06T02:25:00.000Z","type":"session_meta","payload":{"session_id":"loop-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"` + provider + `","git":{"commit_hash":"4926739","branch":"main"}}}`,
		// Same plan re-submitted verbatim each turn: a genuine no-progress
		// thrash loop (ArgsDigestCount==1 < Count), the case the loop gate must
		// still refuse. Distinct-argument progress is covered by
		// TestRunCodexLoopGateAllowsForwardProgressPlanTraffic.
		`{"timestamp":"2026-07-06T02:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_1"}}`,
		`{"timestamp":"2026-07-06T02:25:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:15.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_2"}}`,
		`{"timestamp":"2026-07-06T02:25:16.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:27.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_3"}}`,
		`{"timestamp":"2026-07-06T02:25:28.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_3","output":"Plan updated"}}`,
	})
	return home
}

func codexLauncherCurrentThreadFixture(t *testing.T) (string, string) {
	t.Helper()
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "06")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	threadID := "019f3540-52dd-7001-b559-2818dc14ede6"
	path := filepath.Join(sessionsDir, "rollout-2026-07-06T09-30-00-"+threadID+".jsonl")
	writeCodexLoopFixture(t, path, []string{
		`{"timestamp":"2026-07-06T16:30:00.000Z","type":"session_meta","payload":{"session_id":"019f3540-52dd-7001-b559-2818dc14ede6","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"111ff04","branch":"main"}}}`,
		`{"timestamp":"2026-07-06T16:30:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"git status --short\"}","call_id":"shell_1"}}`,
		`{"timestamp":"2026-07-06T16:30:03.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"shell_1","output":"## main"}}`,
	})
	return home, threadID
}

func TestRunCodexSuccessfulLaunchUsesConciseTimedStatus(t *testing.T) {
	orig := codexLaunchRun
	codexLaunchRun = func(_, _ io.Writer, _, _ []string) int { return 0 }
	t.Cleanup(func() { codexLaunchRun = orig })

	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{"--split", "off", "--loop-gate", "off", "--", "exec", "do x"})
	if rc != 0 {
		t.Fatalf("runCodex rc=%d, want 0; stderr=%s", rc, errb.String())
	}
	got := errb.String()
	for _, unwanted := range []string{"  view        =", "  permissions =", "  command     ="} {
		if strings.Contains(got, unwanted) {
			t.Errorf("normal launch stderr contains verbose preamble %q:\n%s", unwanted, got)
		}
	}
	for _, want := range []string{
		"fak codex: launching Codex through fak guard ...",
		"fak codex: Codex completed successfully in ",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("normal launch stderr missing %q:\n%s", want, got)
		}
	}
}

func TestRunCodexDryRunRetainsCommandDetails(t *testing.T) {
	var out, errb bytes.Buffer
	rc := runCodex(&out, &errb, []string{"--dry-run", "--split", "off", "--", "exec", "do x"})
	if rc != 0 {
		t.Fatalf("runCodex rc=%d, want 0; stderr=%s", rc, errb.String())
	}
	got := errb.String()
	for _, want := range []string{"dry-run - not launching", "  view        =", "  permissions =", "  command     ="} {
		if !strings.Contains(got, want) {
			t.Errorf("dry-run stderr missing %q:\n%s", want, got)
		}
	}
	if !strings.Contains(out.String(), " guard ") || !strings.Contains(out.String(), " exec do x") {
		t.Fatalf("dry-run stdout did not retain runnable command: %q", out.String())
	}
}
