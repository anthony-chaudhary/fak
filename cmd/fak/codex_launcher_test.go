package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestBuildCodexLaunchArgvDefault(t *testing.T) {
	got := buildCodexLaunchArgv("/bin/fak", codexLaunchOptions{
		skipPermissions: true,
		splitMode:       "auto",
		splitWhere:      "bottom",
		splitInterval:   2 * time.Second,
		codexConfig:     true,
	})
	want := []string{
		"/bin/fak", "guard",
		"--split", "auto",
		"--split-where", "bottom",
		"--split-interval", "2s",
		"--",
		"codex",
		"--dangerously-bypass-approvals-and-sandbox",
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
		"--no-audit",
		"--quiet",
		"--local",
		"--gguf", "qwen.gguf",
		"--backend", "cuda",
		"--tokenizer", "tokenizer.json",
		"--codex-config=false",
		"--codex-home", "codex-home",
		"--",
		"codex",
		"--dangerously-bypass-approvals-and-sandbox",
		"exec", "--json", "summarize AGENTS.md",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCodexLaunchArgv advanced = %#v\nwant %#v", got, want)
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
	want := []string{"fak", "guard", "--split", "off", "--split-where", "bottom", "--split-interval", "1s", "--", "codex", "exec", "do x"}
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
		"codex --dangerously-bypass-approvals-and-sandbox exec --json check the repo",
	} {
		if !strings.Contains(gotOut, want) {
			t.Fatalf("dry-run stdout missing %q:\n%s", want, gotOut)
		}
	}
	gotErr := errb.String()
	for _, want := range []string{"agent 80% / fak info 20%", "fak floor is the permission system", "dry-run"} {
		if !strings.Contains(gotErr, want) {
			t.Fatalf("dry-run stderr missing %q:\n%s", want, gotErr)
		}
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
	rc := runCodex(&out, &errb, []string{"--split", "off", "--loop-gate", "off", "--skip-permissions=false", "--", "exec", "do x"})
	if rc != 17 {
		t.Fatalf("runCodex rc=%d, want seam rc 17; stderr=%s", rc, errb.String())
	}
	if len(gotArgv) == 0 || gotArgv[1] != "guard" {
		t.Fatalf("argv was not a guard launch: %#v", gotArgv)
	}
	if strings.Contains(strings.Join(gotArgv, " "), "--dangerously-bypass-approvals-and-sandbox") {
		t.Fatalf("--skip-permissions=false still passed Codex bypass flag: %#v", gotArgv)
	}
	if !strings.HasSuffix(strings.Join(gotArgv, " "), "-- codex exec do x") {
		t.Fatalf("argv tail wrong: %#v", gotArgv)
	}
	if len(gotEnv) == 0 {
		t.Fatal("expected environment to be forwarded to child")
	}
}

func TestRunCodexLoopGateRefusesBeforeSpawn(t *testing.T) {
	home := codexLauncherLoopFixture(t)
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
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("loop-gate stderr missing %q:\n%s", want, got)
		}
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
	t.Helper()
	home := filepath.Join(t.TempDir(), "codex-home")
	sessionsDir := filepath.Join(home, "sessions", "2026", "07", "06")
	if err := os.MkdirAll(sessionsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	loopPath := filepath.Join(sessionsDir, "rollout-2026-07-06T02-25-00-loop.jsonl")
	writeCodexLoopFixture(t, loopPath, []string{
		`{"timestamp":"2026-07-06T02:25:00.000Z","type":"session_meta","payload":{"session_id":"loop-session","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"4926739","branch":"main"}}}`,
		`{"timestamp":"2026-07-06T02:25:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_1"}}`,
		`{"timestamp":"2026-07-06T02:25:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:15.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"two\",\"status\":\"in_progress\"}]}","call_id":"plan_2"}}`,
		`{"timestamp":"2026-07-06T02:25:16.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
		`{"timestamp":"2026-07-06T02:25:27.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"three\",\"status\":\"in_progress\"}]}","call_id":"plan_3"}}`,
		`{"timestamp":"2026-07-06T02:25:28.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_3","output":"Plan updated"}}`,
	})
	return home
}
