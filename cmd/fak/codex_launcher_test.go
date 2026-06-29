package main

import (
	"bytes"
	"io"
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
		"--",
		"codex",
		"--dangerously-bypass-approvals-and-sandbox",
		"exec", "--json", "summarize AGENTS.md",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildCodexLaunchArgv advanced = %#v\nwant %#v", got, want)
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
	rc := runCodex(&out, &errb, []string{"--split", "off", "--skip-permissions=false", "--", "exec", "do x"})
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
