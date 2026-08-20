package main

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestGuardDisableDefaultsToOneRawCodexChild(t *testing.T) {
	opts, code, done := parseGuardDisable("guard", nil, &bytes.Buffer{})
	if done || code != 0 {
		t.Fatalf("parseGuardDisable code=%d done=%v", code, done)
	}
	if opts.Reason != guardDisableDefaultReason || !reflect.DeepEqual(opts.Command, []string{"codex"}) {
		t.Fatalf("default options = %+v", opts)
	}
}

func TestGuardDisableRunsChildWithNestedGuardStateRemoved(t *testing.T) {
	t.Setenv(guardActiveEnv, "1")
	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:4111")
	t.Setenv("OPENAI_BASE_URL", "http://127.0.0.1:4111/v1")
	t.Setenv("OPENAI_API_BASE", "http://127.0.0.1:4111/v1")
	t.Setenv("OPENAI_API_KEY", guardCodexOAuthPlaceholderAPIKey)
	t.Setenv("FAK_GUARD_DENYALL_MODE", "enforce")
	t.Setenv("FAK_AUDIT_JOURNAL", "outer.jsonl")
	t.Setenv("CODEX_THREAD_ID", "outer-thread")
	t.Setenv("GUARD_DISABLE_KEEP", "yes")
	t.Setenv("GO_WANT_GUARD_DISABLE_CHILD", "1")

	var stdout, stderr bytes.Buffer
	code := runGuardDisableWithUsage("guard", strings.NewReader(""), &stdout, &stderr, []string{
		"--reason", "  repair   routing\nnow  ", "--",
		os.Args[0], "-test.run=^TestGuardDisableChildProcess$",
	}, "", nil)
	if code != 23 {
		t.Fatalf("runGuardDisable code=%d, want child exit 23\nstdout=%s\nstderr=%s", code, stdout.String(), stderr.String())
	}
	var got map[string]string
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode child env: %v\n%s", err, stdout.String())
	}
	want := map[string]string{
		codexRawRecoveryEnv:      codexRawRecoveryValue,
		codexLoopHookOverrideEnv: "1",
		"GUARD_DISABLE_KEEP":     "yes",
		guardActiveEnv:           "",
		"ANTHROPIC_BASE_URL":     "",
		"OPENAI_BASE_URL":        "",
		"OPENAI_API_BASE":        "",
		"OPENAI_API_KEY":         "",
		"FAK_GUARD_DENYALL_MODE": "",
		"FAK_AUDIT_JOURNAL":      "",
		"CODEX_THREAD_ID":        "",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("child env = %#v, want %#v", got, want)
	}
	if os.Getenv(guardActiveEnv) != "1" || os.Getenv("ANTHROPIC_BASE_URL") != "http://127.0.0.1:4111" {
		t.Fatal("break-glass mutated the parent environment instead of scoping changes to its child")
	}
	for _, text := range []string{"BREAK-GLASS raw session starting", "reason: repair routing now", "NOT running", "raw session ended (exit 23)", "later launches remain guarded by default"} {
		if !strings.Contains(stderr.String(), text) {
			t.Errorf("stderr missing %q:\n%s", text, stderr.String())
		}
	}
}

func TestGuardDisablePreservesDirectProviderEnvOutsideGuard(t *testing.T) {
	environ := []string{
		"ANTHROPIC_BASE_URL=https://direct.example",
		"OPENAI_API_KEY=real-key",
		"KEEP=yes",
	}
	got := guardDisableChildEnv(environ)
	for _, want := range []string{
		"ANTHROPIC_BASE_URL=https://direct.example",
		"OPENAI_API_KEY=real-key",
		"KEEP=yes",
		codexRawRecoveryEnv + "=" + codexRawRecoveryValue,
		codexLoopHookOverrideEnv + "=1",
	} {
		if !envSliceHasExact(got, want) {
			t.Errorf("child env missing %q: %v", want, got)
		}
	}
}

func TestGuardDisableLaunchFailureReturns127(t *testing.T) {
	var stderr bytes.Buffer
	code := runGuardDisableWithUsage("guard", strings.NewReader(""), &bytes.Buffer{}, &stderr, []string{
		"--reason", "missing child witness", "--", "fak-guard-disable-child-that-does-not-exist",
	}, "", nil)
	if code != 127 {
		t.Fatalf("runGuardDisable launch failure code=%d, want 127\n%s", code, stderr.String())
	}
	for _, want := range []string{`launch "fak-guard-disable-child-that-does-not-exist"`, "raw session ended (exit 127)"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("stderr missing %q:\n%s", want, stderr.String())
		}
	}
}

func TestGuardDisableRoutesBeforeWrappedAgentLookup(t *testing.T) {
	args := "disable --reason repair -- " + guardE2EExitZeroCommand()
	code, out, timedOut := runGuardE2E(t, args, map[string]string{"FAK_USAGE_LOG": "off"})
	if timedOut || code != 0 {
		t.Fatalf("fak guard disable code=%d timedOut=%v\n%s", code, timedOut, out)
	}
	if !strings.Contains(out, "BREAK-GLASS raw session starting") || strings.Contains(out, `\"disable\" is not on your PATH`) {
		t.Fatalf("disable did not route to break-glass handler:\n%s", out)
	}
}

func TestGuardDisableChildProcess(t *testing.T) {
	if os.Getenv("GO_WANT_GUARD_DISABLE_CHILD") != "1" {
		return
	}
	names := []string{
		codexRawRecoveryEnv, codexLoopHookOverrideEnv, "GUARD_DISABLE_KEEP", guardActiveEnv,
		"ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY",
		"FAK_GUARD_DENYALL_MODE", "FAK_AUDIT_JOURNAL", "CODEX_THREAD_ID",
	}
	values := make(map[string]string, len(names))
	for _, name := range names {
		values[name] = os.Getenv(name)
	}
	_ = json.NewEncoder(os.Stdout).Encode(values)
	os.Exit(23)
}

func envSliceHasExact(env []string, want string) bool {
	for _, got := range env {
		if got == want {
			return true
		}
	}
	return false
}
