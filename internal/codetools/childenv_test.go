package codetools

import (
	"context"
	"encoding/json"
	"runtime"
	"strings"
	"testing"
)

// TestBashToolChildEnvStripsGatewayKeys is the #14450 witness for the codetools bash
// engine: with upstream provider credentials and a key-principal variable set in the
// PARENT process, a real bash tool call's `env` output contains none of them, while
// an allowlisted variable still passes through.
func TestBashToolChildEnvStripsGatewayKeys(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-parent-secret-openai")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-parent-secret")
	t.Setenv("FAK_TENANT_PRINCIPAL", "tenant-a")
	t.Setenv("MY_PROVIDER_TOKEN", "tok-parent-secret")
	t.Setenv("LANG", "childenv-test-lang")

	ts, err := New(Config{Root: t.TempDir()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	command := "env"
	if runtime.GOOS == "windows" {
		command = "set"
	}
	body, _ := json.Marshal(BashArgs{Command: command})
	out, bad := ts.Bash(context.Background(), body)
	if bad {
		t.Fatalf("Bash refused: %s", out)
	}
	var res BashResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("unmarshal result: %v (%s)", err, out)
	}
	if res.ExitCode != 0 {
		t.Fatalf("exit code = %d, stderr=%s", res.ExitCode, res.Stderr)
	}
	for _, secret := range []string{"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "FAK_TENANT_PRINCIPAL", "MY_PROVIDER_TOKEN"} {
		if strings.Contains(res.Stdout, secret) {
			t.Errorf("child env leaked parent secret %s:\n%s", secret, res.Stdout)
		}
	}
	if !strings.Contains(res.Stdout, "PATH=") {
		t.Errorf("allowlisted PATH missing from child env:\n%s", res.Stdout)
	}
	if !strings.Contains(res.Stdout, "LANG=childenv-test-lang") {
		t.Errorf("allowlisted LANG did not pass through to the child:\n%s", res.Stdout)
	}
}
