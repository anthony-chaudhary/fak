package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestMacBenchJSONDoesNotLeakBearer(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			_, _ = w.Write([]byte(`{"ok":true}`))
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer super-secret-test-key" {
			t.Fatalf("Authorization = %q", got)
		}
		_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":25,"completion_tokens":8,"total_tokens":33}}`))
	}))
	defer ts.Close()

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"decode-longgen",
		"--gateway", ts.URL,
		"--decode-tokens", "8",
		"--gateway-key-file", "",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runMacBench code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"schema": "fak.macbench.result.v1"`) || !strings.Contains(out, "tok/s") {
		t.Fatalf("unexpected output:\n%s", out)
	}
	if strings.Contains(out, "super-secret-test-key") || strings.Contains(stderr.String(), "super-secret-test-key") {
		t.Fatalf("leaked bearer:\nstdout=%s\nstderr=%s", out, stderr.String())
	}
}

func TestParseIntCSVRejectsBadValues(t *testing.T) {
	if _, err := parseIntCSV("128, nope"); err == nil {
		t.Fatal("expected parse error")
	}
}

func TestMacBenchFetchesGatewayKeyOverSSH(t *testing.T) {
	oldExec := execCommand
	t.Cleanup(func() { execCommand = oldExec })
	execCommand = func(name string, args ...string) *exec.Cmd {
		if name != "ssh" || len(args) == 0 || args[len(args)-1] != "cat ~/.fak-gateway-key" {
			t.Fatalf("unexpected command: %s %v", name, args)
		}
		cmd := exec.Command(os.Args[0], "-test.run=TestMacBenchSSHHelperProcess", "--")
		cmd.Env = append(os.Environ(), "GO_WANT_MACBENCH_SSH_HELPER=1")
		return cmd
	}
	t.Setenv("FAK_GATEWAY_KEY", "")

	key, err := resolveMacBenchKeyForRun(
		"FAK_GATEWAY_KEY",
		"",
		true,
		"user@node-macos-a.local",
		"",
		"http://example.invalid:8080",
		"decode-longgen",
	)
	if err != nil {
		t.Fatalf("resolveMacBenchKeyForRun: %v", err)
	}
	if key != "fetched-macbench-key" {
		t.Fatalf("key = %q", key)
	}
	if got := os.Getenv("FAK_GATEWAY_KEY"); got != "fetched-macbench-key" {
		t.Fatalf("env key = %q", got)
	}
}

func TestMacBenchSSHHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_MACBENCH_SSH_HELPER") != "1" {
		return
	}
	_, _ = os.Stdout.WriteString("fetched-macbench-key\n")
	os.Exit(0)
}
