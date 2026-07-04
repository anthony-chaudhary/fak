package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macbench"
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

func TestMacBenchWatchWritesResultWhenHealthy(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","planner":"inkernel","model":"qwen3.6-27b"}`))
		case "/v1/chat/completions":
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":25,"completion_tokens":4,"total_tokens":29}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer ts.Close()

	result := t.TempDir() + "/macbench-result.json"
	logPath := t.TempDir() + "/macbench-watch.log"
	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{
		"watch",
		"--gateway", ts.URL,
		"--model", "qwen3.6-27b",
		"--gateway-key-file", "",
		"--fetch-key=false",
		"--duration", "1s",
		"--interval", "1ms",
		"--health-timeout", "1s",
		"--run-timeout", "5s",
		"--max-polls", "1",
		"--decode-tokens", "4",
		"--prefill-tokens", "8",
		"--concurrency", "1",
		"--result", result,
		"--log", logPath,
	})
	if code != 0 {
		t.Fatalf("runMacBench watch code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	b, err := os.ReadFile(result)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if !strings.Contains(string(b), `"schema": "fak.macbench.result.v1"`) || !strings.Contains(string(b), `"suite": "all"`) {
		t.Fatalf("unexpected result:\n%s", b)
	}
	if !strings.Contains(stdout.String(), `"suite": "health"`) || !strings.Contains(stdout.String(), `"suite": "all"`) {
		t.Fatalf("watch stdout did not include health and full reports:\n%s", stdout.String())
	}
	logBytes, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if !strings.Contains(string(logBytes), `"suite": "health"`) || !strings.Contains(string(logBytes), `"suite": "all"`) {
		t.Fatalf("watch log did not include health and full reports:\n%s", logBytes)
	}
}

func TestMacBenchWatchLogsKeyErrors(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	var stdout, stderr bytes.Buffer
	code := runMacBenchWatchFull(&stdout, &stderr, macBenchWatchRunOptions{
		gateway:     "http://example.invalid:8080",
		model:       "qwen3.6-27b",
		keyEnv:      "FAK_GATEWAY_KEY",
		keyFile:     t.TempDir(),
		fetchKey:    true,
		sshHost:     "user@node-macos-a.local",
		timeout:     time.Second,
		logPath:     logPath,
		concurrency: 1,
	})
	if code == 0 {
		t.Fatal("expected key error")
	}
	b, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if !strings.Contains(string(b), `"schema": "fak.macbench.watch.event.v1"`) || !strings.Contains(string(b), `"phase": "key"`) {
		t.Fatalf("watch log did not record key error:\n%s", b)
	}
}

func TestMacBenchWatchStatusReportsWaitingWithoutLeakingGateway(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	rep := macbench.Report{
		Schema:      macbench.Schema,
		GeneratedAt: "2026-07-04T07:43:14Z",
		Suite:       macbench.SuiteHealth,
		Gateway:     "<remote-gateway>",
		Model:       "qwen3.6-27b",
		Health:      macbench.Health{Error: `Get "<remote-gateway>/healthz": context deadline exceeded`},
		Errors:      []string{`healthz failed: Get "<remote-gateway>/healthz": context deadline exceeded`},
	}
	if err := writeMacBenchWatchReport(io.Discard, logPath, rep); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"watch-status", "--log", logPath, "--json"})
	if code != 0 {
		t.Fatalf("watch-status code=%d stderr=%s", code, stderr.String())
	}
	var status macBenchWatchStatus
	if err := json.Unmarshal(stdout.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v\n%s", err, stdout.String())
	}
	if status.State != "waiting_for_gateway" || status.Reports != 1 || status.ResultPresent {
		t.Fatalf("status = %+v", status)
	}
	if status.LatestReport == nil || status.LatestReport.Gateway != "<remote-gateway>" {
		t.Fatalf("latest report not surfaced/sanitized: %+v", status.LatestReport)
	}
	if strings.Contains(stdout.String(), "100.64.") || strings.Contains(stdout.String(), "example.invalid") {
		t.Fatalf("status leaked a raw gateway:\n%s", stdout.String())
	}
}

func TestMacBenchWatchStatusReportsCompletedResult(t *testing.T) {
	logPath := t.TempDir() + "/macbench-watch.log"
	resultPath := t.TempDir() + "/macbench-result.json"
	health := macbench.Report{
		Schema:      macbench.Schema,
		GeneratedAt: "2026-07-04T07:43:14Z",
		Suite:       macbench.SuiteHealth,
		Gateway:     "<remote-gateway>",
		Model:       "qwen3.6-27b",
		Health:      macbench.Health{OK: true, Engine: "metal"},
	}
	full := health
	full.GeneratedAt = "2026-07-04T07:45:00Z"
	full.Suite = macbench.SuiteAll
	full.Rows = []macbench.Row{{Name: "decode-256", Kind: "decode-longgen", TokensPerSecond: 12.5}}
	if err := writeMacBenchWatchReport(io.Discard, logPath, health); err != nil {
		t.Fatal(err)
	}
	if err := writeMacBenchResultFile(resultPath, full); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runMacBench(&stdout, &stderr, []string{"watch-status", "--log", logPath, "--result", resultPath, "--json"})
	if code != 0 {
		t.Fatalf("watch-status code=%d stderr=%s", code, stderr.String())
	}
	var status macBenchWatchStatus
	if err := json.Unmarshal(stdout.Bytes(), &status); err != nil {
		t.Fatalf("decode status: %v\n%s", err, stdout.String())
	}
	if status.State != "completed" || !status.ResultPresent || status.Result == nil || status.Result.Suite != macbench.SuiteAll {
		t.Fatalf("status = %+v", status)
	}
	if !strings.Contains(status.NextAction, "record") {
		t.Fatalf("next action should point at recording the result, got %q", status.NextAction)
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
