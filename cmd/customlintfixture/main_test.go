package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// binPath is the freshly compiled customlintfixture binary shared by every test.
// TestMain builds it once so the tests witness the real runtime behaviour of the
// shipped binary rather than an in-process copy of an internal helper.
var binPath string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "customlintfixture-bin-")
	if err != nil {
		fmt.Fprintln(os.Stderr, "mkdtemp:", err)
		os.Exit(1)
	}
	defer os.RemoveAll(dir)
	binPath = filepath.Join(dir, "customlintfixture")
	if os.PathSeparator == '\\' {
		binPath += ".exe"
	}
	build := exec.Command("go", "build", "-o", binPath, ".")
	build.Stdout, build.Stderr = os.Stdout, os.Stderr
	if err := build.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "go build:", err)
		os.Exit(1)
	}
	os.Exit(m.Run())
}

// runBin drives the compiled binary as a subprocess and returns its exit code,
// stdout and stderr. A harness error is reported as -1 with the error text on
// stderr so a broken build cannot be mistaken for a passing assertion.
func runBin(t *testing.T, stdin string, extraEnv []string, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command(binPath, args...)
	cmd.Stdin = strings.NewReader(stdin)
	cmd.Env = append(os.Environ(), extraEnv...)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	err := cmd.Run()
	code := 0
	if err != nil {
		ee, ok := err.(*exec.ExitError)
		if !ok {
			t.Fatalf("run %v: %v (stderr=%s)", args, err, errb.String())
		}
		code = ee.ExitCode()
	}
	return code, out.String(), errb.String()
}

func validRequest(text string) string {
	subject, _ := json.Marshal(map[string]string{"text": text})
	req, _ := json.Marshal(map[string]any{
		"schema":  schema,
		"hook":    "pre-tool",
		"subject": json.RawMessage(subject),
	})
	return string(req)
}

func decodeResponse(t *testing.T, stdout string) response {
	t.Helper()
	var resp response
	if err := json.Unmarshal([]byte(stdout), &resp); err != nil {
		t.Fatalf("stdout is not a valid response: %v\nstdout=%q", err, stdout)
	}
	return resp
}

func TestEchoAllowsCleanRequest(t *testing.T) {
	code, out, errb := runBin(t, validRequest("hello world"), nil, "--mode=echo")
	if code != 0 {
		t.Fatalf("exit=%d want 0; stderr=%q", code, errb)
	}
	resp := decodeResponse(t, out)
	if resp.Schema != schema {
		t.Errorf("schema=%q want %q", resp.Schema, schema)
	}
	if resp.Disposition != "allow" {
		t.Errorf("disposition=%q want allow", resp.Disposition)
	}
	if len(resp.Findings) != 0 {
		t.Errorf("findings=%v want none", resp.Findings)
	}
}

func TestDenyMarkerIsReported(t *testing.T) {
	code, out, errb := runBin(t, validRequest("please deny-me now"), nil, "--mode=echo")
	if code != 0 {
		t.Fatalf("exit=%d want 0; stderr=%q", code, errb)
	}
	resp := decodeResponse(t, out)
	if resp.Disposition != "deny" {
		t.Fatalf("disposition=%q want deny", resp.Disposition)
	}
	if len(resp.Findings) != 1 || resp.Findings[0].ID != "fixture.deny" {
		t.Fatalf("findings=%v want one fixture.deny", resp.Findings)
	}
}

func TestEnvProbeSeesInheritedEnvironment(t *testing.T) {
	code, out, _ := runBin(t, validRequest("env:FIXTURE_PROBE_VAR"), []string{"FIXTURE_PROBE_VAR=1"}, "--mode=echo")
	if code != 0 {
		t.Fatalf("exit=%d want 0", code)
	}
	resp := decodeResponse(t, out)
	if resp.Disposition != "deny" {
		t.Fatalf("disposition=%q want deny", resp.Disposition)
	}
	if len(resp.Findings) != 1 || resp.Findings[0].ID != "fixture.env-visible" {
		t.Fatalf("findings=%v want one fixture.env-visible", resp.Findings)
	}
}

func TestUnknownModeExitsTwo(t *testing.T) {
	code, out, errb := runBin(t, "", nil, "--mode=bogus")
	if code != 2 {
		t.Fatalf("exit=%d want 2; stderr=%q", code, errb)
	}
	if out != "" {
		t.Errorf("stdout=%q want empty", out)
	}
	if !strings.Contains(errb, `unknown mode "bogus"`) {
		t.Errorf("stderr=%q want unknown-mode diagnostic", errb)
	}
}

func TestMalformedRequestExitsTwo(t *testing.T) {
	code, out, errb := runBin(t, "{not json", nil, "--mode=echo")
	if code != 2 {
		t.Fatalf("exit=%d want 2; stderr=%q", code, errb)
	}
	if out != "" {
		t.Errorf("stdout=%q want empty", out)
	}
	if strings.TrimSpace(errb) == "" {
		t.Error("stderr is empty; decode failure was not surfaced")
	}
}

func TestBadSchemaExitsTwo(t *testing.T) {
	req := `{"schema":"wrong/9","hook":"pre-tool","subject":{"text":"hi"}}`
	code, _, errb := runBin(t, req, nil, "--mode=echo")
	if code != 2 {
		t.Fatalf("exit=%d want 2; stderr=%q", code, errb)
	}
	if !strings.Contains(errb, "bad request") {
		t.Errorf("stderr=%q want bad request", errb)
	}
}

func TestCrashModeExitCode(t *testing.T) {
	code, _, errb := runBin(t, "", nil, "--mode=crash")
	if code != 7 {
		t.Fatalf("exit=%d want 7; stderr=%q", code, errb)
	}
	if !strings.Contains(errb, "fixture crash") {
		t.Errorf("stderr=%q want fixture crash", errb)
	}
}

func TestBadFlagExitsTwo(t *testing.T) {
	code, _, _ := runBin(t, "", nil, "--not-a-flag")
	if code != 2 {
		t.Fatalf("exit=%d want 2", code)
	}
}
