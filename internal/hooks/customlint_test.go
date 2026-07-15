package hooks

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func buildCustomLintFixture(t *testing.T) string {
	t.Helper()
	name := "customlintfixture"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	bin := filepath.Join(t.TempDir(), name)
	cmd := exec.Command("go", "build", "-o", bin, "../../cmd/customlintfixture")
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build fixture: %v\n%s", err, out)
	}
	return bin
}

func lintRequest(text string) LintRequest {
	body, _ := json.Marshal(map[string]string{"text": text})
	return LintRequest{Schema: CustomLintSchema, Hook: "pre-tool", Subject: body}
}

func TestCustomLintAllowAndDeny(t *testing.T) {
	bin := buildCustomLintFixture(t)
	spec := CustomLintSpec{Name: "fixture", Command: []string{bin}, OnError: LintErrorDeny}
	if got := RunCustomLint(context.Background(), spec, lintRequest("safe")); !got.Allowed() || got.ErrorKind != "" || got.Duration <= 0 {
		t.Fatalf("allow = %#v", got)
	}
	got := RunCustomLint(context.Background(), spec, lintRequest("please deny-me"))
	if got.Allowed() || got.Response.Disposition != LintDeny || len(got.Response.Findings) != 1 {
		t.Fatalf("deny = %#v", got)
	}
}

func TestCustomLintErrorMode(t *testing.T) {
	bin := buildCustomLintFixture(t)
	cases := []struct {
		name, mode, kind string
		limits           CustomLintLimits
	}{
		{"malformed", "malformed", "malformed_output", CustomLintLimits{}},
		{"overflow", "overflow", "output_overflow", CustomLintLimits{MaxOutputBytes: 64}},
		{"timeout", "sleep", "timeout", CustomLintLimits{Timeout: 20 * time.Millisecond}},
		{"crash", "crash", "crash", CustomLintLimits{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			closed := RunCustomLint(context.Background(), CustomLintSpec{Name: "fixture", Command: []string{bin, "-mode", tc.mode}, OnError: LintErrorDeny, Limits: tc.limits}, lintRequest("safe"))
			if closed.ErrorKind != tc.kind || closed.Allowed() {
				t.Fatalf("closed = %#v", closed)
			}
			open := RunCustomLint(context.Background(), CustomLintSpec{Name: "fixture", Command: []string{bin, "-mode", tc.mode}, OnError: LintErrorAllow, Limits: tc.limits}, lintRequest("safe"))
			if open.ErrorKind != tc.kind || !open.Allowed() || open.Response.Disposition != LintAdvisory {
				t.Fatalf("open = %#v", open)
			}
		})
	}
}

func TestCustomLintScrubsAmbientSecrets(t *testing.T) {
	t.Setenv("FAK_TEST_SECRET", "must-not-pass")
	env := lintEnvironment(nil)
	for _, item := range env {
		if strings.HasPrefix(item, "FAK_TEST_SECRET=") {
			t.Fatalf("secret inherited: %q", item)
		}
	}
	if !hasEnvKey(env, "PATH") {
		t.Fatal("PATH omitted")
	}
	bin := buildCustomLintFixture(t)
	got := RunCustomLint(context.Background(), CustomLintSpec{Name: "fixture", Command: []string{bin}, OnError: LintErrorDeny}, lintRequest("env:FAK_TEST_SECRET"))
	if !got.Allowed() {
		t.Fatalf("ambient secret reached child: %#v", got)
	}
}

func TestCustomLintRejectsInputAndFindingOverflow(t *testing.T) {
	bin := buildCustomLintFixture(t)
	got := RunCustomLint(context.Background(), CustomLintSpec{Name: "fixture", Command: []string{bin}, OnError: LintErrorDeny, Limits: CustomLintLimits{MaxInputBytes: 32}}, lintRequest(strings.Repeat("x", 128)))
	if got.ErrorKind != "input_overflow" || got.Allowed() {
		t.Fatalf("input = %#v", got)
	}
	resp := LintResponse{Schema: CustomLintSchema, Disposition: LintAllow, Findings: make([]LintFinding, 2)}
	if err := validateLintResponse(resp, 1); err == nil || !strings.Contains(err.Error(), "findings") {
		t.Fatalf("error = %v", err)
	}
}
