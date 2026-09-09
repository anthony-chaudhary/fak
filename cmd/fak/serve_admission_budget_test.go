package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestServeAdmissionTokenBudgetDefault(t *testing.T) {
	_, sf := newServeFlagSet()
	policy, err := serveNativeAdmissionPolicy(sf)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := policy.TokenBudget, 8192; got != want {
		t.Fatalf("default native admission token budget = %d, want %d", got, want)
	}
	if got, want := policy, gateway.DefaultAdmissionPolicy(); got != want {
		t.Fatalf("default native admission policy = %+v, want unchanged shipping policy %+v", got, want)
	}
}

func TestServeAdmissionTokenBudgetExplicitEnvelope(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--native-admission-token-budget", "65536"}); err != nil {
		t.Fatal(err)
	}
	controller, message, err := newServeNativeAdmissionController(sf)
	if err != nil {
		t.Fatal(err)
	}
	if message.Source != "serve" || message.Kind != "native-admission-token-budget" || message.Level != "info" || message.Text != "native scheduler admission token budget=65536" {
		t.Fatalf("startup readback = %+v", message)
	}

	lease, err := controller.Acquire(context.Background(), gateway.SeqRequest{TraceID: "issue-9079-admit", Tokens: 22064})
	if err != nil {
		t.Fatalf("22,064-token production footprint refused under explicit 65,536 budget: %v", err)
	}
	if lease == nil {
		t.Fatal("22,064-token production footprint returned no admission lease")
	}
	lease.Release()

	_, err = controller.Acquire(context.Background(), gateway.SeqRequest{TraceID: "issue-9079-reject", Tokens: 65537})
	var admissionErr *gateway.AdmissionError
	if !errors.As(err, &admissionErr) {
		t.Fatalf("above-cap request error = %T %v, want typed *gateway.AdmissionError", err, err)
	}
	if (admissionErr.Verdict != gateway.VerdictShed && admissionErr.Verdict != gateway.VerdictRefused) || admissionErr.Reason != "request tokens 65537 exceed scheduler token budget 65536" {
		t.Fatalf("above-cap typed rejection = %+v", admissionErr)
	}
}

func TestServeAdmissionTokenBudgetRejectsNonPositiveAtStartup(t *testing.T) {
	if value := os.Getenv("FAK_TEST_NATIVE_ADMISSION_TOKEN_BUDGET"); value != "" {
		cmdServe([]string{"--native-admission-token-budget", value, "--print-effective-config"})
		return
	}

	for _, value := range []string{"0", "-1"} {
		t.Run(value, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestServeAdmissionTokenBudgetRejectsNonPositiveAtStartup$")
			cmd.Env = append(os.Environ(), "FAK_TEST_NATIVE_ADMISSION_TOKEN_BUDGET="+value)
			output, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
				t.Fatalf("startup with %s exit = %v; output:\n%s", value, err, output)
			}
			want := fmt.Sprintf("--native-admission-token-budget must be positive (got %s)", value)
			if !strings.Contains(string(output), want) {
				t.Fatalf("startup with %s omitted refusal %q:\n%s", value, want, output)
			}
		})
	}
}

func TestServeEffectiveAdmissionTokenBudget(t *testing.T) {
	_, sf := newServeFlagSet()
	if got := sf.effectiveAdmissionTokenBudget(); got != 8192 {
		t.Fatalf("effective admission token budget default = %d, want shipping admission default 8192", got)
	}

	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--native-admission-token-budget", "4096"}); err != nil {
		t.Fatal(err)
	}
	if got := sf.effectiveAdmissionTokenBudget(); got != 4096 {
		t.Fatalf("effective admission token budget = %d, want 4096", got)
	}

	fs2, sf2 := newServeFlagSet()
	if err := fs2.Parse([]string{"--native-admission-token-budget", "4096", "--context-budget-tokens", "2048"}); err != nil {
		t.Fatal(err)
	}
	if got := sf2.effectiveAdmissionTokenBudget(); got != 2048 {
		t.Fatalf("effective admission token budget = %d, want 2048", got)
	}
}

func TestServeAdmissionMaxTotalTokensOverBudgetFailsFast(t *testing.T) {
	if value := os.Getenv("FAK_TEST_MAX_TOTAL_TOKENS"); value != "" {
		cmdServe([]string{"--max-total-tokens", value, "--print-effective-config"})
		return
	}

	value := "10000"
	cmd := exec.Command(os.Args[0], "-test.run=^TestServeAdmissionMaxTotalTokensOverBudgetFailsFast$")
	cmd.Env = append(os.Environ(), "FAK_TEST_MAX_TOTAL_TOKENS="+value)
	output, err := cmd.CombinedOutput()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
		t.Fatalf("startup with max-total-tokens=%s exit = %v; output:\n%s", value, err, output)
	}
	want := "max_total_tokens (10000) exceeds admission token budget (8192); decrease --max-batch-prefill-tokens or --max-total-tokens"
	if !strings.Contains(string(output), want) {
		t.Fatalf("startup with max-total-tokens=%s omitted refusal %q:\n%s", value, want, output)
	}
}

func TestServeAdmissionTokenBudgetInheritsContextWindow(t *testing.T) {
	// 1. When --native-admission-token-budget is omitted, TokenBudget inherits from --context-budget-tokens
	fs1, sf1 := newServeFlagSet()
	if err := fs1.Parse([]string{"--context-budget-tokens", "32768"}); err != nil {
		t.Fatal(err)
	}
	policy1, err := serveNativeAdmissionPolicy(sf1)
	if err != nil {
		t.Fatal(err)
	}
	if policy1.TokenBudget != 32768 {
		t.Fatalf("TokenBudget = %d, want 32768 inherited from --context-budget-tokens", policy1.TokenBudget)
	}
	if policy1.TokenBudgetProvenance != "context" {
		t.Fatalf("TokenBudgetProvenance = %q, want \"context\"", policy1.TokenBudgetProvenance)
	}

	// 2. When --native-admission-token-budget is omitted, TokenBudget inherits from --ctx alias
	fs2, sf2 := newServeFlagSet()
	if err := fs2.Parse([]string{"--ctx", "65536"}); err != nil {
		t.Fatal(err)
	}
	policy2, err := serveNativeAdmissionPolicy(sf2)
	if err != nil {
		t.Fatal(err)
	}
	if policy2.TokenBudget != 65536 {
		t.Fatalf("TokenBudget = %d, want 65536 inherited from --ctx", policy2.TokenBudget)
	}
	if policy2.TokenBudgetProvenance != "context" {
		t.Fatalf("TokenBudgetProvenance = %q, want \"context\"", policy2.TokenBudgetProvenance)
	}

	// 3. Explicit --native-admission-token-budget declarations continue to take strict precedence
	fs3, sf3 := newServeFlagSet()
	if err := fs3.Parse([]string{"--ctx", "65536", "--native-admission-token-budget", "16384"}); err != nil {
		t.Fatal(err)
	}
	policy3, err := serveNativeAdmissionPolicy(sf3)
	if err != nil {
		t.Fatal(err)
	}
	if policy3.TokenBudget != 16384 {
		t.Fatalf("TokenBudget = %d, want 16384 from explicit --native-admission-token-budget", policy3.TokenBudget)
	}
	if policy3.TokenBudgetProvenance != "explicit" {
		t.Fatalf("TokenBudgetProvenance = %q, want \"explicit\"", policy3.TokenBudgetProvenance)
	}

	// 4. Default when neither is provided remains 8192
	fs4, sf4 := newServeFlagSet()
	if err := fs4.Parse([]string{}); err != nil {
		t.Fatal(err)
	}
	policy4, err := serveNativeAdmissionPolicy(sf4)
	if err != nil {
		t.Fatal(err)
	}
	if policy4.TokenBudget != 8192 {
		t.Fatalf("TokenBudget = %d, want default 8192", policy4.TokenBudget)
	}
	if policy4.TokenBudgetProvenance != "default" {
		t.Fatalf("TokenBudgetProvenance = %q, want \"default\"", policy4.TokenBudgetProvenance)
	}
}
