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
	if admissionErr.Verdict != gateway.VerdictShed || admissionErr.Reason != "request tokens 65537 exceed scheduler token budget 65536" {
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
