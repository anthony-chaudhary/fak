package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
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
	if got := sf2.effectiveAdmissionTokenBudget(); got != 4096 {
		t.Fatalf("effective admission token budget = %d, want independent scheduler cap 4096 (session context budget must not override it)", got)
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

func TestServeAdmissionTokenBudgetMaterializesResolvedModelWindow(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--ctx", "32768"}); err != nil {
		t.Fatal(err)
	}
	if *sf.contextBudgetTokens != 32768 || *sf.nativeContextTokens != 0 {
		t.Fatalf("--ctx crossed token domains: session=%d native-window=%d, want 32768/auto", *sf.contextBudgetTokens, *sf.nativeContextTokens)
	}
	policy, err := serveNativeAdmissionPolicy(sf)
	if err != nil {
		t.Fatal(err)
	}
	if policy.TokenBudget != 8192 || policy.TokenBudgetProvenance != "default" {
		t.Fatalf("pre-resolution admission policy = %+v, want unchanged default independent of session budget", policy)
	}

	*sf.nativeAdmissionTokenBudget = effectiveNativeAdmissionTokenBudget(sf, false, 65536)
	sf.nativeAdmissionProvenance = "context"
	policy, err = serveNativeAdmissionPolicy(sf)
	if err != nil {
		t.Fatal(err)
	}
	if policy.TokenBudget != 65536 || policy.TokenBudgetProvenance != "context" {
		t.Fatalf("materialized admission policy = %+v, want resolved 65536-token model window", policy)
	}

	fs, sf = newServeFlagSet()
	if err := fs.Parse([]string{
		"--ctx", "32768",
		"--native-context-tokens", "65536",
		"--native-admission-token-budget", "16384",
	}); err != nil {
		t.Fatal(err)
	}
	*sf.nativeAdmissionTokenBudget = effectiveNativeAdmissionTokenBudget(sf, true, 65536)
	sf.nativeAdmissionProvenance = "explicit"
	policy, err = serveNativeAdmissionPolicy(sf)
	if err != nil {
		t.Fatal(err)
	}
	if policy.TokenBudget != 16384 || policy.TokenBudgetProvenance != "explicit" {
		t.Fatalf("explicit admission policy = %+v, want independent 16384-token scheduler cap", policy)
	}
	if *sf.contextBudgetTokens != 32768 || *sf.nativeContextTokens != 65536 {
		t.Fatalf("explicit token domains crossed: session=%d native-window=%d, want 32768/65536", *sf.contextBudgetTokens, *sf.nativeContextTokens)
	}

	_, sf = newServeFlagSet()
	if got := effectiveNativeAdmissionTokenBudget(sf, false, 0); got != 8192 {
		t.Fatalf("unresolved model admission budget = %d, want default 8192", got)
	}

	t.Run("local max-total validates against resolved window", func(t *testing.T) {
		for _, tc := range []struct {
			name      string
			maxTotal  string
			wantError string
		}{
			{name: "within resolved cap", maxTotal: "32768"},
			{name: "above resolved cap", maxTotal: "65537", wantError: "max_total_tokens (65537) exceeds admission token budget (65536)"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				fs, sf := newServeFlagSet()
				if err := fs.Parse([]string{"--gguf", "model.gguf", "--max-total-tokens", tc.maxTotal}); err != nil {
					t.Fatal(err)
				}
				*sf.nativeAdmissionTokenBudget = effectiveNativeAdmissionTokenBudget(sf, false, 65536)
				err := validateServeMaxTotalTokens(sf, *sf.nativeAdmissionTokenBudget)
				if tc.wantError == "" && err != nil {
					t.Fatalf("max-total %s rejected after 65536-token model resolution: %v", tc.maxTotal, err)
				}
				if tc.wantError != "" && (err == nil || !strings.Contains(err.Error(), tc.wantError)) {
					t.Fatalf("max-total %s error = %v, want %q", tc.maxTotal, err, tc.wantError)
				}
			})
		}
	})

	t.Run("explicit scheduler cap validates immediately", func(t *testing.T) {
		fs, sf := newServeFlagSet()
		if err := fs.Parse([]string{
			"--gguf", "model.gguf",
			"--native-admission-token-budget", "16384",
			"--max-total-tokens", "16385",
		}); err != nil {
			t.Fatal(err)
		}
		policy, err := serveNativeAdmissionPolicy(sf)
		if err != nil {
			t.Fatal(err)
		}
		err = validateServeMaxTotalTokens(sf, policy.TokenBudget)
		want := "max_total_tokens (16385) exceeds admission token budget (16384)"
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("explicit scheduler envelope error = %v, want %q", err, want)
		}
	})
}

// TestServeAdmissionTokenBudgetInheritsContextWindow pins the three derivation
// outcomes at the materialization seam: an omitted budget follows the resolved
// model context window, an explicit budget strictly wins over it, and a window
// that never resolves falls back to the shipping default.
func TestServeAdmissionTokenBudgetInheritsContextWindow(t *testing.T) {
	tests := []struct {
		name           string
		flags          []string
		explicit       bool
		resolvedWindow int
		provenance     string
		wantBudget     int
		wantProvenance string
	}{
		{name: "omitted budget follows large resolved model window", explicit: false, resolvedWindow: 65536, provenance: "context", wantBudget: 65536, wantProvenance: "context"},
		{name: "explicit budget wins over large resolved model window", flags: []string{"--native-admission-token-budget", "16384"}, explicit: true, resolvedWindow: 65536, provenance: "explicit", wantBudget: 16384, wantProvenance: "explicit"},
		{name: "unresolved window keeps shipping default", explicit: false, resolvedWindow: 0, provenance: "default", wantBudget: 8192, wantProvenance: "default"},
		{name: "explicit budget kept with unresolved window", flags: []string{"--native-admission-token-budget", "16384"}, explicit: true, resolvedWindow: 0, provenance: "explicit", wantBudget: 16384, wantProvenance: "explicit"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fs, sf := newServeFlagSet()
			if len(tc.flags) > 0 {
				if err := fs.Parse(tc.flags); err != nil {
					t.Fatal(err)
				}
			}
			if got := effectiveNativeAdmissionTokenBudget(sf, tc.explicit, tc.resolvedWindow); got != tc.wantBudget {
				t.Fatalf("effectiveNativeAdmissionTokenBudget(explicit=%v, window=%d) = %d, want %d", tc.explicit, tc.resolvedWindow, got, tc.wantBudget)
			}
			*sf.nativeAdmissionTokenBudget = tc.wantBudget
			sf.nativeAdmissionProvenance = tc.provenance
			policy, err := serveNativeAdmissionPolicy(sf)
			if err != nil {
				t.Fatal(err)
			}
			if policy.TokenBudget != tc.wantBudget || policy.TokenBudgetProvenance != tc.wantProvenance {
				t.Fatalf("admission policy = %+v, want budget %d provenance %q", policy, tc.wantBudget, tc.wantProvenance)
			}
		})
	}
}

// TestServeAdmissionTokenBudgetWarningBelowContext pins the operator diagnostic
// for an explicit --native-admission-token-budget below the resolved model
// window: the warning fires with the exact diagnostic text while the explicit
// budget stays strictly in charge of the admission policy.
func TestServeAdmissionTokenBudgetWarningBelowContext(t *testing.T) {
	tests := []struct {
		name       string
		budget     int
		window     int
		wantWarned bool
		wantText   string
	}{
		{name: "explicit budget below context warns and stays in charge", budget: 16384, window: 65536, wantWarned: true, wantText: "WARNING: explicit --native-admission-token-budget (16384) is smaller than resolved native model context window (65536)"},
		{name: "budget equal to window is silent", budget: 65536, window: 65536},
		{name: "budget above window is silent", budget: 131072, window: 65536},
		{name: "non-positive budget is silent", budget: 0, window: 65536},
		{name: "unresolved window is silent", budget: 16384, window: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var stderr bytes.Buffer
			warned := serveExplicitBudgetWarning(&stderr, tc.budget, tc.window)
			if warned != tc.wantWarned {
				t.Fatalf("serveExplicitBudgetWarning(%d, %d) warned = %v, want %v (stderr = %q)", tc.budget, tc.window, warned, tc.wantWarned, stderr.String())
			}
			if tc.wantWarned && !strings.Contains(stderr.String(), tc.wantText) {
				t.Fatalf("warning = %q, want it to contain %q", stderr.String(), tc.wantText)
			}
			if !tc.wantWarned && stderr.String() != "" {
				t.Fatalf("warning = %q, want silence", stderr.String())
			}
			if tc.wantWarned {
				fs, sf := newServeFlagSet()
				if err := fs.Parse([]string{"--native-admission-token-budget", strconv.Itoa(tc.budget)}); err != nil {
					t.Fatal(err)
				}
				policy, err := serveNativeAdmissionPolicy(sf)
				if err != nil {
					t.Fatal(err)
				}
				if policy.TokenBudget != tc.budget || policy.TokenBudgetProvenance != "explicit" {
					t.Fatalf("admission policy after warning = %+v, want explicit budget %d still used", policy, tc.budget)
				}
			}
		})
	}
}
