package main

// manage_admission_budget_test.go — the #10597 witness. `fak manage --gguf …` runs the
// in-process gateway with the admission scheduler's default 8192-token budget, which
// sheds opencode 1.18's ~45k-token floor prompt (429 "request tokens 45358 exceed
// scheduler token budget 8192") with no operator knob — only `fak serve` exposed
// --native-admission-token-budget. manage delegates to the shared guard launcher
// (cmdManageCommand), so these assertions bind THAT surface: the flag is declared with
// serve's authoritative default, the parsed value reaches the gateway's admission policy
// through the same policy seam serve uses, and the startup readback names the launching
// verb. Before the knob existed, the declaration and the constructor did not — this file
// does not compile against that tree, which is the fail-before form for a new seam.

import (
	"context"
	"errors"
	"flag"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// opencodeFloorPromptTokens is opencode 1.18's system+tools floor prompt size from the
// #10597 live run — the request the managed seat shed at the default 8192 budget.
const opencodeFloorPromptTokens = 45358

// TestManageNativeAdmissionTokenBudgetDeclaredOnGuardSurface pins the declaration on the
// launcher manage delegates to, wired through registerGuardAdmissionFlag, and holds the
// default to the gateway's shipping policy (never a literal that can drift from serve's).
func TestManageNativeAdmissionTokenBudgetDeclaredOnGuardSurface(t *testing.T) {
	if !strings.Contains(readEntrypoint(t, "guard.go"), "nativeAdmissionTokenBudget := registerGuardAdmissionFlag(fs)") {
		t.Errorf("guard.go (cmdManageCommand, the launcher `fak manage` delegates to) must register --native-admission-token-budget via registerGuardAdmissionFlag")
	}
	fs := flag.NewFlagSet("manage", flag.ContinueOnError)
	budget := registerGuardAdmissionFlag(fs)
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse empty args: %v", err)
	}
	if got, want := *budget, gateway.DefaultAdmissionPolicy().TokenBudget; got != want {
		t.Fatalf("manage --native-admission-token-budget default = %d, want the shipping policy budget %d (serve's default)", got, want)
	}
}

// TestManageNativeAdmissionTokenBudgetPlumbsToAdmissionPolicy is the behavioral half:
// a value parsed from the manage surface must reach the admission controller the managed
// gateway binds — the same *AdmissionController shape serve installs — proven against the
// issue's real numbers, plus the loud refusal that keeps a typo from arming a zero gate.
func TestManageNativeAdmissionTokenBudgetPlumbsToAdmissionPolicy(t *testing.T) {
	fs := flag.NewFlagSet("manage", flag.ContinueOnError)
	budget := registerGuardAdmissionFlag(fs)
	if err := fs.Parse([]string{"--native-admission-token-budget", "65536"}); err != nil {
		t.Fatal(err)
	}
	controller, message, err := newGuardNativeAdmissionController("manage", *budget)
	if err != nil {
		t.Fatal(err)
	}
	if message.Source != "manage" || message.Kind != "native-admission-token-budget" || message.Level != "info" || message.Text != "native scheduler admission token budget=65536" {
		t.Fatalf("startup readback = %+v", message)
	}
	lease, err := controller.Acquire(context.Background(), gateway.SeqRequest{TraceID: "issue-10597-manage-admit", Tokens: opencodeFloorPromptTokens})
	if err != nil {
		t.Fatalf("opencode's %d-token floor prompt refused under manage --native-admission-token-budget 65536: %v", opencodeFloorPromptTokens, err)
	}
	lease.Release()

	// The other half of the story: at the un-raised default the same floor prompt is
	// exactly the #10597 shed, so the knob is what changed the outcome — not a lax gate.
	defController, _, err := newGuardNativeAdmissionController("manage", gateway.DefaultAdmissionPolicy().TokenBudget)
	if err != nil {
		t.Fatal(err)
	}
	_, err = defController.Acquire(context.Background(), gateway.SeqRequest{TraceID: "issue-10597-manage-shed", Tokens: opencodeFloorPromptTokens})
	var admissionErr *gateway.AdmissionError
	if !errors.As(err, &admissionErr) || admissionErr.Verdict != gateway.VerdictShed {
		t.Fatalf("the un-raised managed gateway must shed the %d-token floor prompt (the #10597 failure the knob escapes), got %v", opencodeFloorPromptTokens, err)
	}

	if _, _, err := newGuardNativeAdmissionController("manage", 0); err == nil || !strings.Contains(err.Error(), "--native-admission-token-budget must be positive") {
		t.Fatalf("non-positive manage budget = %v, want serve's loud refusal", err)
	}
}

// TestManageNativeAdmissionTokenBudgetWiredIntoManagedGateway pins the plumb-through at
// the launcher itself: cmdManageCommand must install the controller on the managed
// gateway (srv.SetAdmissionController) beside the model load profile, or the flag is
// inert on the real seat no matter what the constructor alone does.
func TestManageNativeAdmissionTokenBudgetWiredIntoManagedGateway(t *testing.T) {
	src := readEntrypoint(t, "guard.go")
	for _, want := range []string{
		"newGuardNativeAdmissionController(commandName, *nativeAdmissionTokenBudget)",
		"srv.SetAdmissionController(controller)",
		"srv.AddStartupMessages(admissionNote)",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("guard.go must wire the manage admission knob into the managed gateway: missing %q", want)
		}
	}
}
