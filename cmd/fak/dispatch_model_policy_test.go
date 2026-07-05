package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func TestResolveWorkerModelPolicy_ClaudeDefaultStaysBlank(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}
	got := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false)
	if got.pinned() {
		t.Fatalf("default claude tick must keep the seat default (blank model), got %q", got.Model)
	}
	if got.Source != modelSourceSeatDefault {
		t.Fatalf("source = %q, want %q", got.Source, modelSourceSeatDefault)
	}
	// The full fallback chain is still available for --fallback-model / Layer-2.
	if want := []string{"claude-opus-4-8", "claude-sonnet-5"}; !reflect.DeepEqual(got.Chain, want) {
		t.Fatalf("chain = %v, want %v", got.Chain, want)
	}
}

func TestResolveWorkerModelPolicy_ExplicitPinWins(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	pol := fleetaccounts.DefaultPolicy()
	pol.LaneModels["docs"] = "claude-haiku-4-5-20251001"
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}
	// Explicit pin beats the lane pin AND the benchmark gate.
	got := resolveWorkerModelPolicy("claude", "docs", "claude-opus-4-8", acct, pol, true)
	if got.Model != "claude-opus-4-8" || got.Source != modelSourceExplicit {
		t.Fatalf("explicit pin: got model=%q source=%q", got.Model, got.Source)
	}
	// The primary is dropped from its own downgrade chain.
	if want := []string{"claude-sonnet-5"}; !reflect.DeepEqual(got.Chain, want) {
		t.Fatalf("chain = %v, want %v", got.Chain, want)
	}
}

func TestResolveWorkerModelPolicy_LanePinUnblanks(t *testing.T) {
	pol := fleetaccounts.DefaultPolicy()
	pol.LaneModels["gateway"] = "claude-sonnet-5"
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}
	got := resolveWorkerModelPolicy("claude", "gateway", "", acct, pol, false)
	if got.Model != "claude-sonnet-5" || got.Source != modelSourceLane {
		t.Fatalf("lane pin: got model=%q source=%q", got.Model, got.Source)
	}
}

func TestResolveWorkerModelPolicy_BenchmarkGatePinsProfileThenDefault(t *testing.T) {
	pol := fleetaccounts.DefaultPolicy()
	// With an account model, the benchmark gate pins THAT (a known model to measure).
	withModel := resolveWorkerModelPolicy("claude", "docs", "", dispatchtick.Account{Model: "opus"}, pol, true)
	if withModel.Model != "opus" || withModel.Source != modelSourceProfile {
		t.Fatalf("bench gate w/ account model: got model=%q source=%q", withModel.Model, withModel.Source)
	}
	// Without one, it falls back to the fleet default.
	noModel := resolveWorkerModelPolicy("claude", "docs", "", dispatchtick.Account{}, pol, true)
	if noModel.Model != defaultLaunchModel || noModel.Source != modelSourceDefault {
		t.Fatalf("bench gate w/o account model: got model=%q source=%q", noModel.Model, noModel.Source)
	}
}

func TestResolveWorkerModelPolicy_OpencodePinsAccountModel(t *testing.T) {
	acct := dispatchtick.Account{Model: "zai-coding-plan/glm-5.2"}
	got := resolveWorkerModelPolicy("opencode", "docs", "", acct, fleetaccounts.DefaultPolicy(), false)
	if got.Model != "zai-coding-plan/glm-5.2" || got.Source != modelSourceAccount {
		t.Fatalf("opencode: got model=%q source=%q", got.Model, got.Source)
	}
}
