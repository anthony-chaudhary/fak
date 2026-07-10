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
	got := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, "")
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
	got := resolveWorkerModelPolicy("claude", "docs", "claude-opus-4-8", acct, pol, true, nil, "")
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
	got := resolveWorkerModelPolicy("claude", "gateway", "", acct, pol, false, nil, "")
	if got.Model != "claude-sonnet-5" || got.Source != modelSourceLane {
		t.Fatalf("lane pin: got model=%q source=%q", got.Model, got.Source)
	}
}

func TestResolveWorkerModelPolicy_BenchmarkGatePinsProfileThenDefault(t *testing.T) {
	pol := fleetaccounts.DefaultPolicy()
	// With an account model, the benchmark gate pins THAT (a known model to measure).
	withModel := resolveWorkerModelPolicy("claude", "docs", "", dispatchtick.Account{Model: "opus"}, pol, true, nil, "")
	if withModel.Model != "opus" || withModel.Source != modelSourceProfile {
		t.Fatalf("bench gate w/ account model: got model=%q source=%q", withModel.Model, withModel.Source)
	}
	// Without one, it falls back to the fleet default.
	noModel := resolveWorkerModelPolicy("claude", "docs", "", dispatchtick.Account{}, pol, true, nil, "")
	if noModel.Model != defaultLaunchModel || noModel.Source != modelSourceDefault {
		t.Fatalf("bench gate w/o account model: got model=%q source=%q", noModel.Model, noModel.Source)
	}
}

func TestResolveWorkerModelPolicy_TierProfileUnblanksBelowSeatDefault(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}
	// A hard-tier profile un-blanks the model AND carries ultracode, where without a profile
	// the same claude tick would stay on the seat default.
	prof := dispatchtick.ProfileOpusUltracode
	got := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, &prof, "")
	if got.Source != modelSourceTier {
		t.Fatalf("source = %q, want %q", got.Source, modelSourceTier)
	}
	if got.Model != prof.Model || !got.Ultracode || got.Effort != "" {
		t.Fatalf("tier profile: got model=%q ultracode=%v effort=%q", got.Model, got.Ultracode, got.Effort)
	}
	// The profile's own model is dropped from the downgrade chain it leaves behind.
	for _, m := range got.Chain {
		if m == prof.Model {
			t.Fatalf("tier chain %v must not re-offer the pinned model %q", got.Chain, prof.Model)
		}
	}
}

func TestResolveWorkerModelPolicy_ExplicitAndGatesBeatTier(t *testing.T) {
	prof := dispatchtick.ProfileFableUltracode
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}

	// Explicit pin wins over the tier profile, and drops the tier's effort/ultracode.
	pol := fleetaccounts.DefaultPolicy()
	exp := resolveWorkerModelPolicy("claude", "docs", "claude-sonnet-5", acct, pol, false, &prof, "")
	if exp.Model != "claude-sonnet-5" || exp.Source != modelSourceExplicit || exp.Ultracode {
		t.Fatalf("explicit over tier: got model=%q source=%q ultracode=%v", exp.Model, exp.Source, exp.Ultracode)
	}

	// Lane pin also beats tier.
	pol.LaneModels["docs"] = "claude-haiku-4-5-20251001"
	lane := resolveWorkerModelPolicy("claude", "docs", "", acct, pol, false, &prof, "")
	if lane.Source != modelSourceLane || lane.Ultracode {
		t.Fatalf("lane over tier: got source=%q ultracode=%v", lane.Source, lane.Ultracode)
	}

	// The benchmark gate beats tier too.
	bench := resolveWorkerModelPolicy("claude", "gateway", "", acct, fleetaccounts.DefaultPolicy(), true, &prof, "")
	if bench.Source != modelSourceProfile || bench.Ultracode {
		t.Fatalf("bench over tier: got source=%q ultracode=%v", bench.Source, bench.Ultracode)
	}
}

func TestResolveWorkerModelPolicy_OpencodePinsAccountModel(t *testing.T) {
	acct := dispatchtick.Account{Model: "zai-coding-plan/glm-5.2"}
	got := resolveWorkerModelPolicy("opencode", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, "")
	if got.Model != "zai-coding-plan/glm-5.2" || got.Source != modelSourceAccount {
		t.Fatalf("opencode: got model=%q source=%q", got.Model, got.Source)
	}
	// opencode short-circuits to the account model, so a work-class default is ignored.
	wc := resolveWorkerModelPolicy("opencode", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, dispatchtick.WorkerModelFable)
	if wc.Model != "zai-coding-plan/glm-5.2" || wc.Source != modelSourceAccount {
		t.Fatalf("opencode work-class ignored: got model=%q source=%q", wc.Model, wc.Source)
	}
}

func TestResolveWorkerModelPolicy_WorkClassUnblanksAboveSeatDefault(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}
	// No operator pin, no bench gate, no tier profile: the work-class default un-blanks the
	// model where the same claude tick would otherwise stay on the seat default.
	got := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, dispatchtick.WorkerModelFable)
	if got.Model != dispatchtick.WorkerModelFable || got.Source != modelSourceWorkClass {
		t.Fatalf("work-class default: got model=%q source=%q", got.Model, got.Source)
	}
	// It carries no effort/ultracode (only the tier profile does) and drops its own model
	// from the downgrade chain it leaves behind.
	if got.Ultracode || got.Effort != "" {
		t.Fatalf("work-class default must not set effort/ultracode: %+v", got)
	}
	for _, m := range got.Chain {
		if m == dispatchtick.WorkerModelFable {
			t.Fatalf("work-class chain %v must not re-offer the pinned model", got.Chain)
		}
	}
	// Empty work-class + nothing else keeps the seat default (byte-identical to before).
	seat := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, "")
	if seat.pinned() || seat.Source != modelSourceSeatDefault {
		t.Fatalf("empty work-class must keep seat default, got model=%q source=%q", seat.Model, seat.Source)
	}
}

func TestResolveWorkerModelPolicy_TierAndPinsBeatWorkClass(t *testing.T) {
	acct := dispatchtick.Account{Tag: "gem8", Model: "opus"}

	// A per-issue tier profile beats the work-class default: a hard garden issue tagged
	// tier/T0 still escalates to opus+ultracode rather than the class-default fable.
	prof := dispatchtick.ProfileOpusUltracode
	tier := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, &prof, dispatchtick.WorkerModelFable)
	if tier.Model != prof.Model || tier.Source != modelSourceTier || !tier.Ultracode {
		t.Fatalf("tier over work-class: got model=%q source=%q ultracode=%v", tier.Model, tier.Source, tier.Ultracode)
	}

	// An explicit operator pin beats the work-class default too.
	exp := resolveWorkerModelPolicy("claude", "docs", "claude-opus-4-8", acct, fleetaccounts.DefaultPolicy(), false, nil, dispatchtick.WorkerModelFable)
	if exp.Model != "claude-opus-4-8" || exp.Source != modelSourceExplicit {
		t.Fatalf("explicit over work-class: got model=%q source=%q", exp.Model, exp.Source)
	}

	// A per-lane pin beats it.
	pol := fleetaccounts.DefaultPolicy()
	pol.LaneModels["docs"] = "claude-sonnet-5"
	lane := resolveWorkerModelPolicy("claude", "docs", "", acct, pol, false, nil, dispatchtick.WorkerModelFable)
	if lane.Model != "claude-sonnet-5" || lane.Source != modelSourceLane {
		t.Fatalf("lane over work-class: got model=%q source=%q", lane.Model, lane.Source)
	}

	// The benchmark gate beats it.
	bench := resolveWorkerModelPolicy("claude", "gateway", "", acct, fleetaccounts.DefaultPolicy(), true, nil, dispatchtick.WorkerModelFable)
	if bench.Source != modelSourceProfile {
		t.Fatalf("bench over work-class: got source=%q", bench.Source)
	}
}
