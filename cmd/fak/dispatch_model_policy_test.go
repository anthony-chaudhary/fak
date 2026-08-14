package main

import (
	"io"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func TestDispatchEngineForWorkClass(t *testing.T) {
	tests := []struct {
		name, current, workClass, explicit, want string
	}{
		{name: "gardening defaults codex", current: "claude", workClass: "gardening", want: "codex"},
		{name: "grind defaults codex", current: "claude", workClass: "mechanical_grind", want: "codex"},
		{name: "engineering defaults claude", current: "codex", workClass: "engineering", want: "claude"},
		{name: "rigor defaults claude", current: "codex", workClass: "security_audit", want: "claude"},
		{name: "unknown preserves current", current: "opencode", workClass: "", want: "opencode"},
		{name: "operator pin wins", current: "claude", workClass: "gardening", explicit: "opencode", want: "opencode"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dispatchEngineForWorkClass(tt.current, tt.workClass, tt.explicit); got != tt.want {
				t.Fatalf("dispatchEngineForWorkClass(%q, %q, %q) = %q, want %q", tt.current, tt.workClass, tt.explicit, got, tt.want)
			}
		})
	}
}

func TestParseDispatchTickFlagsRoutesEngineByWorkClass(t *testing.T) {
	t.Run("untagged default is byte-identical", func(t *testing.T) {
		opts, ok, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir()})
		if ok || code != 0 || opts.Backend != "claude" {
			t.Fatalf("ok=%v code=%d backend=%q, want false/0/claude", ok, code, opts.Backend)
		}
	})
	t.Run("grind selects codex", func(t *testing.T) {
		opts, ok, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir(), "--work-kind", "gardening"})
		if ok || code != 0 || opts.Backend != "codex" {
			t.Fatalf("ok=%v code=%d backend=%q, want false/0/codex", ok, code, opts.Backend)
		}
	})
	t.Run("rigor selects claude", func(t *testing.T) {
		opts, ok, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir(), "--work-kind", "security_audit"})
		if ok || code != 0 || opts.Backend != "claude" {
			t.Fatalf("ok=%v code=%d backend=%q, want false/0/claude", ok, code, opts.Backend)
		}
	})
	t.Run("explicit backend wins", func(t *testing.T) {
		opts, ok, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir(), "--backend", "opencode", "--work-kind", "gardening"})
		if ok || code != 0 || opts.Backend != "opencode" || !opts.BackendExplicit {
			t.Fatalf("ok=%v code=%d backend=%q explicit=%v, want false/0/opencode/true", ok, code, opts.Backend, opts.BackendExplicit)
		}
	})
}
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

// --- #3521: the preventive placement gate --------------------------------------------

// TestApplyPlacementGate_GoldenRefusesFableOnChurning reproduces the witnessed placement
// failure: the tier table's BucketUltra profile (fable + ultracode) landing on a hard,
// CHURNING issue. The gate must re-route it to opus BEFORE launch, keep the ultracode
// reasoning posture, stamp the typed reason, and never re-offer fable on the chain.
func TestApplyPlacementGate_GoldenRefusesFableOnChurning(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-fable-5,claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8"}
	prof := dispatchtick.ProfileFableUltracode // exactly what BucketUltra resolves to
	placed := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, &prof, "")
	if placed.Model != dispatchtick.WorkerModelFable || placed.Source != modelSourceTier {
		t.Fatalf("precondition: tier placed model=%q source=%q", placed.Model, placed.Source)
	}

	gated, fired := applyPlacementGate(placed, dispatchtick.ShapeChurning)
	if !fired {
		t.Fatal("fable on a churning slot must be gated before launch")
	}
	if gated.Model != dispatchtick.WorkerModelOpus {
		t.Errorf("re-routed model = %q, want %q", gated.Model, dispatchtick.WorkerModelOpus)
	}
	if gated.Source != modelSourcePlacement {
		t.Errorf("source = %q, want %q", gated.Source, modelSourcePlacement)
	}
	if gated.PlacementReason != dispatchtick.PlacementShapeMismatch {
		t.Errorf("reason = %q, want %q", gated.PlacementReason, dispatchtick.PlacementShapeMismatch)
	}
	// The reasoning posture survives the model swap (a placement wall is model-scoped).
	if !gated.Ultracode {
		t.Error("ultracode must be carried across the placement re-route")
	}
	// The chain never re-offers the safe model, nor the refused one.
	for _, m := range gated.Chain {
		if m == dispatchtick.WorkerModelOpus {
			t.Errorf("chain %v must not re-offer the pinned safe model", gated.Chain)
		}
		if m == dispatchtick.WorkerModelFable {
			t.Errorf("chain %v must not re-offer the model that cannot hold the shape", gated.Chain)
		}
	}
	// The typed reason reaches the tick payload.
	if got := dispatchWorkerModelMap(gated)["placement_reason"]; got != dispatchtick.PlacementShapeMismatch {
		t.Errorf("payload placement_reason = %v, want %q", got, dispatchtick.PlacementShapeMismatch)
	}
}

// TestApplyPlacementGate_ConservativeDegrade pins that an untagged/surgical issue, an
// unpinned seat default, and a capable model all leave the decision byte-identical.
func TestApplyPlacementGate_ConservativeDegrade(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8"}
	fable := dispatchtick.ProfileFableUltracode
	opus := dispatchtick.ProfileOpusUltracode

	tierFable := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, &fable, "")
	tierOpus := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, &opus, "")
	seat := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, "")

	cases := []struct {
		name  string
		p     workerModelPolicy
		shape dispatchtick.WorkShape
	}{
		{"untagged issue (unknown shape)", tierFable, dispatchtick.ShapeUnknown},
		{"surgical issue", tierFable, dispatchtick.ShapeSurgical},
		{"churning issue on a capable model", tierOpus, dispatchtick.ShapeChurning},
		{"unpinned seat default", seat, dispatchtick.ShapeChurning},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, fired := applyPlacementGate(c.p, c.shape)
			if fired {
				t.Fatalf("gate must not fire: %+v", got)
			}
			if !reflect.DeepEqual(got, c.p) {
				t.Errorf("ungated policy must be byte-identical: got %+v want %+v", got, c.p)
			}
			if _, ok := dispatchWorkerModelMap(got)["placement_reason"]; ok {
				t.Error("an ungated decision's payload must carry no placement_reason")
			}
		})
	}
}

// TestApplyPlacementGate_OperatorPinsAreNeverGated: the gate protects the worker from a
// TABLE's choice, not from a human's. An explicit pin, a lane pin, and the benchmark gate
// are deliberate intent and always win, matching the resolver's precedence doctrine.
func TestApplyPlacementGate_OperatorPinsAreNeverGated(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8", Model: dispatchtick.WorkerModelFable}

	explicit := resolveWorkerModelPolicy("claude", "docs", dispatchtick.WorkerModelFable, acct, fleetaccounts.DefaultPolicy(), false, nil, "")
	pol := fleetaccounts.DefaultPolicy()
	pol.LaneModels["docs"] = dispatchtick.WorkerModelFable
	lane := resolveWorkerModelPolicy("claude", "docs", "", acct, pol, false, nil, "")
	bench := resolveWorkerModelPolicy("claude", "gateway", "", acct, fleetaccounts.DefaultPolicy(), true, nil, "")

	for _, c := range []struct {
		name string
		p    workerModelPolicy
	}{
		{"explicit --worker-model pin", explicit},
		{"lane_models pin", lane},
		{"benchmark gate", bench},
	} {
		t.Run(c.name, func(t *testing.T) {
			if c.p.Model != dispatchtick.WorkerModelFable {
				t.Fatalf("precondition: %s should pin fable, got %q", c.name, c.p.Model)
			}
			if got, fired := applyPlacementGate(c.p, dispatchtick.ShapeChurning); fired {
				t.Errorf("operator intent must win the gate, got re-route to %q", got.Model)
			}
		})
	}
}

// TestApplyPlacementGate_WorkClassPinIsGated: the work-class default is an AUTOMATIC pin
// (a table's choice), so it is gated just like the tier profile.
func TestApplyPlacementGate_WorkClassPinIsGated(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-opus-4-8,claude-sonnet-5")
	acct := dispatchtick.Account{Tag: "gem8"}
	wc := resolveWorkerModelPolicy("claude", "docs", "", acct, fleetaccounts.DefaultPolicy(), false, nil, dispatchtick.WorkerModelFable)
	if wc.Source != modelSourceWorkClass {
		t.Fatalf("precondition: source = %q, want %q", wc.Source, modelSourceWorkClass)
	}
	gated, fired := applyPlacementGate(wc, dispatchtick.ShapeChurning)
	if !fired || gated.Model != dispatchtick.WorkerModelOpus || gated.Source != modelSourcePlacement {
		t.Fatalf("work-class fable on churning must re-route: fired=%v %+v", fired, gated)
	}
}
