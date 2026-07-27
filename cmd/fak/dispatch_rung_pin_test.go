package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func seatDefault(chain ...string) workerModelPolicy {
	return workerModelPolicy{Model: "", Chain: chain, Source: modelSourceSeatDefault}
}

func measuredOn(zone modelroute.PlacementZone, model string) *modelroute.Placement {
	return &modelroute.Placement{Model: model, Zone: zone, Measured: true}
}

// The precedence rule, asserted structurally: the ladder may only fill a blank. Every other
// source is somebody's explicit decision, and an automatic placer that overrides one is the
// failure this seam is shaped to make impossible.
func TestTheLadderOnlyPinsAWorkerThatNothingElseClaimed(t *testing.T) {
	for _, source := range []string{
		modelSourceExplicit, modelSourceLane, modelSourceProfile, modelSourceDefault,
		modelSourceAccount, modelSourceDowngrade, modelSourceTier, modelSourceWorkClass,
		modelSourcePlacement, modelSourceRung, "", "some-future-source",
	} {
		before := workerModelPolicy{Model: "operator-choice", Chain: []string{"a", "b"}, Source: source}
		after := placeUnpinnedWorker(before, measuredOn(modelroute.ZoneDevice, "qwen-local"))
		if after.Model != before.Model || after.Source != before.Source {
			t.Errorf("source %q: ladder overrode a decision somebody already made: %+v", source, after)
		}
	}
}

// The load-bearing refusal. An unmeasured placement is the zero-value capability walking to
// the top rung, so applying it would pin the VENDOR model onto every class the fleet has not
// graded yet — worse than the seat default, while looking like the ladder was working.
func TestAnUnmeasuredPlacementLeavesTheWorkerOnItsSeatDefault(t *testing.T) {
	for _, zone := range []modelroute.PlacementZone{modelroute.ZoneDevice, modelroute.ZoneFleet, modelroute.ZoneVendor} {
		before := seatDefault("fallback-1", "fallback-2")
		after := placeUnpinnedWorker(before, &modelroute.Placement{Model: "some-model", Zone: zone, Measured: false})
		if after.Source != modelSourceSeatDefault || after.Model != "" {
			t.Errorf("zone %q: an unmeasured placement pinned %q (source %q)", zone, after.Model, after.Source)
		}
	}
}

func TestAMeasuredPlacementPinsTheRungAndDropsItFromTheDowngradeChain(t *testing.T) {
	before := seatDefault("qwen-local", "sonnet", "opus")
	after := placeUnpinnedWorker(before, measuredOn(modelroute.ZoneDevice, "qwen-local"))
	if after.Model != "qwen-local" {
		t.Fatalf("model = %q, want the placed rung's model", after.Model)
	}
	if after.Source != modelSourceRung {
		t.Errorf("source = %q, want %q", after.Source, modelSourceRung)
	}
	for _, m := range after.Chain {
		if m == "qwen-local" {
			t.Errorf("the pinned model stayed in the downgrade chain %v — a Layer-2 switch would re-dispatch onto the model that just walled", after.Chain)
		}
	}
	if len(after.Chain) != 2 {
		t.Errorf("chain = %v, want the other two entries preserved", after.Chain)
	}
	// The ladder carries no reasoning-effort or workflow-mode opinion; those belong to the
	// per-issue tier profile, and inventing them here would launch a worker with knobs no
	// operator set.
	if after.Effort != "" || after.Ultracode {
		t.Errorf("the ladder invented launch knobs: effort=%q ultracode=%v", after.Effort, after.Ultracode)
	}
	// PlacementReason belongs to the preventive shape-mismatch gate. A pin that never
	// re-routed anything must not claim one.
	if after.PlacementReason != "" {
		t.Errorf("placement reason = %q, want empty", after.PlacementReason)
	}
}

// Not every placement is a saving, and suppressing the expensive answer would make the
// ladder dishonest in the other direction: a measured vendor placement is the ladder saying
// this class needs the horsepower, which is the design's third stratum.
func TestAMeasuredVendorPlacementIsAppliedRatherThanSuppressed(t *testing.T) {
	after := placeUnpinnedWorker(seatDefault("sonnet"), measuredOn(modelroute.ZoneVendor, "opus"))
	if after.Model != "opus" || after.Source != modelSourceRung {
		t.Fatalf("a measured vendor placement was not applied: %+v", after)
	}
}

// Every way of having nothing to say leaves the worker exactly as it was, so a fleet with no
// roster, no evidence, or a placement that named no model is byte-identical to before.
func TestAPlacementWithNothingToSayIsANoOp(t *testing.T) {
	for name, rung := range map[string]*modelroute.Placement{
		"no placement at all":     nil,
		"placement names nothing": {Model: "", Zone: modelroute.ZoneDevice, Measured: true},
		"placement is blank":      {Model: "   ", Zone: modelroute.ZoneDevice, Measured: true},
		"zero value":              {},
	} {
		before := seatDefault("sonnet", "opus")
		after := placeUnpinnedWorker(before, rung)
		if after.Source != modelSourceSeatDefault || after.Model != "" {
			t.Errorf("%s: pinned %q (source %q)", name, after.Model, after.Source)
		}
		if len(after.Chain) != len(before.Chain) {
			t.Errorf("%s: chain changed from %v to %v", name, before.Chain, after.Chain)
		}
	}
}

// The pinned id is the placement's own, never one recovered from the chain or the seat.
func TestThePinIsThePlacementsOwnModel(t *testing.T) {
	after := placeUnpinnedWorker(seatDefault("sonnet", "opus"), measuredOn(modelroute.ZoneFleet, "  glm-5.2  "))
	if after.Model != "glm-5.2" {
		t.Fatalf("model = %q, want the trimmed placed model", after.Model)
	}
	if !after.pinned() {
		t.Errorf("a placed worker did not read as pinned, so Layer-2 and the .model witness would skip it")
	}
}
