package main

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// The placement ladder's un-blanking step (epic #5416 track E).
//
// `Roster.Place` walks down to the cheapest rung that can serve a class of work, and until
// something applied its answer to a launch, nothing moved: the whole three-stratum design
// could be right and every worker would still start on the seat's vendor default. This is
// the step that applies it — for dispatch workers, which is where the volume is.
//
// It is written as a step AFTER `resolveWorkerModelPolicy` rather than as another branch
// inside it, and that is the point rather than a convenience. The rule "the ladder is the
// lowest-precedence source" then holds structurally: this can only act on a decision whose
// Source is the seat default, so an operator pin, a lane pin, the benchmark gate, a
// per-issue tier profile and a work-class default all keep winning no matter where a future
// edit puts things. The alternative — a sixth branch at the bottom of a long if-chain —
// keeps that ordering true only as long as nobody reorders the chain, and the failure mode
// of getting it wrong is an automatic placer silently overriding an operator's explicit
// choice of model.
//
// The load-bearing refusal is `Placement.Measured`. Rule 2 of `Place` is that an unmeasured
// capability may not descend the ladder, so a placement built with no evidence resolves to
// the TOP rung — correctly, since the alternative is asserting a capability nobody measured.
// Applying such a placement would therefore pin the vendor model onto every slot whose work
// class the fleet has not yet graded, which is worse than the seat default it replaced: it
// would look like the ladder was working while it moved traffic in the wrong direction. So
// an unmeasured placement leaves the worker exactly where it was, and this seam starts
// moving traffic only as the track-F evidence corpus fills in. That coupling is deliberate.
//
// A MEASURED placement on the vendor rung IS applied. That is not a failure of the ladder:
// it is the ladder saying this class of work needs the horsepower, which is the third
// stratum of the design and not a case to be suppressed.
//
// What this step cannot check: whether the launching backend can actually reach the pinned
// rung's endpoint. A device- or fleet-rung id is only launchable by a worker whose seat
// routes to that endpoint, and nothing here can verify that — which is why the seam is
// opt-in rather than default-on, and why the fleet trust boundary (#5421, track G) is the
// prerequisite for turning it on broadly. A pin the backend cannot serve walls as
// `model_unknown`, which the Layer-2 downgrade chain already treats as model-switchable.

// modelSourceRung is the placement ladder as an un-blanking source: the resolved automatic
// placement for this slot's work class named a model, and its capability was MEASURED, so
// the worker starts on the cheapest rung the evidence supports instead of the seat default.
// Distinct from modelSourcePlacement, which is the preventive shape-mismatch gate re-routing
// a placement that was already made — this one MAKES the placement.
const modelSourceRung = "placement-rung"

// placeUnpinnedWorker applies a resolved placement to a worker that nothing else pinned.
//
// Pure: policy and placement in, policy out. It returns p unchanged for every input it is
// not certain about, so the seam is a no-op wherever the evidence, the roster or an operator
// has not spoken.
func placeUnpinnedWorker(p workerModelPolicy, rung *modelroute.Placement) workerModelPolicy {
	// Anything other than the seat default is a decision someone already made. The ladder is
	// the lowest-precedence source and may not overwrite one.
	if p.Source != modelSourceSeatDefault {
		return p
	}
	if rung == nil {
		return p
	}
	model := strings.TrimSpace(rung.Model)
	if model == "" {
		return p
	}
	// An unmeasured placement is the zero-value capability walking to the top rung, not a
	// finding about this model. Pinning it would move every ungraded class to the vendor.
	if !rung.Measured {
		return p
	}
	return workerModelPolicy{
		Model:  model,
		Chain:  dropModel(p.Chain, model),
		Source: modelSourceRung,
	}
}
