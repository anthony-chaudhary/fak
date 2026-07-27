package main

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// `fak route --place --spawn-type TYPE` — the delegated half of the placement oracle
// (epic #5416, track E).
//
// `--place` answers where one turn runs. This answers the question the epic's thesis
// actually rests on: where does the work that turn DELEGATES run. In a 100-500 engineer
// shop the sub-agent sweeps, background scouts and review fan-outs are the high-volume
// majority of traffic, so "the bulk of token usage gets covered by self-hosted items"
// either happens here or does not happen.
//
//	fak route --accounts FILE --place --labels work_class=ultra-hard --spawn-type explore
//
// The child's class is NOT inferred from the parent, from the prompt, or from the type
// string's spelling. It is read from the roster's `spawn_classes` block — an operator
// declaring what their own agent types do. An undeclared type is a loud refusal that
// names the fix, because the alternative (defaulting to routine) is how delegation would
// become a way onto a cheap rung without anyone stating what the work is.
//
// Like the rest of `--place`, this moves no traffic. It reports what the ladder WOULD do
// with the operator's roster, before anything on the dispatch path calls PlaceSpawn.

// spawnPlacementReport is the `--json` shape for the delegated half. It sits under a
// pointer field in placementReport so that a run without --spawn-type emits no spawn key
// at all: an absent report means the question was not asked, which must not be
// confusable with a spawn that was placed.
type spawnPlacementReport struct {
	Type  string               `json:"type"`
	Class modelroute.WorkClass `json:"class"`
	// Declared is always true in an emitted report — an undeclared type never gets
	// this far — but it is carried so a consumer reads the class as an operator's
	// declaration rather than as something fak decided.
	Declared  bool                      `json:"declared"`
	Placement modelroute.SpawnPlacement `json:"placement"`
	// SelfHostedDescent is the event the epic counts: a child that both landed on
	// hardware the organization operates AND on a cheaper rung than its parent. It is
	// carried rather than left to the consumer because re-deriving it from the two
	// fields is exactly where a headline number would drift from its definition.
	SelfHostedDescent bool `json:"self_hosted_descent"`
}

// spawnPlacementFor resolves the declared class for a spawn type and places the child
// against the same candidate pool the parent walked.
//
// The parent placement is passed through unchanged. PlaceSpawn records it and does not
// obey it — a sub-agent spawned from a vendor turn must still be able to land on the
// engineer's laptop — and this oracle deliberately does not add a second opinion about
// that on top.
func spawnPlacementFor(r modelroute.Roster, parent modelroute.Placement, spawnType string, candidates []modelroute.Candidate) (spawnPlacementReport, error) {
	class, declared := r.SpawnClassFor(spawnType)
	if !declared {
		return spawnPlacementReport{}, undeclaredSpawnTypeError(r, spawnType)
	}
	sp, err := r.PlaceSpawn(parent, class, candidates)
	if err != nil {
		return spawnPlacementReport{}, err
	}
	return spawnPlacementReport{
		Type:              strings.TrimSpace(spawnType),
		Class:             class,
		Declared:          true,
		Placement:         sp,
		SelfHostedDescent: sp.SelfHostedDescent(),
	}, nil
}

// undeclaredSpawnTypeError says which fix applies, and the two cases need different
// sentences: a roster with no `spawn_classes` block at all has never been told anything
// about sub-agents, while a roster that declares some types and not this one is far more
// likely to be a spelling mismatch with whatever the harness calls its agents.
func undeclaredSpawnTypeError(r modelroute.Roster, spawnType string) error {
	typed := strings.TrimSpace(spawnType)
	declared := declaredSpawnTypes(r)
	if len(declared) == 0 {
		return fmt.Errorf("--spawn-type %q: this roster declares no spawn classes, so no sub-agent type has a work class yet. "+
			"Add a spawn_classes entry naming what that agent type does, e.g. "+
			`{"type": %q, "class": "routine"} — a spawn cannot be placed without one, and defaulting it would be the guess this oracle exists to avoid`,
			typed, typed)
	}
	return fmt.Errorf("--spawn-type %q is not declared in this roster's spawn_classes (declared: %s). "+
		"Add an entry for it, or use the spelling the roster already carries; an undeclared type is left unplaced rather than assumed routine",
		typed, strings.Join(declared, ", "))
}

// declaredSpawnTypes lists the roster's declared spawn types in a deterministic order,
// so the refusal above shows the operator their own vocabulary rather than fak's.
func declaredSpawnTypes(r modelroute.Roster) []string {
	out := make([]string, 0, len(r.SpawnClasses))
	for _, sc := range r.SpawnClasses {
		if t := strings.TrimSpace(sc.Type); t != "" {
			out = append(out, t)
		}
	}
	sort.Strings(out)
	return out
}

// printSpawnPlacement renders the delegated placement for a human, under the parent's
// own report so the two rungs can be read against each other.
func printSpawnPlacement(w io.Writer, rep spawnPlacementReport) {
	sp := rep.Placement
	p := sp.Placement
	fmt.Fprintf(w, "\nSPAWN PLACEMENT  (--spawn-type %s)\n", rep.Type)
	fmt.Fprintf(w, "  work class   %s  [declared by the roster's spawn_classes]\n", rep.Class)
	parent := "(root turn — nothing spawned this)"
	if sp.ParentZone != "" || sp.ParentModel != "" {
		parent = fmt.Sprintf("zone=%s  model=%s", sp.ParentZone, sp.ParentModel)
	}
	fmt.Fprintf(w, "  parent       %s\n", parent)
	fmt.Fprintf(w, "  placed       zone=%s  model=%s\n", p.Zone, p.Model)
	fmt.Fprintf(w, "  target       kind=%s account=%s upstream=%s\n", p.Target.Kind, p.Target.Account, p.Target.UpstreamModel)
	fmt.Fprintf(w, "  self-hosted  %-3s      escalated  %s\n", yesNo(p.SelfHosted()), yesNo(p.Escalated))
	fmt.Fprintf(w, "  tier floor   required=%s optimal=%s  chosen-capability=%s\n",
		p.Choice.RequiredTier, p.Choice.OptimalTier, capabilityCell(p))
	fmt.Fprintf(w, "  relation     %s\n", strings.Join(sp.Reasons, " "))
	fmt.Fprintf(w, "  descent      %s      self-hosted descent  %s\n", yesNo(sp.Descended), yesNo(rep.SelfHostedDescent))
	fmt.Fprintln(w, "  ladder")
	for _, v := range p.Ladder {
		model := v.Model
		if model == "" {
			model = "-"
		}
		fmt.Fprintf(w, "    %-7s %-24s %s\n", v.Zone, model, strings.Join(tallyReasons(v.Reasons), " "))
	}
	// The counterfactual, spelled out. "What would the status quo have done" is the
	// question an operator asks when deciding whether this is a cost change or a
	// correctness one, and the unmeasured case must read as ignorance rather than as a
	// clean bill of health.
	switch {
	case containsReason(sp.Reasons, modelroute.ReasonSpawnInheritUnderTier):
		fmt.Fprintln(w, "  inheriting   would have UNDER-TIERED this child: the parent's model does not clear")
		fmt.Fprintln(w, "               the child's own floor, so today's inherit-the-parent behaviour is a")
		fmt.Fprintln(w, "               floor bypass here, not merely a bigger bill.")
	case containsReason(sp.Reasons, modelroute.ReasonSpawnInheritUnmeasured):
		fmt.Fprintln(w, "  inheriting   UNKNOWN: the parent's capability was never measured, so what inheriting")
		fmt.Fprintln(w, "               would have done cannot be answered. Grade it (--capability / --evidence)")
		fmt.Fprintln(w, "               before reading this line as safe.")
	case containsReason(sp.Reasons, modelroute.ReasonSpawnInheritAdmitted):
		fmt.Fprintln(w, "  inheriting   would have been admitted: the parent's model clears the child's floor,")
		fmt.Fprintln(w, "               so here the argument for re-placing is cost, not safety.")
	}
	if rep.SelfHostedDescent {
		fmt.Fprintln(w, "  note         this is the event epic #5416 counts: a delegated turn moved onto")
		fmt.Fprintln(w, "               hardware the organization operates, one rung below its parent.")
	}
}

// containsReason reports whether a closed reason token is present.
func containsReason(reasons []string, want string) bool {
	for _, r := range reasons {
		if r == want {
			return true
		}
	}
	return false
}
