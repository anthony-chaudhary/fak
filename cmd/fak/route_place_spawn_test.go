package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Witnesses for `fak route --place --spawn-type`, the delegated half of the placement
// oracle (epic #5416, track E).
//
// The property under test is the same one the parent oracle is held to: the surface may
// not overstate what fak knows. For a spawn there is one extra way to overstate it —
// answering for an agent type nobody classified — and that is the first test, because a
// silent default to "routine" would turn delegation into the way onto a cheap rung
// without stating what the work is.

// spawnRoster is the three-rung fixture plus an operator's declarations: one cheap type,
// one that ships code, one that does release/destructive work.
func spawnRoster() modelroute.Roster {
	r := routePlaceRoster()
	r.SpawnClasses = []modelroute.SpawnClass{
		{Type: "explore", Class: modelroute.ClassRoutine},
		{Type: "code-reviewer", Class: modelroute.ClassNormalImpl},
		{Type: "release-cutter", Class: modelroute.ClassSecurityRelease},
	}
	return r
}

// measuredRungs grades all three rungs, which is what makes a descent legal at all: an
// unmeasured candidate may never win a cheap rung, so a spawn test that skipped this
// would be witnessing the unmeasured refusal rather than the spawn placement.
const measuredRungs = "rung-device=t2,rung-fleet=t1,rung-vendor=t0"

// spawnBlock returns just the delegated half of the report. Assertions about where the
// CHILD landed have to be scoped to it: the parent's block uses the same field labels a
// few lines above, so a whole-output match can pass or fail on the wrong rung entirely.
func spawnBlock(t *testing.T, out string) string {
	t.Helper()
	i := strings.Index(out, "SPAWN PLACEMENT")
	if i < 0 {
		t.Fatalf("no spawn block was rendered:\n%s", out)
	}
	return out[i:]
}

func TestAnUndeclaredSpawnTypeIsRefusedNotAssumedRoutine(t *testing.T) {
	// A roster that has never been told anything about sub-agents.
	bare := routePlaceRoster()
	code, out, errOut := routePlaceRunOpts(t, &bare, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: measuredRungs, SpawnType: "explore"})
	if code != 1 {
		t.Fatalf("an undeclared spawn type: exit = %d, want 1; stderr = %q", code, errOut)
	}
	// No half-report: the operator must not read a placement block and believe a spawn
	// was placed somewhere in it.
	if out != "" {
		t.Fatalf("a refused spawn still printed a placement:\n%s", out)
	}
	for _, want := range []string{"spawn_classes", "routine"} {
		if !strings.Contains(errOut, want) {
			t.Errorf("the refusal does not show the fix (%q missing): %q", want, errOut)
		}
	}

	// And a roster that declares OTHER types shows the operator their own vocabulary,
	// because the likely cause is a spelling mismatch with whatever their harness calls
	// its agents — not a missing concept.
	r := spawnRoster()
	code, out, errOut = routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: measuredRungs, SpawnType: "explorer"})
	// A near-miss must not be resolved by prefix, which is the whole reason the lookup
	// is exact: "explore" answering for "explorer" is how a type nobody classified gets
	// a class anyway.
	if code != 1 {
		t.Fatalf("a misspelled spawn type: exit = %d, want 1 (a prefix must not resolve)", code)
	}
	if strings.Contains(out, "SPAWN PLACEMENT") {
		t.Fatalf("a prefix match placed a spawn that was never declared:\n%s", out)
	}
	for _, want := range []string{"explore", "code-reviewer", "release-cutter"} {
		if !strings.Contains(errOut, want) {
			t.Errorf("the refusal does not list declared type %q: %q", want, errOut)
		}
	}
}

func TestADeclaredSpawnDescendsBelowItsParentsRung(t *testing.T) {
	r := spawnRoster()
	// The shape the epic exists for: an expensive parent turn delegating cheap work.
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "ultra-hard"},
		placeOptions{CapSpec: measuredRungs, SpawnType: "explore"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	block := spawnBlock(t, out)
	for _, want := range []string{
		"work class   routine",          // the DECLARED class, not one inferred from the parent
		"spawn_classes",                 // and it says where the declaration came from
		"parent       zone=vendor",      // the parent rung, recorded
		"placed       zone=device",      // the child's own rung, one below it
		modelroute.ReasonSpawnDescended, // in the closed relationship vocabulary
		"self-hosted descent  yes",      // the event the epic counts
		"epic #5416",                    // named, so the number has a definition attached
	} {
		if !strings.Contains(block, want) {
			t.Errorf("the spawn block is missing %q:\n%s", want, block)
		}
	}
	// The parent's own report still stands above it, unchanged: --spawn-type adds a
	// question, it does not replace the one --place already answered.
	if !strings.Contains(out, "PLACEMENT  (fak route --place)") {
		t.Errorf("the parent placement disappeared when a spawn was asked about:\n%s", out)
	}
}

// The safety half, and the reason this is not merely a cost feature. A routine parent
// legitimately on the laptop spawns release/destructive work: inheriting would have run
// it on a T2 device model, and every gate in the package would have been satisfied
// because the child never went through one.
func TestASpawnCannotInheritItsWayUnderTheFloor(t *testing.T) {
	r := spawnRoster()
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: measuredRungs, SpawnType: "release-cutter"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	block := spawnBlock(t, out)
	if !strings.Contains(block, "parent       zone=device") {
		t.Fatalf("the parent was expected on the cheap rung:\n%s", out)
	}
	// The child does NOT stay there.
	if strings.Contains(block, "placed       zone=device") {
		t.Fatalf("release/destructive work was placed on the device rung:\n%s", block)
	}
	for _, want := range []string{
		modelroute.ReasonSpawnRose,
		modelroute.ReasonSpawnInheritUnderTier,
		"floor bypass", // said in words, not only as a token
	} {
		if !strings.Contains(block, want) {
			t.Errorf("the under-tier counterfactual is missing %q:\n%s", want, block)
		}
	}
	if strings.Contains(block, "self-hosted descent  yes") {
		t.Errorf("an escalation was counted as a self-hosted descent:\n%s", block)
	}
}

// An unmeasured parent means the counterfactual CANNOT be answered, and the surface has
// to say so rather than print the reassuring branch. This is the same unearned-zero rule
// the rest of the oracle runs on: absence of a measurement is not a clean bill of health.
func TestAnUnmeasuredParentLeavesTheCounterfactualUnknown(t *testing.T) {
	r := spawnRoster()
	// Only the device rung is graded, so the vendor rung the parent lands on carries no
	// measurement.
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "ultra-hard"},
		placeOptions{CapSpec: "rung-device=t2", SpawnType: "explore"})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	block := spawnBlock(t, out)
	if !strings.Contains(block, modelroute.ReasonSpawnInheritUnmeasured) {
		t.Fatalf("an unmeasured parent did not report the counterfactual as unanswerable:\n%s", block)
	}
	if strings.Contains(block, "would have been admitted") {
		t.Errorf("an ungraded parent was reported as clearing the child's floor:\n%s", block)
	}
	if !strings.Contains(block, "UNKNOWN") {
		t.Errorf("the unknown case is not legible as ignorance:\n%s", block)
	}
}

// ABSENCE DISCIPLINE on the wire: a run that never asked the delegated question must not
// emit a spawn key at all, because a present-but-empty one reads as a spawn that placed
// nowhere.
func TestTheSpawnReportIsAbsentUnlessItWasAsked(t *testing.T) {
	r := spawnRoster()
	code, out, errOut := routePlaceRunOpts(t, &r, map[string]string{"work_class": "routine"},
		placeOptions{CapSpec: measuredRungs, JSON: true})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(out), &raw); err != nil {
		t.Fatalf("--json is not valid JSON: %v\n%s", err, out)
	}
	if _, present := raw["spawn"]; present {
		t.Errorf("a run without --spawn-type emitted a spawn key:\n%s", out)
	}

	code, out, errOut = routePlaceRunOpts(t, &r, map[string]string{"work_class": "ultra-hard"},
		placeOptions{CapSpec: measuredRungs, SpawnType: "explore", JSON: true})
	if code != 0 {
		t.Fatalf("exit = %d, stderr = %q", code, errOut)
	}
	var rep placementReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("--json is not valid JSON: %v\n%s", err, out)
	}
	if rep.Spawn == nil {
		t.Fatalf("the spawn report is absent although --spawn-type was given:\n%s", out)
	}
	if rep.Spawn.Class != modelroute.ClassRoutine || !rep.Spawn.Declared {
		t.Errorf("spawn class = %q declared=%v, want the declared %q",
			rep.Spawn.Class, rep.Spawn.Declared, modelroute.ClassRoutine)
	}
	// The headline number and the two fields it is defined from must agree in the
	// report itself — that is where a drifting definition would first show up.
	sp := rep.Spawn.Placement
	if rep.Spawn.SelfHostedDescent != (sp.Descended && sp.Placement.SelfHosted()) {
		t.Errorf("self_hosted_descent=%v contradicts descended=%v self_hosted=%v",
			rep.Spawn.SelfHostedDescent, sp.Descended, sp.Placement.SelfHosted())
	}
	if !rep.Spawn.SelfHostedDescent {
		t.Errorf("a routine child under a vendor parent should be a self-hosted descent: %+v", sp)
	}
}

// A placement flag that silently does nothing is worse than one that is refused: the
// operator reads a routing answer and believes they saw a spawn placed.
func TestSpawnTypeWithoutPlaceIsRefused(t *testing.T) {
	code, out, errOut := runRT("--spawn-type", "explore")
	if code != 2 {
		t.Fatalf("--spawn-type without --place: exit = %d, want 2 (usage)", code)
	}
	if !strings.Contains(errOut, "--place") {
		t.Errorf("the refusal does not name the missing flag: %q", errOut)
	}
	if strings.Contains(out, "SPAWN") {
		t.Errorf("a spawn block was printed without --place:\n%s", out)
	}
}
