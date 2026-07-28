package main

import (
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// routineTierLabels is a trusted T2 tag pair — the declared work class the ladder grades
// against. Without one the slot has no class and the ladder must refuse rather than place.
var routineTierLabels = []string{"tier/T2-optimal", "tier/T2-required"}

// rungRoot is a workspace with the seam on and no ambient configuration: no roster, no
// journal, default window. Each test adds exactly the inputs it is about.
//
// The reach declaration names all three of placementEvidenceRoster's accounts, so tests
// about EVIDENCE are not silently also testing launchability. The "*" spelling and the
// narrowed and undeclared cases each get their own test rather than riding in here.
func rungRoot(t *testing.T) string {
	t.Helper()
	setDispatchRungPlacement(t, true)
	setDispatchAccountsRoster(t, "")
	t.Setenv(dispatchRungWindowEnv, "")
	t.Setenv(dispatchRungAccountsEnv, "laptop,cluster,frontier")
	return t.TempDir()
}

// setDispatchRungPlacement declares the ladder setting for the duration of one test and
// restores it afterwards — the config-surface counterpart of the t.Setenv this used to need,
// now that the switch is `fak dispatch tick --rung-placement` rather than an environment read.
func setDispatchRungPlacement(t *testing.T, on bool) {
	t.Helper()
	old := dispatchRungPlacement
	dispatchRungPlacement = on
	t.Cleanup(func() { dispatchRungPlacement = old })
}

// reachAll is the whole-roster assertion, for the pure resolver tests that are not about
// which accounts are dialable.
var reachAll = rungReach{All: true}

func withRungRoster(t *testing.T, root string) {
	t.Helper()
	dir := filepath.Join(root, "tools")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model-accounts.json"), []byte(placementEvidenceRoster), 0o644); err != nil {
		t.Fatal(err)
	}
}

// withRungJournal writes outcomes through the real producer, so a change to the journal
// format breaks these tests instead of leaving them passing against a stale hand-written
// fixture.
func withRungJournal(t *testing.T, root string, outcomes []modelroute.TurnOutcome) {
	t.Helper()
	dir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(filepath.Join(dir, dispatchTurnJournalName))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, o := range outcomes {
		if err := modelroute.AppendTurnOutcome(f, o); err != nil {
			t.Fatal(err)
		}
	}
}

// rungTurns is n distinct successful routine turns for one model — enough to clear the
// default grade floor (20 attempts, 80%) when they are all successes. Distinct from the
// route-surface helper next door in taking the PROVENANCE as a parameter, because half of
// what this seam has to get right is which provenance buys nothing.
func rungTurns(model string, n int, age time.Duration, verify modelroute.Verification) []modelroute.TurnOutcome {
	out := make([]modelroute.TurnOutcome, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, modelroute.TurnOutcome{
			ID:      model + "-" + strconv.Itoa(i),
			Model:   model,
			Class:   modelroute.ClassRoutine,
			Success: true,
			Verify:  verify,
			At:      time.Now().Add(-age),
		})
	}
	return out
}

func seatDefaultFor(models ...string) workerModelPolicy {
	return workerModelPolicy{Chain: models, Source: modelSourceSeatDefault}
}

// The seam defaults OFF and every off-ish spelling means off. Turning an automatic placer on
// by default would re-point a fleet's traffic at hardware nobody said it could reach.
func TestTheRungLadderStaysOffUntilAnOperatorTurnsItOn(t *testing.T) {
	// The declaration is on the tick's command surface, not in the ambient environment: an
	// automatic placer armed by a stray exported variable is exactly the accident this seam
	// must not have, and a behavioral setting read via os.LookupEnv is what
	// internal/envconfiglint refuses (CONFIG_NOT_ENV).
	silent, _, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir()})
	if code != 0 {
		t.Fatalf("parse of a bare tick failed with code %d", code)
	}
	if silent.RungPlacement {
		t.Error("an undeclared tick armed the ladder")
	}
	declared, _, code := parseDispatchTickFlags(io.Discard, []string{"--workspace", t.TempDir(), "--rung-placement"})
	if code != 0 {
		t.Fatalf("parse of a declared tick failed with code %d", code)
	}
	if !declared.RungPlacement {
		t.Error("--rung-placement did not reach the options")
	}

	setDispatchRungPlacement(t, false)
	if dispatchRungPlacementEnabled() {
		t.Error("an undeclared setting enabled the ladder")
	}
	setDispatchRungPlacement(t, true)
	if !dispatchRungPlacementEnabled() {
		t.Error("a declared setting did not enable the ladder")
	}
}

// With the seam off the ladder returns no reason at all, so the caller writes no payload key
// and an unconfigured fleet's tick is byte-identical to before this existed. Every OTHER
// silence is a decision and names itself; this one is the absence of a decision.
func TestAnUnconfiguredTickIsByteIdenticalToBeforeTheLadder(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))
	setDispatchRungPlacement(t, false)

	before := seatDefaultFor("opus-5", "glm-5.2")
	after, skip := applyRungPlacement(root, routineTierLabels, before)
	if skip != "" {
		t.Errorf("an off seam reported %q, which the caller would write into every tick payload", skip)
	}
	if !reflect.DeepEqual(after, before) {
		t.Errorf("an off seam changed the policy: %+v -> %+v", before, after)
	}
}

// The design, end to end: a class of work with a graded local model behind it starts on the
// DEVICE rung instead of the seat's vendor default. This is the whole point of the epic —
// the bulk of the tokens served on hardware the company already owns.
func TestTheLadderStartsAGradedRoutineSlotOnTheDeviceRung(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))

	after, skip := applyRungPlacement(root, routineTierLabels, seatDefaultFor("qwen3.6-4b", "opus-5"))
	if skip != "" {
		t.Fatalf("the ladder refused a fully graded slot: %s", skip)
	}
	if after.Model != "qwen3.6-4b" {
		t.Errorf("model = %q, want the graded device-rung model", after.Model)
	}
	if after.Source != modelSourceRung {
		t.Errorf("source = %q, want %q", after.Source, modelSourceRung)
	}
	for _, m := range after.Chain {
		if m == "qwen3.6-4b" {
			t.Errorf("the pinned model stayed on the downgrade chain %v", after.Chain)
		}
	}
}

// PolicyFor floors an UNKNOWN class at T0, which is the right conservatism when choosing a
// floor for work and would be a catastrophe here: it would walk every untagged issue in the
// fleet to the vendor rung. An unclassified slot is refused by name instead.
func TestAnUnclassifiedSlotIsRefusedRatherThanFlooredToTheVendorRung(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))

	for _, labels := range [][]string{nil, {}, {"bug", "area/gateway"}, {"tier/nonsense"}} {
		before := seatDefaultFor("qwen3.6-4b", "opus-5")
		after, skip := applyRungPlacement(root, labels, before)
		if skip != rungSkipNoWorkClass {
			t.Errorf("labels %v: skip = %q, want %q", labels, skip, rungSkipNoWorkClass)
		}
		if !reflect.DeepEqual(after, before) {
			t.Errorf("labels %v: an unclassified slot was moved to %+v", labels, after)
		}
	}
}

// Each way of having nothing to say gets its OWN reason. A ladder that silently does nothing
// is indistinguishable from a broken one, and these strings are what an operator reads out of
// the tick payload when they ask why it did not fire.
func TestEveryWayOfHavingNothingToSayNamesItselfAndMovesNothing(t *testing.T) {
	fresh := rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness)

	for _, c := range []struct {
		name   string
		setup  func(t *testing.T, root string)
		policy workerModelPolicy
		labels []string
		want   string
	}{
		{
			name:   "something already decided this worker's model",
			setup:  func(t *testing.T, root string) { withRungRoster(t, root); withRungJournal(t, root, fresh) },
			policy: workerModelPolicy{Model: "opus-5", Source: modelSourceExplicit},
			want:   rungSkipOutranked,
		},
		{
			name:  "no account roster to place from",
			setup: func(t *testing.T, root string) { withRungJournal(t, root, fresh) },
			want:  rungSkipNoRoster,
		},
		{
			name: "a roster that does not parse",
			setup: func(t *testing.T, root string) {
				if err := os.MkdirAll(filepath.Join(root, "tools"), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(root, "tools", "model-accounts.json"), []byte("{not json"), 0o644); err != nil {
					t.Fatal(err)
				}
				withRungJournal(t, root, fresh)
			},
			want: rungSkipNoRoster,
		},
		{
			name:  "the fleet has produced no turn evidence yet",
			setup: func(t *testing.T, root string) { withRungRoster(t, root) },
			want:  rungSkipNoJournal,
		},
		{
			name:  "a journal with nothing in it",
			setup: func(t *testing.T, root string) { withRungRoster(t, root); withRungJournal(t, root, nil) },
			want:  rungSkipNoEvidence,
		},
		{
			name: "a journal whose rows do not parse at all",
			setup: func(t *testing.T, root string) {
				withRungRoster(t, root)
				withRungJournal(t, root, nil)
				if err := os.WriteFile(filepath.Join(root, dispatchtick.RunsDirName, dispatchTurnJournalName),
					[]byte("{torn\n{also torn\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			},
			want: rungSkipJournalBad,
		},
		{
			// A torn FINAL line must not discard the corpus in front of it: the reader is
			// forgiving on purpose, and a fleet appending to this file mid-write is normal.
			name: "a good corpus with one torn row still places",
			setup: func(t *testing.T, root string) {
				withRungRoster(t, root)
				withRungJournal(t, root, fresh)
				f, err := os.OpenFile(filepath.Join(root, dispatchtick.RunsDirName, dispatchTurnJournalName),
					os.O_APPEND|os.O_WRONLY, 0o644)
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				if _, err := f.WriteString("{\"model\":\"qwen3.6-4b\",\"cl"); err != nil {
					t.Fatal(err)
				}
			},
			want: "",
		},
		{
			name: "an unparseable freshness window",
			setup: func(t *testing.T, root string) {
				withRungRoster(t, root)
				withRungJournal(t, root, fresh)
				t.Setenv(dispatchRungWindowEnv, "a fortnight")
			},
			want: rungSkipBadWindow,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			root := rungRoot(t)
			c.setup(t, root)
			before := c.policy
			if before.Source == "" {
				before = seatDefaultFor("qwen3.6-4b", "opus-5")
			}
			labels := c.labels
			if labels == nil {
				labels = routineTierLabels
			}
			after, skip := applyRungPlacement(root, labels, before)
			if skip != c.want {
				t.Errorf("skip = %q, want %q", skip, c.want)
			}
			if c.want == "" {
				if after.Source != modelSourceRung {
					t.Errorf("the ladder had no complaint and still did not place: %+v", after)
				}
				return
			}
			if !reflect.DeepEqual(after, before) {
				t.Errorf("a refusal moved the worker: %+v -> %+v", before, after)
			}
		})
	}
}

// Capability is a property of a model AS DEPLOYED. A re-quantised local build or a re-pointed
// fleet endpoint keeps the id and changes the thing, so evidence from a quarter ago is not a
// finding about what is running now. Out-of-window evidence must not place anything.
func TestStaleEvidenceIsNotGradedAsCurrentCapability(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, 90*24*time.Hour, modelroute.VerifyWitness))

	before := seatDefaultFor("qwen3.6-4b", "opus-5")
	after, skip := applyRungPlacement(root, routineTierLabels, before)
	if skip != rungSkipNoEvidence {
		t.Errorf("skip = %q, want %q — a 90-day-old corpus graded as current capability", skip, rungSkipNoEvidence)
	}
	if !reflect.DeepEqual(after, before) {
		t.Errorf("stale evidence moved the worker to %+v", after)
	}

	// And the window is the operator's to widen, once they have decided that is what they
	// mean. The same corpus places when they say so.
	t.Setenv(dispatchRungWindowEnv, "365d")
	if after, skip := applyRungPlacement(root, routineTierLabels, before); skip != "" || after.Model != "qwen3.6-4b" {
		t.Errorf("a widened window did not admit the same corpus: model=%q skip=%q", after.Model, skip)
	}
}

// The provenance rule, asserted at the fleet level rather than only in the grader: a model
// self-reporting success 24 times buys no rung at all. Otherwise the cheapest way onto the
// device rung would be for a model to claim it belongs there.
func TestTheModelsOwnWordNeverBuysARung(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyNone))

	before := seatDefaultFor("qwen3.6-4b", "opus-5")
	after, skip := applyRungPlacement(root, routineTierLabels, before)
	if skip != rungSkipUnmeasured {
		t.Errorf("skip = %q, want %q", skip, rungSkipUnmeasured)
	}
	if !reflect.DeepEqual(after, before) {
		t.Errorf("self-reported success moved the worker to %+v", after)
	}
}

// Rule 2 of Place is that an unmeasured capability may not descend, so a roster nobody has
// graded resolves to the TOP rung. That is the correct placement and the wrong thing to
// apply: applying it would move every ungraded class to the vendor while looking like the
// ladder was working. The resolver names it rather than handing it on.
func TestAnUngradedRosterResolvesToTheTopRungAndIsRefused(t *testing.T) {
	roster, err := modelroute.ParseRoster([]byte(placementEvidenceRoster))
	if err != nil {
		t.Fatal(err)
	}
	rung, skip := resolveRungPlacement(roster, modelroute.ClassRoutine, map[string][]modelroute.ClassEvidence{
		"qwen3.6-4b": {{Class: modelroute.ClassRoutine, Attempts: 3, Successes: 3, Verify: modelroute.VerifyWitness}},
	}, reachAll)
	if skip != rungSkipUnmeasured {
		t.Errorf("skip = %q, want %q", skip, rungSkipUnmeasured)
	}
	if rung != nil {
		t.Errorf("an unmeasured placement was handed on: %+v", rung)
	}

	// A roster that binds no models cannot place at all, and says that instead.
	if _, skip := resolveRungPlacement(modelroute.Roster{}, modelroute.ClassRoutine, nil, reachAll); skip != rungSkipRosterEmpty {
		t.Errorf("empty roster skip = %q, want %q", skip, rungSkipRosterEmpty)
	}
}

// The ladder is an AUTOMATIC source, so it belongs on the gated side of applyPlacementGate
// with the tier table and the work-class default. A rung is picked from a corpus of past
// outcomes, which says the model can do this CLASS of work and says nothing about this
// issue's work SHAPE — the question the gate exists to ask.
func TestAnAutomaticRungPinIsGatedLikeTheTierTable(t *testing.T) {
	t.Setenv("FLEET_WORKER_FALLBACK_MODEL", "claude-fable-5,claude-opus-4-8,claude-sonnet-5")
	rung := placeUnpinnedWorker(
		seatDefaultFor(dispatchtick.WorkerModelFable, dispatchtick.WorkerModelOpus),
		measuredOn(modelroute.ZoneFleet, dispatchtick.WorkerModelFable))
	if rung.Source != modelSourceRung {
		t.Fatalf("precondition: source = %q, want %q", rung.Source, modelSourceRung)
	}
	gated, fired := applyPlacementGate(rung, dispatchtick.ShapeChurning)
	if !fired {
		t.Fatal("the ladder's own pin bypassed the preventive shape gate")
	}
	if gated.Model != dispatchtick.WorkerModelOpus || gated.Source != modelSourcePlacement {
		t.Errorf("re-route = %q (source %q), want %q via %q", gated.Model, gated.Source, dispatchtick.WorkerModelOpus, modelSourcePlacement)
	}
}

// A placement is not a launch. Turning the ladder on says "start placing"; it does not say
// which endpoints this fleet's seats can actually dial, and WorkerLaunch carries only a model
// id — no base URL, no credential. So an operator who has declared nothing gets no placement,
// and the refusal names itself. A default of "everything is reachable" would read as a
// boundary while enforcing nothing, which is worse than having none.
func TestTheLadderDialsNothingUntilAnOperatorNamesWhatItCanReach(t *testing.T) {
	for _, undeclared := range []string{"", "   ", ",", " , ,"} {
		root := rungRoot(t)
		withRungRoster(t, root)
		withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))
		t.Setenv(dispatchRungAccountsEnv, undeclared)

		before := seatDefaultFor("qwen3.6-4b", "opus-5")
		after, skip := applyRungPlacement(root, routineTierLabels, before)
		if skip != rungSkipNoReachDecl {
			t.Errorf("%q: skip = %q, want %q", undeclared, skip, rungSkipNoReachDecl)
		}
		if !reflect.DeepEqual(after, before) {
			t.Errorf("%q: an undeclared reach still placed %+v", undeclared, after)
		}
	}

	// Unset is the case an operator reaches by doing nothing, and it is the one that must
	// not place. t.Setenv registers the restore, so unsetting here is safe.
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))
	os.Unsetenv(dispatchRungAccountsEnv)
	if _, skip := applyRungPlacement(root, routineTierLabels, seatDefaultFor("qwen3.6-4b", "opus-5")); skip != rungSkipNoReachDecl {
		t.Errorf("an unset declaration: skip = %q, want %q", skip, rungSkipNoReachDecl)
	}

	// And it is reported ahead of the missing roster and the missing journal. With several
	// things absent at once, "you have not said what this fleet can dial" is the one the
	// operator has to answer anyway — the other two are files a running fleet produces.
	bare := rungRoot(t)
	os.Unsetenv(dispatchRungAccountsEnv)
	if _, skip := applyRungPlacement(bare, routineTierLabels, seatDefaultFor("qwen3.6-4b", "opus-5")); skip != rungSkipNoReachDecl {
		t.Errorf("with nothing configured at all: skip = %q, want %q", skip, rungSkipNoReachDecl)
	}
}

// Narrowing what the fleet can dial costs it the unreachable RUNGS, not the ladder. With the
// device endpoint undeclared and both models graded, the work lands on the fleet rung — still
// self-hosted, which is the epic's actual objective — rather than falling back to the seat's
// vendor default because the cheapest rung happened to be unreachable.
func TestANarrowedReachFallsToTheNextDialableRungNotTheSeatDefault(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, append(
		rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness),
		rungTurns("glm-5.2", 24, time.Minute, modelroute.VerifyWitness)...))
	t.Setenv(dispatchRungAccountsEnv, "cluster")

	after, skip := applyRungPlacement(root, routineTierLabels, seatDefaultFor("qwen3.6-4b", "opus-5"))
	if skip != "" {
		t.Fatalf("a reachable graded rung was refused: %s", skip)
	}
	if after.Model != "glm-5.2" {
		t.Errorf("model = %q, want the fleet-rung model — the device rung was not declared dialable", after.Model)
	}
	if after.Source != modelSourceRung {
		t.Errorf("source = %q, want %q", after.Source, modelSourceRung)
	}
}

// The invariant that makes filtering the pool safe at all: narrowing the reachable set must
// never make a WEAKER model win. Place fixes its top rung from the static zone ladder rather
// than from the pool, so an unmeasured candidate stays barred by rule 2 even when every rung
// above it has been filtered away, and rule 1's tier floor is a property of the work.
func TestNarrowingReachNeverPromotesAWeakerModel(t *testing.T) {
	roster, err := modelroute.ParseRoster([]byte(placementEvidenceRoster))
	if err != nil {
		t.Fatal(err)
	}
	// Graded for ROUTINE work only, which is what a real corpus of small local turns looks
	// like. Nothing here says either model can hold a security release.
	evidence := map[string][]modelroute.ClassEvidence{
		"qwen3.6-4b": {{Class: modelroute.ClassRoutine, Attempts: 40, Successes: 40, Verify: modelroute.VerifyWitness}},
		"glm-5.2":    {{Class: modelroute.ClassRoutine, Attempts: 40, Successes: 40, Verify: modelroute.VerifyWitness}},
	}
	onBox := rungReach{Accounts: map[string]bool{"laptop": true, "cluster": true}}

	// The vendor rung is filtered away, so nothing is left that can hold the class. The
	// answer is a refusal, never "the best rung still standing".
	rung, skip := resolveRungPlacement(roster, modelroute.ClassSecurityRelease, evidence, onBox)
	if rung != nil {
		t.Fatalf("filtering the vendor rung away placed security work on %s (%s)", rung.Model, rung.Zone)
	}
	if skip == "" {
		t.Fatal("a refusal reported no reason")
	}

	// And the same narrowing does not disturb the class those models ARE graded for.
	routine, skip := resolveRungPlacement(roster, modelroute.ClassRoutine, evidence, onBox)
	if skip != "" || routine == nil || routine.Model != "qwen3.6-4b" {
		t.Errorf("routine placement under the same narrowing: skip=%q rung=%+v", skip, routine)
	}
}

// The gate applies to all three rungs by the same rule. The vendor rung is the tempting one
// to exempt — "the frontier API is always reachable" — and exempting it would quietly make an
// undeclared reach mean "send it to the vendor", which is the exact spend this epic exists to
// avoid and the one failure that costs money rather than a retry.
func TestTheVendorRungIsNotExemptFromTheReachGate(t *testing.T) {
	roster, err := modelroute.ParseRoster([]byte(placementEvidenceRoster))
	if err != nil {
		t.Fatal(err)
	}
	// opus-5 is graded for the hardest class there is, so nothing but the declaration stands
	// between this work and the vendor rung.
	evidence := map[string][]modelroute.ClassEvidence{
		"opus-5": {{Class: modelroute.ClassSecurityRelease, Attempts: 40, Successes: 40, Verify: modelroute.VerifyWitness}},
	}
	onBox := rungReach{Accounts: map[string]bool{"laptop": true, "cluster": true}}

	if rung, skip := resolveRungPlacement(roster, modelroute.ClassSecurityRelease, evidence, onBox); rung != nil {
		t.Errorf("an undeclared vendor account was placed on anyway: %s (%s), skip=%q", rung.Model, rung.Zone, skip)
	}
	// Declared, it places — the gate is about what was declared, not about which rung it is.
	withVendor := rungReach{Accounts: map[string]bool{"laptop": true, "cluster": true, "frontier": true}}
	rung, skip := resolveRungPlacement(roster, modelroute.ClassSecurityRelease, evidence, withVendor)
	if skip != "" || rung == nil || rung.Model != "opus-5" {
		t.Errorf("a declared vendor account did not place: skip=%q rung=%+v", skip, rung)
	}
}

// A declaration that selects nothing is a typo, and it must not look like an idle ladder. An
// operator who wrote "lapotp" has a one-character fix; one reading "no graded evidence yet"
// goes looking for turns that are already on disk.
func TestADeclarationThatSelectsNothingNamesItselfRatherThanLookingIdle(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))

	for _, decl := range []string{"lapotp", "LAPTOP", "laptop2,cluster-2"} {
		t.Setenv(dispatchRungAccountsEnv, decl)
		before := seatDefaultFor("qwen3.6-4b", "opus-5")
		after, skip := applyRungPlacement(root, routineTierLabels, before)
		if skip != rungSkipUnreachable {
			t.Errorf("%q: skip = %q, want %q", decl, skip, rungSkipUnreachable)
		}
		if !reflect.DeepEqual(after, before) {
			t.Errorf("%q: moved the worker to %+v", decl, after)
		}
	}
}

// The whole-roster case is available and must be SPELLED. That keeps it an assertion the
// operator made rather than one the code assumed on their behalf.
func TestTheWholeRosterAssertionIsSpelledNotInferred(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))

	t.Setenv(dispatchRungAccountsEnv, "*")
	after, skip := applyRungPlacement(root, routineTierLabels, seatDefaultFor("qwen3.6-4b", "opus-5"))
	if skip != "" || after.Model != "qwen3.6-4b" {
		t.Errorf("the spelled whole-roster assertion did not place: model=%q skip=%q", after.Model, skip)
	}

	reach, ok := dispatchRungReach()
	if !ok || !reach.All {
		t.Errorf(`"*" parsed to %+v (ok=%v), want the whole-roster assertion`, reach, ok)
	}
	// A model the roster does not bind is still not reachable-by-wildcard, because it never
	// enters the candidate pool at all — "*" widens the ACCOUNTS, not the roster.
	if got := reach.filter(modelroute.Roster{}, nil); len(got) != 0 {
		t.Errorf(`"*" invented %d candidates from an empty roster`, len(got))
	}
}

// A dangling binding is a broken roster, and Place is already fail-loud about it. The reach
// filter must not swallow one: converting a misconfiguration into "nothing was reachable"
// hands the operator the wrong cure.
func TestABrokenRosterStaysFailLoudThroughTheReachFilter(t *testing.T) {
	roster := modelroute.Roster{
		Accounts: []modelroute.Account{{ID: "laptop", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"}},
		Bindings: []modelroute.Binding{{Model: "ghost", Account: "nowhere"}},
		Default:  "laptop",
	}
	rung, skip := resolveRungPlacement(roster, modelroute.ClassRoutine, nil,
		rungReach{Accounts: map[string]bool{"laptop": true}})
	if rung != nil {
		t.Fatalf("a dangling binding placed %+v", rung)
	}
	if skip != rungSkipRefused {
		t.Errorf("skip = %q, want %q — the misconfiguration was reported as an empty reach", skip, rungSkipRefused)
	}
}

// The reason vocabulary is closed and its members are distinct: two paths sharing a string
// would make an operator's diagnosis ambiguous exactly when they need it.
func TestTheSkipVocabularyIsClosedAndDistinct(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range []string{
		rungSkipOutranked, rungSkipNoWorkClass, rungSkipBadWindow, rungSkipNoRoster,
		rungSkipRosterEmpty, rungSkipNoJournal, rungSkipJournalBig, rungSkipJournalBad,
		rungSkipNoEvidence, rungSkipRefused, rungSkipUnmeasured, rungSkipNotApplied,
		rungSkipNoReachDecl, rungSkipUnreachable,
	} {
		if r == "" {
			t.Error("the empty reason is reserved for a pin that happened")
		}
		if seen[r] {
			t.Errorf("duplicate skip reason %q", r)
		}
		seen[r] = true
	}
}
