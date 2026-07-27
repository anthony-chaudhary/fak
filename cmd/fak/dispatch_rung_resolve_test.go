package main

import (
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
func rungRoot(t *testing.T) string {
	t.Helper()
	t.Setenv("FLEET_DISPATCH_RUNG_PLACEMENT", "on")
	t.Setenv("FLEET_DISPATCH_ACCOUNTS", "")
	t.Setenv(dispatchRungWindowEnv, "")
	return t.TempDir()
}

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
	for _, off := range []string{"", "0", "off", "false", "no", "disable", "disabled", " OFF "} {
		t.Setenv("FLEET_DISPATCH_RUNG_PLACEMENT", off)
		if dispatchRungPlacementEnabled() {
			t.Errorf("%q enabled the ladder", off)
		}
	}
	for _, on := range []string{"1", "on", "true", "yes", "enabled"} {
		t.Setenv("FLEET_DISPATCH_RUNG_PLACEMENT", on)
		if !dispatchRungPlacementEnabled() {
			t.Errorf("%q did not enable the ladder", on)
		}
	}
	os.Unsetenv("FLEET_DISPATCH_RUNG_PLACEMENT")
	if dispatchRungPlacementEnabled() {
		t.Error("an unset knob enabled the ladder")
	}
}

// With the seam off the ladder returns no reason at all, so the caller writes no payload key
// and an unconfigured fleet's tick is byte-identical to before this existed. Every OTHER
// silence is a decision and names itself; this one is the absence of a decision.
func TestAnUnconfiguredTickIsByteIdenticalToBeforeTheLadder(t *testing.T) {
	root := rungRoot(t)
	withRungRoster(t, root)
	withRungJournal(t, root, rungTurns("qwen3.6-4b", 24, time.Minute, modelroute.VerifyWitness))
	t.Setenv("FLEET_DISPATCH_RUNG_PLACEMENT", "off")

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
	})
	if skip != rungSkipUnmeasured {
		t.Errorf("skip = %q, want %q", skip, rungSkipUnmeasured)
	}
	if rung != nil {
		t.Errorf("an unmeasured placement was handed on: %+v", rung)
	}

	// A roster that binds no models cannot place at all, and says that instead.
	if _, skip := resolveRungPlacement(modelroute.Roster{}, modelroute.ClassRoutine, nil); skip != rungSkipRosterEmpty {
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

// The reason vocabulary is closed and its members are distinct: two paths sharing a string
// would make an operator's diagnosis ambiguous exactly when they need it.
func TestTheSkipVocabularyIsClosedAndDistinct(t *testing.T) {
	seen := map[string]bool{}
	for _, r := range []string{
		rungSkipOutranked, rungSkipNoWorkClass, rungSkipBadWindow, rungSkipNoRoster,
		rungSkipRosterEmpty, rungSkipNoJournal, rungSkipJournalBig, rungSkipJournalBad,
		rungSkipNoEvidence, rungSkipRefused, rungSkipUnmeasured, rungSkipNotApplied,
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
