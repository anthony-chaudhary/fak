// Rung J1 (issue #1903): the end-to-end dogfood fidelity witness. Every relay rung
// below this one is proven in isolation; nothing yet proved the property the epic
// exists for — that a LONG goal survives MANY legs with progress intact. This file is
// that proof, hermetic by construction (issue "Out of scope: No production fleet run;
// a hermetic multi-leg harness"): a real multi-step goal (land ten numbered work
// items in a durable store, one per leg) is driven through >=10 FORCED rotations of
// the H1 driver loop, and the run asserts the two fidelity properties the done
// condition names:
//
//   - 100% attribution: every commit the goal landed is attributable to exactly one
//     leg (the leg's baton carries it as artifact, tombstone anchor AND progress
//     cursor, and the B4 fidelity scorer resolves every pointer), and every leg
//     closure is attributable through the unbroken ParentTrace lineage chain.
//   - The objective pin is BYTE-identical end to end: the `objective` bytes on every
//     rotation's canonical wire equal the seed pin's canonical bytes exactly.
//
// The witness is `go test ./internal/relay -run DogfoodRotations` plus the checked-in
// evidence artifact testdata/dogfood_rotations_evidence.golden, which the run
// re-derives deterministically (no clock, no I/O beyond the golden read, fixture SHAs
// content-derived) and byte-compares. Regenerate with UPDATE_GOLDEN=1 ONLY when the
// relay's wire behavior intentionally changes.
package relay

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// dogfoodRotationCount is the done condition's floor: the goal must survive at least
// ten forced rotations.
const dogfoodRotationCount = 10

// dogfoodSHA derives the deterministic 40-hex fake commit anchor for one landed work
// item — content-addressed from the item label so the whole run (and its evidence
// artifact) is byte-stable across runs, processes and hosts with no clock and no git.
func dogfoodSHA(label string) string {
	sum := sha256.Sum256([]byte("dogfood-1903-" + label))
	return hex.EncodeToString(sum[:])[:40]
}

// dogfoodStore is the hermetic durable store the goal lands work in: the fake
// commit set the resolver verifies against plus the intent-ledger rows a successor
// re-reads through its cursor (D3). It stands in for git + the run ledger; nothing
// here touches a real store.
type dogfoodStore struct {
	verified map[string]bool // fake commit SHAs (and issue refs) that resolve
	rows     []ProgressStep  // intent-ledger rows, in land order
}

func newDogfoodStore() *dogfoodStore {
	return &dogfoodStore{verified: map[string]bool{
		"#1903":                 true, // the tracking issue resolves as an artifact
		dogfoodSHA("item-base"): true, // the pre-goal base anchor
	}}
}

// land records work item n as landed: its commit becomes resolvable and the ledger
// gains the row a successor's verified-progress read returns.
func (s *dogfoodStore) land(n int) string {
	sha := dogfoodSHA(fmt.Sprintf("item-%d", n))
	s.verified[sha] = true
	s.rows = append(s.rows, ProgressStep{Ref: sha, Note: fmt.Sprintf("work item %d landed", n)})
	return sha
}

// ReadProgress is the D3 LedgerReader: it returns the rows as the ledger records
// them — never a number a closing leg asserted.
func (s *dogfoodStore) ReadProgress(string) ([]ProgressStep, error) {
	return append([]ProgressStep(nil), s.rows...), nil
}

// dogfoodLegEvidence is one rotation's row in the evidence artifact: who closed, who
// succeeded it, what it landed, and the two fidelity verdicts for the leg.
type dogfoodLegEvidence struct {
	Leg              int      `json:"leg"`
	ParentTrace      string   `json:"parent_trace"`
	SuccessorTrace   string   `json:"successor_trace"`
	Boundaries       int      `json:"boundaries"`
	Holds            []string `json:"holds"`
	Reason           string   `json:"reason"`
	AtSHA            string   `json:"at_sha"`
	CommitAttributed string   `json:"commit_attributed"`
	FidelityScore    float64  `json:"fidelity_score"`
	PinByteIdentical bool     `json:"pin_byte_identical"`
}

// dogfoodEvidence is the committed artifact the done condition names: >=10 rotations,
// the attribution tally, and the pin-identity verdict, with the per-leg rows behind
// the headline. No maps anywhere, so its JSON is deterministic.
type dogfoodEvidence struct {
	Issue                      string               `json:"issue"`
	Goal                       string               `json:"goal"`
	RelayID                    string               `json:"relay_id"`
	Rotations                  int                  `json:"rotations"`
	CommitsLanded              int                  `json:"commits_landed"`
	CommitsAttributed          int                  `json:"commits_attributed"`
	ClosuresAttributed         int                  `json:"closures_attributed"`
	AttributionRate            string               `json:"attribution_rate"`
	PinID                      string               `json:"pin_id"`
	PinDigest                  string               `json:"pin_digest"`
	PinWire                    string               `json:"pin_wire"`
	PinByteIdenticalAcrossLegs bool                 `json:"pin_byte_identical_across_legs"`
	FinalReason                string               `json:"final_reason"`
	Legs                       []dogfoodLegEvidence `json:"legs"`
}

// runDogfoodRelay drives the hermetic goal end to end — ten legs, each forced to
// rotate (context usage crosses the soft mark every leg), then the final leg that
// reloads the last baton and ends the relay on its done-check — asserting the DoD
// properties inline and returning the derived evidence artifact.
func runDogfoodRelay(t *testing.T) ([]byte, dogfoodEvidence) {
	t.Helper()

	const (
		relayID   = "RLY-DOGFOOD-1903"
		goal      = "Land ten numbered dogfood work items in the durable store, one per leg, surviving every rotation (#1903)."
		doneWhen  = "all ten dogfood work items resolve in the durable store"
		ledgerRef = ".dos/runs/relay-1903.jsonl"
	)
	seedPin := ctxplan.NewObjectivePin("pin-dogfood-1903", goal, 1)
	seedPinWire, err := json.Marshal(seedPin)
	if err != nil {
		t.Fatalf("marshal seed pin: %v", err)
	}

	store := newDogfoodStore()
	resolver := fakeResolver{verified: store.verified}

	var (
		batons  []Baton
		wires   [][]byte
		legs    []dogfoodLegEvidence
		anchor  = dogfoodSHA("item-base")
		trace   = "dogfood-leg-0"
		carried Baton // zero for the first leg
	)

	for i := 0; i < dogfoodRotationCount; i++ {
		item := i
		var seen Orientation
		script := func(o Orientation, b int) (BoundaryObs, error) {
			if b == 0 {
				seen = o
				// Mid-work: the soft mark has already crossed (the FORCED rotation),
				// but a tool call is in flight and the tree is dirty — the leg must
				// hold, not rotate mid-action.
				return BoundaryObs{
					Usage:        BudgetUsage{Context: AxisUsage{Used: 90, Cap: 100}},
					TurnInFlight: true,
					Tree:         TreeStatus{DirtyPaths: []string{"internal/relay/dogfood_item.go"}},
					NextSteps:    []string{fmt.Sprintf("land work item %d", item)},
					Facts:        []LoadBearingFact{{Label: fmt.Sprintf("work item %d design", item)}},
					AtSHA:        anchor,
				}, nil
			}
			// Safe point: the leg's one unit of work has LANDED in the durable
			// store — commit resolvable, ledger row written — so the boundary is
			// green, externalized, and singly-actionable. The armed rotation fires.
			sha := store.land(item)
			return BoundaryObs{
				Usage:     BudgetUsage{Context: AxisUsage{Used: 92, Cap: 100}},
				NextSteps: []string{fmt.Sprintf("land work item %d", item+1)},
				Facts: []LoadBearingFact{{
					Label:   fmt.Sprintf("work item %d landed", item),
					Backing: Artifact{Kind: string(ArtifactCommit), Ref: sha},
				}},
				AtSHA: sha,
				Artifacts: []Artifact{
					{Kind: string(ArtifactCommit), Ref: sha},
					{Kind: string(ArtifactIssue), Ref: "#1903"},
				},
				DoNotRederive: []string{"memory:dogfood-1903-dead-end"},
			}, nil
		}

		var wire []byte
		cfg := LegConfig{
			Incoming:      carried,
			RelayID:       relayID,
			Objective:     seedPin,
			DoneWhen:      doneWhen,
			LedgerRef:     ledgerRef,
			HeldRegion:    []string{"internal/relay/**"},
			TraceID:       trace,
			Triggers:      ArmTriggers{SoftMark: 0.55},
			MaxBoundaries: 4,
			Resolver:      resolver,
			Ledger:        store,
			DoneCheck:     func(string) (bool, error) { return len(store.rows) >= dogfoodRotationCount, nil },
			Work:          script,
			WriteBaton:    func(w []byte) error { wire = append([]byte(nil), w...); return nil },
			Recontinue:    func(Baton) (string, error) { return fmt.Sprintf("dogfood-leg-%d", item+1), nil },
		}
		out, err := DriveLeg(cfg)
		if err != nil {
			t.Fatalf("leg %d: DriveLeg: %v", i, err)
		}
		if out.Reason != "RELAY_ROTATED" || out.Boundaries != 2 {
			t.Fatalf("leg %d: reason=%q boundaries=%d, want a forced RELAY_ROTATED at boundary 2", i, out.Reason, out.Boundaries)
		}
		// Reload fidelity: leg 0 is the first leg; every successor reloads FRESH
		// (never stale) and reads exactly the ledger rows its predecessors landed.
		if i == 0 && !seen.FirstLeg {
			t.Fatalf("leg 0 orientation = %+v, want FirstLeg", seen)
		}
		if i > 0 {
			if seen.FirstLeg || seen.Stale.Stale {
				t.Fatalf("leg %d orientation = %+v, want a fresh successor reload", i, seen)
			}
			if seen.Progress.Verdict != ProgressVerified || len(seen.Progress.Steps) != i {
				t.Fatalf("leg %d verified progress = %q with %d steps, want %d ledger-verified steps",
					i, seen.Progress.Verdict, len(seen.Progress.Steps), i)
			}
		}

		batons = append(batons, out.Baton)
		wires = append(wires, wire)
		legs = append(legs, dogfoodLegEvidence{
			Leg:            i,
			ParentTrace:    out.Baton.ParentTrace,
			SuccessorTrace: out.SuccessorTrace,
			Boundaries:     out.Boundaries,
			Holds:          out.Holds,
			Reason:         out.Baton.Tombstone.Reason,
			AtSHA:          out.Baton.Tombstone.AtSHA,
		})

		carried, trace, anchor = out.Baton, out.SuccessorTrace, out.Baton.Tombstone.AtSHA
	}

	// The final leg reloads the tenth baton and the relay ends on its done-check —
	// the goal is DONE, not merely rotated ten times.
	finalWorked := false
	final, err := DriveLeg(LegConfig{
		Incoming:      carried,
		TraceID:       trace,
		Triggers:      ArmTriggers{SoftMark: 0.55},
		MaxBoundaries: 1,
		Resolver:      resolver,
		Ledger:        store,
		DoneCheck:     func(string) (bool, error) { return len(store.rows) >= dogfoodRotationCount, nil },
		Work: func(Orientation, int) (BoundaryObs, error) {
			finalWorked = true
			return BoundaryObs{}, nil
		},
		WriteBaton: func([]byte) error { return nil },
		Recontinue: func(Baton) (string, error) { return "", nil },
	})
	if err != nil {
		t.Fatalf("final leg: DriveLeg: %v", err)
	}
	if final.Reason != ReasonGoalDone || finalWorked {
		t.Fatalf("final leg: reason=%q worked=%v, want %s with no further work", final.Reason, finalWorked, ReasonGoalDone)
	}
	if final.Orientation.Progress.Verdict != ProgressVerified || len(final.Orientation.Progress.Steps) != dogfoodRotationCount {
		t.Fatalf("final leg progress = %q with %d steps, want all %d ledger-verified",
			final.Orientation.Progress.Verdict, len(final.Orientation.Progress.Steps), dogfoodRotationCount)
	}

	// Attribution audit — commits. Every landed commit must be claimed by EXACTLY one
	// leg's baton, and that baton must anchor on it three ways (artifact, tombstone,
	// progress cursor). The B4 fidelity scorer must resolve every pointer it carries.
	claims := make(map[string][]int)
	for i, b := range batons {
		for _, a := range b.Artifacts {
			if a.Kind == string(ArtifactCommit) {
				claims[a.Ref] = append(claims[a.Ref], i)
			}
		}
	}
	commitsAttributed := 0
	for i := range batons {
		b := batons[i]
		sha := dogfoodSHA(fmt.Sprintf("item-%d", i))
		if got := claims[sha]; len(got) != 1 || got[0] != i {
			t.Fatalf("commit %s claimed by legs %v, want exactly leg %d", sha, got, i)
		}
		if b.Tombstone.AtSHA != sha || b.ProgressCursor.StartSHA != sha {
			t.Fatalf("leg %d anchors: tombstone=%s cursor=%s, want both %s", i, b.Tombstone.AtSHA, b.ProgressCursor.StartSHA, sha)
		}
		f := ScoreBatonFidelity(b, resolver)
		if f.Score != 1.0 || f.Verified != f.Total || f.Total == 0 {
			t.Fatalf("leg %d fidelity = %+v, want every pointer verified", i, f)
		}
		legs[i].CommitAttributed = sha
		legs[i].FidelityScore = f.Score
		commitsAttributed++
	}
	if len(store.rows) != dogfoodRotationCount {
		t.Fatalf("durable store holds %d landed commits, want %d", len(store.rows), dogfoodRotationCount)
	}

	// Attribution audit — closures. The lineage chain must be unbroken: leg i closed
	// under trace dogfood-leg-i, handed off to dogfood-leg-(i+1), with monotonic leg
	// numbers, the carried identity verbatim, and the C4 dead-end index deduped.
	closuresAttributed := 0
	for i, b := range batons {
		wantParent, wantSucc := fmt.Sprintf("dogfood-leg-%d", i), fmt.Sprintf("dogfood-leg-%d", i+1)
		if b.Leg != i || b.ParentTrace != wantParent || legs[i].SuccessorTrace != wantSucc {
			t.Fatalf("leg %d lineage: leg=%d parent=%q successor=%q, want %d %q %q",
				i, b.Leg, b.ParentTrace, legs[i].SuccessorTrace, i, wantParent, wantSucc)
		}
		if b.RelayID != relayID || b.DoneWhen != doneWhen || b.Tombstone.Reason != "RELAY_ROTATED" {
			t.Fatalf("leg %d identity/closure: relay=%q done_when=%q reason=%q, want carried verbatim under RELAY_ROTATED",
				i, b.RelayID, b.DoneWhen, b.Tombstone.Reason)
		}
		if want := []string{"memory:dogfood-1903-dead-end"}; i > 0 && !reflect.DeepEqual(b.DoNotRederive, want) {
			t.Fatalf("leg %d do_not_rederive = %v, want the rediscovered dead end deduped to %v", i, b.DoNotRederive, want)
		}
		closuresAttributed++
	}

	// Pin fidelity: the `objective` bytes on every rotation's canonical wire are
	// byte-identical to the seed pin's canonical bytes — end to end, all ten legs.
	pinIdentical := true
	for i, w := range wires {
		var raw struct {
			Objective json.RawMessage `json:"objective"`
		}
		if err := json.Unmarshal(w, &raw); err != nil {
			t.Fatalf("leg %d: extract objective from wire: %v", i, err)
		}
		same := bytes.Equal(raw.Objective, seedPinWire)
		if !same {
			pinIdentical = false
			t.Errorf("leg %d objective wire drifted:\n got %s\nwant %s", i, raw.Objective, seedPinWire)
		}
		legs[i].PinByteIdentical = same
		if !batons[i].Objective.Verify() || !reflect.DeepEqual(batons[i].Objective, seedPin) {
			t.Fatalf("leg %d objective pin failed Verify or drifted: %+v", i, batons[i].Objective)
		}
	}

	ev := dogfoodEvidence{
		Issue:                      "#1903",
		Goal:                       goal,
		RelayID:                    relayID,
		Rotations:                  len(batons),
		CommitsLanded:              len(store.rows),
		CommitsAttributed:          commitsAttributed,
		ClosuresAttributed:         closuresAttributed,
		AttributionRate:            fmt.Sprintf("%d/%d (%.0f%%)", commitsAttributed, len(store.rows), 100*float64(commitsAttributed)/float64(len(store.rows))),
		PinID:                      seedPin.PinID,
		PinDigest:                  seedPin.Digest,
		PinWire:                    string(seedPinWire),
		PinByteIdenticalAcrossLegs: pinIdentical,
		FinalReason:                final.Reason,
		Legs:                       legs,
	}
	got, err := json.MarshalIndent(ev, "", "  ")
	if err != nil {
		t.Fatalf("marshal evidence: %v", err)
	}
	return append(got, '\n'), ev
}

// TestDogfoodRotationsFidelity is the J1 done-condition witness (issue #1903): the
// hermetic goal survives >=10 forced rotations with 100% attribution and a
// byte-identical objective pin, and the derived evidence matches the committed
// artifact byte for byte.
func TestDogfoodRotationsFidelity(t *testing.T) {
	got, ev := runDogfoodRelay(t)

	// The DoD headline, asserted on the evidence itself so the artifact can never
	// pin a run that fell short.
	if ev.Rotations < dogfoodRotationCount {
		t.Fatalf("rotations = %d, want >= %d", ev.Rotations, dogfoodRotationCount)
	}
	if ev.CommitsAttributed != ev.CommitsLanded || ev.ClosuresAttributed != ev.Rotations {
		t.Fatalf("attribution: commits %d/%d closures %d/%d, want 100%%",
			ev.CommitsAttributed, ev.CommitsLanded, ev.ClosuresAttributed, ev.Rotations)
	}
	if !ev.PinByteIdenticalAcrossLegs {
		t.Fatal("objective pin drifted across legs; the DoD requires byte-identity end to end")
	}
	if ev.FinalReason != ReasonGoalDone {
		t.Fatalf("final reason = %q, want %s", ev.FinalReason, ReasonGoalDone)
	}

	// Golden: the committed evidence artifact. Regenerate with UPDATE_GOLDEN=1.
	golden := filepath.Join("testdata", "dogfood_rotations_evidence.golden")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, got, 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read evidence artifact (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("dogfood evidence drifted from the committed artifact %s — the relay's wire, "+
			"lineage or attribution behavior changed; re-run with UPDATE_GOLDEN=1 ONLY if intended:\n got: %s\nwant: %s",
			golden, got, want)
	}
}

// TestDogfoodRotationsDeterministic re-drives the whole relay twice and asserts the
// derived evidence is bit-identical — the property that makes the committed artifact
// a witness rather than a snapshot of one lucky run.
func TestDogfoodRotationsDeterministic(t *testing.T) {
	a, _ := runDogfoodRelay(t)
	b, _ := runDogfoodRelay(t)
	if !bytes.Equal(a, b) {
		t.Errorf("two hermetic dogfood runs disagree:\n first: %s\nsecond: %s", a, b)
	}
}
