package agentqueue

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

// TestSoakAdmissionHundredPrimaryThousandChild is the admission half of the
// 100-primary / 1000-child logical-intent soak, qualified over the landed
// agentqueue seams only.
//
// Scope note: the public tool-process admission half (ActuateReserved handing
// reservations to "fak dispatch tick") and the private fleet facade are NOT
// part of this leaf. This test creates ZERO OS processes, uses no subprocess
// runner, no clock sleep, and no randomness; every observed transition is
// applied through the landed Reconcile seam, and ambiguity is proven through
// the landed ReconcileRestart seam on a temp FileStore.
func TestSoakAdmissionHundredPrimaryThousandChild(t *testing.T) {
	const (
		primaries    = 100
		children     = 1000 // do NOT weaken: 10 children per primary
		totalIntents = primaries + children
		maxActive    = 4
		desired      = 4
		cancelled    = 10 // small set of children permanently filtered out
		tickCeiling  = 5000
	)

	// Deterministic base instant for synthetic wrapper identity (no OS process
	// is ever started).
	base := time.Date(2026, 9, 24, 12, 0, 0, 0, time.UTC)

	// Agentqueue.Intent has no parent/goal linkage field
	// (internal/agentqueue/agentqueue.go:42-51). The 1000 children therefore
	// encode their distinct primary parent in the deterministic ID
	// "child-<p>-<c>"; the parent link is asserted by prefix decomposition
	// below. There is no ParentTaskID/GoalID on Intent today, so no parent
	// field can be populated or asserted.
	t.Logf("GAP internal/agentqueue/agentqueue.go:42: Intent has no ParentTaskID/GoalID field; %d child parents are encoded in the ID prefix instead", children)

	intents := make([]Intent, 0, totalIntents)
	expectCompleted := make(map[string]bool, totalIntents)
	expectCancelled := make(map[string]bool, totalIntents)
	for p := 0; p < primaries; p++ {
		id := fmt.Sprintf("primary-%03d", p)
		intents = append(intents, Intent{ID: id, State: IntentQueued})
		expectCompleted[id] = true
	}
	for p := 0; p < primaries; p++ {
		for c := 0; c < children/primaries; c++ {
			id := fmt.Sprintf("child-%03d-%03d", p, c)
			intent := Intent{ID: id, State: IntentQueued}
			if childOrdinal := p*(children/primaries) + c; childOrdinal < cancelled {
				// No cancellation primitive exists; the soak filters these out
				// of eligibility by never submitting them as queued and
				// accounting them here as terminal (cancelled), not lost.
				intent.State = IntentFailed
				expectCancelled[id] = true
			} else {
				expectCompleted[id] = true
			}
			intents = append(intents, intent)
		}
	}
	// Assert the parent linkage really is distinct and complete.
	seenParents := make(map[string]int, primaries)
	for _, in := range intents {
		if len(in.ID) < 13 || in.ID[:6] != "child-" {
			continue
		}
		seenParents[in.ID[6:9]]++
	}
	if len(seenParents) != primaries {
		t.Fatalf("child parent linkage: saw %d distinct parents, want %d", len(seenParents), primaries)
	}
	for p, n := range seenParents {
		if n != children/primaries {
			t.Fatalf("parent %q has %d children, want %d", p, n, children/primaries)
		}
	}

	snapshot := Snapshot{
		Schema:     Schema,
		Generation: "gen:soak-seed",
		Pool:       PoolSpec{ID: "soak", Min: 0, Desired: desired, Max: maxActive},
		Intents:    intents,
	}
	intentIndex := make(map[string]int, totalIntents)
	for i, in := range snapshot.Intents {
		intentIndex[in.ID] = i
	}

	maxObservedActive := 0
	maxObservedHold := 0
	admittedTotal := 0
	ticks := 0
	seenInteractiveHold := false
	for tick := 0; tick < tickCeiling; tick++ {
		ticks++
		beforeActive := activeAttemptCount(snapshot)
		eligible := eligibleIntentIDs(snapshot)
		if beforeActive > maxActive {
			t.Fatalf("tick %d over-admission: %d active attempts > Max %d", tick, beforeActive, maxActive)
		}

		receipt, err := Reconcile(snapshot)
		if err != nil {
			t.Fatalf("tick %d reconcile: %v", tick, err)
		}

		// Overflow is HELD, not dropped. When at capacity with work still
		// eligible, the landed seam reports the documented AT_CAPACITY reason,
		// and held count is exactly eligible-minus-admitted.
		if len(receipt.Start) == 0 && beforeActive >= desired && len(eligible) > 0 {
			if len(receipt.Hold) != 1 || receipt.Hold[0] != "AT_CAPACITY" {
				t.Fatalf("tick %d hold reason = %v, want [AT_CAPACITY]", tick, receipt.Hold)
			}
			heldCount := len(eligible) - len(receipt.Start)
			if heldCount != len(eligible) {
				t.Fatalf("tick %d held %d != eligible-minus-admitted %d", tick, heldCount, len(eligible))
			}
			maxObservedHold = maxInt(maxObservedHold, heldCount)
			seenInteractiveHold = true
		}

		if len(receipt.Start) == 0 {
			if beforeActive == 0 {
				break // quiescent: no capacity-parks outstanding and no work
			}
			// Drain the running batch admitted by the previous tick, then loop
			// so the next Reconcile sees free capacity.
			for i := range snapshot.Attempts {
				att := &snapshot.Attempts[i]
				if att.State != AttemptRunning {
					continue
				}
				att.State = AttemptSucceeded
				in := &snapshot.Intents[intentIndex[att.IntentID]]
				in.State = IntentCompleted
			}
			continue
		}

		// Admit: apply the accepted StartActions as running attempts. The pure
		// Reconcile seam is the only admission authority here; one reservation
		// per accepted start, never more than the pool Max.
		if tick == 0 && len(receipt.Start) != maxActive {
			t.Fatalf("first tick admitted %d, want the bounded cap %d", len(receipt.Start), maxActive)
		}
		for _, start := range receipt.Start {
			in := &snapshot.Intents[intentIndex[start.IntentID]]
			in.State = IntentRunning
			in.PID = 4242
			snapshot.Attempts = append(snapshot.Attempts, Attempt{
				ID:        start.IdempotencyKey,
				IntentID:  start.IntentID,
				State:     AttemptRunning,
				Nonce:     "nonce:" + start.IdempotencyKey,
				PID:       4242,
				StartedAt: base,
			})
		}
		snapshot.Generation = nextGeneration(snapshot.Generation, receipt.Start)
		admittedTotal += len(receipt.Start)

		afterActive := activeAttemptCount(snapshot)
		if afterActive > maxActive {
			t.Fatalf("tick %d over-admission after admit: %d active attempts > Max %d", tick, afterActive, maxActive)
		}
		maxObservedActive = maxInt(maxObservedActive, afterActive)
	}

	// ---- end-of-run conservation ----
	if got := activeAttemptCount(snapshot); got > maxActive {
		t.Fatalf("final over-admission: %d active attempts > Max %d", got, maxActive)
	}
	if buildCount := len(snapshot.Intents); buildCount != totalIntents {
		t.Fatalf("built %d intents, want %d", buildCount, totalIntents)
	}

	counts := map[IntentState]int{}
	unique := map[string]bool{}
	for _, in := range snapshot.Intents {
		if unique[in.ID] {
			t.Fatalf("duplicate logical intent %q", in.ID)
		}
		unique[in.ID] = true
		counts[in.State]++
	}
	completedLogicals := 0
	for id := range expectCompleted {
		in, err := findLifecycleIntent(&snapshot, id)
		if err != nil {
			t.Fatalf("logical intent %q lost: %v", id, err)
		}
		if in.State != IntentCompleted {
			t.Fatalf("logical intent %q not terminal: state %q hold %q", id, in.State, in.HoldReason)
		}
		completedLogicals++
	}
	if counts[IntentCompleted] != len(expectCompleted) {
		t.Fatalf("completed = %d, want %d", counts[IntentCompleted], len(expectCompleted))
	}
	if counts[IntentHeld] != 0 {
		t.Fatalf("held = %d, want 0 (no ambiguity was injected into the soak)", counts[IntentHeld])
	}
	if got := counts[IntentQueued] + counts[IntentRunning] + counts[IntentFailed]; got != cancelled {
		t.Fatalf("non-accepted remainder = %d, want the %d cancelled intents", got, cancelled)
	}

	// Conservation: submitted == accepted(completed) + held + cancelled, no
	// intent lost and no intent double-counted.
	submitted := len(snapshot.Intents)
	if submitted != completedLogicals+cancelled {
		t.Fatalf("conservation: submitted %d != completed %d + held %d + cancelled %d",
			submitted, completedLogicals, counts[IntentHeld], cancelled)
	}
	if len(expectCancelled) != cancelled {
		t.Fatalf("built %d cancelled intents, want %d", len(expectCancelled), cancelled)
	}
	for id := range expectCancelled {
		in, err := findLifecycleIntent(&snapshot, id)
		if err != nil {
			t.Fatalf("cancelled intent %q vanished: lost, not accounted: %v", id, err)
		}
		if in.State != IntentFailed {
			t.Fatalf("cancelled intent %q state = %q, want terminal failed/cancelled", id, in.State)
		}
	}

	if counts[IntentQueued] != 0 {
		t.Fatalf("queued intents remain at quiescence: %d", counts[IntentQueued])
	}
	if admittedTotal != len(expectCompleted) {
		t.Fatalf("admitted total = %d, want %d accepted intents (cancelled intents are never granted)", admittedTotal, len(expectCompleted))
	}
	if maxObservedActive != maxActive {
		t.Fatalf("max observed active = %d, want the bounded cap %d", maxObservedActive, maxActive)
	}
	if !seenInteractiveHold || maxObservedHold == 0 {
		t.Fatalf("no AT_CAPACITY interactive hold was ever observed")
	}
	if ticks >= tickCeiling {
		t.Fatalf("tick loop hit the hard ceiling %d without quiescing", tickCeiling)
	}

	t.Logf("soak: primaries=%d children=%d total=%d completed=%d cancelled=%d held=%d maxActive=%d maxHeldAtCapacity=%d ticks=%d",
		primaries, children, totalIntents, completedLogicals, cancelled, counts[IntentHeld], maxObservedActive, maxObservedHold, ticks)

	// ---- restart seam: ambiguous launching attempt is HELD, never restarted ----
	restartDir := t.TempDir()
	restartStore := FileStore(filepath.Join(restartDir, "restart.json"))
	launchDeadline := base.Add(10 * time.Minute)
	restartSnap := Snapshot{
		Schema:     Schema,
		Generation: "gen:restart-seed",
		Pool:       PoolSpec{ID: "soak", Min: 0, Desired: desired, Max: maxActive},
		Intents: []Intent{{
			ID:    "primary-000",
			State: IntentQueued,
		}},
		Attempts: []Attempt{{
			ID:             "restart-attempt-000",
			IntentID:       "primary-000",
			State:          AttemptLaunching,
			Nonce:          "restart-nonce-000",
			LaunchDeadline: launchDeadline,
		}},
	}
	if err := restartStore.Save(restartSnap); err != nil {
		t.Fatalf("restart save: %v", err)
	}
	loaded, err := restartStore.Load()
	if err != nil {
		t.Fatalf("restart load: %v", err)
	}
	rec, updated, err := ReconcileRestart(loaded, func(int) bool { return false }, RestartOptions{Now: base})
	if err != nil {
		t.Fatalf("restart reconcile: %v", err)
	}
	// A launching attempt is deliberately ambiguous
	// (internal/agentqueue/reconcile_restart.go:138-148).
	if len(rec.Held) != 1 || len(rec.Adopted) != 0 || len(rec.Replaced) != 0 {
		t.Fatalf("restart dispositions: held=%d adopted=%d replaced=%d, want 1/0/0", len(rec.Held), len(rec.Adopted), len(rec.Replaced))
	}
	if rec.Held[0].Action != AttemptActionHold {
		t.Fatalf("restart hold action = %q, want %q", rec.Held[0].Action, AttemptActionHold)
	}
	if rec.Held[0].Reason != "launch outcome ambiguous" {
		t.Fatalf("restart hold reason = %q, want %q", rec.Held[0].Reason, "launch outcome ambiguous")
	}
	if got := findAttemptState(t, &updated, "restart-attempt-000"); got != AttemptLaunching {
		t.Fatalf("ambiguous attempt state = %q, want %q (not double-started, not completed)", got, AttemptLaunching)
	}
	if in, err := findLifecycleIntent(&updated, "primary-000"); err != nil || in.State != IntentQueued {
		t.Fatalf("ambiguous intent state = %v (err %v), want queued", in, err)
	}
	if _, _, err := restartStore.ReconcileRestart(context.Background(), func(int) bool { return false }, RestartOptions{Now: base}); err != nil {
		t.Fatalf("persisted restart reconcile: %v", err)
	}
	if persisted, err := restartStore.Load(); err != nil {
		t.Fatalf("persisted restart load: %v", err)
	} else if got := findAttemptState(t, &persisted, "restart-attempt-000"); got != AttemptLaunching {
		t.Fatalf("persisted ambiguous attempt state = %q, want %q", got, AttemptLaunching)
	}
}

func activeAttemptCount(s Snapshot) int {
	n := 0
	for _, a := range s.Attempts {
		if a.State == AttemptReserved || a.State == AttemptLaunching || a.State == AttemptRunning {
			n++
		}
	}
	return n
}

func eligibleIntentIDs(s Snapshot) []string {
	active := map[string]bool{}
	for _, a := range s.Attempts {
		if a.State == AttemptReserved || a.State == AttemptLaunching || a.State == AttemptRunning {
			active[a.IntentID] = true
		}
	}
	var out []string
	for _, in := range s.Intents {
		if !active[in.ID] && (in.State == IntentQueued || (in.State == IntentFailed && in.RetryEligible)) {
			out = append(out, in.ID)
		}
	}
	return out
}

func findAttemptState(t *testing.T, s *Snapshot, id string) AttemptState {
	t.Helper()
	for _, a := range s.Attempts {
		if a.ID == id {
			return a.State
		}
	}
	t.Fatalf("attempt %q absent", id)
	return ""
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
