package model

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime medium est=8s lane=default
func TestV41DSparkAllocatedReferenceRound(t *testing.T) {
	ctx := context.Background()
	m, w, _ := v41DSparkCommitFixture(t)
	lanes := []*v41DSparkReference{v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)}
	oracles := []*v41DSparkReference{v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)}

	// The budget is fixed before the previous proposal exists. This witnesses
	// the allocation boundary, not DeepInfer's measured N-2 budget optimizer.
	const totalQ = 4
	prior, err := lanes[1].Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	previous := [][]float32{append([]float32(nil), prior.ConditionalConfidence[:]...)}
	product := float32(1)
	for _, p := range previous[0] {
		product *= p
		if !(product > 0) {
			t.Fatal("fixture must distinguish prior confidence from a fresh lane")
		}
	}
	// Advance one real committed round so these scores are genuinely older
	// than the proposals whose target verification is about to be allocated.
	for _, f := range append(append([]*v41DSparkReference(nil), lanes...), oracles...) {
		r, c := v41DSparkCommitRound(t, f, []int{1})
		if err := f.FinishCapturedRound(ctx, c, r, 1); err != nil {
			t.Fatal(err)
		}
	}
	remap := []int{-1, 0}
	allocation, err := AllocateSpeculativeVerificationBudget(previous, remap, 5, totalQ)
	if err != nil || !reflect.DeepEqual(allocation, []int{1, 3}) {
		t.Fatalf("fresh/prior allocation: %v, %v", allocation, err)
	}
	swapped, err := AllocateSpeculativeVerificationBudget(previous, []int{0, -1}, 5, totalQ)
	if err != nil || !reflect.DeepEqual(swapped, []int{3, 1}) {
		t.Fatalf("request remapping lost: %v, %v", swapped, err)
	}
	if !reflect.DeepEqual(previous[0], prior.ConditionalConfidence[:]) {
		t.Fatal("allocator mutated the prior proposal snapshot")
	}

	executed := 0
	for lane, f := range lanes {
		proposal, err := f.Propose(ctx, 2)
		if err != nil {
			t.Fatal(err)
		}
		q := allocation[lane]
		ids := append([]int{proposal.Anchor}, proposal.Proposed[:q-1]...)
		r, c := v41DSparkCommitRound(t, f, ids)
		if r.committedCount() != q || len(c.rows) != q || len(f.state.history)-r.baseHistoryLen != q {
			t.Fatalf("lane %d executed outside allocated verification prefix %d", lane, q)
		}
		executed += r.committedCount()

		// Independent serial target oracle: compare the preceding target logits
		// before consuming a draft. Strict > preserves the lowest-index tie.
		g := oracles[lane]
		logits, _, err := m.forwardV41Step(proposal.Anchor, g.state, nil)
		if err != nil {
			t.Fatal(err)
		}
		keep := 1
		for keep < q {
			best := 0
			for token := 1; token < len(logits); token++ {
				if logits[token] > logits[best] {
					best = token
				}
			}
			if best != proposal.Proposed[keep-1] {
				break
			}
			logits, _, err = m.forwardV41Step(best, g.state, nil)
			if err != nil {
				t.Fatal(err)
			}
			keep++
		}
		if err := f.FinishCapturedRound(ctx, c, r, keep); err != nil {
			t.Fatal(err)
		}
		if !r.closed || !c.closed || f.invalid || f.state.activeStepRound != nil ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(g.state)) {
			t.Fatal("allocated verification differs from serial greedy retained prefix")
		}
		// Recompute draft features independently from accepted token history;
		// never seed the oracle ring with rows copied out of the tested capture.
		replay, cold := v41DSparkCommitCold(t, m, w, append([]int(nil), g.state.history...))
		if err := replay.BootstrapCapturedTail(ctx, cold); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(f.ring, replay.ring) || f.next != replay.next || f.prefix != replay.prefix {
			t.Fatal("allocated retained draft features differ from independent cold replay")
		}
		next, err := f.Propose(ctx, 1)
		if err != nil {
			t.Fatal(err)
		}
		want, err := replay.Propose(ctx, 1)
		if err != nil || !reflect.DeepEqual(next, want) {
			t.Fatalf("next proposal after allocated verification differs: %v", err)
		}
	}
	if executed != totalQ {
		t.Fatalf("executed %d target inputs, fixed budget %d", executed, totalQ)
	}

	// Cancellation belongs to this request, not the batch or its sibling.
	f, sibling := lanes[1], lanes[0]
	before, siblingBefore := captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(sibling.state)
	ring, next, prefix := v41DSparkRingCopy(f), f.next, f.prefix
	siblingRing := v41DSparkRingCopy(sibling)
	proposal, err := f.Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	ids := append([]int{proposal.Anchor}, proposal.Proposed[:allocation[1]-1]...)
	r, c := v41DSparkCommitRound(t, f, ids)
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := f.FinishCapturedRound(cancelled, c, r, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancel cause lost: %v", err)
	}
	if !r.closed || r.poisoned || !c.closed || f.invalid || f.state.activeStepRound != nil || f.state.dsparkCapture != nil ||
		!reflect.DeepEqual(before, captureV41ForwardSnapshot(f.state)) || !reflect.DeepEqual(ring, v41DSparkRingCopy(f)) ||
		f.next != next || f.prefix != prefix || !reflect.DeepEqual(siblingBefore, captureV41ForwardSnapshot(sibling.state)) ||
		!reflect.DeepEqual(siblingRing, v41DSparkRingCopy(sibling)) {
		t.Fatal("cancelled allocated round leaked state or touched another request")
	}
	if _, err := f.Propose(ctx, 1); err != nil {
		t.Fatalf("clean rollback retired request: %v", err)
	}
}
