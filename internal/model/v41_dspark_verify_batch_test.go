package model

import (
	"context"
	"errors"
	"reflect"
	"sync"
	"testing"
)

// fak-test:runtime medium est=12s lane=default
func TestV41DSparkGreedyBatchCohort(t *testing.T) {
	ctx := context.Background()
	m, w, shard := v41DSparkCommitFixture(t)
	m.Cfg.SuppressTokens = []int{1}
	refs := []*v41DSparkReference{v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)}
	oracles := []*v41DSparkReference{v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)}
	const totalQ = 4
	prior, err := refs[1].Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	previous := []v41DSparkPreviousLane{{ID: "survivor", Confidence: prior.ConditionalConfidence}}
	for _, f := range append(append([]*v41DSparkReference(nil), refs...), oracles...) {
		r, c := v41DSparkCommitRound(t, f, []int{1})
		if err := f.FinishCapturedRound(ctx, c, r, 1); err != nil {
			t.Fatal(err)
		}
	}
	lanes := []v41DSparkBatchLane{{ID: "fresh", Reference: refs[0], Anchor: 2}, {ID: "survivor", Reference: refs[1], Anchor: 2}}
	var owner sync.Mutex
	var receipt v41DSparkBatchReceipt
	// This callback observes the genuine cohort execution boundary: before the
	// first head commits, the other lane has completed all its layers but still
	// owns its pending token. A loop of serial rounds cannot satisfy this.
	cohortObserved := false
	binding := refs[0].state.headNorm
	refs[0].state.setFinalNorm(func(input []float32) ([]float32, error) {
		a, b := refs[0].state.activeStepRound, refs[1].state.activeStepRound
		if a != nil && b != nil && a.pending != nil && b.pending != nil &&
			b.pending.tx != nil && b.pending.tx.nextLayer == m.Cfg.NumLayers &&
			refs[0].target.backendExecutionScope != nil && refs[0].target.backendExecutionScope == refs[1].target.backendExecutionScope {
			cohortObserved = true
		}
		if binding.normFunc() != nil {
			return binding.normFunc()(input)
		}
		return rmsnorm(input, m.tensor("model.norm.weight"), float32(m.Cfg.RMSNormEps)), nil
	})
	if err := executeV41DSparkGreedyBatch(ctx, &owner, lanes, previous, totalQ, &receipt); err != nil {
		t.Fatal(err)
	}
	refs[0].state.headNorm = binding
	if !cohortObserved || !reflect.DeepEqual(receipt.WaveWidths, []int{2, 1, 1}) {
		t.Fatalf("not a position-wave cohort: %+v, observed=%v", receipt.WaveWidths, cohortObserved)
	}
	for i, f := range refs {
		got := receipt.Lanes[i]
		q := []int{1, 3}[i]
		if got.ID != lanes[i].ID || got.Allocated != q || got.Verified != q || !got.Published || got.Round == nil {
			t.Fatalf("lane receipt: %+v", got)
		}
		proposal, err := oracles[i].Propose(ctx, lanes[i].Anchor)
		if err != nil {
			t.Fatal(err)
		}
		g := oracles[i]
		logits, _, err := m.forwardV41Step(lanes[i].Anchor, g.state, nil)
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
		if q == 3 && keep != 1 {
			t.Fatalf("fixture must reject first draft, retained %d", keep)
		}
		if got.Round.Retained != keep || !reflect.DeepEqual(got.Round.Proposal, proposal) || !reflect.DeepEqual(got.Round.Logits, logits) ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(g.state)) {
			t.Fatalf("lane %s differs from independent serial greedy target", got.ID)
		}
		replay, capture := v41DSparkCommitCold(t, m, w, append([]int(nil), g.state.history...))
		if err := replay.BootstrapCapturedTail(ctx, capture); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(f.ring, replay.ring) || f.next != replay.next || f.prefix != replay.prefix {
			t.Fatal("retained draft ring differs from cold replay")
		}
	}

	projection, err := shard.Reference("mtp.2.attn.wkv.weight")
	if err != nil {
		t.Fatal(err)
	}
	// Reorder current lanes: the previous confidence follows the stable ID,
	// not the current index. Failure before the first publication rolls back
	// both; cancellation preparing the second leaves the first receipt visible.
	lanes[0], lanes[1] = lanes[1], lanes[0]
	for _, mode := range []string{"rollback", "partial-publication", "partial-panic"} {
		t.Run(mode, func(t *testing.T) {
			failAt := 1
			if mode == "rollback" {
				failAt = 0
			}
			before := []*v41ForwardSnapshot{captureV41ForwardSnapshot(lanes[0].Reference.state), captureV41ForwardSnapshot(lanes[1].Reference.state)}
			rings := [2][3][][]float32{v41DSparkRingCopy(lanes[0].Reference), v41DSparkRingCopy(lanes[1].Reference)}
			next := []int{lanes[0].Reference.next, lanes[1].Reference.next}
			prefix := [][32]byte{lanes[0].Reference.prefix, lanes[1].Reference.prefix}
			cancelled, cancel := context.WithCancel(ctx)
			defer cancel()
			original := shard.packed.file.r
			defer func() { shard.packed.file.r = original }()
			fault := errors.New("batch retained draft projection")
			executed := 0
			shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
				f := lanes[failAt].Reference
				if off == projection.Start && f.state.activeStepRound != nil &&
					(failAt == 0 || receipt.Lanes[0].Published) {
					executed = receipt.Lanes[0].Verified + receipt.Lanes[1].Verified
					if failAt == 0 {
						return 0, fault
					}
					if mode == "partial-panic" {
						panic(fault)
					}
					cancel()
				}
				return original.ReadAt(p, off)
			})
			var caught any
			err := func() error {
				defer func() { caught = recover() }()
				return executeV41DSparkGreedyBatch(cancelled, &owner, lanes, previous, totalQ, &receipt)
			}()
			cause := fault
			if failAt == 1 {
				cause = context.Canceled
			}
			if (mode == "partial-panic" && caught != fault) || (mode != "partial-panic" && (caught != nil || !errors.Is(err, cause))) || executed != totalQ || receipt.Lanes[0].Allocated != 3 || receipt.Lanes[1].Allocated != 1 {
				t.Fatalf("cause, budget, or ID remap lost: %v %+v", err, receipt)
			}
			for i, lane := range lanes {
				f := lane.Reference
				published := i < failAt
				if receipt.Lanes[i].Published != published || (receipt.Lanes[i].Round != nil) != published || f.invalid != (mode == "partial-panic" && !published) ||
					f.state.activeStepRound != nil || f.state.dsparkCapture != nil || f.target.backendExecutionScope != nil {
					t.Fatalf("lane %s drain/publication mismatch", lane.ID)
				}
				if !published && (!reflect.DeepEqual(before[i], captureV41ForwardSnapshot(f.state)) || !reflect.DeepEqual(rings[i], v41DSparkRingCopy(f)) || f.next != next[i] || f.prefix != prefix[i]) {
					t.Fatalf("unpublished lane %s changed", lane.ID)
				}
				if published && f.next != next[i]+receipt.Lanes[i].Round.Retained {
					t.Fatal("published prefix hidden by later failure")
				}
			}
			if !owner.TryLock() {
				t.Fatal("batch owner leaked")
			}
			owner.Unlock()
		})
	}
}
