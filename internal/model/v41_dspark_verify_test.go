package model

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime medium est=8s lane=default
func TestV41DSparkExecuteGreedyRound(t *testing.T) {
	ctx := context.Background()
	m, w, shard := v41DSparkCommitFixture(t)
	// Unequal allocations are fixed externally; the executor does not optimize
	// them or turn a stable serving lane into a speculative batch lane.
	for _, q := range []int{1, 3} {
		f, oracle := v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)
		proposal, err := f.Propose(ctx, 2)
		if err != nil {
			t.Fatal(err)
		}
		logits, _, err := m.forwardV41Step(proposal.Anchor, oracle.state, nil)
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
			logits, _, err = m.forwardV41Step(best, oracle.state, nil)
			if err != nil {
				t.Fatal(err)
			}
			keep++
		}
		if q == 3 && keep != 1 {
			t.Fatalf("fixture must exercise first rejection, retained %d", keep)
		}
		got, err := f.ExecuteGreedyRound(ctx, 2, q)
		if err != nil {
			t.Fatal(err)
		}
		if got.Verified != q || got.Retained != keep || !reflect.DeepEqual(got.Proposal, proposal) ||
			!reflect.DeepEqual(got.Logits, logits) || f.invalid || f.state.activeStepRound != nil || f.state.dsparkCapture != nil ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(oracle.state)) {
			t.Fatalf("q=%d: allocated greedy execution differs from independent serial target", q)
		}
		replay, capture := v41DSparkCommitCold(t, m, w, append([]int(nil), oracle.state.history...))
		if err := replay.BootstrapCapturedTail(ctx, capture); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(f.ring, replay.ring) || f.next != replay.next || f.prefix != replay.prefix {
			t.Fatal("retained ring differs from independent cold replay")
		}
	}

	// Tied zero target and Markov heads make draft and target choose token zero.
	// This witnesses full acceptance and the lowest-index tie without replacing
	// proposals or target logits with an acceptance mock.
	clear(m.tensor("lm_head.weight"))
	markov, err := shard.Reference("mtp.2.markov_head.head.weight")
	if err != nil {
		t.Fatal(err)
	}
	packed := shard.packed.file.r
	defer func() { shard.packed.file.r = packed }()
	shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
		n, err := packed.ReadAt(p, off)
		for i := range p[:n] {
			if off+int64(i) >= markov.Start && off+int64(i) < markov.End {
				p[i] = 0
			}
		}
		return n, err
	})
	f, oracle := v41DSparkCommitBoot(t, m, w), v41DSparkCommitBoot(t, m, w)
	got, err := f.ExecuteGreedyRound(ctx, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []int{2, 0, 0} {
		if _, _, err := m.forwardV41Step(id, oracle.state, nil); err != nil {
			t.Fatal(err)
		}
	}
	if got.Verified != 3 || got.Retained != 3 || got.Proposal.Proposed[0] != 0 || got.Proposal.Proposed[1] != 0 ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(oracle.state)) {
		t.Fatal("full acceptance differs from serial greedy target")
	}
	replay, capture := v41DSparkCommitCold(t, m, w, append([]int(nil), oracle.state.history...))
	if err := replay.BootstrapCapturedTail(ctx, capture); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(f.ring, replay.ring) || f.next != replay.next || f.prefix != replay.prefix {
		t.Fatal("fully retained ring differs from cold replay")
	}

	ref, err := shard.Reference("mtp.2.attn.wkv.weight")
	if err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"cancel", "projection-error"} {
		t.Run(mode, func(t *testing.T) {
			before, siblingBefore := captureV41ForwardSnapshot(f.state), captureV41ForwardSnapshot(oracle.state)
			ring, next, prefix := v41DSparkRingCopy(f), f.next, f.prefix
			siblingRing := v41DSparkRingCopy(oracle)
			cancelled, cancel := context.WithCancel(ctx)
			defer cancel()
			original := shard.packed.file.r
			defer func() { shard.packed.file.r = original }()
			fault := errors.New("retained draft projection")
			executed := 0
			shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
				if off == ref.Start && f.state.activeStepRound != nil {
					executed = f.state.activeStepRound.committedCount()
					if mode == "cancel" {
						cancel()
					} else {
						return 0, fault
					}
				}
				return original.ReadAt(p, off)
			})
			got, err := f.ExecuteGreedyRound(cancelled, 2, 3)
			cause := fault
			if mode == "cancel" {
				cause = context.Canceled
			}
			if got != nil || !errors.Is(err, cause) || executed != 3 {
				t.Fatalf("late failure lost cause or allocation: %v, executed %d", err, executed)
			}
			if f.invalid || f.state.activeStepRound != nil || f.state.dsparkCapture != nil ||
				!reflect.DeepEqual(before, captureV41ForwardSnapshot(f.state)) || !reflect.DeepEqual(ring, v41DSparkRingCopy(f)) ||
				f.next != next || f.prefix != prefix || !reflect.DeepEqual(siblingBefore, captureV41ForwardSnapshot(oracle.state)) ||
				!reflect.DeepEqual(siblingRing, v41DSparkRingCopy(oracle)) {
				t.Fatal("failed round leaked state or changed sibling lane")
			}
		})
	}
}
