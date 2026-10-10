package agent

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestExecutionDeadlineUsesOwnedPrefixAndRechecksAttempts(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	ids := synthIDs(p.m.Cfg.VocabSize, 24, 71)
	run := func(ctx context.Context, tokens []int) ([]int, error) {
		var out []int
		_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctx, tokens, 1, 0, 0, 0, nil, 0, 0, nil, func(id int) bool {
			out = append(out, id)
			return false
		})
		return out, err
	}
	want, err := run(context.Background(), ids)
	if err != nil || len(want) != 1 {
		t.Fatalf("seed completion=%v error=%v", want, err)
	}
	calls := 0
	refused := errors.New("deadline fixture refusal")
	ctx := p.WithExecutionDeadlineAdmission(context.Background(), func(_ context.Context, prompt, cached, output int) error {
		calls++
		if prompt != len(ids) || output != 1 {
			t.Fatalf("admission prompt/output=%d/%d", prompt, output)
		}
		if calls == 1 {
			if cached != len(ids) {
				t.Fatalf("warm admission cached=%d, want %d", cached, len(ids))
			}
			p.mu.Lock()
			removed := p.tree.EvictPrefix(ids)
			p.mu.Unlock()
			if removed == 0 {
				t.Fatal("fixture did not evict the acquired source prefix")
			}
			return nil
		}
		if cached != 0 {
			t.Fatalf("new attempt retained stale warm credit: %d", cached)
		}
		return refused
	})
	got, err := run(ctx, ids)
	if err != nil || !slices.Equal(got, want) {
		t.Fatalf("owned prefix after eviction produced %v, error=%v, want %v", got, err, want)
	}
	different := slices.Clone(ids)
	different[0] = (different[0] + 1) % p.m.Cfg.VocabSize
	got, err = run(ctx, different)
	if !errors.Is(err, refused) || len(got) != 0 || calls != 2 {
		t.Fatalf("second attempt tokens=%v error=%v calls=%d", got, err, calls)
	}
	other := reusePlanner(false, false, tinyCfg())
	if err := other.checkExecutionDeadline(ctx, len(ids), len(ids), 1); err == nil || calls != 2 {
		t.Fatalf("another planner used request credit: error=%v calls=%d", err, calls)
	}
	p.SetSpeculativeEngine(&model.SpeculativeEngine{})
	if p.ExecutionDeadlineAdmissionSupported() {
		t.Fatal("configured speculative planner advertises prefix credit")
	}
	_, err = p.generateReusedRecovering(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
	if !errors.Is(err, refused) || calls != 3 {
		t.Fatalf("changed speculative route bypassed cold verdict: error=%v calls=%d", err, calls)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestExecutionDeadlineExactHitWithoutLogitsCreditsRefeed(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	ids := synthIDs(p.m.Cfg.VocabSize, 12, 93)
	sess := p.m.NewSession()
	sess.Prefill(ids)
	defer sess.Close()
	root, _ := p.tree.Lookup(nil)
	leaf := p.tree.Insert(root, ids, sess.Cache)
	p.tree.Done(leaf)
	refused := errors.New("stop before refeed")
	calls := 0
	ctx := p.WithExecutionDeadlineAdmission(context.Background(), func(_ context.Context, prompt, cached, output int) error {
		calls++
		if prompt != len(ids) || cached != len(ids)-1 || output != 1 {
			t.Fatalf("exact hit without logits credited %d/%d/%d", prompt, cached, output)
		}
		return refused
	})
	_, _, _, _, _, _, _, _, err := p.generateReusedContextWithBias(ctx, ids, 1, 0, 0, 0, nil, 0, 0, nil, nil)
	if !errors.Is(err, refused) || calls != 1 {
		t.Fatalf("admission error=%v calls=%d", err, calls)
	}
}
