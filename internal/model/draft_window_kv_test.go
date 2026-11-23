package model

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestDraftWindowKV(t *testing.T) {
	t.Parallel()
	const (
		window      = 8
		blockTokens = 4
		maxStep     = 4
	)

	t.Run("per-layer backend picker", func(t *testing.T) {
		got, err := PickDraftLayerBackends([]int{0, 128, 0, 64}, KVPrecisionFP16)
		if err != nil {
			t.Fatal(err)
		}
		want := []DraftLayerBackend{DraftLayerFullContextMLA, DraftLayerWindowedMLA, DraftLayerFullContextMLA, DraftLayerWindowedMLA}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("PickDraftLayerBackends = %v, want %v", got, want)
		}
		// A windowed layer must not fall back to the full-context path.
		for _, kv := range []KVPrecision{KVPrecisionFP32, KVPrecisionBF16} {
			if got, err := PickDraftLayerBackends([]int{64}, kv); err != nil || got[0] != DraftLayerWindowedMLA {
				t.Fatalf("windowed layer under %q = %v, %v", kv, got, err)
			}
		}
		for _, kv := range []KVPrecision{KVPrecisionQ8_0, KVPrecisionQ4_0, "", "fp8_packed"} {
			if got, err := PickDraftLayerBackends([]int{0, 128}, kv); !errors.Is(err, ErrDraftWindowKVConfig) {
				t.Fatalf("windowed layer under %q picked %v, %v; want refusal", kv, got, err)
			}
			if got, err := PickDraftLayerBackends([]int{0, 0}, kv); err != nil || got[0] != DraftLayerFullContextMLA || got[1] != DraftLayerFullContextMLA {
				t.Fatalf("full-context layers under %q = %v, %v", kv, got, err)
			}
		}
		if _, err := PickDraftLayerBackends([]int{-1}, KVPrecisionFP16); !errors.Is(err, ErrDraftWindowKVConfig) {
			t.Fatalf("negative window admitted: %v", err)
		}
	})

	t.Run("separate budget accounting", func(t *testing.T) {
		plan, err := PlanDraftWindowKV(100, 2, window, blockTokens, maxStep)
		if err != nil {
			t.Fatal(err)
		}
		// ceil((8-1+4)/4)+1 = 4 blocks per sequence, two sequences.
		if plan != (DraftWindowKVPlan{TargetBlocks: 92, DraftBlocks: 8, DraftBlocksPerSeq: 4}) {
			t.Fatalf("plan = %+v", plan)
		}
		if plan.TargetBlocks+plan.DraftBlocks != 100 {
			t.Fatalf("shares overlap or leak: %+v", plan)
		}
		pool, err := NewDraftWindowKVPool(window, blockTokens, plan.DraftBlocks)
		if err != nil {
			t.Fatal(err)
		}
		// Three draft sequences against a group sized for two: the third is
		// refused at the group's own capacity even though the target share has
		// 92 free blocks.
		for seq := 0; seq < 2; seq++ {
			if _, err := pool.Append(seq, 16); err != nil {
				t.Fatalf("sequence %d: %v", seq, err)
			}
		}
		if pool.Used() != 8 || pool.Capacity() != 8 {
			t.Fatalf("used %d of %d, want 8 of 8", pool.Used(), pool.Capacity())
		}
		if _, err := pool.Append(2, 1); !errors.Is(err, ErrDraftWindowKVBudget) {
			t.Fatalf("third sequence admitted past the draft budget: %v", err)
		}
		if pool.Used() != 8 || pool.SeqBlocks(2) != 0 {
			t.Fatalf("refused append changed the pool: used=%d seq2=%d", pool.Used(), pool.SeqBlocks(2))
		}
		if _, _, ok := pool.Retained(2); ok {
			t.Fatal("refused sequence left an entry behind")
		}
		// A group that would starve the target is refused at plan time.
		if _, err := PlanDraftWindowKV(8, 2, window, blockTokens, maxStep); !errors.Is(err, ErrDraftWindowKVBudget) {
			t.Fatalf("plan with no target share admitted: %v", err)
		}
	})

	t.Run("window-bounded eviction", func(t *testing.T) {
		perSeq, err := DraftWindowKVBlocksPerSeq(window, blockTokens, maxStep)
		if err != nil || perSeq != 4 {
			t.Fatalf("blocks per sequence = %d, %v; want 4", perSeq, err)
		}
		pool, err := NewDraftWindowKVPool(window, blockTokens, perSeq)
		if err != nil {
			t.Fatal(err)
		}
		const seq = 7
		end, released := 0, 0
		for step := 0; step < 400; step++ {
			tokens := 1 + step%maxStep
			before := end
			n, err := pool.Append(seq, tokens)
			if err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			released += n
			end += tokens
			if got := pool.SeqBlocks(seq); got > perSeq {
				t.Fatalf("step %d: sequence holds %d blocks, bound is %d", step, got, perSeq)
			}
			start, gotEnd, ok := pool.Retained(seq)
			if !ok || gotEnd != end {
				t.Fatalf("step %d: retained end %d, %v; want %d", step, gotEnd, ok, end)
			}
			// Everything the step's first new position attends to is still
			// held, and nothing older than one block behind that survives.
			oldest := before - (window - 1)
			if oldest < 0 {
				oldest = 0
			}
			if start > oldest {
				t.Fatalf("step %d: retained from %d, but position %d is inside the window", step, start, oldest)
			}
			if oldest-start >= blockTokens {
				t.Fatalf("step %d: retained from %d, a whole block behind window start %d", step, start, oldest)
			}
		}
		// A full-context layer would hold every block of the sequence.
		fullContext := (end + blockTokens - 1) / blockTokens
		if pool.Used() > perSeq || fullContext <= 10*perSeq {
			t.Fatalf("used %d blocks for %d tokens (full context %d)", pool.Used(), end, fullContext)
		}
		if released != pool.Evicted() || released != fullContext-pool.SeqBlocks(seq) {
			t.Fatalf("released %d, pool evicted %d, want %d", released, pool.Evicted(), fullContext-pool.SeqBlocks(seq))
		}
	})

	t.Run("eviction domain is the owning sequence", func(t *testing.T) {
		pool, err := NewDraftWindowKVPool(window, blockTokens, 4)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Append(1, 12); err != nil {
			t.Fatal(err)
		}
		startA, endA, _ := pool.Retained(1)
		// Sequence 2 needs room; sequence 1's in-window blocks are not taken.
		if _, err := pool.Append(2, 8); !errors.Is(err, ErrDraftWindowKVBudget) {
			t.Fatalf("append admitted by evicting a peer: %v", err)
		}
		if s, e, ok := pool.Retained(1); !ok || s != startA || e != endA || pool.SeqBlocks(1) != 3 || pool.Evicted() != 0 {
			t.Fatalf("peer disturbed: retained [%d,%d) ok=%v blocks=%d evicted=%d", s, e, ok, pool.SeqBlocks(1), pool.Evicted())
		}
		if got := pool.Release(1); got != 3 || pool.Used() != 0 {
			t.Fatalf("release freed %d blocks, used now %d", got, pool.Used())
		}
		if _, err := pool.Append(2, 8); err != nil {
			t.Fatalf("append after peer release: %v", err)
		}
		// A refused step on a sequence that already holds blocks, and whose
		// window would have slid, must not apply the slide.
		if _, err := pool.Append(2, 4); err != nil {
			t.Fatal(err)
		}
		if _, err := pool.Append(3, 4); err != nil {
			t.Fatal(err)
		}
		start2, end2, _ := pool.Retained(2)
		blocks2, used, evicted := pool.SeqBlocks(2), pool.Used(), pool.Evicted()
		if _, err := pool.Append(2, 8); !errors.Is(err, ErrDraftWindowKVBudget) {
			t.Fatalf("over-budget step on a held sequence admitted: %v", err)
		}
		if s, e, _ := pool.Retained(2); s != start2 || e != end2 || pool.SeqBlocks(2) != blocks2 || pool.Used() != used || pool.Evicted() != evicted {
			t.Fatalf("refused step moved the sequence: retained [%d,%d) blocks=%d used=%d evicted=%d", s, e, pool.SeqBlocks(2), pool.Used(), pool.Evicted())
		}
		if pool.Release(99) != 0 {
			t.Fatal("releasing an unknown sequence freed blocks")
		}
	})

	t.Run("fail closed", func(t *testing.T) {
		for _, args := range [][3]int{{0, 4, 4}, {8, 0, 4}, {8, 4, 0}} {
			if _, err := NewDraftWindowKVPool(args[0], args[1], args[2]); !errors.Is(err, ErrDraftWindowKVConfig) {
				t.Fatalf("NewDraftWindowKVPool%v admitted: %v", args, err)
			}
			if _, err := DraftWindowKVBlocksPerSeq(args[0], args[1], args[2]); !errors.Is(err, ErrDraftWindowKVConfig) {
				t.Fatalf("DraftWindowKVBlocksPerSeq%v admitted: %v", args, err)
			}
		}
		if _, err := PlanDraftWindowKV(100, 0, window, blockTokens, maxStep); !errors.Is(err, ErrDraftWindowKVConfig) {
			t.Fatalf("plan for zero sequences admitted: %v", err)
		}
		var zero DraftWindowKVPool
		if _, err := zero.Append(0, 1); !errors.Is(err, ErrDraftWindowKVConfig) {
			t.Fatalf("zero-value pool admitted an append: %v", err)
		}
		pool, _ := NewDraftWindowKVPool(window, blockTokens, 4)
		if _, err := pool.Append(0, 0); !errors.Is(err, ErrDraftWindowKVConfig) {
			t.Fatalf("empty append admitted: %v", err)
		}
	})
}
