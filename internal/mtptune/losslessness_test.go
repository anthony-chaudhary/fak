package mtptune

import (
	"errors"
	"testing"
)

// TestSpecDecodeLosslessness proves the three load-bearing properties of the
// speculative accept path: (a) a rejected draft prefix rolls committed KV back
// to the pre-step length and rejected tokens never commit; (b) the accepted
// prefix is bit-exact with base greedy decoding; (c) the anchor-as-first layout
// is refused with the typed sentinel.
func TestSpecDecodeLosslessness(t *testing.T) {
	t.Run("rejected prefix rolls back KV", func(t *testing.T) {
		kv := NewKVCache(7, 8, 9)
		base := kv.Len()

		// Draft positions 0 and 1 match the target; positions 2 and 3 do not.
		// TargetTokens[0] is the bonus/anchor token; TargetTokens[i+1] is the
		// target's own token for Draft[i].
		step := SpecStep{
			Anchor:       9,
			Draft:        []int{10, 11, 99, 98},
			TargetTokens: []int{10, 10, 11, 12, 13},
		}
		res, err := VerifySpecStep(LayoutBonus1N, kv, step)
		if err != nil {
			t.Fatalf("VerifySpecStep refused a valid step: %v", err)
		}
		if res.Accepted != 2 || res.Rejected != 2 {
			t.Fatalf("accepted=%d rejected=%d, want 2 and 2", res.Accepted, res.Rejected)
		}
		// The 1+N contract commits the target's own bonus token plus the
		// accepted prefix; rejected drafts never commit. So committed length is
		// base + 1 (bonus) + accepted, and the rejected suffix is rolled back.
		if kv.Len() != base+1+res.Accepted {
			t.Fatalf("committed KV len=%d, want base(%d)+1+accepted(%d)", kv.Len(), base, res.Accepted)
		}
		committed := kv.Snapshot()
		for _, tok := range committed {
			if tok == 99 || tok == 98 {
				t.Fatalf("rejected draft token reached committed KV: %v", committed)
			}
		}
		// RolledBack names the notional full-panel positions for the rejected
		// suffix: base+1+accepted, base+1+accepted+1, ... These are the positions
		// a naive append-all-then-truncate would have undone; the real undo is
		// KV.Rollback(base), and the committed length is base+len(Emitted).
		wantRolledBack := []int{base + 1 + res.Accepted, base + 2 + res.Accepted}
		if len(res.RolledBack) != len(wantRolledBack) {
			t.Fatalf("RolledBack=%v, want %v", res.RolledBack, wantRolledBack)
		}
		for i := range wantRolledBack {
			if res.RolledBack[i] != wantRolledBack[i] {
				t.Fatalf("RolledBack=%v, want %v", res.RolledBack, wantRolledBack)
			}
		}

		// A step where the very first draft token is rejected commits exactly the
		// target's own correction token: base + 1, never a rejected draft.
		kv2 := NewKVCache(3, 4, 5)
		before := kv2.Len()
		step2 := SpecStep{Anchor: 5, Draft: []int{42, 43}, TargetTokens: []int{6, 77, 78}}
		res2, err := VerifySpecStep(LayoutBonus1N, kv2, step2)
		if err != nil {
			t.Fatalf("VerifySpecStep refused: %v", err)
		}
		if res2.Accepted != 0 || res2.Rejected != 2 {
			t.Fatalf("accepted=%d rejected=%d, want 0 and 2", res2.Accepted, res2.Rejected)
		}
		if kv2.Len() != before+1 {
			t.Fatalf("committed KV len=%d, want base(%d)+1 (target correction only)", kv2.Len(), before)
		}
		if got := kv2.Snapshot(); len(got) == 0 || got[len(got)-1] != 6 {
			t.Fatalf("expected target correction 6 committed, got %v", got)
		}
	})

	t.Run("full acceptance commits the 1+N stream", func(t *testing.T) {
		kv := NewKVCache(7)
		step := SpecStep{Anchor: 7, Draft: []int{10, 11, 12}, TargetTokens: []int{9, 10, 11, 12}}
		res, err := VerifySpecStep(LayoutBonus1N, kv, step)
		if err != nil {
			t.Fatalf("VerifySpecStep: %v", err)
		}
		if res.Accepted != 3 || res.Rejected != 0 {
			t.Fatalf("accepted=%d rejected=%d, want 3 and 0", res.Accepted, res.Rejected)
		}
		got := kv.Snapshot()
		want := []int{7, 9, 10, 11, 12}
		if len(got) != len(want) {
			t.Fatalf("committed=%v want %v", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("committed=%v want %v", got, want)
			}
		}
	})

	t.Run("accepted prefix is bit-exact with base greedy", func(t *testing.T) {
		// Independent base oracle: three logits rows. Base greedy emits the
		// argmax of each row: 2, 0, 1.
		rows := [][]float64{
			{0.1, 0.2, 9.0, 0.0},
			{5.0, 0.1, 0.2, 0.0},
			{0.0, 4.0, 0.1, 0.2},
		}
		base := BaseGreedyDecode(rows)
		want := []int{2, 0, 1}
		if len(base) != len(want) {
			t.Fatalf("oracle len=%d want %d", len(base), len(want))
		}
		for i := range want {
			if base[i] != want[i] {
				t.Fatalf("oracle[%d]=%d want %d", i, base[i], want[i])
			}
		}

		// A perfect draft over the same positions: Draft[i] matches base[i+1],
		// and TargetTokens[i] == base[i] (TargetTokens[0] is the anchor's bonus
		// token, TargetTokens[i+1] is the target's own token for Draft[i]).
		kv := NewKVCache()
		step := SpecStep{
			Anchor:       base[0],
			Draft:        []int{base[1], base[2]},
			TargetTokens: base,
		}
		res, err := VerifySpecStep(LayoutBonus1N, kv, step)
		if err != nil {
			t.Fatalf("VerifySpecStep refused a perfect draft: %v", err)
		}
		got := kv.Snapshot()
		if len(got) != len(base) {
			t.Fatalf("committed stream len=%d want %d: %v", len(got), len(base), got)
		}
		for i := range base {
			if got[i] != base[i] {
				t.Fatalf("committed stream diverges from base at %d: got %v want %v", i, got, base)
			}
		}
		if res.Accepted != len(step.Draft) || res.Rejected != 0 {
			t.Fatalf("perfect draft should accept all: accepted=%d rejected=%d", res.Accepted, res.Rejected)
		}
	})

	t.Run("partial rejection stays bit-exact with base greedy", func(t *testing.T) {
		// The base oracle emits 2, 0, 1 from three rows. Draft the first token
		// wrong and the second right: position 0 is rejected, so the accepted
		// prefix is empty and the stream must still emit the target's own token
		// at that position (the correction), then continue from base.
		rows := [][]float64{
			{0.1, 0.2, 9.0, 0.0},
			{5.0, 0.1, 0.2, 0.0},
			{0.0, 4.0, 0.1, 0.2},
		}
		base := BaseGreedyDecode(rows) // {2, 0, 1}

		kv := NewKVCache()
		// Draft[0]=99 is rejected; Draft[1]=1 matches the target's own token
		// after the correction. TargetTokens mirrors base exactly, so the
		// emitted stream must equal base regardless of the rejected draft.
		step := SpecStep{
			Anchor:       base[0],
			Draft:        []int{99, base[2]},
			TargetTokens: base,
		}
		res, err := VerifySpecStep(LayoutBonus1N, kv, step)
		if err != nil {
			t.Fatalf("VerifySpecStep refused: %v", err)
		}
		if res.Accepted != 0 {
			t.Fatalf("rejected first draft accepted=%d, want 0", res.Accepted)
		}
		if res.Next != base[0] {
			t.Fatalf("correction token=%d, want target's own %d", res.Next, base[0])
		}
		if len(res.RolledBack) != 2 {
			t.Fatalf("RolledBack=%v, want the full rejected suffix", res.RolledBack)
		}
		// Only the target's own correction token is committed: a rejected draft
		// never reaches committed KV.
		got := kv.Snapshot()
		if len(got) != 1 || got[0] != base[0] {
			t.Fatalf("committed stream=%v, want just the target correction [%d]", got, base[0])
		}
		for _, tok := range got {
			if tok == 99 {
				t.Fatalf("rejected draft token 99 reached committed KV: %v", got)
			}
		}
	})

	t.Run("anchor-as-first is refused with typed error", func(t *testing.T) {
		kv := NewKVCache(1, 2)
		step := SpecStep{Anchor: 2, Draft: []int{3}, TargetTokens: []int{2, 3}}
		if _, err := VerifySpecStep(LayoutAnchorFirst, kv, step); !errors.Is(err, ErrSpecDecodeAnchorCorruption) {
			t.Fatalf("VerifySpecStep err=%v, want errors.Is ErrSpecDecodeAnchorCorruption", err)
		}
		if _, err := VerifySpecStep(SpecTokenLayout("bogus"), kv, step); !errors.Is(err, ErrSpecDecodeAnchorCorruption) {
			t.Fatalf("unknown layout err=%v, want ErrSpecDecodeAnchorCorruption", err)
		}
		if err := (SpecDecodeConfig{Enabled: true, DraftLen: 4, Layout: LayoutAnchorFirst}).Validate(); !errors.Is(err, ErrSpecDecodeAnchorCorruption) {
			t.Fatalf("config Validate err=%v, want ErrSpecDecodeAnchorCorruption", err)
		}
	})

	t.Run("spec decode is off by default and gated", func(t *testing.T) {
		def := DefaultSpecDecodeConfig()
		if def.Enabled {
			t.Fatal("DefaultSpecDecodeConfig must be Enabled:false")
		}
		if _, err := def.Engage(); !errors.Is(err, ErrSpecDecodeDisabled) {
			t.Fatalf("default Engage err=%v, want ErrSpecDecodeDisabled", err)
		}
		if _, err := (SpecDecodeConfig{}).Engage(); !errors.Is(err, ErrSpecDecodeDisabled) {
			t.Fatalf("zero-value Engage err=%v, want ErrSpecDecodeDisabled", err)
		}
		on := SpecDecodeConfig{Enabled: true, DraftLen: 8, Layout: LayoutBonus1N}
		k, err := on.Engage()
		if err != nil || k != 8 {
			t.Fatalf("enabled valid config Engage=%d,%v want 8,nil", k, err)
		}
	})
}

// The block-verification sweep witness (TestBlockVerificationSweep) lives in
// blockverify_test.go; this file owns only the losslessness invariant.
