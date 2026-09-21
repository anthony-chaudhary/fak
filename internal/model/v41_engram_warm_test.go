package model

import (
	"errors"
	"testing"
)

// TestV41EngramPrefixWarm is the CW-11 witness for
// Model.WarmV41EngramPrefix: prefetching the rows a token prefix will consume
// must land residency in the very caches the reduced forward reads, must not
// advance any live hash history, and must leave the forward numerically
// unchanged.
func TestV41EngramPrefixWarm(t *testing.T) {
	m, layout := v41ReducedEngramModel(t)
	tokens := []int{1, 3, 2, 0, 3, 1}

	// The reduced fixture wires a 4-row budget, too small to retain this prefix.
	// Re-wire the same stage over a source budgeted to hold it, so the residency
	// contract is exercised rather than the eviction bound.
	reqIDs := v41EngramReference(layout, tokens, nil)
	distinct := map[uint32]struct{}{}
	for _, row := range reqIDs {
		distinct[row] = struct{}{}
	}
	src := &v41EngramMemorySource{packed: v41EngramTestPackedRows(int(layout.Rows[0])), rows: int(layout.Rows[0])}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(V41EngramPackedRowBytes)*int64(len(distinct)+1)); err != nil {
		t.Fatalf("widen Engram stage budget: %v", err)
	}

	// The forward must be admissible before we warm its stage.
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("admission before warm = %v, want nil", err)
	}

	// A cold warm must read rows and leave every requested row resident.
	cold, err := m.WarmV41EngramPrefix(tokens, nil)
	if err != nil {
		t.Fatalf("cold warm: %v", err)
	}
	if cold.Layers != 1 {
		t.Fatalf("cold warm touched %d layers, want 1", cold.Layers)
	}
	if cold.Requested == 0 || cold.Misses == 0 || cold.BytesRead == 0 {
		t.Fatalf("cold warm did no work: %+v", cold)
	}
	if cold.Hits != 0 {
		t.Fatalf("a fresh cache reported %d hits, want 0", cold.Hits)
	}
	if cold.Requested != int64(len(distinct)) {
		t.Fatalf("cold warm requested %d distinct rows, want %d", cold.Requested, len(distinct))
	}

	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("stage vanished after warm")
	}
	cache := stage.caches[0]
	if len(reqIDs) != len(tokens)*stage.columns*len(stage.caches) {
		t.Fatalf("reference hash produced %d ids, want %d", len(reqIDs), len(tokens)*stage.columns*len(stage.caches))
	}

	// Every distinct requested address must now be resident: a re-warm is hits-only.
	before := cache.Stats()
	warm, err := m.WarmV41EngramPrefix(tokens, nil)
	if err != nil {
		t.Fatalf("re-warm: %v", err)
	}
	if !warm.Resident() {
		t.Fatalf("re-warm was not fully resident: %+v", warm)
	}
	if warm.Misses != 0 || warm.BytesRead != 0 {
		t.Fatalf("re-warm re-read backing rows: %+v", warm)
	}
	if warm.Requested != int64(len(distinct)) {
		t.Fatalf("re-warm requested %d distinct rows, want %d", warm.Requested, len(distinct))
	}
	after := cache.Stats()
	if after.Hits <= before.Hits {
		t.Fatalf("re-warm did not hit the warmed cache: before=%+v after=%+v", before, after)
	}

	// Warming must not disturb the forward: logits over the warmed prefix must
	// equal an independently built un-warmed model's logits.
	warmed := m.Forward(tokens)
	if warmed == nil || len(warmed.Logits) != len(tokens) {
		t.Fatalf("warmed Forward returned %v positions, want %d", warmed, len(tokens))
	}
	control, _ := v41ReducedEngramModel(t)
	want := control.Forward(tokens)
	if want == nil || len(want.Logits) != len(tokens) {
		t.Fatalf("control Forward returned %v positions, want %d", want, len(tokens))
	}
	for pos := range tokens {
		v41LogitsClose(t, "engram-warm-forward", warmed.Logits[pos], want.Logits[pos])
	}

	// An invalid token must fail without warming anything.
	if _, err := m.WarmV41EngramPrefix([]int{len(layout.TokenMap)}, nil); err == nil {
		t.Fatal("warm accepted an out-of-range token")
	} else if !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("out-of-range warm error = %v, want ErrV41NativeUnsupported", err)
	}
}

// TestV41EngramPrefixWarmBoundedBudget pins the honest bound: when the stage's
// cache is smaller than the prefix's working set, the warm reports non-resident
// rather than claiming a residency it did not establish. This is the
// eviction-bound counterpart to the residency contract above.
func TestV41EngramPrefixWarmBoundedBudget(t *testing.T) {
	m, _ := v41ReducedEngramModel(t) // 4-row budget, deliberately too small
	tokens := []int{1, 3, 2, 0, 3, 1}

	delta, err := m.WarmV41EngramPrefix(tokens, nil)
	if err != nil {
		t.Fatalf("bounded warm: %v", err)
	}
	if delta.Requested == 0 {
		t.Fatalf("bounded warm requested nothing: %+v", delta)
	}
	if delta.Resident() {
		t.Fatalf("4-row cache claimed residency for a larger working set: %+v", delta)
	}
	if delta.Misses != int64(delta.Requested) {
		t.Fatalf("bounded warm hit a cache that cannot retain it: %+v", delta)
	}
}

// TestV41EngramPrefixWarmUnsupported pins the cold-fallback contract: every
// model shape that cannot establish Engram residency refuses with
// ErrV41NativeUnsupported and a zero delta, so a caller keeps serving un-warmed.
func TestV41EngramPrefixWarmUnsupported(t *testing.T) {
	t.Run("no V4.1 config", func(t *testing.T) {
		m := &Model{Cfg: Config{}}
		delta, err := m.WarmV41EngramPrefix([]int{1}, nil)
		if !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("error = %v, want ErrV41NativeUnsupported", err)
		}
		if delta != (V41EngramWarmDelta{}) {
			t.Fatalf("delta = %+v, want zero", delta)
		}
	})

	t.Run("nil model", func(t *testing.T) {
		var m *Model
		if _, err := m.WarmV41EngramPrefix([]int{1}, nil); !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("error = %v, want ErrV41NativeUnsupported", err)
		}
	})

	t.Run("declared but unwired stage", func(t *testing.T) {
		m, _ := v41ReducedEngramModel(t)
		v41EngramStages.Delete(m)
		delta, err := m.WarmV41EngramPrefix([]int{1, 3}, nil)
		if !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("error = %v, want ErrV41NativeUnsupported", err)
		}
		if delta != (V41EngramWarmDelta{}) {
			t.Fatalf("delta = %+v, want zero", delta)
		}
	})

	t.Run("no declared Engram layers", func(t *testing.T) {
		m, _ := v41ReducedEngramModel(t)
		m.Cfg.DeepSeekV41.EngramLayerIDs = nil
		if _, err := m.WarmV41EngramPrefix([]int{1, 3}, nil); !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("error = %v, want ErrV41NativeUnsupported", err)
		}
	})
}
