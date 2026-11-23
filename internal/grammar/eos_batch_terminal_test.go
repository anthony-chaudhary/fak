package grammar

import (
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestEOSBatchTerminal(t *testing.T) {
	const eos = 0
	// Three live rows with distinct allowed sets, all containing EOS.
	batch, err := NewBatchedGrammarStep(eos,
		[]int{eos, 1, 2},
		[]int{eos, 3, 4},
		[]int{eos, 5, 6},
	)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("EOS freezes row k only", func(t *testing.T) {
		// Advance rows 0 and 2 with live tokens so they have distinct state.
		if err := batch.Step(0, 1, []int{eos, 7}); err != nil {
			t.Fatal(err)
		}
		if err := batch.Step(2, 5, []int{eos, 8}); err != nil {
			t.Fatal(err)
		}
		// Row 1 hits EOS: it must freeze, rows 0 and 2 must be untouched.
		if err := batch.Step(1, eos, nil); err != nil {
			t.Fatal(err)
		}
		f1, err := batch.Finished(1)
		if err != nil || !f1 {
			t.Fatalf("row 1 not finished: %v %v", f1, err)
		}
		f0, _ := batch.Finished(0)
		f2, _ := batch.Finished(2)
		if f0 || f2 {
			t.Fatalf("live rows marked finished: row0=%v row2=%v", f0, f2)
		}
		for k, want := range []bool{true, false, true} {
			if live, err := batch.Live(k); err != nil || live != want {
				t.Fatalf("Live(%d) = %v, %v; want %v", k, live, err, want)
			}
		}
		// Row 0 and 2 kept their advanced live sets; row 1 is masked to {eos}.
		if m, _ := batch.Mask(0); !reflect.DeepEqual(m, []int{eos, 7}) {
			t.Fatalf("row 0 mask disturbed: %v", m)
		}
		if m, _ := batch.Mask(2); !reflect.DeepEqual(m, []int{eos, 8}) {
			t.Fatalf("row 2 mask disturbed: %v", m)
		}
		if m, _ := batch.Mask(1); !reflect.DeepEqual(m, []int{eos}) {
			t.Fatalf("row 1 not masked to terminal-only: %v", m)
		}
	})

	t.Run("finished row masked from further transitions", func(t *testing.T) {
		// Row 1 is finished; any further step on it is refused and leaves
		// the batch unchanged.
		before := snapshot(batch)
		if err := batch.Step(1, eos, nil); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("step on finished row admitted: %v", err)
		}
		if err := batch.Step(1, 3, []int{9}); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("live token on finished row admitted: %v", err)
		}
		if got := snapshot(batch); !reflect.DeepEqual(got, before) {
			t.Fatalf("refusal changed batch state")
		}
		// The finished row's mask stays terminal-only.
		if m, _ := batch.Mask(1); !reflect.DeepEqual(m, []int{eos}) {
			t.Fatalf("finished row mask drifted: %v", m)
		}
	})

	t.Run("live rows continue after a sibling EOS", func(t *testing.T) {
		// After row 1 froze, rows 0 and 2 still advance normally.
		if err := batch.Step(0, 7, []int{eos, 10}); err != nil {
			t.Fatalf("live row 0 stopped advancing: %v", err)
		}
		if m, _ := batch.Mask(0); !reflect.DeepEqual(m, []int{eos, 10}) {
			t.Fatalf("row 0 did not advance: %v", m)
		}
		if err := batch.Step(2, 8, []int{eos}); err != nil {
			t.Fatalf("live row 2 stopped advancing: %v", err)
		}
		if m, _ := batch.Mask(2); !reflect.DeepEqual(m, []int{eos}) {
			t.Fatalf("row 2 did not advance: %v", m)
		}
	})

	t.Run("fail-closed on out-of-range row", func(t *testing.T) {
		for _, k := range []int{-1, len(batch.rows), len(batch.rows) + 5} {
			if err := batch.Step(k, eos, nil); !errors.Is(err, ErrEOSBatchTerminal) {
				t.Fatalf("out-of-range row %d step admitted: %v", k, err)
			}
			if _, err := batch.Mask(k); !errors.Is(err, ErrEOSBatchTerminal) {
				t.Fatalf("out-of-range row %d mask admitted: %v", k, err)
			}
			if _, err := batch.Finished(k); !errors.Is(err, ErrEOSBatchTerminal) {
				t.Fatalf("out-of-range row %d finished admitted: %v", k, err)
			}
			if _, err := batch.Live(k); !errors.Is(err, ErrEOSBatchTerminal) {
				t.Fatalf("out-of-range row %d live admitted: %v", k, err)
			}
		}
	})

	t.Run("fail-closed on token outside live set", func(t *testing.T) {
		// Row 0's live set is {eos,10}; token 99 is not allowed.
		before := snapshot(batch)
		if err := batch.Step(0, 99, []int{11}); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("token outside live set admitted: %v", err)
		}
		if got := snapshot(batch); !reflect.DeepEqual(got, before) {
			t.Fatalf("refusal changed batch state")
		}
	})

	t.Run("EOS outside the live set is not a terminal transition", func(t *testing.T) {
		// The grammar decides when a row may end: once row 0's live set drops
		// EOS, an EOS token is an ordinary disallowed token for that row.
		if err := batch.Step(0, 10, []int{12, 13}); err != nil {
			t.Fatal(err)
		}
		before := snapshot(batch)
		if err := batch.Step(0, eos, nil); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("EOS outside the live set admitted: %v", err)
		}
		if f, _ := batch.Finished(0); f {
			t.Fatal("row 0 finished on an EOS its grammar did not allow")
		}
		if got := snapshot(batch); !reflect.DeepEqual(got, before) {
			t.Fatalf("refusal changed batch state")
		}
	})

	t.Run("constructor and zero value refuse", func(t *testing.T) {
		if _, err := NewBatchedGrammarStep(eos); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("empty batch admitted: %v", err)
		}
		if _, err := NewBatchedGrammarStep(eos, []int{1, 2}); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("row without terminal admitted: %v", err)
		}
		if _, err := NewBatchedGrammarStep(eos, []int{}); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("empty allowed set admitted: %v", err)
		}
		var zero BatchedGrammarStep
		if err := zero.Step(0, eos, nil); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("zero value step admitted: %v", err)
		}
		if _, err := zero.Mask(0); !errors.Is(err, ErrEOSBatchTerminal) {
			t.Fatalf("zero value mask admitted: %v", err)
		}
	})
}

// snapshot returns a comparable copy of the batch's per-row state.
func snapshot(b *BatchedGrammarStep) []RowState {
	out := make([]RowState, len(b.rows))
	for i := range b.rows {
		out[i] = RowState{
			allowed:  append([]int(nil), b.rows[i].allowed...),
			finished: b.rows[i].finished,
		}
	}
	return out
}
