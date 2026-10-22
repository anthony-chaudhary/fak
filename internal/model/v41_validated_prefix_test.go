package model

import (
	"errors"
	"math"
	"testing"
)

func v41RecordCombinedScans(t *testing.T) *int {
	t.Helper()
	scanned := -1
	prior := v41CombinedAttentionScanProbe
	v41CombinedAttentionScanProbe = func(elements int) { scanned = elements }
	t.Cleanup(func() { v41CombinedAttentionScanProbe = prior })
	return &scanned
}

// The memo only removes rescans of immutable publications: outputs and
// refusals match the unmemoized composer bit for bit, and scan work per step is
// the window plus newly published rows, independent of the history length.
// fak-test:runtime fast est=50ms lane=default
func TestV41CombinedAttentionValidatedPrefix(t *testing.T) {
	const dim, windowRows = 2, 3
	scanned := v41RecordCombinedScans(t)
	row := func(i int) []float32 { return []float32{float32(i%7) * 0.125, -float32(i%5) * 0.25} }
	publish := func(t *testing.T, s *V41AttentionState, i int) {
		t.Helper()
		if err := s.publishUpdates([]V41AttentionStateUpdate{{Ref: V41AttentionStateRef{LayerID: 0, Ratio: 1, IsKVSource: true}, Latent: row(i)}}); err != nil {
			t.Fatal(err)
		}
	}
	newState := func(t *testing.T, n int) *V41AttentionState {
		t.Helper()
		s, err := NewV41AttentionState(dim, 4)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			publish(t, s, i)
		}
		return s
	}
	window := func() [][]float32 {
		out := make([][]float32, windowRows)
		for i := range out {
			out[i] = []float32{0.5, float32(i) * 0.25}
		}
		return out
	}
	attend := func(t *testing.T, s *V41AttentionState, win [][]float32, fresh ...[]float32) (int, error) {
		t.Helper()
		compressed, _ := s.borrowKVSourceRows(0)
		compressed = append(compressed, fresh...)
		ids := make([]int32, len(compressed))
		for i := range ids {
			ids[i] = int32(i)
		}
		pos := len(compressed) + len(win)
		q, sink := []float32{0.75, -0.5}, []float32{0.125}
		want, wantErr := v41CombinedAttentionWithContinuation(0, pos, 1, 1, dim, q, sink, win, compressed, ids, 0.5, nil, nil, nil)
		*scanned = -1
		got, err := v41CombinedAttentionValidated(0, pos, 1, 1, dim, q, sink, win, compressed, ids, 0.5, nil, nil, nil, &s.validatedPrefix)
		if (err == nil) != (wantErr == nil) || !v41SameBits(got, want) {
			t.Fatalf("memo changed the composer: got=%v err=%v want=%v wantErr=%v", got, err, want, wantErr)
		}
		return *scanned, err
	}

	t.Run("scan-work-independent-of-history", func(t *testing.T) {
		for _, n := range []int{4, 64} {
			s := newState(t, n)
			if got, err := attend(t, s, window()); err != nil || got != (windowRows+n)*dim {
				t.Fatalf("n=%d first call scanned %d err=%v, want full %d", n, got, err, (windowRows+n)*dim)
			}
			for step := 0; step < 3; step++ {
				publish(t, s, n+step)
				if got, err := attend(t, s, window()); err != nil || got != (windowRows+1)*dim {
					t.Fatalf("n=%d step=%d scanned %d err=%v, want window plus one new row %d", n, step, got, err, (windowRows+1)*dim)
				}
			}
			if got, err := attend(t, s, window()); err != nil || got != windowRows*dim {
				t.Fatalf("n=%d unchanged history scanned %d err=%v, want window only %d", n, got, err, windowRows*dim)
			}
			if s.validatedPrefix.rows != n+3 {
				t.Fatalf("n=%d memo covers %d rows, want %d", n, s.validatedPrefix.rows, n+3)
			}
		}
	})

	t.Run("fresh-nonfinite-rows-refuse", func(t *testing.T) {
		s := newState(t, 8)
		if _, err := attend(t, s, window()); err != nil {
			t.Fatal(err)
		}
		nan := float32(math.NaN())
		for name, call := range map[string]func() error{
			"compressed": func() error { _, err := attend(t, s, window(), []float32{nan, 0}); return err },
			"window": func() error {
				win := window()
				win[0][1] = float32(math.Inf(1))
				_, err := attend(t, s, win)
				return err
			},
		} {
			var stage *V41ForwardError
			if err := call(); !errors.As(err, &stage) || stage.Stage != v41StageAttention || stage.Layer != 0 {
				t.Fatalf("%s non-finite row was not refused at the attention stage: %v", name, err)
			}
			if s.validatedPrefix.rows != 8 {
				t.Fatalf("%s refusal moved the memo to %d rows", name, s.validatedPrefix.rows)
			}
		}
	})

	t.Run("rollback-rescans", func(t *testing.T) {
		s := newState(t, 6)
		if _, err := attend(t, s, window()); err != nil {
			t.Fatal(err)
		}
		undo := stageV41StepPublication(s.kvPublications, s.kvPublishedEnd, 0)
		publish(t, s, 100)
		if got, err := attend(t, s, window()); err != nil || got != (windowRows+1)*dim {
			t.Fatalf("prospective row scanned %d err=%v", got, err)
		}
		undo.restore()
		if got, err := attend(t, s, window()); err != nil || got != (windowRows+6)*dim {
			t.Fatalf("shorter history after rollback scanned %d err=%v, want full", got, err)
		}
		undo = stageV41StepPublication(s.kvPublications, s.kvPublishedEnd, 0)
		publish(t, s, 101)
		if _, err := attend(t, s, window()); err != nil {
			t.Fatal(err)
		}
		undo.restore()
		publish(t, s, 102)
		if got, err := attend(t, s, window()); err != nil || got != (windowRows+7)*dim {
			t.Fatalf("replacement row at the anchor key scanned %d err=%v, want full", got, err)
		}
	})

	t.Run("restore-and-reset-rescan", func(t *testing.T) {
		s := newState(t, 6)
		if _, err := attend(t, s, window()); err != nil {
			t.Fatal(err)
		}
		restored := s.clone()
		if restored.validatedPrefix.rows != 0 {
			t.Fatal("snapshot clone carried the attention memo")
		}
		restored.validatedPrefix = s.validatedPrefix
		if got, err := attend(t, restored, window()); err != nil || got != (windowRows+6)*dim {
			t.Fatalf("restored rows under a live memo scanned %d err=%v, want full", got, err)
		}
		s.Reset()
		if s.validatedPrefix.rows != 0 {
			t.Fatal("reset kept the attention memo")
		}
		for i := 0; i < 6; i++ {
			publish(t, s, i)
		}
		if got, err := attend(t, s, window()); err != nil || got != (windowRows+6)*dim {
			t.Fatalf("republished rows after reset scanned %d err=%v, want full", got, err)
		}
	})
}

// The decode role step is the memo's caller: across consecutive positions it
// skips exactly the rows earlier steps proved finite and stays bit-identical to
// a run whose memo is cleared before every layer.
// fak-test:runtime fast est=2s lane=default
func TestV41RoleStepValidatedPrefix(t *testing.T) {
	m := v41ReaderWidthFixture(t)
	scanned := v41RecordCombinedScans(t)
	type call struct{ scanned, prior, rows int }
	run := func(memo bool) ([]call, [][]float32) {
		st := v41RoleContinuationSeed(t, m, []int{1, 2, 3, 4})
		var calls []call
		var outs [][]float32
		for pos := 4; pos < 10; pos++ {
			x, streams := v41LayerStepInputs(t, m, 1+pos%4, true)
			scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
			for l := 0; l < m.Cfg.NumLayers; l++ {
				plan, err := m.v41AttentionPlan(l)
				if err != nil {
					t.Fatal(err)
				}
				state := st.layerState(l)
				if !memo {
					state.validatedPrefix = v41ValidatedPrefix{}
				}
				prior := state.validatedPrefix.rows
				*scanned = -1
				work, err := m.prepareV41LayerStepRole(l, plan, x, streams, pos, state, st.attn, scratch)
				if err != nil {
					t.Fatal(err)
				}
				if err = work.complete(nil); err != nil {
					t.Fatal(err)
				}
				calls = append(calls, call{*scanned, prior, state.validatedPrefix.rows})
			}
			outs = append(outs, append([]float32(nil), x...))
			for _, stream := range streams {
				outs = append(outs, append([]float32(nil), stream...))
			}
		}
		return calls, outs
	}
	control, controlOut := run(false)
	memo, memoOut := run(true)
	if len(memo) != len(control) || len(memoOut) != len(controlOut) {
		t.Fatal("memo run took a different number of layer steps")
	}
	for i := range memoOut {
		if !v41SameBits(memoOut[i], controlOut[i]) {
			t.Fatalf("memo changed step output %d", i)
		}
	}
	hd := m.Cfg.HeadDim
	skipped := 0
	for i, got := range memo {
		pos := 4 + i/m.Cfg.NumLayers
		want := control[i]
		if got.scanned < 0 || want.scanned < 0 || got.rows != (pos+1)/2 || want.rows != got.rows {
			t.Fatalf("call %d at pos %d: memo=%+v control=%+v, want %d recorded rows", i, pos, got, want, (pos+1)/2)
		}
		if pos > 4 && got.prior != pos/2 {
			t.Fatalf("call %d at pos %d: memo entered with %d rows, want the previous step's %d", i, pos, got.prior, pos/2)
		}
		if want.scanned-got.scanned != got.prior*hd {
			t.Fatalf("call %d at pos %d: skipped %d elements, want the %d-row memo prefix", i, pos, want.scanned-got.scanned, got.prior)
		}
		skipped += got.prior
	}
	if skipped == 0 {
		t.Fatal("decode never skipped an already validated row")
	}
}
