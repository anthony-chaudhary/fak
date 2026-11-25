//go:build darwin && arm64 && cgo

package inklingmetal

import (
	"os"
	"slices"
	"testing"
	"time"
)

const laneSteps = 32

// Unequal lengths put the lanes at different positions and KV pages.
func lanePrompts() [][]int32 {
	caps := []int32{371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884, 154328, 13, 149917, 151451, 149879, 153630, 149884}
	fib := []int32{1719, 220, 46389, 7, 77, 8, 25, 198, 149862, 172706, 72358, 149876, 149934, 12, 258, 181713, 150107, 13, 70646, 198}
	return [][]int32{caps, fib, caps[:13], slices.Concat(fib, caps)}
}

func openTargetOnly(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("FAK_INKLING2_TEST_GGUF")
	if path == "" {
		t.Skip("set FAK_INKLING2_TEST_GGUF to run on Apple GPU")
	}
	m, err := OpenWithDraft(path, "")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })
	return m
}

func prefilled(t *testing.T, m *Model, prompts [][]int32) ([]*Session, []int32) {
	t.Helper()
	sessions := make([]*Session, len(prompts))
	first := make([]int32, len(prompts))
	for i, prompt := range prompts {
		s, err := m.NewSession(128)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { s.Close() })
		sessions[i] = s
		if first[i], err = s.PrefillID(prompt); err != nil {
			t.Fatal(err)
		}
	}
	return sessions, first
}

// fak-test:runtime slow est=90s
func TestBatchLanesMatchSoloDecode(t *testing.T) {
	m := openTargetOnly(t)
	prompts := lanePrompts()

	type stepper func(sessions []*Session, ids []int32) ([]int32, error)
	each := func(one func(s *Session, id int32) (int32, error)) stepper {
		return func(sessions []*Session, ids []int32) ([]int32, error) {
			out := make([]int32, len(ids))
			for i, s := range sessions {
				next, err := one(s, ids[i])
				if err != nil {
					return nil, err
				}
				out[i] = next
			}
			return out, nil
		}
	}
	twoCalls := each(func(s *Session, id int32) (int32, error) {
		next, err := s.BeginBlockIDs([]int32{id})
		if err != nil {
			return 0, err
		}
		return next[0], s.Commit(1)
	})
	oneCall := each(func(s *Session, id int32) (int32, error) {
		next, err := m.stepBatch([]*Session{s}, []int32{id}, false, nil)
		if err != nil {
			return 0, err
		}
		return next[0], nil
	})
	// The first step compiles its pipelines, so rates cover the steps after it.
	run := func(prompts [][]int32, step stepper) ([][]int32, float64) {
		t.Helper()
		sessions, ids := prefilled(t, m, prompts)
		got := make([][]int32, len(prompts))
		var start time.Time
		for k := 0; k < laneSteps; k++ {
			if k == 1 {
				start = time.Now()
			}
			next, err := step(sessions, ids)
			if err != nil {
				t.Fatal(err)
			}
			for i, id := range next {
				got[i] = append(got[i], id)
			}
			ids = next
		}
		rate := float64(len(prompts)*(laneSteps-1)) / time.Since(start).Seconds()
		for _, s := range sessions {
			s.Close()
		}
		return got, rate
	}

	solo := make([][]int32, len(prompts))
	for i := range prompts {
		got, _ := run(prompts[i:i+1], twoCalls)
		solo[i] = got[0]
	}
	for _, lanes := range []int{1, 2, 4} {
		serial, serialRate := run(prompts[:lanes], twoCalls)
		single, singleRate := run(prompts[:lanes], oneCall)
		batch, batchRate := run(prompts[:lanes], m.StepBatch)
		wide, wideRate := run(prompts[:lanes], m.StepBatchWide)
		agree := 0
		for i := 0; i < lanes; i++ {
			for name, got := range map[string][]int32{"interleaved sessions": serial[i], "one-call steps": single[i], "batched lanes": batch[i]} {
				if !slices.Equal(got, solo[i]) {
					t.Fatalf("lanes=%d lane %d: %s diverge from the prompt alone\n got %v\nwant %v", lanes, i, name, got, solo[i])
				}
			}
			for k, id := range wide[i] {
				if id != solo[i][k] {
					break
				}
				agree++
			}
		}
		// Up to two rows a pass, a wide step is the exact one.
		if lanes <= 2 && agree != lanes*laneSteps {
			t.Fatalf("lanes=%d: a wide step of column-tile rows diverges from the prompts alone", lanes)
		}
		l := float64(lanes)
		t.Logf("lanes=%d aggregate tok/s: batched=%.2f wide=%.2f serialized_two_call=%.2f serialized_one_call=%.2f; per-lane tok/s: batched=%.2f wide=%.2f serialized_two_call=%.2f serialized_one_call=%.2f; wide ids equal to solo before first divergence: %d/%d",
			lanes, batchRate, wideRate, serialRate, singleRate, batchRate/l, wideRate/l, serialRate/l, singleRate/l, agree, lanes*laneSteps)
	}
	t.Logf("batch scratch held by the model: %d bytes", m.batchBytes())

	sessions, ids := prefilled(t, m, prompts[:1])
	if _, err := m.StepBatch([]*Session{sessions[0], sessions[0]}, []int32{ids[0], ids[0]}); err == nil {
		t.Fatal("a session repeated across lanes was accepted")
	}
}

// Each lane is fed the tokens its prompt alone chose, so every step's scores
// compare against that prompt's own.
//
// fak-test:runtime slow est=60s
func TestBatchLanesScoresMatchSolo(t *testing.T) {
	m := openTargetOnly(t)
	prompts := lanePrompts()
	type row struct {
		fed    int32
		scores []float32
	}
	solo := make([][]row, len(prompts))
	for p := range prompts {
		sessions, first := prefilled(t, m, prompts[p:p+1])
		s, id := sessions[0], first[0]
		for k := 0; k < laneSteps; k++ {
			scores, err := s.BeginBlock([]int32{id})
			if err != nil {
				t.Fatal(err)
			}
			if err := s.Commit(1); err != nil {
				t.Fatal(err)
			}
			solo[p] = append(solo[p], row{id, scores[0]})
			id = int32(slices.Index(scores[0], slices.Max(scores[0])))
		}
		s.Close()
	}
	type drift struct {
		rows, moved, flipped int
		worst                float32
	}
	compare := func(d *drift, want, have []float32) {
		var worst float32
		for j := range want {
			worst = max(worst, want[j]-have[j], have[j]-want[j])
		}
		d.rows++
		if worst > 0 {
			d.moved++
		}
		if slices.Index(have, slices.Max(have)) != slices.Index(want, slices.Max(want)) {
			d.flipped++
		}
		d.worst = max(d.worst, worst)
	}
	lanes := func(order []int, wide bool) drift {
		t.Helper()
		picked := make([][]int32, len(order))
		for i, p := range order {
			picked[i] = prompts[p]
		}
		sessions, _ := prefilled(t, m, picked)
		scores := make([]float32, len(order)*Vocabulary)
		ids := make([]int32, len(order))
		var d drift
		for k := 0; k < laneSteps; k++ {
			for i, p := range order {
				ids[i] = solo[p][k].fed
			}
			if _, err := m.stepBatch(sessions, ids, wide, scores); err != nil {
				t.Fatal(err)
			}
			for i, p := range order {
				compare(&d, solo[p][k].scores, scores[i*Vocabulary:(i+1)*Vocabulary])
			}
		}
		for _, s := range sessions {
			s.Close()
		}
		return d
	}
	// Equal prompts route every lane to the same experts.
	for _, order := range [][]int{{3}, {3, 3}, {0, 1, 2}, {0, 1, 2, 3}, {3, 2, 1, 0}, {3, 3, 3, 3}} {
		if d := lanes(order, false); d.moved != 0 {
			t.Fatalf("prompts %v: batched scores left the prompts' own in %d of %d rows, by up to %g", order, d.moved, d.rows, d.worst)
		}
		d := lanes(order, true)
		if len(order) <= 2 && d.moved != 0 {
			t.Fatalf("prompts %v: a wide step of column-tile rows left the prompts' own scores in %d of %d rows", order, d.moved, d.rows)
		}
		t.Logf("prompts %v wide step: rows=%d scores_moved=%d argmax_flipped=%d worst_abs_score_diff=%g", order, d.rows, d.moved, d.flipped, d.worst)
	}
	// The same rounding in one session: a block of rows against its single steps.
	for _, block := range []int{2, 4} {
		sessions, _ := prefilled(t, m, prompts[3:])
		var d drift
		for k := 0; k < laneSteps; k += block {
			ids := make([]int32, block)
			for r := range ids {
				ids[r] = solo[3][k+r].fed
			}
			scores, err := sessions[0].BeginBlock(ids)
			if err != nil {
				t.Fatal(err)
			}
			if err := sessions[0].Commit(block); err != nil {
				t.Fatal(err)
			}
			for r := range ids {
				compare(&d, solo[3][k+r].scores, scores[r])
			}
		}
		sessions[0].Close()
		t.Logf("one session, blocks of %d rows: rows=%d scores_moved=%d argmax_flipped=%d worst_abs_score_diff=%g", block, d.rows, d.moved, d.flipped, d.worst)
	}
}
