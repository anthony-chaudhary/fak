package ggufload

import (
	"context"
	"slices"
	"testing"
)

// scriptedSession returns script[i] from the i-th forward (prefill is forward 0).
type scriptedSession struct {
	script []int
	calls  int
}

func (s *scriptedSession) Prefill(context.Context, []int) (int, error) { return s.next(), nil }
func (s *scriptedSession) Step(int) (int, error)                       { return s.next(), nil }
func (s *scriptedSession) Close() error                                { return nil }

func (s *scriptedSession) next() int {
	s.calls++
	return s.script[s.calls-1]
}

func TestInkling2SessionGreedyStopsAtEOS(t *testing.T) {
	const eos = 99
	cases := []struct {
		name     string
		script   []int
		n, eos   int
		want     []int
		steps    int
		forwards int
		reachesN bool
	}{
		{"stops before eos", []int{5, 6, 7, eos, 8, 9}, 16, eos, []int{5, 6, 7}, 3, 4, false},
		{"eos from prefill", []int{eos, 1}, 16, eos, []int{}, 0, 1, false},
		{"limit before eos", []int{5, 6, 7, eos}, 2, eos, []int{5, 6}, 1, 2, true},
		{"no eos decodes n", []int{5, eos, 7, 8}, 4, -1, []int{5, eos, 7, 8}, 3, 4, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := &scriptedSession{script: c.script}
			out, st, err := inkling2GreedyStepsUntil(s, []int{1, 2}, c.n, c.eos)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(out, c.want) {
				t.Errorf("ids %v, want %v", out, c.want)
			}
			if st.DecodeSteps != c.steps || s.calls != c.forwards {
				t.Errorf("decode steps %d over %d forwards, want %d over %d", st.DecodeSteps, s.calls, c.steps, c.forwards)
			}
			if (len(out) == c.n) != c.reachesN {
				t.Errorf("reached the limit %v, want %v", len(out) == c.n, c.reachesN)
			}
		})
	}
}
