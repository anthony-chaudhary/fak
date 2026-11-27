//go:build cuda && cgo

package ggufload

import (
	"context"
	"fmt"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// FAK_INKLING2_SPEC_ORACLE=1: option (b) of the spec parity contract. The batched
// verify is token-parity, not bit-parity, with the per-token step, so speculative
// greedy must equal plain greedy up to the first position where the two disagree,
// and that position must be an admissible near tie by the oracle rule (plain top-2
// gap under inkling2OracleNearTieNats, and the speculative run took the runner-up).
// Past a tie the two runs legitimately diverge, so the comparison stops there.
// Each oracle case is run with a perfect drafter at depth 4.
func TestInkling2CUDASpecNearTieOracle(t *testing.T) {
	m, be := openInkling2CUDAForTest(t, "FAK_INKLING2_SPEC_ORACLE")
	defer m.Close()
	t.Setenv("FAK_INKLING2_BATCHED_PREFILL", "1")
	// Spec decode downgrades auto-f16 KV to f32; pin f32 so the plain scalar reference matches.
	t.Setenv("FAK_INKLING2_KV", "f32")
	const n, depth = 128, 4
	for _, c := range inkling2OracleCases {
		t.Run(c.name, func(t *testing.T) {
			ref, runnerUp, margin, err := inkling2ResidentGreedyMargins(m, be, c.prompt, n)
			if err != nil {
				t.Fatal(err)
			}
			got, sst, err := inkling2ResidentSpecGreedy(m, be, c.prompt, n, depth, func(*inkling2Resident) (inkling2Drafter, error) {
				return &inkling2OracleDrafter{ref: ref}, nil
			})
			if err != nil {
				t.Fatal(err)
			}
			at, nearTie, why := inkling2CohortDivergence(ref, runnerUp, margin, got)
			if at >= 0 && !nearTie {
				if candGap, gerr := inkling2SpecVerifyGapAt(m, be, c.prompt, ref, depth, at, got[at]); gerr == nil {
					nearTie = inkling2NearTie(margin[at], got[at] == runnerUp[at], candGap)
					why = fmt.Sprintf("%s; spec verify gap for scalar %d is %.4f nats", why, ref[at], candGap)
				}
			}
			if at < 0 {
				t.Logf("spec depth %d matches plain greedy over %d tokens: %.2f tok/s, accepted %d/%d",
					depth, len(got), sst.DecodeTokensPerSecond(), sst.Accepted, sst.Drafted)
				return
			}
			if !nearTie {
				t.Fatalf("spec depth %d diverges from plain greedy: %s", depth, why)
			}
			t.Logf("spec depth %d diverges at token %d on a near tie: %s", depth, at, why)
		})
	}
}

// inkling2SpecVerifyGapAt replays speculative greedy with the oracle drafter up
// to divergent step and returns how many softcapped nats ref[step] sits below
// the verify step's top-1 token when that top-1 equals gotTok.
func inkling2SpecVerifyGapAt(m *Inkling2LoopModel, be compute.Backend, prompt, ref []int, depth, step, gotTok int) (float64, error) {
	s, err := newInkling2ResidentWithSpec(m, be, true)
	if err != nil {
		return 0, err
	}
	defer s.close()
	if err := s.enableSpec(depth); err != nil {
		return 0, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	next, err := s.prefill(context.Background(), prompt)
	if err != nil {
		return 0, err
	}
	if s.graph {
		warm := make([]int, depth+1)
		for i := range warm {
			warm[i] = next
		}
		if _, err := s.verify(warm); err != nil {
			return 0, err
		}
		if err := s.commitVerify(len(warm), 0); err != nil {
			return 0, err
		}
	}
	d := &inkling2OracleDrafter{ref: ref}
	out := make([]int, 0, step+1)
	for len(out) <= step {
		out = append(out, next)
		if len(out) > step {
			break
		}
		props := d.Draft(out, depth)
		fed := append([]int{next}, props...)
		g, err := s.verify(fed)
		if err != nil {
			return 0, err
		}
		for k := range fed {
			if len(out)+k == step {
				logits, err := s.dev.Ink2Read(at(s.verifyLogits, k*s.output.rows), s.hp.VocabUnpadded)
				if err != nil {
					return 0, err
				}
				top, _ := inkling2TopTwo(logits)
				if top != gotTok {
					return 0, fmt.Errorf("spec verify top %d != decoded %d at step %d", top, gotTok, step)
				}
				return float64(logits[top] - logits[ref[step]]), nil
			}
		}
		a := 0
		for a < len(props) && props[a] == g[a] {
			a++
		}
		if err := s.commitVerify(len(fed), a+1); err != nil {
			return 0, err
		}
		out = append(out, props[:a]...)
		next = g[a]
	}
	return 0, fmt.Errorf("step %d not reached in spec verify", step)
}
