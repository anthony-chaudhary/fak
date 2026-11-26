//go:build cuda && cgo

package ggufload

import (
	"testing"
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
