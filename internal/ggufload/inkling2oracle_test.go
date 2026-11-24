package ggufload

import (
	"fmt"
	"strings"
	"testing"
)

// inkling2OracleNearTieNats is the widest oracle top-2 logprob gap at which a
// divergence counts as a near tie rather than a fak bug. Changing only
// llama.cpp's ubatch moved its own top-2 margin by up to 0.25 nats on the
// oracle steps whose margin is under 2 nats, and flipped passphrase-recall-1946
// step 10 (0.24 nats). The bound is twice that self-noise, still well short of
// the multi-nat gaps a real numeric bug produces.
const inkling2OracleNearTieNats = 0.5

// inkling2OracleMaxNearTies bounds re-anchoring: a case that needs more near
// ties than this to reach the end is drifting, not tying.
const inkling2OracleMaxNearTies = 2

type inkling2OracleCase struct {
	name           string
	prompt, greedy []int
	// runnerUp[i] is the oracle's second-ranked token at greedy step i and
	// margin[i] the top-1 minus top-2 logprob there, in nats, both from the
	// same llama.cpp run as greedy (pre-sampling n_probs).
	runnerUp []int
	margin   []float64
}

type inkling2OracleVerdict string

const (
	inkling2OraclePass     inkling2OracleVerdict = "PASS"
	inkling2OracleNearTie  inkling2OracleVerdict = "NEAR_TIE"
	inkling2OracleMismatch inkling2OracleVerdict = "FAIL"
)

// inkling2OracleDivergence compares got, decoded after prompt+greedy[:from],
// with greedy[from:]. It returns the first divergent greedy index (-1 when
// none) and whether that divergence is an admissible near tie: the oracle's
// margin there is under inkling2OracleNearTieNats and fak picked the oracle's
// runner-up.
func inkling2OracleDivergence(c inkling2OracleCase, from int, got []int) (at int, nearTie bool, why string) {
	want := c.greedy[from:]
	if len(got) != len(want) {
		return from + min(len(got), len(want)), false, fmt.Sprintf("decoded %d tokens, oracle has %d", len(got), len(want))
	}
	for k := range got {
		if got[k] == want[k] {
			continue
		}
		i := from + k
		why = fmt.Sprintf("greedy token %d = %d, llama.cpp = %d (runner-up %d, margin %.4f nats, near-tie bound %.2f)",
			i, got[k], want[k], c.runnerUp[i], c.margin[i], inkling2OracleNearTieNats)
		return i, c.margin[i] < inkling2OracleNearTieNats && got[k] == c.runnerUp[i], why
	}
	return -1, false, ""
}

// inkling2OracleCompare runs decode from the prompt and, at each admissible
// near tie, re-anchors on the oracle's token by decoding again from
// prompt+greedy[:at+1], so the steps after a tie are still compared.
func inkling2OracleCompare(c inkling2OracleCase, decode func(prompt []int, n int) ([]int, error)) (inkling2OracleVerdict, []string, error) {
	if len(c.runnerUp) != len(c.greedy) || len(c.margin) != len(c.greedy) {
		return inkling2OracleMismatch, nil, fmt.Errorf("%s: oracle has %d greedy, %d runner-up, %d margin entries", c.name, len(c.greedy), len(c.runnerUp), len(c.margin))
	}
	var ties []string
	for from := 0; from < len(c.greedy); {
		prompt := append(append([]int(nil), c.prompt...), c.greedy[:from]...)
		got, err := decode(prompt, len(c.greedy)-from)
		if err != nil {
			return inkling2OracleMismatch, ties, err
		}
		at, nearTie, why := inkling2OracleDivergence(c, from, got)
		switch {
		case at < 0:
			from = len(c.greedy)
		case !nearTie:
			return inkling2OracleMismatch, ties, fmt.Errorf("%s (got %v)", why, got)
		case len(ties) == inkling2OracleMaxNearTies:
			return inkling2OracleMismatch, ties, fmt.Errorf("more than %d near ties: %s", inkling2OracleMaxNearTies, why)
		default:
			ties = append(ties, why)
			from = at + 1
		}
	}
	if len(ties) > 0 {
		return inkling2OracleNearTie, ties, nil
	}
	return inkling2OraclePass, nil, nil
}

// runInkling2Oracle reports one ORACLE_VERDICT line per case; a NEAR_TIE is
// green but never logged as PASS.
func runInkling2Oracle(t *testing.T, c inkling2OracleCase, decode func(prompt []int, n int) ([]int, error)) {
	t.Helper()
	v, ties, err := inkling2OracleCompare(c, decode)
	if err != nil {
		t.Fatalf("ORACLE_VERDICT %s %s: %v (near ties so far: %v)", c.name, v, err, ties)
	}
	if v == inkling2OracleNearTie {
		t.Logf("ORACLE_VERDICT %s %s: %s", c.name, v, strings.Join(ties, "; "))
		return
	}
	t.Logf("ORACLE_VERDICT %s %s", c.name, v)
}

// replayDecode serves canned decodes keyed by how many oracle tokens were
// forced onto the prompt.
func replayDecode(c inkling2OracleCase, byFrom map[int][]int) func([]int, int) ([]int, error) {
	return func(prompt []int, n int) ([]int, error) {
		out, ok := byFrom[len(prompt)-len(c.prompt)]
		if !ok {
			return nil, fmt.Errorf("unexpected re-anchor at %d", len(prompt)-len(c.prompt))
		}
		return out[:n], nil
	}
}

func syntheticInkling2OracleCase() inkling2OracleCase {
	return inkling2OracleCase{
		name:     "synthetic",
		prompt:   []int{1, 2},
		greedy:   []int{10, 11, 12, 13},
		runnerUp: []int{20, 21, 22, 23},
		margin:   []float64{3, 0.21, 4, 2.5},
	}
}

func TestInkling2OracleExactMatchIsPass(t *testing.T) {
	c := syntheticInkling2OracleCase()
	v, ties, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {10, 11, 12, 13}}))
	if err != nil || v != inkling2OraclePass || len(ties) != 0 {
		t.Fatalf("verdict %s ties %v err %v, want PASS", v, ties, err)
	}
}

func TestInkling2OracleRunnerUpUnderBoundIsNearTieAndReanchors(t *testing.T) {
	c := syntheticInkling2OracleCase()
	// fak takes the runner-up at the 0.21-nat step, then follows its own path;
	// the re-anchored decode from greedy[:2] must match the rest.
	v, ties, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{
		0: {10, 21, 99, 99},
		2: {12, 13},
	}))
	if err != nil || v != inkling2OracleNearTie || len(ties) != 1 {
		t.Fatalf("verdict %s ties %v err %v, want one NEAR_TIE", v, ties, err)
	}
}

func TestInkling2OracleDivergenceAfterReanchorStillFails(t *testing.T) {
	c := syntheticInkling2OracleCase()
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{
		0: {10, 21, 99, 99},
		2: {12, 23},
	}))
	if err == nil || v != inkling2OracleMismatch {
		t.Fatalf("verdict %s err %v, want FAIL at the 2.5-nat step after re-anchor", v, err)
	}
}

func TestInkling2OracleLargeMarginDivergenceFails(t *testing.T) {
	c := syntheticInkling2OracleCase()
	// fak picks the runner-up where llama.cpp led by 3 nats: a real bug.
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {20, 11, 12, 13}}))
	if err == nil || v != inkling2OracleMismatch {
		t.Fatalf("verdict %s err %v, want FAIL", v, err)
	}
	// Just over the bound is still a failure.
	c.margin[0] = inkling2OracleNearTieNats + 1e-6
	if v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {20, 11, 12, 13}})); err == nil || v != inkling2OracleMismatch {
		t.Fatalf("margin just over the bound: verdict %s err %v, want FAIL", v, err)
	}
}

func TestInkling2OracleSmallMarginNonRunnerUpFails(t *testing.T) {
	c := syntheticInkling2OracleCase()
	// Under the bound, but fak picked neither of the oracle's top two.
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {10, 77, 12, 13}}))
	if err == nil || v != inkling2OracleMismatch {
		t.Fatalf("verdict %s err %v, want FAIL", v, err)
	}
}

func TestInkling2OracleNearTieBudgetIsBounded(t *testing.T) {
	c := inkling2OracleCase{
		name:     "ties",
		prompt:   []int{1},
		greedy:   []int{10, 11, 12, 13},
		runnerUp: []int{20, 21, 22, 23},
		margin:   []float64{0.1, 0.1, 0.1, 0.1},
	}
	v, ties, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{
		0: {20, 0, 0, 0},
		1: {21, 0, 0},
		2: {22, 0},
	}))
	if err == nil || v != inkling2OracleMismatch || len(ties) != inkling2OracleMaxNearTies {
		t.Fatalf("verdict %s ties %v err %v, want FAIL after %d near ties", v, ties, err, inkling2OracleMaxNearTies)
	}
}

func TestInkling2OracleFixtureCarriesMargins(t *testing.T) {
	for _, c := range inkling2OracleCases {
		if len(c.runnerUp) != len(c.greedy) || len(c.margin) != len(c.greedy) {
			t.Fatalf("%s: %d greedy, %d runner-up, %d margin", c.name, len(c.greedy), len(c.runnerUp), len(c.margin))
		}
		for i, m := range c.margin {
			if m < 0 || c.runnerUp[i] == c.greedy[i] {
				t.Fatalf("%s step %d: margin %v runner-up %d greedy %d", c.name, i, m, c.runnerUp[i], c.greedy[i])
			}
		}
	}
}
