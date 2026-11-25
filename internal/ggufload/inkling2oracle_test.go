package ggufload

import (
	"fmt"
	"math"
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

// inkling2NearTie judges a greedy flip from both engines' side: the reference
// led the candidate's token, its runner-up, by under the bound, or the
// reference's token sits under the bound below the candidate's own top-1.
// Reduction-order noise only needs one of the two distributions to be near
// tied to flip the pick, while a real numeric bug moves both by nats. candGap
// is +Inf when the candidate's logits are not available.
func inkling2NearTie(refMargin float64, gotIsRefRunnerUp bool, candGap float64) bool {
	return (gotIsRefRunnerUp && refMargin < inkling2OracleNearTieNats) || candGap < inkling2OracleNearTieNats
}

// inkling2OracleGap teacher-forces forced after prompt the way the decode under
// test reached that position, and returns fak's top-1 token there and how far
// tok's logprob sits below it, in nats. The fak-side clause counts only when
// that top-1 is the token the decode picked.
type inkling2OracleGap func(prompt, forced []int, tok int) (top int, gap float64, err error)

// inkling2OracleDivergence compares got, decoded after prompt+greedy[:from],
// with greedy[from:]. It returns the first divergent greedy index (-1 when
// none) and whether that divergence is an admissible near tie by
// inkling2NearTie. A nil gap judges by the oracle's margin alone.
func inkling2OracleDivergence(c inkling2OracleCase, from int, got []int, gap inkling2OracleGap) (at int, nearTie bool, why string, err error) {
	want := c.greedy[from:]
	if len(got) != len(want) {
		return from + min(len(got), len(want)), false, fmt.Sprintf("decoded %d tokens, oracle has %d", len(got), len(want)), nil
	}
	for k := range got {
		if got[k] == want[k] {
			continue
		}
		i := from + k
		fakGap, fakSide := math.Inf(1), "fak gap n/a"
		if gap != nil {
			prompt := append(append([]int(nil), c.prompt...), c.greedy[:from]...)
			top, g, err := gap(prompt, c.greedy[from:i], want[k])
			if err != nil {
				return i, false, "", err
			}
			fakSide = fmt.Sprintf("fak top %d, llama.cpp's token %.4f nats below it", top, g)
			// The gap only describes the pick when the teacher-forced top-1 is the
			// token fak decoded; otherwise fak's own distribution says nothing about it.
			if top == got[k] {
				fakGap = g
			} else {
				fakSide += fmt.Sprintf(", but fak decoded %d", got[k])
			}
		}
		why = fmt.Sprintf("greedy token %d = %d, llama.cpp = %d (runner-up %d, margin %.4f nats; %s; near-tie bound %.2f)",
			i, got[k], want[k], c.runnerUp[i], c.margin[i], fakSide, inkling2OracleNearTieNats)
		return i, inkling2NearTie(c.margin[i], got[k] == c.runnerUp[i], fakGap), why, nil
	}
	return -1, false, "", nil
}

// inkling2OracleCompare runs decode from the prompt and, at each admissible
// near tie, re-anchors on the oracle's token by decoding again from
// prompt+greedy[:at+1], so the steps after a tie are still compared.
func inkling2OracleCompare(c inkling2OracleCase, decode func(prompt []int, n int) ([]int, error), gap inkling2OracleGap) (inkling2OracleVerdict, []string, error) {
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
		at, nearTie, why, err := inkling2OracleDivergence(c, from, got, gap)
		if err != nil {
			return inkling2OracleMismatch, ties, err
		}
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
func runInkling2Oracle(t *testing.T, c inkling2OracleCase, decode func(prompt []int, n int) ([]int, error), gap inkling2OracleGap) {
	t.Helper()
	v, ties, err := inkling2OracleCompare(c, decode, gap)
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
	v, ties, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {10, 11, 12, 13}}), nil)
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
	}), nil)
	if err != nil || v != inkling2OracleNearTie || len(ties) != 1 {
		t.Fatalf("verdict %s ties %v err %v, want one NEAR_TIE", v, ties, err)
	}
}

func TestInkling2OracleDivergenceAfterReanchorStillFails(t *testing.T) {
	c := syntheticInkling2OracleCase()
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{
		0: {10, 21, 99, 99},
		2: {12, 23},
	}), nil)
	if err == nil || v != inkling2OracleMismatch {
		t.Fatalf("verdict %s err %v, want FAIL at the 2.5-nat step after re-anchor", v, err)
	}
}

func TestInkling2OracleLargeMarginDivergenceFails(t *testing.T) {
	c := syntheticInkling2OracleCase()
	// fak picks the runner-up where llama.cpp led by 3 nats: a real bug.
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {20, 11, 12, 13}}), nil)
	if err == nil || v != inkling2OracleMismatch {
		t.Fatalf("verdict %s err %v, want FAIL", v, err)
	}
	// Just over the bound is still a failure.
	c.margin[0] = inkling2OracleNearTieNats + 1e-6
	if v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {20, 11, 12, 13}}), nil); err == nil || v != inkling2OracleMismatch {
		t.Fatalf("margin just over the bound: verdict %s err %v, want FAIL", v, err)
	}
}

func TestInkling2OracleSmallMarginNonRunnerUpFails(t *testing.T) {
	c := syntheticInkling2OracleCase()
	// Under the bound, but fak picked neither of the oracle's top two.
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {10, 77, 12, 13}}), nil)
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
	}), nil)
	if err == nil || v != inkling2OracleMismatch || len(ties) != inkling2OracleMaxNearTies {
		t.Fatalf("verdict %s ties %v err %v, want FAIL after %d near ties", v, ties, err, inkling2OracleMaxNearTies)
	}
}

// passphraseInkling2OracleCase is a flip llama.cpp's margin alone refuses: it
// leads fak's pick, its runner-up, by 0.548 nats, just over the bound, while
// fak's own top-1 leads llama.cpp's token by only 0.351.
func passphraseInkling2OracleCase() inkling2OracleCase {
	return inkling2OracleCase{
		name:     "passphrase-shape",
		prompt:   []int{1, 2},
		greedy:   []int{10, 150008, 12},
		runnerUp: []int{20, 149884, 22},
		margin:   []float64{3, 0.548, 4},
	}
}

// fixedGap answers every gap query with one fak-side gap and records the
// teacher-forced position it was asked about.
func fixedGap(top int, g float64, asked *[][]int) inkling2OracleGap {
	return func(prompt, forced []int, tok int) (int, float64, error) {
		*asked = append(*asked, append(append(append([]int(nil), prompt...), forced...), tok))
		return top, g, nil
	}
}

func TestInkling2OracleFakSideGapUnderBoundIsNearTie(t *testing.T) {
	c := passphraseInkling2OracleCase()
	decode := replayDecode(c, map[int][]int{0: {10, 149884, 99}, 2: {12}})
	var asked [][]int
	v, ties, err := inkling2OracleCompare(c, decode, fixedGap(149884, 0.351, &asked))
	if err != nil || v != inkling2OracleNearTie || len(ties) != 1 {
		t.Fatalf("verdict %s ties %v err %v, want one NEAR_TIE from fak's 0.351-nat gap", v, ties, err)
	}
	if want := []int{1, 2, 10, 150008}; len(asked) != 1 || fmt.Sprint(asked[0]) != fmt.Sprint(want) {
		t.Fatalf("gap asked %v, want one query at prompt+greedy[:1] for llama.cpp's token %v", asked, want)
	}
	if !strings.Contains(ties[0], "0.3510 nats below") || !strings.Contains(ties[0], "margin 0.5480") {
		t.Fatalf("tie %q does not report both sides' gaps", ties[0])
	}
	// The llama.cpp margin alone still refuses it, as before.
	if v, _, err := inkling2OracleCompare(c, decode, nil); err == nil || v != inkling2OracleMismatch {
		t.Fatalf("without fak's gap: verdict %s err %v, want FAIL on the 0.548-nat margin", v, err)
	}
}

func TestInkling2OracleWideGapOnBothSidesFails(t *testing.T) {
	c := passphraseInkling2OracleCase()
	c.margin[1] = 0.9
	decode := replayDecode(c, map[int][]int{0: {10, 149884, 99}})
	for _, g := range []float64{0.8, inkling2OracleNearTieNats + 1e-6} {
		var asked [][]int
		v, _, err := inkling2OracleCompare(c, decode, fixedGap(149884, g, &asked))
		if err == nil || v != inkling2OracleMismatch {
			t.Fatalf("llama margin 0.9, fak gap %v: verdict %s err %v, want FAIL", g, v, err)
		}
	}
}

func TestInkling2OracleFakSideGapRescuesOracleTopTwoMiss(t *testing.T) {
	c := passphraseInkling2OracleCase()
	// fak picked a token outside llama.cpp's top two, but llama.cpp's token is
	// within the bound of fak's top-1: fak itself is near tied there.
	decode := replayDecode(c, map[int][]int{0: {10, 77, 99}, 2: {12}})
	var asked [][]int
	if v, _, err := inkling2OracleCompare(c, decode, fixedGap(77, 0.1, &asked)); err != nil || v != inkling2OracleNearTie {
		t.Fatalf("verdict %s err %v, want NEAR_TIE", v, err)
	}
	asked = nil
	if v, _, err := inkling2OracleCompare(c, decode, fixedGap(77, 2, &asked)); err == nil || v != inkling2OracleMismatch {
		t.Fatalf("fak gap 2: verdict %s err %v, want FAIL", v, err)
	}
}

func TestInkling2OracleFakSideGapNeedsFakTopToBeItsPick(t *testing.T) {
	c := passphraseInkling2OracleCase()
	// Teacher-forced, fak's top-1 is llama.cpp's own token (gap 0), yet the
	// decode picked 77: the gap says nothing about that pick.
	decode := replayDecode(c, map[int][]int{0: {10, 77, 99}, 2: {12}})
	var asked [][]int
	v, _, err := inkling2OracleCompare(c, decode, fixedGap(150008, 0, &asked))
	if err == nil || v != inkling2OracleMismatch || !strings.Contains(err.Error(), "but fak decoded 77") {
		t.Fatalf("verdict %s err %v, want FAIL naming the decoded token", v, err)
	}
}

func TestInkling2OracleGapErrorFails(t *testing.T) {
	c := passphraseInkling2OracleCase()
	boom := func([]int, []int, int) (int, float64, error) { return 0, 0, fmt.Errorf("logits unavailable") }
	v, _, err := inkling2OracleCompare(c, replayDecode(c, map[int][]int{0: {10, 149884, 99}}), boom)
	if err == nil || v != inkling2OracleMismatch || !strings.Contains(err.Error(), "logits unavailable") {
		t.Fatalf("verdict %s err %v, want the gap error as FAIL", v, err)
	}
}

func TestInkling2NearTieAcceptsEitherSide(t *testing.T) {
	const b = inkling2OracleNearTieNats
	for _, tc := range []struct {
		margin   float64
		runnerUp bool
		gap      float64
		want     bool
	}{
		{0.548, true, 0.351, true},
		{0.351, true, 0.548, true},
		{0.6, true, 0.6, false},
		{0.1, false, math.Inf(1), false},
		{0.1, true, math.Inf(1), true},
		{b, true, b, false},
	} {
		if got := inkling2NearTie(tc.margin, tc.runnerUp, tc.gap); got != tc.want {
			t.Errorf("inkling2NearTie(%v, %v, %v) = %v, want %v", tc.margin, tc.runnerUp, tc.gap, got, tc.want)
		}
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
