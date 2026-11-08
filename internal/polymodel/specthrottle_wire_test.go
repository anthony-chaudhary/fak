package polymodel

import (
	"reflect"
	"testing"
)

// specthrottle_wire_test.go — witness that the adaptive draft-length throttle is WIRED into
// the live SpecDecode loop (evergreen-spec-decode-auto): the loop must feed each draft
// round's (accepted, proposed) counts to the bound throttle and cap the NEXT round's draft
// at its verdict, stopping drafting entirely when acceptance collapses — while the emitted
// stream stays token-identical to plain greedy decode (losslessness preserved under
// adaptation).

// regimeDrafter proposes k tokens: while len(committed) < switchAt it proposes the target's
// own greedy continuation (a perfect drafter against the stateful oracle); at or past
// switchAt every proposal is shifted by one (guaranteed mismatch at its own position, kept
// inside [1, vocab-1]). It models a workload whose acceptance collapses partway through the
// run — the exact case the adaptive throttle exists for.
func regimeDrafter(k, switchAt int) Drafter {
	return func(committed []int) []int {
		d := make([]int, k)
		ctx := append([]int(nil), committed...)
		for j := range d {
			t := oracleStep(ctx)
			if len(committed)+j < switchAt {
				d[j] = t
			} else {
				d[j] = t%(specOracleVocab-1) + 1
			}
			ctx = append(ctx, t)
		}
		return d
	}
}

func TestSpecDecodeThrottleAdaptsAndStops(t *testing.T) {
	const (
		promptLen = 4
		budget    = 60
		switchAt  = 30 // committed length at which acceptance collapses
		staticCap = 8
	)
	prompt := make([]int, promptLen)
	for i := range prompt {
		prompt[i] = 1 + i
	}
	draft := regimeDrafter(8, switchAt)

	// Static baseline: fixed K=8 cap, no throttle — keeps drafting full depth even after
	// acceptance collapses to zero.
	staticRun, err := SpecDecode(prompt, draft, oracleVerifier, SpecDecodeConfig{MaxNewTokens: budget, MaxDraft: staticCap})
	if err != nil {
		t.Fatalf("static SpecDecode: %v", err)
	}

	// Adaptive: the same loop with the acceptance-hysteresis throttle bound. High early
	// acceptance grows K toward the max; the collapse shrinks it to the floor and stops.
	th := NewDraftLengthThrottle(DraftLengthThrottleConfig{Window: 2, Low: 0.3, High: 0.7, Floor: 1, Max: 8, Step: 1, StopStreak: 3, Start: 4})
	adaptiveRun, err := SpecDecode(prompt, draft, oracleVerifier, SpecDecodeConfig{MaxNewTokens: budget, MaxDraft: staticCap, Throttle: th})
	if err != nil {
		t.Fatalf("throttled SpecDecode: %v", err)
	}

	// LOSSLESSNESS PRESERVED: adaptation changes only how far rounds draft, never the stream —
	// both runs must equal plain sequential greedy decode of the target.
	want := oracleGreedy(prompt, budget)
	if !reflect.DeepEqual(staticRun.Output, want) {
		t.Fatalf("static output diverged from greedy: got %v want %v", staticRun.Output, want)
	}
	if !reflect.DeepEqual(adaptiveRun.Output, want) {
		t.Fatalf("throttled output diverged from greedy: got %v want %v", adaptiveRun.Output, want)
	}

	// The throttle must have learned the collapse: ended fully stopped (drafting off).
	if th.Drafting() || th.Length() != 0 {
		t.Fatalf("throttle still drafting after acceptance collapse: length=%d rate=%.2f", th.Length(), th.Rate())
	}

	// And it saved verify work late in the run: fewer drafted tokens overall...
	if adaptiveRun.DraftedTokens >= staticRun.DraftedTokens {
		t.Errorf("throttled run drafted %d tokens, not below static %d", adaptiveRun.DraftedTokens, staticRun.DraftedTokens)
	}
	// ...at the cost of more rounds (plain single-token decode once drafting stopped).
	if adaptiveRun.Rounds <= staticRun.Rounds {
		t.Errorf("throttled run took %d rounds, not above static %d", adaptiveRun.Rounds, staticRun.Rounds)
	}
}
