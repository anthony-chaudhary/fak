package polymodel

import (
	"errors"
	"testing"
)

// specdecode_test.go — the #4877 done-condition witness: the live draft→verify→accept→
// rollback loop (SpecDecode) produces output TOKEN-IDENTICAL to plain sequential greedy
// decode of the target, for ANY drafter, and reports a mean acceptance length > 1 when the
// drafting bought throughput. The witness named by the issue is
// `go test ./internal/polymodel -run SpecDecodeLossless`.
//
// The target is a deterministic ORACLE (a pure function of the committed context) standing
// in for model.Session.VerifyForward — the loop is engine-agnostic, so the same control
// structure a real target drives is exercised here with no weights, no GPU, no backend.
// The oracle IS the target: the reference greedy decode and the verify pass are both built
// from it, so "lossless" means the loop's commit/rollback bookkeeping reproduces the
// target's own greedy stream exactly. The drafters are INDEPENDENT of the target's quality
// — that independence is the property under test.

const specOracleVocab = 64

// oracleStep is the deterministic target: the token it would greedily emit next given the
// committed context. It depends on the last token and the position so the sequence is
// non-trivial (not a fixed point), and it never emits token 0 (reserved as an EOS the
// lossless runs disable) because the +1 bias keeps it in [1, vocab).
func oracleStep(committed []int) int {
	last := 0
	if len(committed) > 0 {
		last = committed[len(committed)-1]
	}
	return 1 + (last*31+len(committed)*7+13)%(specOracleVocab-1)
}

// oracleVerifier is the target's verify pass built from the oracle: the target argmax at
// the len(draft)+1 panel positions — index 0 after the committed prefix, index i after
// committed+draft[:i]. This is exactly the argmax vector model.Session.VerifyForward yields
// (with the current-position logits prepended).
func oracleVerifier(committed, draft []int) []int {
	ctx := append([]int(nil), committed...)
	ta := make([]int, 0, len(draft)+1)
	ta = append(ta, oracleStep(ctx))
	for _, dt := range draft {
		ctx = append(ctx, dt)
		ta = append(ta, oracleStep(ctx))
	}
	return ta
}

// oracleGreedy is plain sequential greedy decode of the target — the reference the
// speculative loop must reproduce token-for-token.
func oracleGreedy(prompt []int, n int) []int {
	ctx := append([]int(nil), prompt...)
	out := make([]int, 0, n)
	for len(out) < n {
		t := oracleStep(ctx)
		out = append(out, t)
		ctx = append(ctx, t)
	}
	return out
}

// perfectDrafter reproduces the oracle exactly, so every drafted token matches the
// target's argmax → a K-token draft advances K+1 real tokens per round (max acceptance).
func perfectDrafter(k int) Drafter {
	return func(committed []int) []int {
		ctx := append([]int(nil), committed...)
		d := make([]int, 0, k)
		for j := 0; j < k; j++ {
			t := oracleStep(ctx)
			d = append(d, t)
			ctx = append(ctx, t)
		}
		return d
	}
}

// partialDrafter is perfect for the first `good` tokens of each draft and deliberately
// wrong after, so acceptance is EXACTLY `good` drafts + 1 correction every round — a
// provable mean acceptance length of good+1 (> 1) with guaranteed rollbacks of K-good KV.
func partialDrafter(k, good int) Drafter {
	return func(committed []int) []int {
		ctx := append([]int(nil), committed...)
		d := make([]int, 0, k)
		for j := 0; j < k; j++ {
			t := oracleStep(ctx)
			if j >= good {
				t = 1 + (t % (specOracleVocab - 1)) // shift off the correct token → forced reject
			}
			d = append(d, t)
			ctx = append(ctx, t)
		}
		return d
	}
}

// wrongDrafter proposes a first token that CANNOT match the target's argmax (target+1),
// so acceptance is 0 every round: mean acceptance length is exactly 1.0 and every round
// rolls back all K drafted KV positions — the hardest rollback stress, still lossless.
func wrongDrafter(k int) Drafter {
	return func(committed []int) []int {
		first := 1 + (oracleStep(committed) % (specOracleVocab - 1)) // guaranteed != oracleStep(committed)
		d := make([]int, k)
		for j := range d {
			d[j] = first
		}
		return d
	}
}

// TestSpecDecodeLossless is the #4877 witness: across four drafter qualities the loop
// output is token-identical to sequential greedy, the mean acceptance length is reported,
// the rollback hook fires exactly the accounted KV positions, and the rollback path is
// genuinely exercised (non-vacuous).
func TestSpecDecodeLossless(t *testing.T) {
	const n, k = 24, 4
	prompt := []int{2, 9, 5, 40, 17}
	want := oracleGreedy(prompt, n)

	cases := []struct {
		name          string
		draft         Drafter
		wantMeanAbove float64 // MeanAcceptanceLength must exceed this
		wantEvictPos  bool    // rollback path must have run (EvictKV > 0)
	}{
		{"perfect", perfectDrafter(k), 1.0, false}, // all accepted → advance k+1/round, no rollback
		{"partial", partialDrafter(k, 2), 1.0, true},
		{"wrong", wrongDrafter(k), 0.0, true}, // acceptance 1.0 exactly (not > 1) → wantMeanAbove 0
		{"none", nil, 0.0, false},             // plain decode, no drafts, no rollback
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var evictSeen int
			run, err := SpecDecode(prompt, tc.draft, oracleVerifier, SpecDecodeConfig{
				MaxNewTokens: n,
				MaxDraft:     k,
				Rollback:     func(e int) { evictSeen += e },
			})
			if err != nil {
				t.Fatalf("SpecDecode: %v", err)
			}
			// (1) LOSSLESS: token-identical to sequential greedy over the full budget.
			if len(run.Output) != n {
				t.Fatalf("emitted %d tokens, want %d", len(run.Output), n)
			}
			for i := 0; i < n; i++ {
				if run.Output[i] != want[i] {
					t.Fatalf("LOSSLESS VIOLATED at token %d: spec=%d greedy=%d (drafter=%s)",
						i, run.Output[i], want[i], tc.name)
				}
			}
			// (2) ACCEPTANCE-LENGTH metric emitted and > 1 where drafting helped.
			if run.MeanAcceptanceLength <= tc.wantMeanAbove {
				t.Fatalf("mean acceptance length = %.3f, want > %.3f (drafter=%s)",
					run.MeanAcceptanceLength, tc.wantMeanAbove, tc.name)
			}
			// The metric must equal emitted/rounds by construction.
			if run.Rounds == 0 || run.MeanAcceptanceLength != float64(len(run.Output))/float64(run.Rounds) {
				t.Fatalf("mean acceptance length %.3f != emitted/rounds %d/%d",
					run.MeanAcceptanceLength, len(run.Output), run.Rounds)
			}
			// (3) The Rollback hook fired exactly the accounted KV positions (non-vacuous
			// where rejections are guaranteed).
			if evictSeen != run.EvictKV {
				t.Fatalf("rollback hook saw %d KV, run accounted %d", evictSeen, run.EvictKV)
			}
			if tc.wantEvictPos && run.EvictKV == 0 {
				t.Fatalf("drafter=%s should have rolled back rejected drafts, but EvictKV=0 (vacuous)", tc.name)
			}
		})
	}
}

// TestSpecDecodeLosslessPerfectMaxAcceptance proves the perfect drafter advances the full
// K+1 real tokens per verify pass — the maximum speculative speedup, and a concrete mean
// acceptance length well above 1.
func TestSpecDecodeLosslessPerfectMaxAcceptance(t *testing.T) {
	const n, k = 20, 4
	prompt := []int{7, 3, 3, 1}
	run, err := SpecDecode(prompt, perfectDrafter(k), oracleVerifier, SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k})
	if err != nil {
		t.Fatalf("SpecDecode: %v", err)
	}
	// A perfect drafter accepts all k and takes the bonus, so each round advances k+1=5.
	// n=20 is a multiple of 5, so it is exactly n/(k+1)=4 rounds with mean 5.0.
	if run.Rounds != n/(k+1) {
		t.Fatalf("rounds = %d, want %d (perfect drafter advances k+1/round)", run.Rounds, n/(k+1))
	}
	if run.MeanAcceptanceLength != float64(k+1) {
		t.Fatalf("mean acceptance length = %.3f, want %.1f", run.MeanAcceptanceLength, float64(k+1))
	}
	if run.EvictKV != 0 {
		t.Fatalf("perfect drafter should evict nothing, got %d", run.EvictKV)
	}
}

// TestSpecDecodeLosslessPartialAcceptanceExact proves the partial drafter's acceptance is
// exactly `good`+1 tokens per round with K-good rolled back — a deterministic, non-flaky
// acceptance-length assertion.
func TestSpecDecodeLosslessPartialAcceptanceExact(t *testing.T) {
	const n, k, good = 24, 4, 2
	prompt := []int{11, 4}
	run, err := SpecDecode(prompt, partialDrafter(k, good), oracleVerifier, SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k})
	if err != nil {
		t.Fatalf("SpecDecode: %v", err)
	}
	// good accepted + 1 correction = good+1 real tokens per round; n divisible by good+1.
	if adv := good + 1; run.Rounds != n/adv || run.MeanAcceptanceLength != float64(adv) {
		t.Fatalf("rounds=%d mean=%.3f, want rounds=%d mean=%.1f", run.Rounds, run.MeanAcceptanceLength, n/adv, float64(adv))
	}
	// K-good drafts rejected every round → (k-good) evicted per round.
	if wantEvict := (k - good) * run.Rounds; run.EvictKV != wantEvict {
		t.Fatalf("evicted %d KV, want %d", run.EvictKV, wantEvict)
	}
}

// TestSpecDecodeStopToken proves the run halts once an EOS token is committed, even below
// the MaxNewTokens budget, and does not emit past it.
func TestSpecDecodeStopToken(t *testing.T) {
	prompt := []int{1}
	// Find the greedy stream, pick a token in it as the stop token.
	stream := oracleGreedy(prompt, 30)
	stop := stream[5]
	// The first occurrence of `stop` bounds where the run must end.
	firstStop := 0
	for firstStop < len(stream) && stream[firstStop] != stop {
		firstStop++
	}
	run, err := SpecDecode(prompt, perfectDrafter(4), oracleVerifier, SpecDecodeConfig{
		MaxNewTokens: 30, MaxDraft: 4, StopToken: stop, StopEnabled: true,
	})
	if err != nil {
		t.Fatalf("SpecDecode: %v", err)
	}
	if len(run.Output) == 0 || run.Output[len(run.Output)-1] != stop {
		t.Fatalf("run should end on the stop token %d, got tail %v", stop, run.Output)
	}
	if len(run.Output) != firstStop+1 {
		t.Fatalf("run emitted %d tokens, want %d (stop at first occurrence)", len(run.Output), firstStop+1)
	}
	// Everything before the stop is still greedy-identical.
	for i := 0; i < firstStop; i++ {
		if run.Output[i] != stream[i] {
			t.Fatalf("token %d = %d, want greedy %d", i, run.Output[i], stream[i])
		}
	}
}

// TestSpecDecodeDrafterOwnsContext addresses the #4877 confusion-risk: a drafter running
// its OWN (here, deliberately shorter/independent) context still yields lossless output,
// because every drafted token is gated by the target's verify pass. A drafter that ignores
// most of the committed context and proposes from a truncated window cannot corrupt the
// stream — it only lowers acceptance.
func TestSpecDecodeDrafterOwnsContext(t *testing.T) {
	const n, k = 18, 3
	prompt := []int{5, 8, 2, 9}
	want := oracleGreedy(prompt, n)
	// A drafter that only ever looks at the LAST committed token (its own tiny context
	// window), independent of the target's full-context session.
	shortCtxDrafter := func(committed []int) []int {
		var tiny []int
		if len(committed) > 0 {
			tiny = committed[len(committed)-1:]
		}
		ctx := append([]int(nil), tiny...)
		d := make([]int, 0, k)
		for j := 0; j < k; j++ {
			t := oracleStep(ctx)
			d = append(d, t)
			ctx = append(ctx, t)
		}
		return d
	}
	run, err := SpecDecode(prompt, shortCtxDrafter, oracleVerifier, SpecDecodeConfig{MaxNewTokens: n, MaxDraft: k})
	if err != nil {
		t.Fatalf("SpecDecode: %v", err)
	}
	if len(run.Output) != n {
		t.Fatalf("emitted %d, want %d", len(run.Output), n)
	}
	for i := 0; i < n; i++ {
		if run.Output[i] != want[i] {
			t.Fatalf("LOSSLESS VIOLATED at %d: spec=%d greedy=%d (short-context drafter)", i, run.Output[i], want[i])
		}
	}
}

// TestSpecDecodeMaxDraftCap proves an over-eager drafter's proposal is truncated to
// MaxDraft, so the loop never verifies more than the configured depth.
func TestSpecDecodeMaxDraftCap(t *testing.T) {
	prompt := []int{3}
	// Drafter proposes 10 tokens; MaxDraft caps K at 2, so at most 3 real tokens/round and
	// at most 2 drafted/round.
	over := perfectDrafter(10)
	run, err := SpecDecode(prompt, over, oracleVerifier, SpecDecodeConfig{MaxNewTokens: 12, MaxDraft: 2})
	if err != nil {
		t.Fatalf("SpecDecode: %v", err)
	}
	if run.DraftedTokens > 2*run.Rounds {
		t.Fatalf("drafted %d over %d rounds exceeds MaxDraft=2/round", run.DraftedTokens, run.Rounds)
	}
	// Perfect drafter capped at K=2 advances 3/round; still lossless.
	want := oracleGreedy(prompt, 12)
	for i := range want {
		if run.Output[i] != want[i] {
			t.Fatalf("capped run not lossless at %d: %d != %d", i, run.Output[i], want[i])
		}
	}
}

// TestSpecDecodeErrors proves the loop's guards: a nil Verifier refuses, an empty-argmax
// (contract-violating) Verifier is caught rather than spinning, and a zero budget is a
// clean no-op.
func TestSpecDecodeErrors(t *testing.T) {
	if _, err := SpecDecode([]int{1}, perfectDrafter(2), nil, SpecDecodeConfig{MaxNewTokens: 4}); !errors.Is(err, ErrNoVerifier) {
		t.Fatalf("nil verifier should be ErrNoVerifier, got %v", err)
	}
	stalled := func(committed, draft []int) []int { return nil } // violates len==len(draft)+1
	if _, err := SpecDecode([]int{1}, nil, stalled, SpecDecodeConfig{MaxNewTokens: 4}); !errors.Is(err, ErrVerifierStalled) {
		t.Fatalf("empty-argmax verifier should be ErrVerifierStalled, got %v", err)
	}
	run, err := SpecDecode([]int{1}, perfectDrafter(2), oracleVerifier, SpecDecodeConfig{MaxNewTokens: 0})
	if err != nil || len(run.Output) != 0 || run.Rounds != 0 {
		t.Fatalf("zero budget should be a clean no-op, got run=%+v err=%v", run, err)
	}
}
