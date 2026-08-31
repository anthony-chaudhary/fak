package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// TestSpecDecodeGreedyMatchesGreedyAndAccepts is the #5098 witness for the pure engine
// binding: greedy speculative decode over live Sessions is TOKEN-IDENTICAL to plain greedy
// decode (Session.Generate), and — because a same-weights drafter predicts the target's own
// argmax at every position — the mean acceptance length is > 1 (drafting bought throughput).
func TestSpecDecodeGreedyMatchesGreedyAndAccepts(t *testing.T) {
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	n, k := 24, 4

	want := m.NewSession().Generate(prompt, n) // reference: plain greedy decode

	target := m.NewSession()
	drafter := m.NewSession()
	run, err := SpecDecodeGreedy(target, drafter, prompt, n, k)
	if err != nil {
		t.Fatalf("SpecDecodeGreedy: %v", err)
	}
	if len(run.Output) != len(want) {
		t.Fatalf("len(output)=%d want %d", len(run.Output), len(want))
	}
	for i := range want {
		if run.Output[i] != want[i] {
			t.Fatalf("token %d: spec=%d greedy=%d (speculative decode is NOT lossless)", i, run.Output[i], want[i])
		}
	}
	if run.AcceptedDrafts == 0 {
		t.Fatalf("AcceptedDrafts=0: a same-weights drafter must accept drafts")
	}
	if run.MeanAcceptanceLength <= 1.0 {
		t.Fatalf("MeanAcceptanceLength=%v, want >1 (a same-weights drafter accepts every draft)", run.MeanAcceptanceLength)
	}
}

// coResidentPool builds a residency Pool with two same-family, prefill-shareable models — a
// big verifier and a cheaper small drafter — so PickDrafter/BridgeRoles resolve them as a
// co-resident speculation pair. Both map to sessions of the SAME synthetic weights in the
// tests below, which makes the drafter a perfect predictor (acceptance > 1) while keeping the
// residency descriptors genuinely distinct.
func coResidentPool() *polymodel.Pool {
	pool := polymodel.NewPool(1 << 30)
	pool.Admit(polymodel.Model{ID: "big", Family: "fam", WeightBytes: 100, PrefixDigest: "d"})
	pool.Admit(polymodel.Model{ID: "small", Family: "fam", WeightBytes: 10, PrefixDigest: "d"})
	return pool
}

// TestSpecDecodeGreedyResolvedGateDefaultOff is the #5098 gate assertion: with FAK_POLYMODEL
// unset, the request-path entry NEVER speculates (ok=false), even with a fully co-resident
// drafter available — the caller falls back to plain self-decode.
func TestSpecDecodeGreedyResolvedGateDefaultOff(t *testing.T) {
	if polymodel.Enabled() {
		t.Skip("FAK_POLYMODEL is set in the environment; the default-off assertion needs it unset")
	}
	m := NewSynthetic(cfgV(32, 2, 2, 1, 16, 64))
	pool := coResidentPool()
	sessions := map[polymodel.ModelID]*Session{"big": m.NewSession(), "small": m.NewSession()}

	run, drafter, ok, err := SpecDecodeGreedyResolved([]int{1, 2, 3, 4}, 8, 4, "big", pool, sessions)
	if err != nil {
		t.Fatalf("resolved (gate off): %v", err)
	}
	if ok {
		t.Fatalf("gate default-off VIOLATED: speculation ran with FAK_POLYMODEL unset (drafter=%q, run=%+v)", drafter, run)
	}
}

// TestSpecDecodeGreedyResolvedRunsWhenEnabled proves the gated request path, once opted in via
// FAK_POLYMODEL, resolves the cheapest co-resident drafter (PickDrafter → "small") and runs a
// lossless speculative decode with acceptance > 1.
func TestSpecDecodeGreedyResolvedRunsWhenEnabled(t *testing.T) {
	t.Setenv(polymodel.FlagEnv, "on")
	m := NewSynthetic(cfgV(64, 4, 4, 2, 16, 128))
	pool := coResidentPool()
	sessions := map[polymodel.ModelID]*Session{"big": m.NewSession(), "small": m.NewSession()}
	prompt := []int{1, 2, 3, 4, 5, 6, 7, 8}
	n, k := 24, 4

	want := m.NewSession().Generate(prompt, n)
	run, drafter, ok, err := SpecDecodeGreedyResolved(prompt, n, k, "big", pool, sessions)
	if err != nil {
		t.Fatalf("resolved (gate on): %v", err)
	}
	if !ok {
		t.Fatalf("gate on + co-resident drafter: expected speculation to run")
	}
	if drafter != "small" {
		t.Fatalf("PickDrafter chose %q, want the cheapest co-resident 'small'", drafter)
	}
	if len(run.Output) != len(want) {
		t.Fatalf("len(output)=%d want %d", len(run.Output), len(want))
	}
	for i := range want {
		if run.Output[i] != want[i] {
			t.Fatalf("token %d: spec=%d greedy=%d (resolved path is NOT lossless)", i, run.Output[i], want[i])
		}
	}
	if run.MeanAcceptanceLength <= 1.0 {
		t.Fatalf("MeanAcceptanceLength=%v, want >1", run.MeanAcceptanceLength)
	}
}

type sessionMTPForward struct {
	vocab     int
	failAtPos int
	failErr   error
	calls     []fakeQwen35MTPForwardCall
	closes    int
}

func (f *sessionMTPForward) Forward(pos int, hidden, embedding []float32) ([]float32, error) {
	f.calls = append(f.calls, fakeQwen35MTPForwardCall{
		pos:       pos,
		hidden:    append([]float32(nil), hidden...),
		embedding: append([]float32(nil), embedding...),
	})
	if f.failErr != nil && pos == f.failAtPos {
		return nil, f.failErr
	}
	logits := make([]float32, f.vocab)
	logits[(pos+1)%f.vocab] = 1
	return logits, nil
}

func (f *sessionMTPForward) Close() { f.closes++ }

func TestSpecDecodeGreedyQwen35MTPOptInMatchesTargetGreedy(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	n := 8

	wantSession := m.NewSession()
	wantSession.captureTargetHidden = true
	want := wantSession.Generate(prompt, n)
	t.Cleanup(wantSession.Close)

	target := m.NewSession()
	t.Cleanup(target.Close)
	run, err := SpecDecodeGreedyQwen35MTP(target, prompt, n, 1)
	if err != nil {
		t.Fatalf("opt-in Qwen3.8 MTP decode: %v", err)
	}
	if !reflect.DeepEqual(run.Output, want) {
		t.Fatalf("MTP output = %v, want target-greedy %v", run.Output, want)
	}
	assertQwen35MTPTargetStateEqual(t, target, wantSession)
	if run.Rounds == 0 {
		t.Fatal("opt-in Qwen3.8 MTP decode performed no speculative rounds")
	}
}

func TestSpecDecodeGreedyQwen35MTPBlockAcceptance(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	depth := 3

	for _, tc := range []struct {
		name     string
		accepted int
	}{
		{name: "zero", accepted: 0},
		{name: "partial", accepted: 1},
		{name: "full", accepted: depth},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := m.NewSession()
			target.captureTargetHidden = true
			t.Cleanup(target.Close)
			before := target.Prefill(prompt)
			tx, err := beginQwen35MTPTargetTransaction(target, before)
			if err != nil {
				t.Fatal(err)
			}

			draft := make([]int, depth)
			targetArgmax := make([]int, depth+1)
			targetArgmax[0] = argmaxF32(before)
			logits := append([]float32(nil), before...)
			for i := range draft {
				if i < tc.accepted {
					draft[i] = argmaxF32(logits)
				} else {
					draft[i] = (argmaxF32(logits) + 1) % m.Cfg.VocabSize
				}
				logits = target.Step(draft[i])
				targetArgmax[i+1] = argmaxF32(logits)
			}
			if err := tx.Abort(); err != nil {
				t.Fatal(err)
			}

			tx, err = beginQwen35MTPTargetTransaction(target, before)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := tx.Verify(draft); err != nil {
				t.Fatal(err)
			}
			verifyReceipt := tx.VerificationReceipt()
			if verifyReceipt.Engine != targetVerificationEngine || verifyReceipt.Path != targetVerificationQwen38Path ||
				verifyReceipt.TargetVerificationOperations != 1 || verifyReceipt.TargetDecodeSteps != 0 ||
				!verifyReceipt.OneOperation {
				t.Fatalf("verification receipt = %+v, want one fak-native Qwen3.8 target operation", verifyReceipt)
			}
			res := polymodel.AcceptGreedy(draft, targetArgmax)
			if res.Accepted != tc.accepted {
				t.Fatalf("accepted = %d, want %d", res.Accepted, tc.accepted)
			}
			if res.Accepted+res.EvictKV != len(draft) {
				t.Fatalf("accepted %d + rejected %d != proposed %d", res.Accepted, res.EvictKV, len(draft))
			}
			if _, err := tx.Commit(res.Accepted); err != nil {
				t.Fatal(err)
			}
			commitReceipt := tx.VerificationReceipt()
			if commitReceipt.AcceptedTokens != tc.accepted || commitReceipt.RejectedTokens != depth-tc.accepted {
				t.Fatalf("commit receipt accepted/rejected = %d/%d, want %d/%d",
					commitReceipt.AcceptedTokens, commitReceipt.RejectedTokens, tc.accepted, depth-tc.accepted)
			}
			if !commitReceipt.Accounting.Rollback.Measured || !commitReceipt.Accounting.Synchronization.Measured {
				t.Fatalf("commit receipt omitted rollback/synchronization accounting: %+v", commitReceipt.Accounting)
			}
			if commitReceipt.EndToEndMeasured() {
				t.Fatal("verification receipt incorrectly supports an end-to-end speedup claim")
			}

			correction := targetArgmax[res.Accepted]
			got := append(append([]int(nil), draft[:res.Accepted]...), correction)
			wantSession := m.NewSession()
			wantSession.captureTargetHidden = true
			t.Cleanup(wantSession.Close)
			wantLogits := wantSession.Prefill(prompt)
			normalizeSnapshotForTest(t, wantSession)
			want := make([]int, 0, len(got))
			for range got {
				token := argmaxF32(wantLogits)
				want = append(want, token)
				wantLogits = wantSession.Step(token)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("emitted block = %v, want target-only %v", got, want)
			}
			if next := argmaxF32(target.Step(correction)); next != argmaxF32(wantLogits) {
				t.Fatalf("next-token continuation = %d, want %d", next, argmaxF32(wantLogits))
			}
			assertQwen35MTPTargetStateEqual(t, target, wantSession)
		})
	}
}

func TestQwen35MTPSpeculativeTargetTransaction(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	seed := m.NewSession()
	seed.captureTargetHidden = true
	logits := seed.Prefill(prompt)
	draft := make([]int, 2)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = seed.Step(draft[i])
	}
	seed.Close()

	for _, tc := range []struct {
		name      string
		accepted  int
		abort     bool
		verifyErr bool
	}{
		{name: "zero acceptance", accepted: 0},
		{name: "partial acceptance", accepted: 1},
		{name: "full acceptance", accepted: 2},
		{name: "verifier error", abort: true, verifyErr: true},
		{name: "cancellation", abort: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := m.NewSession()
			target.captureTargetHidden = true
			t.Cleanup(target.Close)
			before := target.Prefill(prompt)
			tx, err := beginQwen35MTPTargetTransaction(target, before)
			if err != nil {
				t.Fatal(err)
			}
			if tc.verifyErr {
				tx.verify = func([]int) ([][]float32, TargetVerificationReceipt, error) {
					panic("injected verifier failure")
				}
			}
			_, verifyErr := tx.Verify(draft)
			if tc.verifyErr && verifyErr == nil {
				t.Fatal("injected verifier failure was not surfaced")
			}
			if !tc.verifyErr && verifyErr != nil {
				t.Fatal(verifyErr)
			}
			if tc.abort {
				if err := tx.Abort(); err != nil {
					t.Fatal(err)
				}
				if err := tx.Abort(); err != nil {
					t.Fatalf("idempotent abort: %v", err)
				}
			} else if _, err := tx.Commit(tc.accepted); err != nil {
				t.Fatal(err)
			}
			if !tx.closed || tx.snapshot != nil || tx.closeCount != 1 {
				t.Fatalf("transaction ownership = closed:%v snapshot:%p closes:%d, want closed/nil/1", tx.closed, tx.snapshot, tx.closeCount)
			}

			want := m.NewSession()
			want.captureTargetHidden = true
			t.Cleanup(want.Close)
			want.Prefill(prompt)
			if !tc.abort {
				for _, token := range draft[:tc.accepted] {
					want.Step(token)
				}
			}
			assertQwen35MTPTargetStateEqual(t, target, want)
		})
	}
}

func assertQwen35MTPTargetStateEqual(t *testing.T, got, want *Session) {
	t.Helper()
	if got.captureTargetHidden != want.captureTargetHidden ||
		!reflect.DeepEqual(got.Cache, want.Cache) ||
		!reflect.DeepEqual(got.targetHidden, want.targetHidden) ||
		!reflect.DeepEqual(got.targetHiddenTokens, want.targetHiddenTokens) {
		t.Fatalf("target state differs: cache len %d/%d hidden %d/%d hidden tokens %v/%v", got.Cache.Len(), want.Cache.Len(), len(got.targetHidden), len(want.targetHidden), got.targetHiddenTokens, want.targetHiddenTokens)
	}
}

func TestQwen35MTPSpeculativeTargetTransactionFailureAtomicity(t *testing.T) {
	m := qwen38HybridMTPEnabledSyntheticModel(t)
	prompt := []int{0, 1}
	panicErr := "injected transaction failure"
	seed := m.NewSession()
	seed.captureTargetHidden = true
	logits := seed.Prefill(prompt)
	draft := make([]int, 3)
	for i := range draft {
		draft[i] = argmaxF32(logits)
		logits = seed.Step(draft[i])
	}
	seed.Close()

	t.Run("verifier panic restores before returning", func(t *testing.T) {
		target := m.NewSession()
		target.captureTargetHidden = true
		t.Cleanup(target.Close)
		before := target.Prefill(prompt)
		want := m.NewSession()
		want.captureTargetHidden = true
		t.Cleanup(want.Close)
		want.Prefill(prompt)
		normalizeSnapshotForTest(t, want)

		tx, err := beginQwen35MTPTargetTransaction(target, before)
		if err != nil {
			t.Fatal(err)
		}
		tx.verify = func(draft []int) ([][]float32, TargetVerificationReceipt, error) {
			target.VerifyForward(draft, nil, nil)
			panic(panicErr)
		}
		rows, err := tx.Verify(draft)
		if err == nil {
			t.Fatal("verifier panic was not surfaced")
		}
		if rows != nil {
			t.Fatalf("failed verification exposed logits: %v", rows)
		}
		if !tx.closed || tx.snapshot != nil || tx.closeCount != 1 {
			t.Fatalf("failed verify ownership = closed:%v snapshot:%p closes:%d, want closed/nil/1", tx.closed, tx.snapshot, tx.closeCount)
		}
		assertQwen35MTPTargetStateEqual(t, target, want)
	})

	t.Run("commit replay panic restores whole block", func(t *testing.T) {
		target := m.NewSession()
		target.captureTargetHidden = true
		t.Cleanup(target.Close)
		before := target.Prefill(prompt)
		want := m.NewSession()
		want.captureTargetHidden = true
		t.Cleanup(want.Close)
		want.Prefill(prompt)
		normalizeSnapshotForTest(t, want)

		tx, err := beginQwen35MTPTargetTransaction(target, before)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := tx.Verify(draft); err != nil {
			t.Fatal(err)
		}
		steps := 0
		tx.step = func(token int) []float32 {
			steps++
			if steps == 2 {
				panic(panicErr)
			}
			return target.Step(token)
		}
		got, err := tx.Commit(len(draft))
		if err == nil {
			t.Fatal("commit replay panic was not surfaced")
		}
		if got != nil {
			t.Fatalf("failed commit exposed logits: %v", got)
		}
		if !tx.closed || tx.snapshot != nil || tx.closeCount != 1 {
			t.Fatalf("failed commit ownership = closed:%v snapshot:%p closes:%d, want closed/nil/1", tx.closed, tx.snapshot, tx.closeCount)
		}
		assertQwen35MTPTargetStateEqual(t, target, want)
	})
}

func TestSpecDecodeGreedyQwen35MTPPassesPromptAndRefreshesEvaluatedHiddenHistory(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{2, 1}
	var prefixes [][]int
	var hidden [][]float32
	var forwards []*sessionMTPForward

	build := func(_ *Model, k int, targetHidden Qwen35MTPTargetHidden, embedding Qwen35MTPTokenEmbedding) (*Qwen35MTPDrafter, error) {
		recordHidden := func(prefix []int) ([]float32, error) {
			value, err := targetHidden(prefix)
			if err == nil {
				prefixes = append(prefixes, append([]int(nil), prefix...))
				hidden = append(hidden, append([]float32(nil), value...))
			}
			return value, err
		}
		return newQwen35MTPDrafter(k, recordHidden, embedding, func() (qwen35MTPDraftForward, error) {
			forward := &sessionMTPForward{vocab: m.Cfg.VocabSize, failAtPos: -1}
			forwards = append(forwards, forward)
			return forward, nil
		})
	}

	target := m.NewSession()
	t.Cleanup(target.Close)
	run, err := specDecodeGreedyQwen35MTP(target, prompt, 6, 1, build)
	if err != nil {
		t.Fatalf("Qwen3.8 MTP decode with recording forward: %v", err)
	}
	if len(prefixes) < len(prompt)+1 {
		t.Fatalf("target-hidden callback prefixes = %v, want prompt plus a subsequent-round extension", prefixes)
	}
	if !reflect.DeepEqual(prefixes[0], prompt[:1]) {
		t.Fatalf("first speculative target prefix = %v, want real prompt prefix %v", prefixes[0], prompt[:1])
	}
	if !reflect.DeepEqual(prefixes[len(prompt)-1], prompt) {
		t.Fatalf("first-round committed prefix = %v, want full real prompt %v", prefixes[len(prompt)-1], prompt)
	}

	greedyHistory := append(append([]int(nil), prompt...), run.Output...)
	baseline := m.NewSession()
	t.Cleanup(baseline.Close)
	if err := evaluateQwen35MTPTargetPrefix(baseline, greedyHistory); err != nil {
		t.Fatalf("evaluate target-greedy hidden baseline: %v", err)
	}
	for i, prefix := range prefixes {
		if !tokenPrefix(prefix, greedyHistory) {
			t.Fatalf("hidden callback prefix %v is not evaluated target-greedy history %v", prefix, greedyHistory)
		}
		pos := len(prefix) - 1
		want, err := baseline.TargetHiddenAt(pos)
		if err != nil {
			t.Fatalf("baseline hidden at position %d: %v", pos, err)
		}
		assertFloat32BitsEqual(t, "subsequent-round target hidden", want, hidden[i])
	}
	if len(forwards) != 1 || forwards[0].closes != 1 {
		t.Fatalf("MTP resource ownership = %d forwards, closes=%v; want one forward closed once", len(forwards), func() int {
			if len(forwards) == 0 {
				return 0
			}
			return forwards[0].closes
		}())
	}
}

func TestSpecDecodeGreedyQwen35MTPSurfacesRuntimeErrorAndCloses(t *testing.T) {
	m := qwen35MTPEnabledSyntheticModel(t)
	prompt := []int{0, 2}
	runtimeFailure := errors.New("injected native MTP forward failure")
	var forward *sessionMTPForward
	build := func(_ *Model, k int, targetHidden Qwen35MTPTargetHidden, embedding Qwen35MTPTokenEmbedding) (*Qwen35MTPDrafter, error) {
		return newQwen35MTPDrafter(k, targetHidden, embedding, func() (qwen35MTPDraftForward, error) {
			forward = &sessionMTPForward{vocab: m.Cfg.VocabSize, failAtPos: len(prompt) - 1, failErr: runtimeFailure}
			return forward, nil
		})
	}

	target := m.NewSession()
	t.Cleanup(target.Close)
	run, err := specDecodeGreedyQwen35MTP(target, prompt, 8, 1, build)
	if !errors.Is(err, runtimeFailure) {
		t.Fatalf("runtime error = %v, want injected MTP failure", err)
	}
	if len(run.Output) != 0 {
		t.Fatalf("output after first-round MTP failure = %v, want no committed tokens", run.Output)
	}
	if forward == nil || forward.closes != 1 {
		t.Fatalf("failed MTP forward close count = %v, want 1", func() int {
			if forward == nil {
				return 0
			}
			return forward.closes
		}())
	}
	want := m.NewSession()
	want.captureTargetHidden = true
	t.Cleanup(want.Close)
	want.Prefill(prompt)
	assertQwen35MTPTargetStateEqual(t, target, want)
}

func TestSpecDecodeGreedyQwen35MTPRejectsIneligibleAndUnsafeTargets(t *testing.T) {
	baseCfg := cfgV(8, 1, 2, 1, 4, 16)
	baseCfg.ModelType = "qwen3_5_text"
	baseCfg.MTPNumHiddenLayers = 1

	t.Run("no retained MTP", func(t *testing.T) {
		target := NewSynthetic(baseCfg).NewSession()
		t.Cleanup(target.Close)
		unsupported := assertQwen35MTPUnsupported(t, func() error {
			_, err := SpecDecodeGreedyQwen35MTP(target, []int{1}, 2, 1)
			return err
		}())
		if unsupported.Reason != "mtp-tensors-not-retained" {
			t.Fatalf("unsupported reason = %q, want mtp-tensors-not-retained", unsupported.Reason)
		}
	})

	t.Run("accelerated target has no exact hidden capture", func(t *testing.T) {
		target := qwen35MTPEnabledSyntheticModel(t).NewSession()
		target.Quant = true
		t.Cleanup(target.Close)
		assertQwen35MTPUnsupported(t, func() error {
			_, err := SpecDecodeGreedyQwen35MTP(target, []int{1}, 2, 1)
			return err
		}())
	})

	t.Run("one layer cannot safely draft two tokens", func(t *testing.T) {
		target := qwen35MTPEnabledSyntheticModel(t).NewSession()
		t.Cleanup(target.Close)
		assertQwen35MTPUnsupported(t, func() error {
			_, err := SpecDecodeGreedyQwen35MTP(target, []int{1}, 2, 2)
			return err
		}())
	})

	t.Run("empty prompt", func(t *testing.T) {
		target := qwen35MTPEnabledSyntheticModel(t).NewSession()
		t.Cleanup(target.Close)
		if _, err := SpecDecodeGreedyQwen35MTP(target, nil, 2, 1); !errors.Is(err, ErrQwen35MTPEmptyPrefix) {
			t.Fatalf("empty-prompt error = %v, want ErrQwen35MTPEmptyPrefix", err)
		}
	})
}
