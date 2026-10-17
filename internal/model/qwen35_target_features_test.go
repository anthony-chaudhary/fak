package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime medium est=2s lane=default
func TestQwen35TargetFeaturesExactGreedyAndEveryAcceptedPrefix(t *testing.T) {
	m := qwen35TargetFeatureTestModel(t)
	prompt, layers := []int{3, 7, 11}, []int{0, 2, 3}
	for accepted := 0; accepted <= qwen35TargetFeatureMaxTokens; accepted++ {
		t.Run(fmt.Sprintf("accepted-%d", accepted), func(t *testing.T) {
			target, plain, manual, proposer := m.NewSession(), m.NewSession(), m.NewSession(), m.NewSession()
			for _, s := range []*Session{target, plain, manual, proposer} {
				t.Cleanup(s.Close)
				s.captureTargetHidden = true
			}
			before := target.Prefill(prompt)
			wantBoundary := plain.Prefill(prompt)
			manual.Prefill(prompt)
			proposalLogits := proposer.Prefill(prompt)
			normalizeSnapshotForTest(t, target)
			normalizeSnapshotForTest(t, plain)
			draft := make([]int, qwen35TargetFeatureMaxTokens)
			for i := range draft {
				draft[i] = argmaxF32(proposalLogits)
				proposalLogits = proposer.Step(draft[i])
			}
			if accepted < len(draft) {
				draft[accepted] = (draft[accepted] + 1) % m.Cfg.VocabSize
			}
			f, err := beginQwen35TargetFeatureTransaction(target, before, layers)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = f.Abort() })
			verified, err := f.Verify(context.Background(), draft)
			if err != nil {
				t.Fatal(err)
			}
			gotAccepted, current := 0, before
			for i, token := range draft {
				if token != argmaxF32(current) {
					break
				}
				gotAccepted++
				current = verified[i]
			}
			if gotAccepted != accepted {
				t.Fatalf("accepted %d, want %d", gotAccepted, accepted)
			}
			gotBoundary, features, err := f.Commit(context.Background(), gotAccepted)
			if err != nil || len(features) != accepted {
				t.Fatalf("commit rows=%d err=%v", len(features), err)
			}
			for i, token := range draft[:accepted] {
				if token != argmaxF32(wantBoundary) {
					t.Fatalf("token %d differs from plain greedy", i)
				}
				wantBoundary = plain.Step(token)
				wantFeatures := qwen35TargetFeaturesManualStep(manual, token, layers)
				assertQwen35TargetFeatureRow(t, wantFeatures, features[i])
			}
			assertFloat32BitsEqual(t, "commit boundary", wantBoundary, gotBoundary)
			assertQwen35MTPTargetStateEqual(t, target, plain)
			if target.qwen35TargetFeatureTxn != nil || target.qwen35TargetLayerCapture != nil || f.rows != nil || f.tx.closeCount != 1 {
				t.Fatal("closed transaction retained session ownership or staged features")
			}
			// A correction/bonus token and further continuation must use the exact
			// same logits and recurrent state as ordinary single-token decoding.
			for i := 0; i < 3; i++ {
				token := argmaxF32(wantBoundary)
				if token != argmaxF32(gotBoundary) {
					t.Fatalf("continuation token %d differs from plain greedy", i)
				}
				wantBoundary, gotBoundary = plain.Step(token), target.Step(token)
				assertFloat32BitsEqual(t, "continuation", wantBoundary, gotBoundary)
			}
			assertQwen35MTPTargetStateEqual(t, target, plain)
		})
	}
}

func qwen35TargetFeaturesManualStep(s *Session, token int, layers []int) qwen35TargetFeatureRow {
	position, cfg := s.Cache.Len(), s.M.Cfg
	x := append([]float32(nil), s.M.embedRows()[token*cfg.HiddenSize:(token+1)*cfg.HiddenSize]...)
	scaleEmbedInPlace(x, cfg)
	row := qwen35TargetFeatureRow{Position: position, Token: token, Layers: append([]int(nil), layers...)}
	for layer := 0; layer < cfg.NumLayers; layer++ {
		cos, sin := ropeRowForLayer(cfg, layer, position)
		x = s.blockStep(layer, position, x, cos, sin, f32Kernel{s.M})
		for _, want := range layers {
			if want == layer {
				row.Hidden = append(row.Hidden, append([]float32(nil), x...))
			}
		}
	}
	s.Cache.appendPosition(position, token)
	s.rememberTargetHidden(position, token, x)
	return row
}

func assertQwen35TargetFeatureRow(t *testing.T, want, got qwen35TargetFeatureRow) {
	t.Helper()
	if want.Position != got.Position || want.Token != got.Token || !reflect.DeepEqual(want.Layers, got.Layers) || len(want.Hidden) != len(got.Hidden) {
		t.Fatalf("feature identity mismatch at position %d", want.Position)
	}
	for i := range want.Hidden {
		assertFloat32BitsEqual(t, fmt.Sprintf("post-block layer %d", want.Layers[i]), want.Hidden[i], got.Hidden[i])
	}
}

// fak-test:runtime medium est=1s lane=default
func TestQwen35TargetFeaturesCancelAbortAndReuse(t *testing.T) {
	for _, mode := range []string{"before-verify", "during-verify", "before-commit", "during-replay", "abort", "verify-panic"} {
		t.Run(mode, func(t *testing.T) {
			m := qwen35TargetFeatureTestModel(t)
			target, plain := m.NewSession(), m.NewSession()
			t.Cleanup(target.Close)
			t.Cleanup(plain.Close)
			target.captureTargetHidden, plain.captureTargetHidden = true, true
			before := target.Prefill([]int{3, 7})
			plain.Prefill([]int{3, 7})
			normalizeSnapshotForTest(t, target)
			normalizeSnapshotForTest(t, plain)
			f, err := beginQwen35TargetFeatureTransaction(target, before, []int{0, 3})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = f.Abort() })
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var verifyContext context.Context = ctx
			if mode == "before-verify" {
				cancel()
			}
			if mode == "during-verify" {
				verifyContext = &qwen35FeatureCancelContext{Context: ctx, cancel: cancel, target: target, atLength: 3, armed: true}
			}
			if mode == "verify-panic" {
				native := f.tx.verify
				f.tx.verify = func(tokens []int) ([][]float32, TargetVerificationReceipt, error) {
					_, _, err := native(tokens)
					if err != nil {
						panic(err)
					}
					panic("injected failure after actual target steps and feature capture")
				}
			}
			rows, verifyErr := f.Verify(verifyContext, []int{11, 13, 17})
			switch mode {
			case "before-verify", "during-verify":
				if !errors.Is(verifyErr, context.Canceled) || rows != nil {
					t.Fatalf("canceled verify rows=%v err=%v", rows, verifyErr)
				}
			case "verify-panic":
				if verifyErr == nil || rows != nil {
					t.Fatal("failed verification exposed output")
				}
			default:
				if verifyErr != nil {
					t.Fatal(verifyErr)
				}
				if mode == "abort" {
					if err := f.Abort(); err != nil {
						t.Fatal(err)
					}
				} else {
					var commitContext context.Context = ctx
					if mode == "before-commit" {
						cancel()
					} else {
						commitContext = &qwen35FeatureCancelContext{Context: ctx, cancel: cancel, target: target, armLength: 2, atLength: 3}
					}
					logits, features, err := f.Commit(commitContext, 2)
					if !errors.Is(err, context.Canceled) || logits != nil || features != nil {
						t.Fatalf("canceled commit published output: err=%v", err)
					}
				}
			}
			if err := f.Abort(); err != nil || f.tx.closeCount != 1 || target.qwen35TargetFeatureTxn != nil || target.qwen35TargetLayerCapture != nil || f.rows != nil {
				t.Fatalf("abort ownership err=%v closeCount=%d", err, f.tx.closeCount)
			}
			assertQwen35MTPTargetStateEqual(t, target, plain)
			// Reuse the same target after failure, then fork its complete prefix.
			logits := target.Step(19)
			assertFloat32BitsEqual(t, "post-abort reuse", plain.Step(19), logits)
			again, err := beginQwen35TargetFeatureTransaction(target, logits, []int{1, 3})
			if err != nil {
				t.Fatal(err)
			}
			if err := again.Abort(); err != nil {
				t.Fatal(err)
			}
			snapshot, err := target.PrefixSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(snapshot.Close)
			reuse := m.NewSession()
			t.Cleanup(reuse.Close)
			if err := snapshot.Restore(reuse); err != nil {
				t.Fatal(err)
			}
			assertFloat32BitsEqual(t, "complete prefix reuse", plain.Step(23), reuse.Step(23))
		})
	}
}

// Trigger real cancellation at native state transitions, without sleeps or an
// alternate target kernel. armLength defers cancellation until restore/replay.
type qwen35FeatureCancelContext struct {
	context.Context
	cancel    context.CancelFunc
	target    *Session
	armLength int
	atLength  int
	armed     bool
}

func (c *qwen35FeatureCancelContext) Err() error {
	if c.target.Cache.Len() == c.armLength {
		c.armed = true
	}
	if c.armed && c.target.Cache.Len() == c.atLength {
		c.cancel()
	}
	return c.Context.Err()
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35TargetFeaturesStopBoundaryAndIndependentStorage(t *testing.T) {
	m := qwen35TargetFeatureTestModel(t)
	target, plain := m.NewSession(), m.NewSession()
	t.Cleanup(target.Close)
	t.Cleanup(plain.Close)
	before := target.Prefill([]int{3, 7})
	want := plain.Prefill([]int{3, 7})
	normalizeSnapshotForTest(t, target)
	normalizeSnapshotForTest(t, plain)
	f, err := beginQwen35TargetFeatureTransaction(target, before, []int{0, 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := beginQwen35TargetFeatureTransaction(target, before, []int{0}); err == nil {
		t.Fatal("nested transaction admitted")
	}
	if _, err := f.Verify(context.Background(), []int{11, 13, 17}); err != nil {
		t.Fatal(err)
	}
	// A caller stop/budget at the first verified token commits only that prefix.
	got, features, err := f.Commit(context.Background(), 1)
	if err != nil || len(features) != 1 {
		t.Fatalf("stop boundary features=%d err=%v", len(features), err)
	}
	want = plain.Step(11)
	assertFloat32BitsEqual(t, "stop prefix", want, got)
	assertQwen35MTPTargetStateEqual(t, target, plain)
	features[0].Hidden[0][0] = float32(math.Inf(1))
	features[0].Layers[0] = -1
	assertFloat32BitsEqual(t, "feature ownership", plain.Step(19), target.Step(19))
	if _, _, err := f.Commit(context.Background(), 1); err == nil {
		t.Fatal("double commit admitted")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35TargetFeaturesRejectInvalidBeforeMutation(t *testing.T) {
	for _, layers := range [][]int{nil, {-1}, {4}, {1, 1}, {2, 1}, {0, 1, 2, 3, 4, 5}} {
		m := qwen35TargetFeatureTestModel(t)
		s := m.NewSession()
		t.Cleanup(s.Close)
		before := s.Prefill([]int{3})
		saved := s.Cache.Clone()
		if _, err := beginQwen35TargetFeatureTransaction(s, before, layers); err == nil {
			t.Fatalf("invalid layers %v admitted", layers)
		}
		assertKVCacheUnchanged(t, "invalid layers", saved, s.Cache)
		s.Close()
	}
	for _, tokens := range [][]int{nil, {-1}, {97}, make([]int, 9)} {
		m := qwen35TargetFeatureTestModel(t)
		s := m.NewSession()
		t.Cleanup(s.Close)
		before := s.Prefill([]int{3})
		saved := s.Cache.Clone()
		f, err := beginQwen35TargetFeatureTransaction(s, before, []int{0})
		if err != nil {
			t.Fatal(err)
		}
		if rows, err := f.Verify(context.Background(), tokens); err == nil || rows != nil {
			t.Fatalf("invalid tokens %v admitted", tokens)
		}
		assertKVCacheUnchanged(t, "invalid tokens", saved, s.Cache)
		if s.qwen35TargetFeatureTxn != nil {
			t.Fatal("invalid request retained transaction ownership")
		}
		s.Close()
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35TargetFeaturesIndependentConcurrentSessions(t *testing.T) {
	m := qwen35TargetFeatureTestModel(t)
	for worker := 0; worker < 4; worker++ {
		t.Run(fmt.Sprintf("session-%d", worker), func(t *testing.T) {
			t.Parallel()
			target, plain := m.NewSession(), m.NewSession()
			t.Cleanup(target.Close)
			t.Cleanup(plain.Close)
			before := target.Prefill([]int{worker + 1, 7})
			plain.Prefill([]int{worker + 1, 7})
			f, err := beginQwen35TargetFeatureTransaction(target, before, []int{0, 3})
			if err != nil {
				t.Fatal(err)
			}
			if _, err := f.Verify(context.Background(), []int{11, 13}); err != nil {
				t.Fatal(err)
			}
			got, features, err := f.Commit(context.Background(), 1)
			if err != nil || len(features) != 1 {
				t.Fatalf("concurrent commit features=%d err=%v", len(features), err)
			}
			assertFloat32BitsEqual(t, "independent concurrent session", plain.Step(11), got)
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35TargetFeaturesRefuseUnqualifiedRoutes(t *testing.T) {
	if _, err := beginQwen35TargetFeatureTransaction(nil, nil, []int{0}); err == nil {
		t.Fatal("nil session admitted")
	}
	for _, tc := range []struct {
		name   string
		change func(*Session)
	}{
		{"q8-weights", func(s *Session) { s.Quant = true }},
		{"q4-weights", func(s *Session) { s.Q4 = true }},
		{"q4k-weights", func(s *Session) { s.Q4K = true }},
		{"gptq-weights", func(s *Session) { s.GPTQ = true }},
		{"f16-route", func(s *Session) { s.F16 = true }},
		{"metal-route", func(s *Session) { s.Metal = true }},
		{"metal-q4k-route", func(s *Session) { s.MetalQ4K = true }},
		{"mixed-placement", func(s *Session) { s.DenseGPULayers = 1 }},
		{"device-only", func(s *Session) { s.executionPolicy = ExecutionPolicyDeviceOnly }},
		{"distributed", func(s *Session) { s.M.epRanks = 2 }},
		{"ds41", func(s *Session) { s.M.Cfg.ModelType = "deepseek_v41" }},
		{"missing-lineage", func(s *Session) { s.Cache.lineage.ids = nil }},
		{"closed", func(s *Session) { s.Close() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := qwen35TargetFeatureTestModel(t)
			s := m.NewSession()
			t.Cleanup(s.Close)
			before := s.Prefill([]int{3})
			tc.change(s)
			saved := s.Cache.Clone()
			if _, err := beginQwen35TargetFeatureTransaction(s, before, []int{0}); err == nil {
				t.Fatal("unqualified route admitted")
			}
			assertKVCacheUnchanged(t, "refused route", saved, s.Cache)
			if s.qwen35TargetFeatureTxn != nil || s.qwen35TargetLayerCapture != nil {
				t.Fatal("refused route acquired feature ownership")
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestQwen35TargetFeaturesInvalidCommitRollsBack(t *testing.T) {
	for _, accepted := range []int{-1, 3} {
		m := qwen35TargetFeatureTestModel(t)
		s := m.NewSession()
		t.Cleanup(s.Close)
		before := s.Prefill([]int{3})
		saved := s.Cache.Clone()
		f, err := beginQwen35TargetFeatureTransaction(s, before, []int{0, 3})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Verify(context.Background(), []int{7, 11}); err != nil {
			t.Fatal(err)
		}
		if logits, rows, err := f.Commit(context.Background(), accepted); err == nil || logits != nil || rows != nil {
			t.Fatalf("invalid accepted prefix %d exposed output", accepted)
		}
		assertKVCacheUnchanged(t, "invalid commit", saved, s.Cache)
		if s.qwen35TargetFeatureTxn != nil || f.rows != nil || f.tx.closeCount != 1 {
			t.Fatal("invalid commit retained ownership")
		}
	}
}

// Register model teardown before any session cleanup. testing runs a parent's
// cleanup after its subtests (including parallel subtests), and cleanup LIFO
// closes sessions before their model. Treat an active-session error as a leak.
func qwen35TargetFeatureTestModel(t *testing.T) *Model {
	t.Helper()
	m := NewSynthetic(qwen35HybridTestCfg())
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("CloseWeights: %v", err)
		}
	})
	return m
}
