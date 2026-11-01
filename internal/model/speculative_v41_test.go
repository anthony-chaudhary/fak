package model

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=500ms lane=default
func TestParallelVerifyV41RetainsAcceptedPrefix(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	t.Cleanup(func() { _ = m.CloseWeights() })

	source := m.NewSession()
	t.Cleanup(source.Close)
	if len(source.Prefill([]int{1, 2, 3})) == 0 {
		t.Fatal("prefill returned no logits")
	}
	seed := captureV41ForwardSnapshot(source.v41State())
	newBranch := func() *Session {
		s := m.NewSession()
		t.Cleanup(s.Close)
		s.v41Forward = seed.clone().restore()
		s.v41State()
		return s
	}
	target, fullOracle, keptOracle := newBranch(), newBranch(), newBranch()

	first := 4 % m.Cfg.VocabSize
	firstRow := append([]float32(nil), fullOracle.Step(first)...)
	second := (argmaxF32(firstRow) + 1) % m.Cfg.VocabSize
	secondRow := append([]float32(nil), fullOracle.Step(second)...)
	keptOracle.Step(first)
	last := make([]float32, m.Cfg.VocabSize)
	last[first] = 1

	got, err := ParallelVerifyKernel(context.Background(), target, nil,
		DraftProposal{Tokens: []int{first, second}}, last, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.NumAccepted != 1 || !reflect.DeepEqual(got.AcceptedTokens, []int{first}) ||
		got.CorrectionToken != argmaxF32(firstRow) ||
		!reflect.DeepEqual(got.TargetLogits, [][]float32{firstRow, secondRow}) {
		t.Fatalf("verification result differs from serial oracle: accepted=%v count=%d correction=%d rows_match=%t",
			got.AcceptedTokens, got.NumAccepted, got.CorrectionToken,
			reflect.DeepEqual(got.TargetLogits, [][]float32{firstRow, secondRow}))
	}
	if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), captureV41ForwardSnapshot(keptOracle.v41State())) {
		t.Fatal("rejected suffix remained in V4.1 retained state")
	}
	next := 7 % m.Cfg.VocabSize
	if targetNext, oracleNext := target.Step(next), keptOracle.Step(next); !reflect.DeepEqual(targetNext, oracleNext) ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), captureV41ForwardSnapshot(keptOracle.v41State())) {
		t.Fatal("continued logits or state differ from accepted-prefix oracle")
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestParallelVerifyV41CancellationRestoresState(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	t.Cleanup(func() { _ = m.CloseWeights() })
	target := m.NewSession()
	t.Cleanup(target.Close)
	if len(target.Prefill([]int{1, 2, 3})) == 0 {
		t.Fatal("prefill returned no logits")
	}
	before := captureV41ForwardSnapshot(target.v41State())
	last := make([]float32, m.Cfg.VocabSize)
	last[4%m.Cfg.VocabSize] = 1
	ctx := &speculativeCancelContext{Context: context.Background(), allowed: 2}

	_, err := ParallelVerifyKernel(ctx, target, nil,
		DraftProposal{Tokens: []int{4 % m.Cfg.VocabSize, 5 % m.Cfg.VocabSize}}, last, nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context cancellation", err)
	}
	if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), before) {
		t.Fatal("canceled verification retained a speculative V4.1 token")
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestParallelVerifyV41PanicRestoresState(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	t.Cleanup(func() { _ = m.CloseWeights() })
	target := m.NewSession()
	t.Cleanup(target.Close)
	if len(target.Prefill([]int{1, 2, 3})) == 0 {
		t.Fatal("prefill returned no logits")
	}
	sentinel := errors.New("selected final norm panic")
	target.v41State().setFinalNorm(func([]float32) ([]float32, error) { panic(sentinel) })
	before := captureV41ForwardSnapshot(target.v41State())
	first := 4 % m.Cfg.VocabSize
	last := make([]float32, m.Cfg.VocabSize)
	last[first] = 1

	panicked := recoverError(func() {
		_, _ = ParallelVerifyKernel(context.Background(), target, nil,
			DraftProposal{Tokens: []int{first}}, last, nil, nil)
	})
	if !errors.Is(panicked, sentinel) {
		t.Fatalf("panic = %v, want selected final norm panic", panicked)
	}
	if target.v41State().activeStepRound != nil ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), before) {
		t.Fatal("panicked verification retained V4.1 round state")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestParallelVerifyV41PreservesStepAccounting(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	f := newV41BatchFixture(t)
	target := f.lanes[0].Session
	first := f.lanes[0].Token
	last := make([]float32, f.m.Cfg.VocabSize)
	last[first] = 1
	beforeTokens := f.m.V41ExpertFaultAttribution().Decode.Tokens
	oldProbe := v41IncrementalStepProbe
	probes := 0
	v41IncrementalStepProbe = func() { probes++ }
	defer func() { v41IncrementalStepProbe = oldProbe }()

	round, err := ParallelVerifyKernelOwned(context.Background(), target, nil,
		DraftProposal{Tokens: []int{first}}, last, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer round.Abort()
	got := round.Result
	if err := round.Finish(got.NumAccepted); err != nil {
		t.Fatal(err)
	}
	if round.PanelWork == nil || round.PanelWork.LayerPanels != f.m.Cfg.NumLayers || round.PanelWork.HeadRows != 1 || round.PanelWork.HeadProjections != 1 || round.PanelWork.SharedHeadTiles != 0 {
		t.Fatalf("panel accounting differs: %+v", round.PanelWork)
	}
	if got.NumAccepted != 1 || f.b.beginCalls != 1 || f.b.ends != 1 || probes != 0 ||
		f.m.v41ExpertFaultPhase() != V41PhaseUnknown ||
		f.m.V41ExpertFaultAttribution().Decode.Tokens != beforeTokens+1 {
		t.Fatalf("accounting differs: accepted=%d batches=%d/%d probes=%d phase=%v tokens=%d",
			got.NumAccepted, f.b.beginCalls, f.b.ends, probes, f.m.v41ExpertFaultPhase(),
			f.m.V41ExpertFaultAttribution().Decode.Tokens-beforeTokens)
	}
}

// fak-test:runtime medium est=5s lane=default
func TestParallelVerifyV41BatchErrorRestoresState(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	f := newV41BatchFixture(t)
	target := f.lanes[0].Session
	fault := errors.New("injected step batch flush failure")
	target.Backend = &speculativeStepBatchFaultBackend{executionOwnerTestBackend: f.b, fault: fault}
	target.v41Forward.callbackOwner = nil
	target.v41State()
	before := captureV41ForwardSnapshot(target.v41State())
	beforeTokens := f.m.V41ExpertFaultAttribution().Decode.Tokens
	first := f.lanes[0].Token
	last := make([]float32, f.m.Cfg.VocabSize)
	last[first] = 1

	_, err := ParallelVerifyKernel(context.Background(), target, nil,
		DraftProposal{Tokens: []int{first}}, last, nil, nil)
	var batchErr *v41StepBatchError
	if !errors.As(err, &batchErr) || !errors.Is(err, fault) {
		t.Fatalf("error = %v, want typed step batch failure", err)
	}
	if f.b.beginCalls != 1 || f.b.ends != 1 || f.m.v41ExpertFaultPhase() != V41PhaseUnknown ||
		f.m.V41ExpertFaultAttribution().Decode.Tokens != beforeTokens || target.v41State().activeStepRound != nil ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), before) {
		t.Fatal("batch failure did not balance accounting and restore the round")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestParallelVerifyV41IndexerFailureRestoresBeforeRetirement(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	f := newV41BatchFixture(t)
	target := f.lanes[0].Session
	st := target.v41State()
	primary := &V41IndexerScoreOperationError{Layer: 0, Cause: errors.New("selected second-token indexer failure")}
	tokens := 0
	st.setIndexerScore(func(layer int, q, keys, weights []float32, heads, dim, rows int) ([]float32, error) {
		if layer == 0 {
			tokens++
			if tokens == 2 {
				return nil, primary
			}
		}
		return V41IndexerScore(q, keys, weights, heads, dim, rows)
	})
	before := captureV41ForwardSnapshot(st)
	closeObserved := false
	f.b.hook = func(site string) {
		if site != "flush" {
			return
		}
		closeObserved = true
		if st.activeStepRound != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("backend retirement began before the speculative round was restored")
		}
	}
	first, second := f.lanes[0].Token, (f.lanes[0].Token+1)%f.m.Cfg.VocabSize
	last := make([]float32, f.m.Cfg.VocabSize)
	last[first] = 1

	panicked := recoverError(func() {
		_, _ = ParallelVerifyKernel(context.Background(), target, nil,
			DraftProposal{Tokens: []int{first, second}}, last, nil, nil)
	})
	var operation *BackendForwardOperationError
	if tokens != 2 || !errors.Is(panicked, primary) || !errors.As(panicked, &operation) ||
		!closeObserved || !target.BackendSessionClosed() || target.halFailure != operation ||
		st.activeStepRound != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
		t.Fatalf("selected retirement differs: token_calls=%d panic=%v closed=%t close_observed=%t",
			tokens, panicked, target.BackendSessionClosed(), closeObserved)
	}
}

type speculativeStepBatchFaultBackend struct {
	*executionOwnerTestBackend
	fault error
}

func (b *speculativeStepBatchFaultBackend) EndV41StepBatch() error {
	_ = b.executionOwnerTestBackend.EndV41StepBatch()
	return b.fault
}

type speculativeCancelContext struct {
	context.Context
	checks  int
	allowed int
}

func (c *speculativeCancelContext) Err() error {
	c.checks++
	if c.checks > c.allowed {
		return context.Canceled
	}
	return nil
}

// Five bounded real-consumer cases; fixtures execute the native model, not a
// replacement verifier. This is software correctness, not physical GPU parity.
// fak-test:runtime medium est=10s lane=default
func TestV41SpeculativePanelContract(t *testing.T) {
	t.Run("all-accepted-layer-major", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		t.Cleanup(func() { _ = m.CloseWeights() })
		target, oracle := m.NewSession(), m.NewSession()
		t.Cleanup(target.Close)
		t.Cleanup(oracle.Close)
		prefix := []int{1, 2, 3}
		boundary := target.Prefill(prefix)
		next := oracle.Prefill(prefix)
		var draft []int
		var wantRows [][]float32
		for i := 0; i < 16; i++ {
			tok := argmaxF32(next)
			draft = append(draft, tok)
			next = append([]float32(nil), oracle.Step(tok)...)
			wantRows = append(wantRows, next)
		}
		var layers []int
		target.v41State().setDenseProjection(func(l int, _ string, _ []float32, _, _, _ int) ([]float32, v41DenseProjectionOutcome, error) {
			if l >= 0 {
				layers = append(layers, l)
			}
			return nil, v41ProjectionDeclined, nil
		})
		oldProbe := v41IncrementalStepProbe
		steps := 0
		v41IncrementalStepProbe = func() { steps++ }
		defer func() { v41IncrementalStepProbe = oldProbe }()
		round, err := ParallelVerifyKernelOwned(context.Background(), target, prefix, DraftProposal{Tokens: draft}, boundary, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer round.Abort()
		if round.Result.NumAccepted != 16 || !reflect.DeepEqual(round.Result.TargetLogits, wantRows) {
			t.Fatal("panel logits/acceptance differ from native serial oracle")
		}
		if round.PanelWork == nil || round.PanelWork.LayerPanels != m.Cfg.NumLayers || round.PanelWork.HeadRows != 16 || round.PanelWork.HeadProjections != 16 || round.PanelWork.SharedHeadTiles != 0 || steps != 0 {
			t.Fatalf("not genuine layer-major work: %+v steps=%d", round.PanelWork, steps)
		}
		if len(layers) == 0 {
			t.Fatal("native projection-order witness did not execute")
		}
		for i := 1; i < len(layers); i++ {
			if layers[i] < layers[i-1] {
				t.Fatalf("token-major layer restart: %v", layers)
			}
		}
		if !round.Receipt.OneOperation || round.Receipt.TargetVerificationOperations != 1 || round.Receipt.TargetDecodeSteps != 0 {
			t.Fatalf("untruthful target receipt: %+v", round.Receipt)
		}
		if err := round.Finish(16); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), captureV41ForwardSnapshot(oracle.v41State())) {
			t.Fatal("all-accepted state differs")
		}
		bonus := round.Result.CorrectionToken
		if bonus != argmaxF32(next) || !reflect.DeepEqual(target.Step(bonus), oracle.Step(bonus)) {
			t.Fatal("single bonus/continuation differs")
		}
	})
	t.Run("first-rejected-no-work", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		t.Cleanup(func() { _ = m.CloseWeights() })
		target := m.NewSession()
		t.Cleanup(target.Close)
		boundary := target.Prefill([]int{1, 2, 3})
		before := captureV41ForwardSnapshot(target.v41State())
		bad := (argmaxF32(boundary) + 1) % m.Cfg.VocabSize
		round, err := ParallelVerifyKernelOwned(context.Background(), target, nil, DraftProposal{Tokens: []int{bad}}, boundary, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer round.Abort()
		if round.Result.NumAccepted != 0 || round.PanelWork != nil || round.Receipt.TargetVerificationOperations != 0 || round.Receipt.TargetDecodeSteps != 0 {
			t.Fatal("known rejection executed target work")
		}
		if err := round.Finish(0); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), before) {
			t.Fatal("known rejection mutated continuation")
		}
	})
	t.Run("middle-rejected-publication-boundary", func(t *testing.T) {
		m := v41CompressorTestFixtureIndex(t, 128)
		m.Cfg.Window = []int{2, 2}
		t.Cleanup(func() { _ = m.CloseWeights() })
		target, full, kept := m.NewSession(), m.NewSession(), m.NewSession()
		t.Cleanup(target.Close)
		t.Cleanup(full.Close)
		t.Cleanup(kept.Close)
		prefix := []int{1, 2, 3}
		boundary := target.Prefill(prefix)
		next := full.Prefill(prefix)
		kept.Prefill(prefix)
		var draft []int
		var rows [][]float32
		for i := 0; i < 4; i++ {
			tok := argmaxF32(next)
			if i == 2 {
				tok = (tok + 1) % m.Cfg.VocabSize
			}
			draft = append(draft, tok)
			next = append([]float32(nil), full.Step(tok)...)
			rows = append(rows, next)
			if i < 2 {
				kept.Step(tok)
			}
		}
		round, err := ParallelVerifyKernelOwned(context.Background(), target, prefix, DraftProposal{Tokens: draft}, boundary, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		defer round.Abort()
		if round.Result.NumAccepted != 2 || !reflect.DeepEqual(round.Result.TargetLogits, rows) {
			t.Fatal("causal source/reader rows differ across publication boundary")
		}
		if err := round.Finish(2); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), captureV41ForwardSnapshot(kept.v41State())) {
			t.Fatal("rejected rows remain in ring/compressor/publications")
		}
		tok := round.Result.CorrectionToken
		if !reflect.DeepEqual(target.Step(tok), kept.Step(tok)) {
			t.Fatal("strict-prefix continuation differs")
		}
	})
	t.Run("generate-emitted-budget", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 2)
		t.Cleanup(func() { _ = m.CloseWeights() })
		target, oracle := m.NewSession(), m.NewSession()
		t.Cleanup(target.Close)
		t.Cleanup(oracle.Close)
		prefix := []int{1, 2, 3}
		next := oracle.Prefill(prefix)
		var draft []int
		var rows [][]float32
		for i := 0; i < 4; i++ {
			tok := argmaxF32(next)
			draft = append(draft, tok)
			next = append([]float32(nil), oracle.Step(tok)...)
			rows = append(rows, next)
		}
		generator := NewMTPProposalGeneratorWithFn(func(context.Context, []int, int) ([]int, error) { return append([]int(nil), draft...), nil })
		engine := NewSpeculativeEngine(target, generator, SpeculativeEngineConfig{MaxDraft: 4})
		got, err := engine.Generate(context.Background(), prefix, 2)
		if err != nil {
			t.Fatal(err)
		}
		kept := m.NewSession()
		t.Cleanup(kept.Close)
		kept.Prefill(prefix)
		kept.Step(draft[0])
		kept.Step(draft[1])
		if !reflect.DeepEqual(got, draft[:2]) || !reflect.DeepEqual(engine.LastLogits(), rows[1]) || !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), captureV41ForwardSnapshot(kept.v41State())) {
			t.Fatal("Generate retained an accepted-but-unemitted suffix or stale engine boundary")
		}
		tok := argmaxF32(rows[1])
		if !reflect.DeepEqual(target.Step(tok), kept.Step(tok)) {
			t.Fatal("budget continuation differs")
		}
	})
	t.Run("admission-versus-selected-fault", func(t *testing.T) {
		t.Run("pure-admission-decline", func(t *testing.T) {
			m := v41IncrementalPlainModel(t, 2)
			t.Cleanup(func() { _ = m.CloseWeights() })
			target := m.NewSession()
			t.Cleanup(target.Close)
			boundary := target.Prefill([]int{1, 2, 3})
			before := captureV41ForwardSnapshot(target.v41State())
			draft := make([]int, 33)
			for i := range draft {
				draft[i] = argmaxF32(boundary)
			}
			round, err := ParallelVerifyKernelOwned(context.Background(), target, nil, DraftProposal{Tokens: draft}, boundary, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer round.Abort()
			if round.Result.NumAccepted != 0 || round.Result.CorrectionToken != argmaxF32(boundary) || round.Receipt.DowngradeReason == "" || round.Receipt.TargetVerificationOperations != 0 || round.Receipt.TargetDecodeSteps != 0 {
				t.Fatal("unsupported panel did not explicitly decline before work")
			}
			if err := round.Finish(0); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(target.v41State()), before) {
				t.Fatal("unsupported panel mutated state")
			}
		})
		t.Run("selected-fault-rollback-before-retirement", TestParallelVerifyV41IndexerFailureRestoresBeforeRetirement)
	})
}
