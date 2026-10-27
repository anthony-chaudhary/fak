package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// Real CPU Q4_K checkpoint-tier weights exercise the ordinary warm serial
// Session prefill route. This measures allocation, not hardware throughput.
// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41SuffixPrefillReusesProjectionScratch(t *testing.T) {
	t.Setenv(v41SuffixUbatchEnv, "")
	t.Setenv(v41SuffixRefoldMaxHistoryEnv, "")
	t.Setenv(v41ActivationTraceEnvVar, "")
	m := v41IncrementalExpertFixture(t, false, true)
	defer m.CloseWeights()
	source := m.NewSession()
	defer source.Close()
	prefix := []int{1, 2, 3}
	source.Prefill(prefix)
	if !source.v41IncrementalEligible() || m.expertCheckpoint == nil {
		t.Fatal("fixture must seed the real streamed-expert incremental route")
	}
	cached, err := source.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer cached.Close()
	sourceState := captureV41ForwardSnapshot(source.v41Forward)
	branch := func(t *testing.T) *Session {
		t.Helper()
		fork, err := cached.CloneForRestore()
		if err != nil {
			t.Fatal(err)
		}
		defer fork.Close()
		s := m.NewSession()
		if err := fork.Restore(s); err != nil {
			s.Close()
			t.Fatal(err)
		}
		t.Cleanup(s.Close)
		if !s.v41IncrementalEligible() {
			t.Fatal("restored branch lost incremental state")
		}
		return s
	}
	equalLogits := func(t *testing.T, got, want []float32) {
		t.Helper()
		if len(got) != m.Cfg.VocabSize || len(want) != len(got) {
			t.Fatalf("logit widths %d and %d, want %d", len(got), len(want), m.Cfg.VocabSize)
		}
		for i := range got {
			g, w := float64(got[i]), float64(want[i])
			if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("logits[%d]=%g want finite bit-identical %g", i, g, w)
			}
		}
	}
	equalState := func(t *testing.T, got, want *Session) {
		t.Helper()
		if !reflect.DeepEqual(captureV41ForwardSnapshot(got.v41Forward), captureV41ForwardSnapshot(want.v41Forward)) {
			t.Fatal("retained history/window/recurrent/publication state differs from fresh-scratch reference")
		}
	}
	for _, wantLogits := range []bool{true, false} {
		name := "Prefill"
		if !wantLogits {
			name = "PrefillNoLogits"
		}
		t.Run(name, func(t *testing.T) {
			actual, reference := branch(t), branch(t)
			suffix := []int{4, 5, 6, 7, 0, 1, 2, 3}
			runActual := func(ids []int) []float32 {
				if wantLogits {
					return actual.Prefill(ids)
				}
				actual.PrefillNoLogits(ids)
				return nil
			}
			read, restore := v41SuffixPrefillCalls(t)
			defer restore()
			var got, want []float32
			readsBefore := m.expertCheckpoint.Stats().Reads
			actualBytes := v41MeasureAlloc(func() { got = runActual(suffix) })
			actualReads := m.expertCheckpoint.Stats().Reads - readsBefore
			if read() != len(suffix) || actualReads == 0 {
				t.Fatalf("ordinary serial steps=%d, tier reads=%d", read(), actualReads)
			}
			readsBefore = m.expertCheckpoint.Stats().Reads
			legacyBytes := v41MeasureAlloc(func() { want = v41SuffixFreshScratchReference(reference, suffix, wantLogits) })
			legacyReads := m.expertCheckpoint.Stats().Reads - readsBefore
			if read() != 2*len(suffix) || actualReads != legacyReads {
				t.Fatalf("scratch reuse changed execution: steps=%d reads=%d/%d", read(), actualReads, legacyReads)
			}
			// The reference omits the public wrapper's extra admission, so
			// comparing against it is conservative. Require half the exact
			// redundant expert-expansion bytes, not a timing or noise threshold.
			redundant := uint64(len(suffix)-1) * 3 * uint64(m.Cfg.HiddenSize) * uint64(m.Cfg.MoEIntermediateSize) * 4
			if legacyBytes <= actualBytes || legacyBytes-actualBytes < redundant/2 {
				t.Fatalf("per-token scratch still discarded: actual=%d fresh=%d expected saved >=%d", actualBytes, legacyBytes, redundant/2)
			}
			t.Logf("serial suffix=%d tier_reads=%d allocation_bytes fresh=%d reused=%d", len(suffix), actualReads, legacyBytes, actualBytes)
			if wantLogits {
				equalLogits(t, got, want)
			} else if got != nil || want != nil {
				t.Fatal("no-logits path returned logits")
			}
			equalState(t, actual, reference)
			saved := append([]float32(nil), got...)
			// A second chunk starts a new call-scoped scratch, while its state
			// continues the first chunk. No projection buffer belongs to KV.
			tail := []int{4, 5, 6}
			nextGot := runActual(tail)
			nextWant := v41SuffixFreshScratchReference(reference, tail, wantLogits)
			if wantLogits {
				equalLogits(t, nextGot, nextWant)
				equalLogits(t, got, saved)
			}
			equalState(t, actual, reference)
			equalLogits(t, actual.Step(7), reference.Step(7))
			equalState(t, actual, reference)
			if !reflect.DeepEqual(sourceState, captureV41ForwardSnapshot(source.v41Forward)) {
				t.Fatal("restored suffix changed the cached prefix owner")
			}
		})
	}
	t.Run("selected-second-token-failure", func(t *testing.T) {
		actual, reference := branch(t), branch(t)
		reference.Step(4)
		sentinel := errors.New("selected second-token expert failure")
		firstCalls, failCalls := 0, 0
		st := actual.v41State()
		fail := func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
			failCalls++
			return nil, v41GateUpError, sentinel
		}
		st.expertGateUp = func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
			firstCalls++
			// Decline to the real host math for token one, but replace the
			// binding for token two. A reused scratch must rebind each token.
			st.expertGateUp = fail
			return nil, v41GateUpDeclined, nil
		}
		read, restore := v41SuffixPrefillCalls(t)
		defer restore()
		err := panicAsError(func() { actual.Prefill([]int{4, 5}) })
		var selected *V41ExpertOperationError
		if !errors.Is(err, sentinel) || !errors.As(err, &selected) || read() != 1 || firstCalls != m.Cfg.NumExpertsPerTok*m.Cfg.NumLayers || failCalls != 1 {
			t.Fatalf("selected fault/binding boundary: err=%v steps=%d first=%d failed=%d", err, read(), firstCalls, failCalls)
		}
		equalState(t, actual, reference)
		st.expertGateUp = nil
		equalLogits(t, actual.Step(5), reference.Step(5))
		equalState(t, actual, reference)
	})
}

// This is the immutable e745 prefillV41Suffix body, with only its declaration
// adapted to a test free function. It keeps the real per-token transaction,
// batch, selected-failure/refold and accounting behavior, not replacement math.
func v41SuffixFreshScratchReference(s *Session, ids []int, wantLogits bool) []float32 {
	st := s.v41State()
	startLen := len(st.history)
	var logits []float32
	for i, id := range ids {
		var got []float32
		var err error
		func() {
			endBatch := s.v41BeginStepBatch()
			bodyComplete := false
			defer func() {
				v41FinishStepBatch(endBatch, &err, bodyComplete)
				if err != nil {
					got = nil
				}
			}()
			got, _, err = s.M.forwardV41StepCore(id, st, &v41ProjScratch{}, false, wantLogits && i == len(ids)-1)
			bodyComplete = true
		}()
		if err != nil {
			if v41SuffixFailureIsSelected(s.M, err) {
				panic(err)
			}
			return s.prefillV41Refold(st, startLen, ids, wantLogits)
		}
		s.M.v41NoteWorkload(v41WorkloadSuffixSerial, 1, 1)
		if v41PrefillStepProbe != nil {
			v41PrefillStepProbe()
		}
		if i == len(ids)-1 {
			logits = got
		}
	}
	s.M.v41NoteExpertFaultToken(len(ids))
	return logits
}
