package model

import (
	"fmt"
	"os"
	"strconv"
)

// v41SuffixUbatchEnv selects the batched suffix prefill. A positive value is the
// shortest suffix it takes (at least 2); empty, zero or invalid keeps the
// per-token suffix.
const v41SuffixUbatchEnv = "FAK_V41_SUFFIX_UBATCH"

// v41SuffixUbatchMaxRows bounds the open step transactions of one ubatch; a
// longer suffix runs as consecutive, evenly sized ubatches.
const v41SuffixUbatchMaxRows = 1024

// v41SuffixUbatchProbe observes each completed ubatch's row count in tests.
var v41SuffixUbatchProbe func(rows int)

func v41SuffixUbatchMinRows() int {
	n, err := strconv.Atoi(os.Getenv(v41SuffixUbatchEnv))
	if err != nil || n <= 0 {
		return 0
	}
	return max(n, 2)
}

// prefillV41SuffixBatched is the expert-grouped form of prefillV41Suffix. Each
// suffix position keeps its own step transaction, but the layer loop is outer:
// every position prepares the layer and publishes its window row, the ubatch's
// routed experts are contracted expert-major once, then every position
// finishes the layer. Positions commit in order, so the session ends exactly
// where the per-token suffix would. It reports false, with the session
// unchanged, when the batched route does not apply.
func (s *Session) prefillV41SuffixBatched(ids []int, wantLogits bool) ([]float32, bool) {
	minRows := v41SuffixUbatchMinRows()
	st := s.v41State()
	if minRows == 0 || len(ids) < minRows || s.M.v41EPRequested() || st.activeStepRound != nil || v41ActivationTraceEnvPath() != "" {
		return nil, false
	}
	startLen := len(st.history)
	chunks := (len(ids) + v41SuffixUbatchMaxRows - 1) / v41SuffixUbatchMaxRows
	size := (len(ids) + chunks - 1) / chunks
	var logits []float32
	for done := 0; done < len(ids); done += size {
		end := min(done+size, len(ids))
		got, err := s.v41SuffixUbatch(ids[done:end], wantLogits && end == len(ids))
		if err != nil {
			if v41SuffixFailureIsSelected(s.M, err) {
				panic(err)
			}
			// A failed ubatch may leave earlier positions committed or a ring row
			// restored out of order, so rebuild every layer state from history.
			return s.prefillV41Refold(st, startLen, ids, wantLogits), true
		}
		logits = got
	}
	s.M.v41NoteExpertFaultToken(len(ids))
	return logits, true
}

// v41SuffixUbatch runs one ubatch inside one step-batch window.
func (s *Session) v41SuffixUbatch(ids []int, wantLogits bool) (logits []float32, err error) {
	endBatch := s.v41BeginStepBatch()
	bodyComplete := false
	defer func() {
		v41FinishStepBatch(endBatch, &err, bodyComplete)
		if err != nil {
			logits = nil
		}
	}()
	logits, err = s.v41SuffixUbatchBody(ids, wantLogits)
	bodyComplete = true
	return logits, err
}

func (s *Session) v41SuffixUbatchBody(ids []int, wantLogits bool) (logits []float32, err error) {
	m, st := s.M, s.v41State()
	txs := make([]*v41StepTransaction, 0, len(ids))
	// Uncommitted positions roll back newest first, so every layer's journal
	// restores the cursor and ring row it staged.
	defer func() {
		for k := len(txs) - 1; k >= 0; k-- {
			txs[k].abortAfterFailure()
		}
	}()
	base := st.history
	view := append(base[:len(base):len(base)], ids...)
	admissionReusable := false
	for k, id := range ids {
		// A step takes its position and Engram context from history; each
		// position begins as if the earlier suffix positions had committed.
		st.history = view[:len(base)+k]
		// Reuse shape admission only across callback-free construction. A
		// selected embedding, initial-mHC or clock hook may change model wiring;
		// even if it removes itself, its successor must re-admit normally.
		callbackFree := st.gatherOps.embeddingGatherFunc() == nil && st.mhcDevice == nil && v41ActivationTraceEnvPath() == ""
		if ledger := v41ExpertFaultLedgerOf(m, false); ledger != nil {
			ledger.mu.Lock()
			callbackFree = callbackFree && ledger.nowNanos == nil
			ledger.mu.Unlock()
		}
		var tx *v41StepTransaction
		var berr error
		if admissionReusable && callbackFree {
			tx, berr = m.beginV41StepAdmitted(id, st, &v41ProjScratch{})
		} else {
			tx, berr = m.beginV41Step(id, st, &v41ProjScratch{})
		}
		admissionReusable = callbackFree && berr == nil
		st.history = base
		if berr != nil {
			return nil, berr
		}
		txs = append(txs, tx)
	}
	scratch := &v41ProjScratch{}
	scratch.bindLayerSessionBindings(st)
	defer scratch.clearLayerSessionBindings()
	works := make([]*v41LayerStepWork, len(txs))
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for k, tx := range txs {
			if err = tx.prepareLayer(); err != nil {
				return nil, err
			}
			if err = tx.publishPendingKV(); err != nil {
				return nil, err
			}
			works[k] = tx.pendingLayer
		}
		routed, rerr := m.v41SuffixUbatchRouted(l, works, st, scratch)
		if rerr != nil {
			return nil, rerr
		}
		for k, tx := range txs {
			if err = tx.finishLayer(routed[k]); err != nil {
				return nil, err
			}
		}
	}
	if v41SuffixUbatchProbe != nil {
		v41SuffixUbatchProbe(len(txs))
	}
	for k, tx := range txs {
		if wantLogits && k == len(txs)-1 {
			logits, err = tx.finalize()
		} else {
			err = tx.finalizeNoLogits()
		}
		if err != nil {
			return nil, err
		}
	}
	return logits, nil
}

// v41SuffixUbatchRouted contracts one layer's routed picks for the whole
// ubatch through the cold-prefill grouped contraction: each touched expert is
// materialized once and contracts every row assigned to it, and each
// position's weighted sum is replayed in its original slot order.
func (m *Model) v41SuffixUbatchRouted(l int, works []*v41LayerStepWork, st *v41ForwardState, scratch *v41ProjScratch) ([][]float32, error) {
	inputs := make([][]float32, len(works))
	picks := make([][]routePick, len(works))
	for k, w := range works {
		if w == nil || w.layer != l || w.full != works[0].full {
			return nil, v41StageErr(v41StageMoE, l, fmt.Errorf("%w: suffix ubatch layer continuations disagree", ErrV41ForwardStage))
		}
		inputs[k], picks[k] = w.input, w.picks
	}
	scratch.v41LayerCacheReset()
	scratch.v41SizeExpertLayerBudget(m)
	routed := make([][]float32, len(works))
	if err := m.v41ContractRoutedGroupedInto(l, inputs, picks, scratch, m.Cfg, routed, st, false); err != nil {
		return nil, err
	}
	return routed, nil
}
