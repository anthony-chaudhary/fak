package model

import (
	"fmt"
	"os"
	"reflect"
	"slices"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// qwen35BatchDiagEnabled gates the model-level batch-step stage-timing diagnostic
// line (fak_batch_model_diag) used by the q2kxl concurrency-collapse investigation.
// Off by default; FAK_BATCH_DIAG=1 turns it on. Pure observation: no behavior change.
var qwen35BatchDiagEnabled = os.Getenv("FAK_BATCH_DIAG") == "1"

// qwen35HybridQ4KBatchStep is an optional exact native batch override.
// Without an override, dispatch requires the existing backend capability.
var qwen35HybridQ4KBatchStep func(*BatchSession, []int) ([][]float32, bool)

// Qwen35IndependentDecodeCohortLimit returns the existing independent decode
// width only when the complete offered cohort has known, private owners. It is
// an observation, not an ownership transfer: the caller must serialize session
// mutation through submission. It never uploads rows, adopts missing owners,
// retains snapshots, clones COW state, or invokes a forward operation.
//
// Check the whole cohort before splitting it. Per-group checks alone would miss
// KV or recurrent aliases on opposite sides of a group boundary. Platform
// overrides and unknown/shared ownership retain the established unsplit route.
func (bs *BatchSession) Qwen35IndependentDecodeCohortLimit() int {
	if bs == nil || bs.M == nil || len(bs.Seqs) < 2 || qwen35HybridQ4KBatchStep != nil ||
		!bs.M.Cfg.IsQwen35Hybrid() || bs.M.Cfg.BlockTopology != PreNorm || bs.M.Cfg.IsMoE() {
		return 0
	}
	cfg := bs.M.Cfg
	layout, err := ComputeQwen35StateLayout(cfg, "f32")
	if err != nil || cfg.NumLayers <= 0 || layout.ConvElements == 0 {
		return 0
	}
	// Only nonnil pointer-backed identities can establish known ownership. This
	// also refuses noncomparable interface implementations without panicking.
	seen := make(map[any]bool)
	unique := func(v any) bool {
		if !qwen35IndependentDecodePointer(v) || seen[v] {
			return false
		}
		seen[v] = true
		return true
	}
	var backend compute.Backend
	stores := make([]compute.KVStore, len(bs.Seqs))
	states := make([][]compute.Qwen35SequenceState, len(bs.Seqs))
	for i, session := range bs.Seqs {
		if session == nil || session.M != bs.M || session.halClosed || session.halFailure != nil ||
			!unique(session) || !unique(session.halKV) || !unique(session.qwen35HAL) ||
			!qwen35IndependentDecodePointer(session.Backend) {
			return 0
		}
		if i == 0 {
			backend = session.Backend
			if _, ok := qwen35IndependentDecodeBatchFor(backend); !ok {
				return 0
			}
		} else if session.Backend != backend {
			return 0
		}
		q := session.qwen35HAL
		stores[i] = session.halKV
		states[i] = make([]compute.Qwen35SequenceState, cfg.NumLayers)
		valid := func() bool {
			q.mu.Lock()
			defer q.mu.Unlock()
			if q.sequenceFailure != nil || len(q.layers) != cfg.NumLayers ||
				!qwen35IndependentDecodePointer(q.backend) || any(q.backend) != any(backend) {
				return false
			}
			linear := 0
			for layer := range q.layers {
				if !cfg.isLinearAttnLayer(layer) {
					if state := q.layers[layer]; state.owner != nil || state.conv.Buf() != nil || state.recurrent.Buf() != nil {
						return false
					}
					continue
				}
				linear++
				state := &q.layers[layer]
				owner := state.owner
				if !unique(owner) {
					return false
				}
				private := func() bool {
					owner.mu.Lock()
					defer owner.mu.Unlock()
					if owner.refs != 1 || owner.graphInvalid || owner.graphMutating ||
						!qwen35IndependentDecodePointer(owner.backend) || owner.backend != backend {
						return false
					}
					pairs := []struct {
						state, owned compute.Tensor
						shape        []int
					}{
						{state.conv, owner.conv, []int{layout.ConvRows, layout.ConvDim}},
						{state.recurrent, owner.recurrent, []int{layout.RecurrentHeads, layout.KeyHeadDim, layout.ValueHeadDim}},
					}
					for _, pair := range pairs {
						if !unique(pair.owned.Buf()) || !qwen35IndependentDecodePointer(pair.state.Buf()) ||
							pair.state.Buf() != pair.owned.Buf() || pair.state.Dtype != compute.F32 || pair.owned.Dtype != compute.F32 ||
							pair.state.Layout != compute.RowMajor || pair.owned.Layout != compute.RowMajor ||
							pair.state.Quant != nil || pair.owned.Quant != nil ||
							!slices.Equal(pair.state.Shape, pair.shape) || !slices.Equal(pair.owned.Shape, pair.shape) {
							return false
						}
					}
					return true
				}()
				if !private {
					return false
				}
				states[i][layer] = compute.Qwen35SequenceState{Conv: state.conv, Recurrent: state.recurrent}
			}
			return linear > 0
		}()
		if !valid {
			return 0
		}
	}
	// Distinct opaque handles do not prove physical storage independence. Require
	// positive whole-cohort backend evidence, outside the model ownership locks.
	// The caller's session serialization still spans this query and submissions.
	proof, ok := backend.(interface {
		Qwen35IndependentDecodeCohortOwned([]compute.KVStore, [][]compute.Qwen35SequenceState) bool
	})
	if !ok || !proof.Qwen35IndependentDecodeCohortOwned(stores, states) {
		return 0
	}
	return 8
}

func qwen35IndependentDecodePointer(v any) bool {
	if v == nil {
		return false
	}
	rv := reflect.ValueOf(v)
	return rv.Kind() == reflect.Pointer && !rv.IsNil()
}

func (bs *BatchSession) qwen35HybridQ4KBatchOK(batch int) bool {
	if bs == nil || bs.M == nil || batch < 2 || batch > 8 ||
		!bs.M.Cfg.IsQwen35Hybrid() || bs.M.Cfg.BlockTopology != PreNorm || bs.M.Cfg.IsMoE() {
		return false
	}
	if qwen35HybridQ4KBatchStep != nil {
		return true
	}
	// Keep platform overrides authoritative and unsupported scalar cohorts on
	// their established path, without materializing a batch request.
	if len(bs.Seqs) == 0 || bs.Seqs[0] == nil {
		return false
	}
	_, ok := qwen35IndependentDecodeBatchFor(bs.Seqs[0].Backend)
	return ok
}

func (bs *BatchSession) stepBatchQwen35HybridQ4K(ids []int) [][]float32 {
	step := qwen35HybridQ4KBatchStep
	if step == nil {
		step = stepBatchQwen35HybridQ4KIndependent
	}
	out, accepted := step(bs, ids)
	if !accepted {
		// A decline occurs before any model state mutation, so scalar replay is safe.
		return bs.stepBatchSerial(ids, "backend_declined")
	}
	return out
}

type qwen35IndependentDecodeBatchBackend interface {
	compute.Qwen35IndependentDecodeBatchBackend
}

func qwen35IndependentDecodeBatchFor(backend any) (qwen35IndependentDecodeBatchBackend, bool) {
	batch, ok := backend.(qwen35IndependentDecodeBatchBackend)
	return batch, ok && batch.Qwen35IndependentDecodeBatchPath() == compute.Qwen35IndependentDecodeBatchPath
}

// qwen35CohortMutation makes every recurrent pair private across every given
// session, installs those protected handles into the lane requests, and keeps
// them all locked through the whole-cohort callback. It flattens the per-lane
// nested mutateSequence recursion into one scope: an N-lane step pays one
// states allocation and one lock/unlock pass instead of N nested frames, while
// retaining the invariant that no other goroutine can observe a partially
// mutated lane during the batched forward. Locks are taken in lane order - the
// same global order the nested form established - so admission is unchanged.
func qwen35CohortMutation(backend compute.Backend, sessions []*Session, lanes []compute.Qwen35SequencePrefillRequest, fn func() error) error {
	var cleanup qwen35HALLayerRelease
	complete := false
	defer func() {
		if complete {
			cleanup.free()
		} else {
			suppressBackendCleanupPanic(cleanup.free)
		}
	}()
	n := len(sessions)
	locked := make([][]*qwen35HALLayerOwner, n)
	states := make([][]compute.Qwen35SequenceState, n)
	unlockAll := func() {
		for i := n - 1; i >= 0; i-- {
			for j := len(locked[i]) - 1; j >= 0; j-- {
				locked[i][j].mu.Unlock()
			}
			sessions[i].qwen35HAL.mu.Unlock()
		}
	}
	err := func() error {
		for i, s := range sessions {
			q := s.qwen35HAL
			q.mu.Lock()
			states[i] = make([]compute.Qwen35SequenceState, len(q.layers))
			locked[i] = make([]*qwen35HALLayerOwner, 0, len(q.layers))
			for layer := range q.layers {
				state := &q.layers[layer]
				if state.conv.Buf() == nil && state.recurrent.Buf() == nil {
					continue // non-linear layer: empty pair, nothing to protect
				}
				inheritedGraphFailure := state.graphInheritedFailureLocked()
				owner, mutableErr := state.mutableOwnerLocked(backend, &cleanup)
				if mutableErr != nil {
					return fmt.Errorf("lane %d layer %d: %w", i, layer, mutableErr)
				}
				locked[i] = append(locked[i], owner)
				owner.graphInvalid = owner.graphInvalid || inheritedGraphFailure
				states[i][layer] = compute.Qwen35SequenceState{Conv: owner.conv, Recurrent: owner.recurrent}
			}
			lanes[i].States = states[i]
		}
		// Clone failures above issue no mutation attempt. Once the callback starts,
		// every locked pair participates and an abnormal exit invalidates them all.
		for i := range locked {
			for _, owner := range locked[i] {
				owner.graphMutationBeginLocked()
			}
		}
		graphComplete := false
		defer func() {
			for i := range locked {
				for _, owner := range locked[i] {
					owner.graphMutationFinishLocked(graphComplete)
				}
			}
		}()
		mutationErr := fn()
		graphComplete = mutationErr == nil
		return mutationErr
	}()
	unlockAll()
	complete = err == nil
	return err
}

func failQwen35IndependentDecodeBatchSessions(sessions []*Session, cause error) {
	for _, s := range sessions {
		if s == nil {
			continue
		}
		if s.halFailure == nil {
			s.halFailure = cause
		}
		if s.halClosed {
			continue
		}
		// Retire every participating session even if one backend cleanup panics.
		func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
	}
}

// qwen35DecodeBatchRequests builds every lane's one-token decode request in one
// pass instead of one per-lane builder frame. A resident embedding table is
// shared by every lane exactly as before; a packed (Q2_K) or oversized table
// gathers all token rows in one host pass. A backend advertising
// Qwen35IndependentDecodeSharedEmbeddingPanel uploads one [lanes, hidden] panel
// whose row i each lane borrows; every other backend keeps one uploaded row
// buffer per lane. ok=false is a decline before any state mutation; release
// retires everything the call uploaded.
func (bs *BatchSession) qwen35DecodeBatchRequests(ids []int) ([]compute.Qwen35SequencePrefillRequest, func(), bool) {
	n := len(ids)
	lead := bs.Seqs[0]
	cfg := lead.M.Cfg
	packed := lead.M.Q2KEmbedding != nil
	shape := []int{cfg.VocabSize, cfg.HiddenSize}
	if meta, ok := lead.M.manifest["model.embed_tokens.weight"]; ok && len(meta.Shape) == 2 {
		shape = meta.Shape
	}
	if !packed && deviceEmbeddingTableFits(lead.Backend, shape) {
		lanes := make([]compute.Qwen35SequencePrefillRequest, n)
		for i, s := range bs.Seqs {
			lanes[i] = s.qwen35SequencePrefillRequest([]int{ids[i]}, true)
		}
		return lanes, func() {}, true
	}
	rows, ok := lead.Backend.(compute.Qwen35SequenceEmbeddingRowsBackend)
	if !ok || rows.Qwen35SequenceEmbeddingRowsPath() != compute.Qwen35SequenceEmbeddingRowsPath {
		return nil, nil, false
	}
	shared := false
	if panel, ok := lead.Backend.(compute.Qwen35IndependentDecodeSharedEmbeddingPanel); ok {
		shared = panel.Qwen35IndependentDecodeSharedEmbeddingPanel()
	}
	var panel compute.Tensor
	var releases []func()
	releaseAll := func() {
		for _, release := range releases {
			release()
		}
	}
	if shared {
		data, err := lead.qwen35EmbeddingRows(ids)
		if err != nil {
			return nil, nil, false
		}
		panel = lead.uploadHostF32([]int{n, cfg.HiddenSize}, data, compute.MemoryActivation, "qwen35-decode-batch-embedding-panel")
		owner := lead.Backend
		releases = append(releases, func() { owner.Free(panel) })
	}
	lanes := make([]compute.Qwen35SequencePrefillRequest, n)
	for i, s := range bs.Seqs {
		req := s.qwen35SequencePrefillRequestWithOutput([]int{ids[i]}, true, compute.Tensor{}, false, 0, true)
		if shared {
			req.TokenEmbedding = panel
			req.TokenEmbeddingRows = true
			req.TokenEmbeddingRow = i
			req.TokenEmbeddingVocab = cfg.VocabSize
		} else {
			data, err := s.qwen35EmbeddingRows([]int{ids[i]})
			if err != nil {
				releaseAll()
				return nil, nil, false
			}
			row := s.uploadHostF32([]int{1, cfg.HiddenSize}, data, compute.MemoryActivation, "qwen35-decode-batch-embedding-row")
			owner := s.Backend
			releases = append(releases, func() { owner.Free(row) })
			req.TokenEmbedding = row
			req.TokenEmbeddingRows = true
			req.TokenEmbeddingVocab = cfg.VocabSize
		}
		lanes[i] = req
	}
	return lanes, releaseAll, true
}

func stepBatchQwen35HybridQ4KIndependent(bs *BatchSession, ids []int) ([][]float32, bool) {
	if bs == nil || bs.M == nil || len(ids) < 2 || len(ids) > 8 || len(ids) != len(bs.Seqs) {
		return nil, false
	}
	diagT0 := time.Now()
	var diagPrep, diagLineage, diagCall, diagRead, diagSplit time.Duration
	var be compute.Backend
	for i, s := range bs.Seqs {
		// Failed cleanup can retain state pointers; they do not reopen a retired lane.
		s.ensureOpenBackendSession()
		if s == nil || s.M != bs.M || s.Backend == nil || s.qwen35HAL == nil || s.halKV == nil {
			return nil, false
		}
		if i == 0 {
			be = s.Backend
		} else if s.Backend != be {
			return nil, false
		}
	}
	lanes, release, ok := bs.qwen35DecodeBatchRequests(ids)
	if !ok {
		return nil, false
	}
	defer release()
	finishes := make([]func(), 0, len(ids))
	diagPrep = time.Since(diagT0)
	batch, ok := qwen35IndependentDecodeBatchFor(be)
	if !ok {
		return nil, false
	}
	for i, s := range bs.Seqs {
		finishes = append(finishes, s.beginHALTokenLineageWrite([]int{ids[i]}))
	}
	defer func() {
		for i := len(finishes) - 1; i >= 0; i-- {
			finishes[i]()
		}
	}()
	diagLineage = time.Since(diagT0) - diagPrep
	var result compute.Qwen35IndependentDecodeBatchResult
	var diagBackend time.Duration
	callErr := qwen35CohortMutation(be, bs.Seqs, lanes, func() error {
		backendT0 := time.Now()
		var err error
		result, err = batch.Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest{Path: compute.Qwen35IndependentDecodeBatchPath, Lanes: lanes})
		diagBackend = time.Since(backendT0)
		return err
	})
	diagCall = time.Since(diagT0) - diagPrep - diagLineage
	if err := callErr; err != nil {
		if compute.Qwen35IndependentDecodeBatchMutated(err) {
			failQwen35IndependentDecodeBatchSessions(bs.Seqs, err)
			panic(err)
		}
		return nil, false
	}
	if result.Lanes != len(ids) || result.SharedProjectionPanels <= 0 || result.ProjectionMACs <= 0 || result.SerialFallbacks != 0 || result.Transfers.ActivationD2HBytes != 0 || result.Logits.Buf() == nil || len(result.Logits.Shape) != 2 || result.Logits.Shape[0] != len(ids) || result.Logits.Shape[1] != bs.M.Cfg.VocabSize {
		cause := fmt.Errorf("model: malformed Qwen Vulkan independent decode result: lanes=%d panels=%d fallback=%d shape=%v", result.Lanes, result.SharedProjectionPanels, result.SerialFallbacks, result.Logits.Shape)
		failQwen35IndependentDecodeBatchSessions(bs.Seqs, cause)
		panic(cause.Error()) // preserve the existing initial string-panic contract
	}
	flat := be.Read(result.Logits)
	be.Free(result.Logits)
	diagRead = time.Since(diagT0) - diagPrep - diagLineage - diagCall
	if len(flat) != len(ids)*bs.M.Cfg.VocabSize {
		cause := fmt.Errorf("model: malformed Qwen Vulkan independent decode logits")
		failQwen35IndependentDecodeBatchSessions(bs.Seqs, cause)
		panic(cause.Error()) // retries retain cause through the session failure latch
	}
	bs.lastStepSharedPanels = result.SharedProjectionPanels
	bs.lastStepMACs = result.ProjectionMACs
	rows := splitScaledLogits(nil, flat, len(ids), bs.M.Cfg.VocabSize, bs.M.Cfg)
	diagSplit = time.Since(diagT0) - diagPrep - diagLineage - diagCall - diagRead
	if qwen35BatchDiagEnabled {
		// This dispatcher span excludes the deferred lineage and embedding-row cleanup.
		fmt.Fprintf(os.Stderr, "fak_batch_model_diag ids=%d prep_us=%d lineage_us=%d call_us=%d backend_us=%d hal_us=%d read_us=%d split_us=%d total_us=%d\n",
			len(ids), diagPrep.Microseconds(), diagLineage.Microseconds(), diagCall.Microseconds(), diagBackend.Microseconds(), (diagCall - diagBackend).Microseconds(), diagRead.Microseconds(), diagSplit.Microseconds(), time.Since(diagT0).Microseconds())
	}
	return rows, true
}
