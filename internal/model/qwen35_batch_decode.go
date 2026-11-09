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

// qwen35DecodeBatchLaneRequest builds one lane's one-token decode request. A resident
// embedding table is shared by every lane; a packed (Q2_K) or oversized table instead
// uploads this lane's gathered row, as the sequence prefill does, and returns its release.
// ok=false is a decline before any state mutation.
func (s *Session) qwen35DecodeBatchLaneRequest(id int) (compute.Qwen35SequencePrefillRequest, func(), bool) {
	packed := s.M.Q2KEmbedding != nil
	shape := []int{s.M.Cfg.VocabSize, s.M.Cfg.HiddenSize}
	if meta, ok := s.M.manifest["model.embed_tokens.weight"]; ok && len(meta.Shape) == 2 {
		shape = meta.Shape
	}
	if !packed && deviceEmbeddingTableFits(s.Backend, shape) {
		return s.qwen35SequencePrefillRequest([]int{id}, true), nil, true
	}
	if rows, ok := s.Backend.(compute.Qwen35SequenceEmbeddingRowsBackend); !ok || rows.Qwen35SequenceEmbeddingRowsPath() != compute.Qwen35SequenceEmbeddingRowsPath {
		return compute.Qwen35SequencePrefillRequest{}, nil, false
	}
	data, err := s.qwen35EmbeddingRows([]int{id})
	if err != nil {
		return compute.Qwen35SequencePrefillRequest{}, nil, false
	}
	row := s.uploadHostF32([]int{1, s.M.Cfg.HiddenSize}, data, compute.MemoryActivation, "qwen35-decode-batch-embedding-row")
	return s.qwen35SequencePrefillRequestWithEmbedding([]int{id}, true, row, true, s.M.Cfg.VocabSize), func() { s.Backend.Free(row) }, true
}

func stepBatchQwen35HybridQ4KIndependent(bs *BatchSession, ids []int) ([][]float32, bool) {
	if bs == nil || bs.M == nil || len(ids) < 2 || len(ids) > 8 || len(ids) != len(bs.Seqs) {
		return nil, false
	}
	diagT0 := time.Now()
	var diagPrep, diagLineage, diagCall, diagRead, diagSplit time.Duration
	var be compute.Backend
	lanes := make([]compute.Qwen35SequencePrefillRequest, len(ids))
	finishes := make([]func(), 0, len(ids))
	var releases []func()
	defer func() {
		for _, release := range releases {
			release()
		}
	}()
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
		req, release, ok := s.qwen35DecodeBatchLaneRequest(ids[i])
		if !ok {
			return nil, false
		}
		if release != nil {
			releases = append(releases, release)
		}
		lanes[i] = req
	}
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
	var call func(int) error
	call = func(i int) error {
		if i == len(bs.Seqs) {
			var err error
			backendT0 := time.Now()
			result, err = batch.Qwen35IndependentDecodeBatch(compute.Qwen35IndependentDecodeBatchRequest{Path: compute.Qwen35IndependentDecodeBatchPath, Lanes: lanes})
			diagBackend = time.Since(backendT0)
			return err
		}
		_, err := bs.Seqs[i].qwen35HAL.mutateSequence(be, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
			lanes[i].States = states
			return compute.Qwen35SequencePrefillResult{}, call(i + 1)
		})
		return err
	}
	callErr := call(0)
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
