package model

import (
	"fmt"
	"math"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const qwen35DeviceVerifyMaxDraft = 4

func qwen35DevicePanelAdvertised(s *Session) bool {
	if s == nil || s.Backend == nil {
		return false
	}
	_, ok := s.Backend.(compute.Qwen35SequenceAllLogitsBackend)
	return ok
}

// verifyQwen35DevicePanel advances the live device KV and recurrent state with
// one bounded sequence invocation and reads back only its K logits rows. Its
// caller owns the pre-round PrefixSnapshot and must commit or restore it.
func (s *Session) verifyQwen35DevicePanel(ids []int, boundaryLogits []float32) (rows [][]float32, receipt TargetVerificationReceipt, err error) {
	receipt = TargetVerificationReceipt{
		Schema:      targetVerificationReceiptSchema,
		Engine:      targetVerificationEngine,
		Path:        targetVerificationDecodePath,
		DraftTokens: len(ids),
	}
	setup := time.Now()
	if s == nil || s.M == nil || s.Backend == nil || s.halKV == nil || s.qwen35HAL == nil {
		receipt.Accounting.Setup = measuredSpeculativeCost(setup)
		return nil, receipt, targetVerificationDowngrade("device target verification requires a live Qwen3.8 hybrid backend session")
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	if !s.M.Cfg.IsQwen35Hybrid() {
		err = targetVerificationDowngrade("device target verification requires the Qwen3.8 hybrid architecture")
	} else if len(ids) < 1 || len(ids) > qwen35DeviceVerifyMaxDraft {
		err = targetVerificationDowngrade("device target verification requires 1..4 draft tokens")
	} else if len(boundaryLogits) != s.M.Cfg.VocabSize {
		err = targetVerificationDowngrade("device target boundary logits width differs from vocabulary")
	}
	for _, value := range boundaryLogits {
		if err == nil && (math.IsNaN(float64(value)) || math.IsInf(float64(value), 0)) {
			err = ErrSpeculativeNonFiniteLogits
		}
	}
	for _, id := range ids {
		if err == nil && (id < 0 || id >= s.M.Cfg.VocabSize) {
			err = targetVerificationDowngrade("device target draft token is outside vocabulary")
		}
	}
	seq, advertised, seqErr := qwen35SequencePrefillBackend(s.Backend)
	if err == nil && (seqErr != nil || !advertised) {
		if seqErr != nil {
			err = seqErr
		} else {
			err = targetVerificationDowngrade("backend does not advertise the resident Qwen3.8 sequence operation")
		}
	}
	allRows, allRowsOK := s.Backend.(compute.Qwen35SequenceAllLogitsBackend)
	if err == nil && (!allRowsOK || allRows.Qwen35SequenceAllLogitsPath() != compute.Qwen35SequenceAllLogitsPath) {
		err = targetVerificationDowngrade("backend does not advertise Qwen3.8 all-row final projection")
	}
	if err == nil {
		if _, split := s.validateDenseGPULayers(); split {
			err = targetVerificationDowngrade("device target verification excludes split host/device layer placement")
		}
	}
	receipt.Accounting.Setup = measuredSpeculativeCost(setup)
	if err != nil {
		return nil, receipt, err
	}
	if !s.qwen35SequenceOutputHeadFits() {
		return nil, receipt, targetVerificationDowngrade("device target output head exceeds the backend buffer cap")
	}

	embedShape := []int{s.M.Cfg.VocabSize, s.M.Cfg.HiddenSize}
	if s.M.Q2KEmbedding != nil {
		embedShape = []int{len(ids), s.M.Q2KEmbedding.Hidden()}
	} else if meta, ok := s.M.manifest["model.embed_tokens.weight"]; ok && len(meta.Shape) == 2 {
		embedShape = meta.Shape
	}
	request := compute.Qwen35SequencePrefillRequest{}
	if s.M.Q2KEmbedding != nil || !deviceEmbeddingTableFits(s.Backend, embedShape) {
		embeddingRows, ok := s.Backend.(compute.Qwen35SequenceEmbeddingRowsBackend)
		if !ok || embeddingRows.Qwen35SequenceEmbeddingRowsPath() != compute.Qwen35SequenceEmbeddingRowsPath {
			return nil, receipt, targetVerificationDowngrade("bounded target embeddings require the resident embedding-row operation")
		}
		data, gatherErr := s.qwen35EmbeddingRows(ids)
		if gatherErr != nil {
			return nil, receipt, fmt.Errorf("model: gather Qwen3.8 device verification embeddings: %w", gatherErr)
		}
		panel := s.uploadHostF32([]int{len(ids), s.M.Cfg.HiddenSize}, data, compute.MemoryActivation, "qwen35-device-verify-embedding-rows")
		defer s.Backend.Free(panel)
		request = s.qwen35SequencePrefillRequestWithEmbedding(ids, false, panel, true, s.M.Cfg.VocabSize)
	} else {
		request = s.qwen35SequencePrefillRequest(ids, false)
	}
	request.NeedAllLogits = true
	receipt.Accounting.Setup = measuredSpeculativeCost(setup)

	startPos := s.halKV.Len()
	finishLineage := s.beginHALTokenLineageWrite(ids)
	defer finishLineage()
	defer s.retireRequestResources()
	started := time.Now()
	receipt.Path = targetVerificationQwen38DevicePath
	receipt.TargetVerificationOperations = 1
	result, callErr := seq.Qwen35SequencePrefill(request)
	if callErr != nil {
		receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
		return nil, receipt, &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardQwen35GDN, Path: compute.Qwen35SequencePrefillPath, Layer: -1, Stage: "device target verification", Cause: callErr}
	}
	vocab := s.M.Cfg.VocabSize
	if result.Tokens != len(ids) || s.halKV.Len() != startPos+len(ids) || result.LastHidden.Buf() == nil || !result.LastHidden.Ready() || result.LogitsRows.Buf() == nil || !result.LogitsRows.Ready() || len(result.LogitsRows.Shape) != 2 || result.LogitsRows.Shape[0] != len(ids) || result.LogitsRows.Shape[1] != vocab {
		receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
		return nil, receipt, fmt.Errorf("model: malformed Qwen3.8 device verification result: tokens=%d want=%d kv_len=%d want=%d logits_shape=%v", result.Tokens, len(ids), s.halKV.Len(), startPos+len(ids), result.LogitsRows.Shape)
	}
	flat := s.Backend.Read(result.LogitsRows)
	if len(flat) != len(ids)*vocab {
		receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
		return nil, receipt, fmt.Errorf("model: Qwen3.8 device verification read %d logits, want %d", len(flat), len(ids)*vocab)
	}
	// rows below retain one host copy after the backend tensor is retired. This
	// is known live result memory, not a process-peak or transfer-bandwidth claim.
	receipt.Accounting.KnownMemoryBytes += int64(len(flat)) * 4
	rows = make([][]float32, len(ids))
	for i := range rows {
		rows[i] = append([]float32(nil), flat[i*vocab:(i+1)*vocab]...)
		for _, value := range rows[i] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
				return nil, receipt, ErrSpeculativeNonFiniteLogits
			}
		}
	}
	s.halStep += len(ids)
	s.halLogitsWarm = true
	receipt.Accounting.TargetVerification = measuredSpeculativeCost(started)
	receipt.OneOperation = true
	return rows, receipt, nil
}
