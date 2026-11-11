package model

import "github.com/anthony-chaudhary/fak/internal/compute"

type qwen35HostMetadata struct {
	model   *Model
	backend compute.Backend
	quant   bool
	f16     bool
	q4k     bool
	request compute.Qwen35SequencePrefillRequest
}

func (s *Session) qwen35HostMetadataRequest() compute.Qwen35SequencePrefillRequest {
	q := s.qwen35HAL
	if q == nil {
		return s.buildQwen35HostMetadata()
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	m := q.hostMetadata
	if m == nil || m.model != s.M || m.backend != s.Backend || m.quant != s.Quant || m.f16 != s.F16 || m.q4k != s.Q4K {
		m = &qwen35HostMetadata{
			model: s.M, backend: s.Backend, quant: s.Quant, f16: s.F16, q4k: s.Q4K,
			request: s.buildQwen35HostMetadata(),
		}
		q.hostMetadata = m
	}
	return m.request
}

// qwen35SequencePrefillRequestWithOutput fills the per-request fields of a
// sequence-prefill request over the cached host-metadata template. Output is
// present only for callers that consume the vocabulary head; the advertised
// headless prefix path keeps it absent.
func (s *Session) qwen35SequencePrefillRequestWithOutput(ids []int, needLogits bool, embedding compute.Tensor, embeddingRows bool, embeddingVocab int, includeOutput bool) compute.Qwen35SequencePrefillRequest {
	req := s.qwen35HostMetadataRequest()
	// Backends receive request-owned slices; their writes cannot corrupt the cached template.
	req.Layers = append([]compute.Qwen35SequenceLayer(nil), req.Layers...)
	req.RoPEThetaForLayer = append([]float64(nil), req.RoPEThetaForLayer...)
	req.States = make([]compute.Qwen35SequenceState, len(req.Layers))
	req.TokenIDs = append([]int(nil), ids...)
	req.StartPos, req.KV = s.halKV.Len(), s.halKV
	req.TokenEmbedding, req.TokenEmbeddingRows, req.TokenEmbeddingVocab = embedding, embeddingRows, embeddingVocab
	req.NeedLogits, req.CaptureRawHidden = needLogits, s.captureTargetHidden
	if includeOutput {
		req.Output = s.lmHeadMatHAL()
	}
	return req
}
