package model

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var qwen35HostMetadataRequestSink compute.Qwen35SequencePrefillRequest

func qwen35HostMetadataSessions(tb testing.TB, lanes int) []*Session {
	tb.Helper()
	m := NewSynthetic(qwen35HybridTestCfg())
	be := newSequencePrefillBackend(m)
	sessions := make([]*Session, lanes)
	for i := range sessions {
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			tb.Fatal(err)
		}
		sessions[i] = s
		tb.Cleanup(s.Close)
	}
	return sessions
}

// fak-test:runtime fast est=1s
func TestQwen35HostMetadataPreservesRequestOwnership(t *testing.T) {
	sessions := qwen35HostMetadataSessions(t, 2)
	s := sessions[0]
	ids := []int{3}
	embedding := s.tokenEmbeddingHAL()
	first := s.qwen35SequencePrefillRequestWithEmbedding(ids, true, embedding, false, 0)
	warm := s.qwen35SequencePrefillRequestWithEmbedding(ids, true, embedding, false, 0)
	if !reflect.DeepEqual(first, warm) {
		t.Fatal("warm request differs from the initial request")
	}
	norm, theta := warm.Layers[0].InputNorm.Buf(), warm.RoPEThetaForLayer[0]
	ids[0] = 9
	first.TokenIDs[0] = 11
	first.Layers[0].InputNorm = compute.Tensor{}
	first.RoPEThetaForLayer[0]++
	first.States[0].Conv = embedding
	if warm.TokenIDs[0] != 3 || warm.Layers[0].InputNorm.Buf() != norm || warm.RoPEThetaForLayer[0] != theta || warm.States[0].Conv.Buf() != nil {
		t.Fatal("requests share mutable slice storage")
	}
	s.Prefill([]int{3, 7})
	s.captureTargetHidden = true
	row := compute.NewF32(s.Backend, []int{1, s.M.Cfg.HiddenSize}, make([]float32, s.M.Cfg.HiddenSize))
	next := s.qwen35SequencePrefillRequestWithEmbedding([]int{5}, false, row, true, s.M.Cfg.VocabSize)
	if next.TokenIDs[0] != 5 || next.StartPos != 2 || next.KV != s.halKV || next.TokenEmbedding.Buf() != row.Buf() || !next.TokenEmbeddingRows || next.TokenEmbeddingVocab != s.M.Cfg.VocabSize || next.NeedLogits || !next.CaptureRawHidden {
		t.Fatal("cached metadata retained a prior request's dynamic fields")
	}
	if next.Layers[0].InputNorm.Buf() != norm || next.RoPEThetaForLayer[0] != theta || next.States[0].Conv.Buf() != nil {
		t.Fatal("request mutation corrupted subsequent metadata")
	}
	other := sessions[1].qwen35SequencePrefillRequestWithEmbedding([]int{8}, true, embedding, false, 0)
	if other.KV == next.KV || other.StartPos != 0 || other.CaptureRawHidden {
		t.Fatal("independent sessions shared dynamic request state")
	}
}

// fak-test:runtime fast est=1s
func TestQwen35HostMetadataFollowsStateLifetime(t *testing.T) {
	s := qwen35HostMetadataSessions(t, 1)[0]
	s.qwen35SequencePrefillRequest([]int{3}, true)
	first := s.qwen35HAL.hostMetadata
	if first == nil {
		t.Fatal("request preparation did not retain immutable metadata")
	}
	s.Quant = !s.Quant
	s.qwen35SequencePrefillRequest([]int{3}, true)
	if s.qwen35HAL.hostMetadata == first {
		t.Fatal("weight mode change retained the previous metadata")
	}
	state := s.qwen35HAL
	s.Close()
	if state.hostMetadata != nil {
		t.Fatal("closed state retained borrowed weight descriptors")
	}
}

// fak-test:runtime fast est=3s lane=optin
func BenchmarkQwen35HostMetadataBatchRequest(b *testing.B) {
	for _, lanes := range []int{2, 8} {
		b.Run(fmt.Sprintf("lanes%d", lanes), func(b *testing.B) {
			sessions := qwen35HostMetadataSessions(b, lanes)
			prepare := func() {
				for i, s := range sessions {
					req := s.qwen35SequencePrefillRequest([]int{i + 3}, true)
					qwen35HostMetadataRequestSink = req
				}
			}
			prepare()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				prepare()
			}
		})
	}
}
