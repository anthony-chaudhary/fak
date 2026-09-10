package model

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type cappedSequencePrefillWithoutEmbeddingRows struct {
	*sequencePrefillWithoutEmbeddingRows
	maxBufferBytes int64
}

type quantizedHeadCappedSequencePrefillBackend struct {
	*cappedSequencePrefillBackend
	maxUploadBytes       int64
	oversizedUploadBytes int64
}

func (b *quantizedHeadCappedSequencePrefillBackend) Caps() compute.Caps {
	caps := b.cappedSequencePrefillBackend.Caps()
	caps.UploadDtype = true
	return caps
}

func (b *quantizedHeadCappedSequencePrefillBackend) recordUpload(t compute.Tensor, as compute.Dtype) {
	bytes := int64(t.Numel() * as.Bytes())
	if as == compute.Q8_0 && t.Quant != nil {
		bytes += int64(len(t.Quant.Scale) * compute.F32.Bytes())
	}
	if bytes > b.maxUploadBytes {
		b.maxUploadBytes = bytes
	}
	if bytes > b.MaxWeightBufferBytes() {
		b.oversizedUploadBytes = bytes
	}
}

func (b *quantizedHeadCappedSequencePrefillBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	b.recordUpload(t, as)
	return b.cappedSequencePrefillBackend.Upload(t, as)
}

func (b *quantizedHeadCappedSequencePrefillBackend) UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	b.recordUpload(t, as)
	return b.cappedSequencePrefillBackend.UploadClass(t, as, class, site)
}

type quantizedHeadCappedSequencePrefillWithoutEmbeddingRows struct {
	*cappedSequencePrefillWithoutEmbeddingRows
}

func (b *quantizedHeadCappedSequencePrefillWithoutEmbeddingRows) Caps() compute.Caps {
	caps := b.cappedSequencePrefillWithoutEmbeddingRows.Caps()
	caps.UploadDtype = true
	return caps
}

func (b *cappedSequencePrefillWithoutEmbeddingRows) MaxWeightBufferBytes() int64 {
	return b.maxBufferBytes
}

func TestQwen35SequencePrefillF32EmbeddingRowsBypassFullTableCap(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.EmbedScale = 2.5
	m := NewSynthetic(cfg)
	// Quantize the tied output projection so it fits below the synthetic single-buffer
	// cap while the source F32 input table does not. Production Q4_K artifacts likewise
	// carry a separately packed output head.
	m.Quantize()
	base := newSequencePrefillBackend(m)
	ids := []int{0, cfg.VocabSize / 2, cfg.VocabSize - 1, cfg.VocabSize / 2}
	panelBytes := int64(len(ids) * cfg.HiddenSize * compute.F32.Bytes())
	fullBytes := int64(cfg.VocabSize * cfg.HiddenSize * compute.F32.Bytes())
	if panelBytes >= fullBytes {
		t.Fatalf("fixture panel=%d must be smaller than full embedding=%d", panelBytes, fullBytes)
	}
	be := &quantizedHeadCappedSequencePrefillBackend{
		cappedSequencePrefillBackend: &cappedSequencePrefillBackend{
			sequencePrefillBackend: base,
			maxBufferBytes:         fullBytes - 1,
		},
	}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.Quant = true

	table := append([]float32(nil), m.embedRows()...)
	want := make([]float32, len(ids)*cfg.HiddenSize)
	for row, id := range ids {
		for col := 0; col < cfg.HiddenSize; col++ {
			want[row*cfg.HiddenSize+col] = table[id*cfg.HiddenSize+col] * cfg.embedScale()
		}
	}

	logits := s.prefillHAL(ids, false)
	if logits != nil {
		t.Fatalf("no-logits prefill returned %d logits", len(logits))
	}
	if base.calls != 1 {
		t.Fatalf("sequence calls=%d, want one batched call", base.calls)
	}
	req := base.requests[0]
	if !reflect.DeepEqual(req.TokenIDs, ids) {
		t.Fatalf("request token ids=%v, want %v", req.TokenIDs, ids)
	}
	if !req.TokenEmbeddingRows || req.TokenEmbeddingVocab != cfg.VocabSize {
		t.Fatalf("request embedding rows=%t vocab=%d, want true/%d", req.TokenEmbeddingRows, req.TokenEmbeddingVocab, cfg.VocabSize)
	}
	wantShape := []int{len(ids), cfg.HiddenSize}
	if !reflect.DeepEqual(req.TokenEmbedding.Shape, wantShape) {
		t.Fatalf("request embedding shape=%v, want %v", req.TokenEmbedding.Shape, wantShape)
	}
	got := base.Read(req.TokenEmbedding)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ordered/scaled embedding panel mismatch: max|delta|=%g", maxAbsDelta(got, want))
	}
	for _, site := range base.sites {
		if site == "hal-weight model.embed_tokens.weight" {
			t.Fatal("oversized whole F32 embedding table was uploaded")
		}
	}
	if site := base.tensorSites[req.TokenEmbedding.Buf()]; site != "qwen35-sequence-f32-embedding-rows" {
		t.Fatalf("embedding panel upload site=%q", site)
	}
	head := m.q8w[m.headName()]
	if head == nil || req.Output.Dtype != compute.Q8_0 {
		t.Fatalf("output head dtype=%s resident=%t, want bounded Q8_0", req.Output.Dtype, head != nil)
	}
	if headBytes := q8ResidentBytes(head); headBytes > be.MaxWeightBufferBytes() {
		t.Fatalf("packed output head bytes=%d exceed cap=%d", headBytes, be.MaxWeightBufferBytes())
	}
	if be.oversizedUploadBytes != 0 || be.maxUploadBytes == 0 {
		t.Fatalf("model upload bound max=%d oversized=%d cap=%d", be.maxUploadBytes, be.oversizedUploadBytes, be.MaxWeightBufferBytes())
	}
	if gotBytes := int64(req.TokenEmbedding.Numel() * req.TokenEmbedding.Dtype.Bytes()); gotBytes != panelBytes || gotBytes > be.MaxWeightBufferBytes() {
		t.Fatalf("embedding panel bytes=%d, want %d within cap=%d", gotBytes, panelBytes, be.MaxWeightBufferBytes())
	}
	if frees := base.freeCalls[req.TokenEmbedding.Buf()]; frees != 1 {
		t.Fatalf("embedding panel free count=%d, want exactly 1", frees)
	}
	if req.StartPos != 0 || s.halStep != len(ids) || s.halKV.Len() != len(ids) {
		t.Fatalf("sequence state start=%d step=%d kv=%d, want 0/%d/%d", req.StartPos, s.halStep, s.halKV.Len(), len(ids), len(ids))
	}
	status, present := s.Qwen35SequencePrefillRouteStatus()
	if !present || status.RequestedPath != compute.Qwen35SequencePrefillPath || status.EffectivePath != compute.Qwen35SequencePrefillPath || status.FallbackActive || !status.NativePerformanceQualifying || status.PackedEmbeddingRows || status.EmbeddingPanelBytes != panelBytes {
		t.Fatalf("F32 embedding-row route status=%+v present=%t, want qualifying panel bytes=%d", status, present, panelBytes)
	}

	moreIDs := []int{7, 7}
	if got := s.prefillHAL(moreIDs, false); got != nil {
		t.Fatalf("appended no-logits prefill returned %d logits", len(got))
	}
	if base.calls != 2 {
		t.Fatalf("sequence calls after append=%d, want one additional batched call", base.calls)
	}
	appended := base.requests[1]
	if appended.StartPos != len(ids) || s.halStep != len(ids)+len(moreIDs) || s.halKV.Len() != len(ids)+len(moreIDs) {
		t.Fatalf("appended state start=%d step=%d kv=%d, want %d/%d/%d", appended.StartPos, s.halStep, s.halKV.Len(), len(ids), len(ids)+len(moreIDs), len(ids)+len(moreIDs))
	}
	wantAppended := make([]float32, len(moreIDs)*cfg.HiddenSize)
	for row, id := range moreIDs {
		for col := 0; col < cfg.HiddenSize; col++ {
			wantAppended[row*cfg.HiddenSize+col] = table[id*cfg.HiddenSize+col] * cfg.embedScale()
		}
	}
	if got := base.Read(appended.TokenEmbedding); !reflect.DeepEqual(got, wantAppended) {
		t.Fatalf("appended ordered/scaled embedding panel mismatch: max|delta|=%g", maxAbsDelta(got, wantAppended))
	}
}

func TestQwen35SequencePrefillF32EmbeddingRowsFailClosed(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	ids := []int{3, 7, 3}

	t.Run("row capability absent", func(t *testing.T) {
		m := NewSynthetic(cfg)
		m.Quantize()
		base := &sequencePrefillWithoutEmbeddingRows{recordingQwen35Backend: newRecordingQwen35Backend(m)}
		be := &quantizedHeadCappedSequencePrefillWithoutEmbeddingRows{
			cappedSequencePrefillWithoutEmbeddingRows: &cappedSequencePrefillWithoutEmbeddingRows{
				sequencePrefillWithoutEmbeddingRows: base,
				maxBufferBytes:                      int64(cfg.VocabSize*cfg.HiddenSize*compute.F32.Bytes()) - 1,
			},
		}
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		s.Quant = true

		_, used, err := s.tryQwen35SequencePrefill(ids, false)
		if err != nil || used || base.calls != 0 {
			t.Fatalf("row-incapable route used=%t err=%v sequence_calls=%d, want clean fallback", used, err, base.calls)
		}
		status, present := s.Qwen35SequencePrefillRouteStatus()
		if !present || status.DeclineReason != Qwen35SequencePrefillDeclineF32EmbeddingRowsUnsupported || !status.FallbackActive || status.NativePerformanceQualifying || status.PackedEmbeddingRows {
			t.Fatalf("row-incapable route status=%+v present=%t", status, present)
		}
	})

	t.Run("row panel exceeds cap", func(t *testing.T) {
		longIDs := make([]int, 40)
		for i := range longIDs {
			longIDs[i] = i % cfg.VocabSize
		}
		longPanelBytes := int64(len(longIDs) * cfg.HiddenSize * compute.F32.Bytes())
		m := NewSynthetic(cfg)
		m.Quantize()
		base := newSequencePrefillBackend(m)
		be := &quantizedHeadCappedSequencePrefillBackend{
			cappedSequencePrefillBackend: &cappedSequencePrefillBackend{
				sequencePrefillBackend: base,
				maxBufferBytes:         longPanelBytes - 1,
			},
		}
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		s.Quant = true
		if headBytes := q8ResidentBytes(m.q8w[m.headName()]); headBytes > be.MaxWeightBufferBytes() {
			t.Fatalf("fixture packed head bytes=%d exceed cap=%d", headBytes, be.MaxWeightBufferBytes())
		}

		_, used, err := s.tryQwen35SequencePrefill(longIDs, false)
		if err != nil || used || base.calls != 0 {
			t.Fatalf("over-cap row panel used=%t err=%v sequence_calls=%d, want clean fallback", used, err, base.calls)
		}
		status, present := s.Qwen35SequencePrefillRouteStatus()
		if !present || status.DeclineReason != Qwen35SequencePrefillDeclineF32EmbeddingPanelCap || !status.FallbackActive || status.NativePerformanceQualifying || status.PackedEmbeddingRows || status.EmbeddingPanelBytes != longPanelBytes {
			t.Fatalf("over-cap row-panel status=%+v present=%t, want panel bytes=%d", status, present, longPanelBytes)
		}
	})
}

func TestQwen35SequencePrefillF32EmbeddingRowsRejectMalformedTable(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cases := []struct {
		name       string
		mutate     func(*tensorMeta)
		wantDetail string
	}{
		{name: "non-F32 dtype", mutate: func(meta *tensorMeta) { meta.Dtype = "F16" }, wantDetail: "want F32"},
		{name: "unaligned offset", mutate: func(meta *tensorMeta) { meta.Offset++ }, wantDetail: "not 4-byte aligned"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSynthetic(cfg)
			m.Quantize()
			meta := m.manifest["model.embed_tokens.weight"]
			tc.mutate(&meta)
			m.manifest["model.embed_tokens.weight"] = meta
			base := newSequencePrefillBackend(m)
			be := &quantizedHeadCappedSequencePrefillBackend{cappedSequencePrefillBackend: &cappedSequencePrefillBackend{sequencePrefillBackend: base, maxBufferBytes: int64(cfg.VocabSize*cfg.HiddenSize*compute.F32.Bytes()) - 1}}
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			s.Quant = true

			_, used, err := s.tryQwen35SequencePrefill([]int{0, 1}, false)
			var opErr *BackendForwardOperationError
			if !used || !errors.As(err, &opErr) || opErr.Stage != "F32 embedding row gather" || !strings.Contains(err.Error(), tc.wantDetail) {
				t.Fatalf("malformed table used=%t error=%T %v, want typed detail %q", used, err, err, tc.wantDetail)
			}
			if base.calls != 0 {
				t.Fatalf("malformed table reached sequence backend %d times", base.calls)
			}
		})
	}
}

func TestQwen35SequencePrefillF32EmbeddingRowsRejectInvalidTokenIDs(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cases := []struct {
		name string
		ids  []int
	}{
		{name: "negative", ids: []int{-1, 0}},
		{name: "at-vocab", ids: []int{0, cfg.VocabSize}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := NewSynthetic(cfg)
			m.Quantize()
			base := newSequencePrefillBackend(m)
			be := &quantizedHeadCappedSequencePrefillBackend{cappedSequencePrefillBackend: &cappedSequencePrefillBackend{sequencePrefillBackend: base, maxBufferBytes: int64(cfg.VocabSize*cfg.HiddenSize*compute.F32.Bytes()) - 1}}
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			s.Quant = true

			_, used, err := s.tryQwen35SequencePrefill(tc.ids, false)
			if !used || err == nil {
				t.Fatalf("invalid ids=%v used=%t err=%v, want advertised typed error", tc.ids, used, err)
			}
			var opErr *BackendForwardOperationError
			if !errors.As(err, &opErr) || opErr.Stage != "F32 embedding row gather" || !strings.Contains(err.Error(), "token ID") {
				t.Fatalf("invalid ids=%v error=%T %v, want typed F32 row-gather bounds error", tc.ids, err, err)
			}
			if base.calls != 0 {
				t.Fatalf("invalid ids=%v reached sequence backend %d times", tc.ids, base.calls)
			}
		})
	}
}

func TestQwen35SequencePrefillF32EmbeddingFullTablePathWhenItFits(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	base := newSequencePrefillBackend(m)
	fullBytes := int64(cfg.VocabSize * cfg.HiddenSize * compute.F32.Bytes())
	be := &cappedSequencePrefillBackend{sequencePrefillBackend: base, maxBufferBytes: fullBytes}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	ids := []int{3, 7}
	_, used, err := s.tryQwen35SequencePrefill(ids, false)
	if err != nil || !used || base.calls != 1 {
		t.Fatalf("full-table route used=%t err=%v sequence_calls=%d", used, err, base.calls)
	}
	req := base.requests[0]
	if req.TokenEmbeddingRows || !reflect.DeepEqual(req.TokenEmbedding.Shape, []int{cfg.VocabSize, cfg.HiddenSize}) {
		t.Fatalf("full-table request rows=%t shape=%v", req.TokenEmbeddingRows, req.TokenEmbedding.Shape)
	}
	if site := base.tensorSites[req.TokenEmbedding.Buf()]; site != "hal-weight model.embed_tokens.weight" {
		t.Fatalf("full-table embedding upload site=%q", site)
	}
	status, present := s.Qwen35SequencePrefillRouteStatus()
	if !present || status.FallbackActive || !status.NativePerformanceQualifying || status.PackedEmbeddingRows || status.EmbeddingPanelBytes != 0 {
		t.Fatalf("full-table route status=%+v present=%t", status, present)
	}
}
