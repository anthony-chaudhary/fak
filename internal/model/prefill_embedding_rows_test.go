package model

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"
)

func prefillEmbeddingFixture(t *testing.T, format packedEmbeddingFormat, prism bool) (*Model, [][]int, int, int) {
	t.Helper()
	const H, vocab = 256, 8
	P := parThreshold/H + 1
	prompts := make([][]int, 2)
	for b := range prompts {
		prompts[b] = make([]int, P)
		for i := range prompts[b] {
			prompts[b][i] = (i*3 + b) % vocab
		}
	}
	manifest, raw := synthBuildRaw([]synthTensor{{"model.embed_tokens.weight", []int{vocab, H}}, {"projection.weight", []int{2, H}}}, func(_ string, next func() float32) float32 { return next() })
	m := &Model{manifest: manifest, raw: raw, Cfg: Config{HiddenSize: H, VocabSize: vocab, EmbedScale: 1.75}}
	t.Cleanup(func() { _ = m.CloseWeights() })
	if format != 0 {
		blockWeights, blockBytes := format.blockWeights(), format.blockBytes()
		payload := make([]byte, vocab*(H/blockWeights)*blockBytes)
		lcgBytes(payload, uint64(format)+778)
		for off := 0; off < len(payload); off += blockBytes {
			switch format {
			case packedEmbeddingQ4K:
				binary.LittleEndian.PutUint16(payload[off:], 0x3400)
				binary.LittleEndian.PutUint16(payload[off+2:], 0x3000)
			case packedEmbeddingPQ2:
				binary.LittleEndian.PutUint16(payload[off:], 0x3400)
			case packedEmbeddingQ6K:
				binary.LittleEndian.PutUint16(payload[off+blockBytes-2:], 0x3400)
			default:
				binary.LittleEndian.PutUint16(payload[off+blockBytes-4:], 0x3400)
				binary.LittleEndian.PutUint16(payload[off+blockBytes-2:], 0x3000)
			}
		}
		var err error
		m.Q2KEmbedding, err = newPackedEmbedding(payload, vocab, H, format)
		if err != nil {
			t.Fatal(err)
		}
	}
	if prism {
		signs := make([]int, H)
		for i := range signs {
			signs[i] = 1
			if i%3 == 1 {
				signs[i] = -1
			}
		}
		if err := m.SetPrismHadamard(PrismHadamardSpec{BlockSize: 64, SignWidths: []int{H}, SignValues: signs, WeightNames: []string{"projection.weight"}, InverseNames: []string{"model.embed_tokens.weight"}}); err != nil {
			t.Fatal(err)
		}
	}
	return m, prompts, P, H
}

// fak-test:runtime fast est=300ms
func TestPrefillEmbeddingRowsKeepOriginalGatherInverseScale(t *testing.T) {
	for _, format := range []packedEmbeddingFormat{0, packedEmbeddingQ2K, packedEmbeddingQ4K, packedEmbeddingPQ2, packedEmbeddingQ6K} {
		for _, prism := range []bool{false, true} {
			m, prompts, P, H := prefillEmbeddingFixture(t, format, prism)
			want, got := make([]float32, len(prompts)*P*H+3), make([]float32, len(prompts)*P*H+3)
			for i := len(got) - 3; i < len(got); i++ {
				want[i], got[i] = 123, 123
			}
			rawBefore := bytes.Clone(m.raw)
			var packedBefore []byte
			if m.Q2KEmbedding != nil {
				packedBefore = bytes.Clone(m.Q2KEmbedding.raw)
			}
			m.embedRectRowsInto(want, prompts, P, H, m.Cfg)
			if !m.tryPrefillEmbedRectRows(got, prompts, P, H, m.Cfg, 4) {
				t.Fatal("validated bulk fixture declined")
			}
			assertFloat32BitsEqual(t, "bulk embedding", want, got)
			if !bytes.Equal(m.raw, rawBefore) || (m.Q2KEmbedding != nil && !bytes.Equal(m.Q2KEmbedding.raw, packedBefore)) {
				t.Fatal("borrowed weight storage changed")
			}
		}
	}
}

// fak-test:runtime fast est=60ms
func TestPrefillEmbeddingAdmissionDeclinesBeforeWrites(t *testing.T) {
	m, prompts, P, H := prefillEmbeddingFixture(t, 0, true)
	dst := make([]float32, len(prompts)*P*H)
	decline := func(label string, ids [][]int, p, h, workers int) {
		t.Helper()
		for i := range dst {
			dst[i] = 99
		}
		if m.tryPrefillEmbedRectRows(dst, ids, p, h, m.Cfg, workers) {
			t.Fatal(label + " admitted")
		}
		for _, v := range dst {
			if v != 99 {
				t.Fatal(label + " changed output")
			}
		}
	}
	decline("serial budget", prompts, P, H, 1)
	decline("short output", prompts, P, H*2, 4)
	decline("negative width", prompts, P, -1, 4)
	bad := append([][]int(nil), prompts...)
	bad[1] = bad[1][:P-1]
	decline("ragged", bad, P, H, 4)
	oldID := prompts[1][1]
	prompts[1][1] = -1
	decline("negative token", prompts, P, H, 4)
	prompts[1][1] = 8
	decline("large token", prompts, P, H, 4)
	prompts[1][1] = oldID
	oldBlock := m.prism.blockSize
	m.prism.blockSize = 3
	decline("invalid inverse block", prompts, P, H, 4)
	m.prism.blockSize = oldBlock
	oldProbe := tensorLookupProbe
	t.Cleanup(func() { tensorLookupProbe = oldProbe })
	// Retain lookup observer count/order by declining before even resolving the
	// f32 view. The function type is inferred from the existing observer variable.
	tensorLookupProbe = func(_ tensorLookupKind) { t.Fatal("fast admission invoked lookup observer") }
	decline("lookup observer", prompts, P, H, 4)
	tensorLookupProbe = oldProbe
}

// fak-test:runtime fast est=80ms
func TestPrefillEmbeddingRefusesEmbeddingAndTokenAliases(t *testing.T) {
	m, prompts, P, H := prefillEmbeddingFixture(t, 0, false)
	work := len(prompts) * P * H
	// A real embedding view that is also the output would make row scheduling
	// observable; keep the original sequential copy semantics instead.
	manifest, raw := synthBuildRaw([]synthTensor{{"model.embed_tokens.weight", []int{len(prompts) * P, H}}}, func(_ string, next func() float32) float32 { return next() })
	aliasModel := &Model{manifest: manifest, raw: raw}
	t.Cleanup(func() { _ = aliasModel.CloseWeights() })
	embed := aliasModel.embedRows()
	before := append([]float32(nil), embed...)
	if aliasModel.tryPrefillEmbedRectRows(embed, prompts, P, H, m.Cfg, 4) {
		t.Fatal("embedding/output alias admitted")
	}
	assertFloat32BitsEqual(t, "declined embedding", before, embed)
	dst := make([]float32, work)
	// Deliberately adversarial cross-typed alias, used only to witness the
	// address-only guard; production code never casts uintptr back to a pointer.
	ids := unsafe.Slice((*int)(unsafe.Pointer(unsafe.SliceData(dst))), P)
	input := [][]int{ids, prompts[1]}
	if m.tryPrefillEmbedRectRows(dst, input, P, H, m.Cfg, 4) {
		t.Fatal("token/output alias admitted")
	}
	for _, v := range dst {
		if v != 0 {
			t.Fatal("alias decline wrote output")
		}
	}
	// Packed raw bytes are equally immutable. Size a genuine constructor-owned
	// table so the adversarial destination view fits entirely inside its payload.
	vocab := (work*4 + q2kBlockBytes - 1) / q2kBlockBytes
	packed, err := NewQ2KEmbedding(make([]byte, vocab*q2kBlockBytes), vocab, H)
	if err != nil {
		t.Fatal(err)
	}
	packedModel := &Model{Q2KEmbedding: packed}
	packedDst := unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(packed.raw))), work)
	beforePacked := bytes.Clone(packed.raw)
	if packedModel.tryPrefillEmbedRectRows(packedDst, prompts, P, H, m.Cfg, 4) {
		t.Fatal("packed/output alias admitted")
	}
	if !bytes.Equal(packed.raw, beforePacked) {
		t.Fatal("packed alias decline wrote source")
	}
}

// fak-test:runtime slow est=2s
func TestRectangularPrefillEmbeddingWorkerParity(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	cfg := Config{HiddenSize: 64, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 16, IntermediateSize: 64, VocabSize: 128, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1}
	m := NewSynthetic(cfg)
	t.Cleanup(func() { _ = m.CloseWeights() })
	m.Quantize()
	P := (parThreshold + 2*cfg.HiddenSize - 1) / (2 * cfg.HiddenSize)
	prompts := [][]int{make([]int, P), make([]int, P)}
	for b := range prompts {
		for i := range prompts[b] {
			prompts[b][i] = (i*7 + b) % cfg.VocabSize
		}
	}
	if gotP, ok := rectangularNoLogitsPrefillLen(prompts); !ok || gotP != P {
		t.Fatal("fixture misses rectangular prompt admission")
	}
	if len(prompts)*P*cfg.HiddenSize < parThreshold {
		t.Fatal("fixture misses parallel embedding threshold")
	}
	for _, quant := range []bool{false, true} {
		if !batchRectFastPathOK(cfg, quant) {
			t.Fatal("fixture misses rectangular production route")
		}
		if err := SetWorkers(1); err != nil {
			t.Fatal(err)
		}
		ref := m.NewBatchSession(2)
		ref.SetQuant(quant)
		ref.PrefillEachNoLogits(prompts)
		if err := SetWorkers(4); err != nil {
			t.Fatal(err)
		}
		got := m.NewBatchSession(2)
		got.SetQuant(quant)
		got.PrefillEachNoLogits(prompts)
		for b := range prompts {
			a, z := ref.Seqs[b].Cache, got.Seqs[b].Cache
			if a.Len() != z.Len() {
				t.Fatal("ordinary cache length changed")
			}
			for l := 0; l < cfg.NumLayers; l++ {
				assertFloat32BitsEqual(t, "ordinary K", a.K[l], z.K[l])
				assertFloat32BitsEqual(t, "ordinary Kraw", a.Kraw[l], z.Kraw[l])
				assertFloat32BitsEqual(t, "ordinary V", a.V[l], z.V[l])
			}
		}
	}
}
