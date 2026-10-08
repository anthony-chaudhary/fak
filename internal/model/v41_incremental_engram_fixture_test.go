package model

import (
	"errors"
	"testing"
)

var v41IncrementalEngramReadFault = errors.New("incremental Engram fixture read fault")

type v41IncrementalEngramSource struct {
	base      *v41EngramMemorySource
	fail      bool
	malformed bool
	onRead    func()
}

func (s *v41IncrementalEngramSource) RowBytes() int { return s.base.RowBytes() }

func (s *v41IncrementalEngramSource) ReadRows(start, count int, dst []byte) (int, error) {
	if s.fail {
		return 0, v41IncrementalEngramReadFault
	}
	n, err := s.base.ReadRows(start, count, dst)
	if err == nil && s.onRead != nil {
		onRead := s.onRead
		s.onRead = nil
		onRead()
	}
	if err == nil && s.malformed {
		for r := 0; r < count; r++ {
			dst[r*V41EngramPackedRowBytes+256] = 255
		}
	}
	return n, err
}

func v41IncrementalEngramFixture(t *testing.T, compressed bool) (*Model, []*v41IncrementalEngramSource) {
	t.Helper()
	m, _ := v41FullEngramModel(t)
	for l := 1; l < 3; l++ {
		for name, meta := range m.manifest {
			const prefix = "model.layers.0."
			if len(name) >= len(prefix) && name[:len(prefix)] == prefix {
				target := layerName(l, name[len(prefix):])
				m.manifest[target] = meta
			}
		}
	}
	m.Cfg.NumLayers = 3
	m.Cfg.Window = []int{-1}
	d := m.Cfg.DeepSeekV41
	d.CompressRatios = []int{0, 0, 0}
	d.KVSourceLayerIDs = nil
	d.IndexSourceLayerIDs = nil
	d.EngramLayerIDs = []int{0, 2}
	d.EngramNumEmbeddings = []int{408, 408}
	if compressed {
		d.CompressRatios = []int{2, 2, 2}
		H := m.Cfg.HiddenSize
		var tensors []synthTensor
		for l := 0; l < m.Cfg.NumLayers; l++ {
			tensors = append(tensors,
				synthTensor{layerName(l, "attn.compressor.wkv.weight"), []int{v41KVLoraRank, H}},
				synthTensor{layerName(l, "attn.compressor.wgate.weight"), []int{v41KVLoraRank, H}},
				synthTensor{layerName(l, "attn.compressor.norm.weight"), []int{v41KVLoraRank}},
			)
		}
		man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
			if hasSuffix(name, "compressor.norm.weight") {
				return 1
			}
			return synthMatmulFill(name, next)
		})
		for name, meta := range man {
			meta.Offset += len(m.raw)
			m.manifest[name] = meta
		}
		m.raw = append(m.raw, raw...)
	}
	layout := V41EngramLayout{
		TokenMap: make([]uint32, m.Cfg.VocabSize), CompressedVocab: uint32(m.Cfg.VocabSize), PadID: 0,
		Rows: []uint32{408, 408}, Multipliers: [][]uint64{{17, 13, 19, 11}, {23, 7, 29, 5}},
		Primes: [][]uint32{make([]uint32, 24), make([]uint32, 24)}, MaxNgramSize: 4, HeadsPerNgram: 8,
	}
	for i := range layout.TokenMap {
		layout.TokenMap[i] = uint32(i)
	}
	for l := range layout.Primes {
		for i := range layout.Primes[l] {
			layout.Primes[l][i] = 17
		}
	}
	sources := make([]*v41IncrementalEngramSource, 2)
	rows := make([]V41EngramRowSource, 2)
	for l := range sources {
		sources[l] = &v41IncrementalEngramSource{base: &v41EngramMemorySource{packed: v41EngramTestPackedRows(408), rows: 408}}
		rows[l] = sources[l]
	}
	if err := m.wireV41Engram(layout, rows, int64(V41EngramPackedRowBytes)); err != nil {
		t.Fatal(err)
	}
	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("fixture admission: %v", err)
	}
	return m, sources
}

func v41IncrementalEngramRequests(m *Model) int64 {
	var total int64
	for _, cache := range m.v41EngramStageFor().caches {
		s := cache.Stats()
		total += s.Hits + s.Misses
	}
	return total
}

func v41IncrementalEngramCold(t *testing.T, m *Model, history []int) []float32 {
	t.Helper()
	act, err := m.forwardV41(history, nil)
	if err != nil {
		t.Fatalf("cold oracle: %v", err)
	}
	return act.Logits[len(act.Logits)-1]
}
