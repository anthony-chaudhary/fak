package model

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=500ms lane=default
// Source estimate: two four-by-256 Q3_K down projections, three requested
// activation rows, and no model load, backend dispatch, or goroutines.
func TestV41GroupedDownReusesMaterialization(t *testing.T) {
	const hidden, intermediate, experts = 4, qkK, 2
	cfg := Config{HiddenSize: hidden, MoEIntermediateSize: intermediate, HeadDim: 32}
	inputs := [][]float32{{.25, -.5, .75, 1}, {-.125, .25, .5, .75}, {.5, .125, -.25, 1}}
	picks := [][]routePick{
		{{expert: 1, weight: .25}, {expert: 0, weight: .75}},
		{{expert: 0, weight: .5}, {expert: 1, weight: .125}},
		{{expert: 1, weight: .75}, {expert: 0, weight: .25}},
	}
	requested := 0
	distinct := map[int]bool{}
	for _, row := range picks {
		requested += len(row)
		for _, pick := range row {
			distinct[pick.expert] = true
		}
	}
	stride := hidden * q3kBlockBytes
	raw := make([]byte, experts*stride)
	decoded := make([][]float32, experts)
	for expert := range decoded {
		part := raw[expert*stride : (expert+1)*stride]
		for i := range part {
			part[i] = byte(i*37 + expert*11)
		}
		for off := 0; off < len(part); off += q3kBlockBytes {
			binary.LittleEndian.PutUint16(part[off+q3kBlockBytes-2:], 0x1800)
		}
		decoded[expert] = make([]float32, hidden*intermediate)
		DequantQ3K(decoded[expert], part)
	}
	newModel := func(t *testing.T, config Config) *Model {
		t.Helper()
		tier := NewExpertCheckpointTier(0)
		if err := tier.AddShard(bytes.NewReader(raw), int64(len(raw)), []FusedExpertTensor{{
			Name: "blk.0.ffn_down_exps.weight", Layer: 0, Proj: "down_proj", Arch: "deepseek41",
			Quant: ExpertCheckpointQ3K, Experts: experts, Rows: hidden, Cols: intermediate,
		}}); err != nil {
			t.Fatal(err)
		}
		return &Model{Cfg: config, expertCheckpoint: tier}
	}
	gate := func(_ int, _ string, x []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		h := make([]float32, intermediate)
		for i := range h {
			h[i] = x[i%len(x)] * float32(i%5+1) / 8
		}
		return h, v41GateUpHandled, nil
	}
	for _, mode := range []string{"reduced", "full_contract"} {
		t.Run(mode, func(t *testing.T) {
			config := cfg
			full := mode == "full_contract"
			if full {
				config.HeadDim = v41KVLoraRank
			}
			// The numerical control runs each requested pick independently with
			// already decoded immutable weights. It cannot use the grouped memo.
			oracle := &Model{Cfg: config}
			want := make([][]float32, len(inputs))
			for token, input := range inputs {
				want[token] = make([]float32, hidden)
				values := make([][]float32, len(picks[token]))
				for slot, pick := range picks[token] {
					if full {
						var err error
						values[slot], err = oracle.v41FullRoutedExpert(0, "ffn.experts."+itoa(pick.expert), input, pick.weight, config, gate, nil, nil,
							func() ([]float32, error) { return decoded[pick.expert], nil }, true)
						if err != nil {
							t.Fatal(err)
						}
					} else {
						h, _, _ := gate(0, "", input)
						for i, v := range matRows(decoded[pick.expert], h, hidden, intermediate) {
							want[token][i] += pick.weight * v
						}
					}
				}
				if full {
					var err error
					want[token], err = v41FullRoutedSum(0, picks[token], values, hidden)
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			for _, budget := range []struct {
				name  string
				bytes int64
			}{{"disabled", 0}, {"undersized", int64(hidden*intermediate*4 - 1)}} {
				t.Run(budget.name, func(t *testing.T) {
					m := newModel(t, config)
					scratch := &v41ProjScratch{expertLayerCacheBytes: budget.bytes}
					gateCalls, downCalls := 0, 0
					st := &v41ForwardState{
						expertGateUp: func(l int, stem string, x []float32) ([]float32, v41ExpertGateUpOutcome, error) {
							gateCalls++
							return gate(l, stem, x)
						},
						expertDown: func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
							downCalls++
							return nil, v41DownDeclined, nil
						},
					}
					for invocation := 1; invocation <= 2; invocation++ {
						got := make([][]float32, len(inputs))
						if err := m.v41ContractRoutedGrouped(0, inputs, picks, scratch, config, got, st); err != nil {
							t.Fatal(err)
						}
						nonzero := false
						for row := range want {
							if len(got[row]) != len(want[row]) {
								t.Fatalf("row=%d width=%d want=%d", row, len(got[row]), len(want[row]))
							}
							for i, v := range want[row] {
								nonzero = nonzero || v != 0
								if !finite32(got[row][i]) || math.Float32bits(got[row][i]) != math.Float32bits(v) {
									t.Fatalf("row=%d lane=%d got=%g want=%g", row, i, got[row][i], v)
								}
							}
						}
						if !nonzero || gateCalls != invocation*requested || downCalls != invocation*requested {
							t.Fatalf("nonzero=%t gate=%d down=%d requested=%d", nonzero, gateCalls, downCalls, invocation*requested)
						}
						if len(scratch.exp1) != 0 || len(scratch.exp3) != 0 || len(scratch.layerExperts) != 0 || scratch.layerExpertBytes != 0 || scratch.expertLayerCacheBytes != budget.bytes {
							t.Fatal("host gate/up materialized or layer-cache policy changed")
						}
						stats := m.expertCheckpoint.Stats()
						wantReads := invocation * len(distinct)
						if stats.Reads != wantReads || stats.BytesRead != int64(wantReads*stride) || stats.Hits != 0 {
							t.Fatalf("grouped down repeated materialization: invocation=%d reads=%d bytes=%d want_reads=%d want_bytes=%d gate=%d down=%d parity=true", invocation, stats.Reads, stats.BytesRead, wantReads, wantReads*stride, gateCalls, downCalls)
						}
					}
				})
			}
		})
	}
	t.Run("selected_down_error_stays_before_weight_read", func(t *testing.T) {
		m := newModel(t, cfg)
		sentinel := errors.New("selected down refusal")
		calls := 0
		st := &v41ForwardState{expertGateUp: gate, expertDown: func(int, string, []float32) ([]float32, v41ExpertDownOutcome, error) {
			calls++
			return nil, v41DownError, sentinel
		}}
		out := make([][]float32, len(inputs))
		err := m.v41ContractRoutedGrouped(0, inputs, picks, &v41ProjScratch{}, cfg, out, st)
		if !errors.Is(err, sentinel) || calls != 1 || m.expertCheckpoint.Stats().Reads != 0 {
			t.Fatalf("selected failure=%v calls=%d reads=%d", err, calls, m.expertCheckpoint.Stats().Reads)
		}
		for _, row := range out {
			for _, v := range row {
				if v != 0 {
					t.Fatal("selected failure published a partial grouped result")
				}
			}
		}
	})
}
