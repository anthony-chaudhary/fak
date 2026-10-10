package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41RotaryTripleRegression pins the three V4.1 rotary fixes landed on
// 2026-10-09 in one place, so reverting any one of them turns this test red:
//
//   - 56346cb8 fix(model): correct V4.1 per-layer rotary tables
//     (ratio-0 layers: base theta, no YaRN; nonzero ratios: compressed theta + YaRN)
//   - cbf95b03 fix(model): restore V4.1 inverse attention rotation
//     (the attention output is conjugate-rotated at its query position before
//     the grouped output projection)
//   - 72ade8c9 fix(model): fix compressed rotary cache publication
//     (published compressed latent/index rows carry the rotation at the group's
//     FIRST absolute position, applied after normalization/index projection)
//
// Every expected value is computed here from the closed-form rotary equation
// (or the independent v41OracleRopeTable transcription); no production table or
// rotation helper feeds an expected value. One synthetic two-layer compressed
// model (ratio 2, layer 0 KV/index source, layer 1 reader) carries the forward
// checks; its base and compressed thetas differ so a wrong-base table is caught
// on the executed path too. This is a host software witness, not a device or
// full-model parity claim.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41RotaryTripleRegression(t *testing.T) {
	const (
		baseTheta       = 256.0
		compressedTheta = 4096.0
	)
	// angleTable is the independent unscaled interleaved-pair table.
	angleTable := func(pos int, theta float64, rd int) (cos, sin []float32) {
		cos, sin = make([]float32, rd/2), make([]float32, rd/2)
		for j := range cos {
			a := float64(pos) / math.Pow(theta, float64(2*j)/float64(rd))
			cos[j], sin[j] = float32(math.Cos(a)), float32(math.Sin(a))
		}
		return cos, sin
	}
	// rotateTail rotates (sign=+1) or conjugate-rotates (sign=-1) the trailing rd
	// components of each headDim-wide head as adjacent complex pairs.
	rotateTail := func(row []float32, headDim, rd, pos int, theta float64, sign float32) []float32 {
		out := append([]float32(nil), row...)
		cos, sin := angleTable(pos, theta, rd)
		for h := 0; h+headDim <= len(out); h += headDim {
			for j := range cos {
				i := h + headDim - rd + 2*j
				c, s := cos[j], sign*sin[j]
				a, b := out[i], out[i+1]
				out[i] = float32(a*c) - float32(b*s)
				out[i+1] = float32(b*c) + float32(a*s)
			}
		}
		return out
	}

	t.Run("per-layer-tables-56346cb8", func(t *testing.T) {
		_, cfg := readDeepSeekV41Config(t)
		ratios := cfg.DeepSeekV41.CompressRatios
		if len(ratios) < 21 || ratios[0] != 0 || ratios[2] == 0 || cfg.RopeScaling != "yarn" ||
			cfg.RopeTheta == cfg.DeepSeekV41.CompressRopeTheta {
			t.Fatal("official V4.1 fixture no longer has distinct plain/compressed rotary regimes")
		}
		for _, layer := range []int{0, 1, 2, 20} {
			for _, pos := range []int{1, 7, 65536} {
				gotCos, gotSin := v41RopeTableForLayer(cfg, layer, pos)
				wantCos, wantSin := v41OracleRopeTable(t, cfg, layer, pos)
				if len(gotCos) != cfg.QKRopeHeadDim/2 || len(gotSin) != len(gotCos) || len(wantCos) != len(gotCos) {
					t.Fatalf("layer %d: table width %d, want %d", layer, len(gotCos), cfg.QKRopeHeadDim/2)
				}
				for j := range gotCos {
					if math.Abs(float64(gotCos[j]-wantCos[j])) > 1e-6 || math.Abs(float64(gotSin[j]-wantSin[j])) > 1e-6 {
						t.Fatalf("layer %d (ratio %d) pos %d pair %d: table (%g,%g), want (%g,%g)",
							layer, ratios[layer], pos, j, gotCos[j], gotSin[j], wantCos[j], wantSin[j])
					}
				}
			}
		}
		// Ratio-0 layer 0 must be the bare base-theta table (no YaRN, unit amplitude).
		cos0, sin0 := v41RopeTableForLayer(cfg, 0, 7)
		bareCos, bareSin := angleTable(7, cfg.RopeTheta, cfg.QKRopeHeadDim)
		if !reflect.DeepEqual(cos0, bareCos) || !reflect.DeepEqual(sin0, bareSin) {
			t.Fatal("ratio-0 layer table is not the bare base-theta table")
		}
		cos2, sin2 := v41RopeTableForLayer(cfg, 2, 7)
		if reflect.DeepEqual(cos0, cos2) && reflect.DeepEqual(sin0, sin2) {
			t.Fatal("plain and compressed layers selected the same rotary table")
		}
	})

	// One tiny synthetic compressed model drives the two executed-path checks.
	m := v41CompressorTestFixtureIndex(t, 128)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	m.Cfg.RopeTheta, m.Cfg.DeepSeekV41.CompressRopeTheta = baseTheta, compressedTheta
	m.Cfg.RopeThetaPerLayer = nil
	m.Cfg.RopeScaling, m.Cfg.LongRope = "", nil
	m.Cfg.RopeFactor, m.Cfg.RopeOrigContext = 0, 0
	hd, rd, heads := m.Cfg.HeadDim, m.Cfg.QKRopeHeadDim, m.Cfg.NumHeads
	if rd < 4 || rd >= hd || m.Cfg.DeepSeekV41.CompressRatios[0] != 2 || m.Cfg.DeepSeekV41.CompressRatios[1] != 2 {
		t.Fatal("synthetic fixture geometry changed")
	}

	t.Run("inverse-attention-output-cbf95b03", func(t *testing.T) {
		if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
			t.Skip("inverse-output software witness requires cpu-ref")
		}
		// Round trip: an independent forward rotation followed by the production
		// inverse is the identity on the whole head (prefix untouched).
		row := make([]float32, heads*hd)
		for i := range row {
			row[i] = float32(i%17)/8 - 1
		}
		for _, layer := range []int{0, 1} {
			for _, pos := range []int{1, 3, 1000} {
				rotated := rotateTail(row, hd, rd, pos, compressedTheta, 1)
				if reflect.DeepEqual(rotated, row) {
					t.Fatal("forward rotation fixture is the identity")
				}
				v41InverseAttentionOutputInPlace(m.Cfg, layer, pos, rotated, heads, hd)
				for i := range row {
					if math.Abs(float64(rotated[i]-row[i])) > 2e-6 {
						t.Fatalf("layer %d pos %d: rotate->inverse[%d]=%g, want %g", layer, pos, i, rotated[i], row[i])
					}
				}
			}
		}

		// Forward: the grouped projection must consume the contraction output
		// conjugate-rotated at the query's absolute position with the layer's
		// (compressed) theta, on both the source and the reader layer.
		st := &v41ForwardState{}
		positions := map[int]int{}
		var contracted []float32
		contractedLayer, contractions, projections := -1, 0, 0
		changed := false
		st.sharedAttention = func(layer int, r v41SharedAttentionRequest) ([]float32, error) {
			out, err := V41SparseAttentionSink(r.q, r.kv, r.sink, r.idx, r.plain)
			if err != nil {
				return nil, err
			}
			contracted, contractedLayer = append([]float32(nil), out...), layer
			contractions++
			return out, nil
		}
		st.groupedOutput = func(layer int, out []float32, h, headDim, groups, rank, dim int) ([]float32, v41DenseProjectionOutcome, error) {
			if layer != contractedLayer || contractions != projections+1 || len(out) != len(contracted) {
				t.Fatalf("grouped projection did not immediately consume layer %d's contraction", layer)
			}
			pos := positions[layer]
			positions[layer]++
			want := rotateTail(contracted, headDim, rd, pos, compressedTheta, -1)
			for i := range want {
				if delta := math.Abs(float64(out[i]) - float64(want[i])); math.IsNaN(delta) || delta > 1e-6+1e-6*math.Abs(float64(want[i])) {
					t.Fatalf("layer %d pos %d: projection input[%d]=%g, want inverse-rotated %g (raw %g)",
						layer, pos, i, out[i], want[i], contracted[i])
				}
				if math.Abs(float64(want[i])-float64(contracted[i])) > 1e-4 {
					changed = true
				}
			}
			projections++
			return make([]float32, dim), v41ProjectionHandled, nil
		}
		if _, err := m.forwardV41([]int{1, 2, 3, 4}, st); err != nil {
			t.Fatal(err)
		}
		if positions[0] != 4 || positions[1] != 4 || !changed {
			t.Fatalf("forward witness incomplete: positions=%v discriminating=%v", positions, changed)
		}
	})

	t.Run("compressed-publication-72ade8c9", func(t *testing.T) {
		latents, keys := map[int][][]float32{}, map[int][][]float32{}
		st := &v41ForwardState{}
		st.compressorNorm = func(l int, input, gain []float32, eps float32) ([]float32, error) {
			out := v41CompressorNormRefTail(input, gain, eps, "")
			latents[l] = append(latents[l], append([]float32(nil), out...))
			return out, nil
		}
		st.indexKeyNorm = func(l int, input, gain []float32, eps float32) ([]float32, error) {
			out := v41IndexKeyNormReference(input, gain, eps)
			keys[l] = append(keys[l], append([]float32(nil), out...))
			return out, nil
		}
		if _, err := m.forwardV41([]int{1, 2, 3, 4}, st); err != nil {
			t.Fatal(err)
		}
		bf16Tail := func(row []float32, width int) []float32 {
			for i := len(row) - width; i < len(row); i++ {
				row[i] = v41CompressorNormRefCast(row[i])
			}
			return row
		}
		publishedLatent := func(group, pos int) []float32 {
			return bf16Tail(rotateTail(latents[0][group], hd, rd, pos, compressedTheta, 1), rd)
		}
		publishedKey := func(group, pos int) []float32 {
			key := bf16Tail(append([]float32(nil), keys[0][group]...), len(keys[0][group]))
			return bf16Tail(rotateTail(key, len(key), rd, pos, compressedTheta, 1), rd)
		}
		rows, ok := st.layerState(0).KVSourceRows(0)
		idx, idxOK := st.layerState(0).IndexKeys(0)
		if !ok || !idxOK || len(rows) != 2 || len(idx) != 2 || len(latents[0]) != 2 || len(keys[0]) != 2 {
			t.Fatalf("expected two published groups: rows=%d keys=%d latents=%d", len(rows), len(idx), len(latents[0]))
		}
		shared, _ := st.attn.KVSourceRows(0)
		sharedIdx, _ := st.attn.IndexKeys(0)
		if !reflect.DeepEqual(rows, shared) || !reflect.DeepEqual(idx, sharedIdx) {
			t.Fatal("owner and registry published different rows")
		}
		for group := range rows {
			first, closing := group*2, group*2+1
			want := publishedLatent(group, first)
			// Group 0's first position is 0 (identity), so only later groups can
			// separate "rotated at group-first" from "never rotated".
			unrotated := bf16Tail(append([]float32(nil), latents[0][group]...), rd)
			if reflect.DeepEqual(want, publishedLatent(group, closing)) || (first > 0 && reflect.DeepEqual(want, unrotated)) {
				t.Fatalf("group %d: fixture cannot distinguish group-first rotation from closing/unrotated", group)
			}
			if !reflect.DeepEqual(rows[group][:hd-rd], latents[0][group][:hd-rd]) {
				t.Fatalf("group %d: published latent prefix changed", group)
			}
			if !reflect.DeepEqual(rows[group], want) {
				t.Fatalf("group %d: published latent rope tail %v, want rotation at group-first position %d %v",
					group, rows[group][hd-rd:], first, want[hd-rd:])
			}
			if wantKey := publishedKey(group, first); !reflect.DeepEqual(idx[group], wantKey) {
				t.Fatalf("group %d: published index key rope tail %v, want %v", group,
					idx[group][len(wantKey)-rd:], wantKey[len(wantKey)-rd:])
			}
		}
	})
}
