package model

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

// The fixture uses real original-format bytes and all128/top3 draft experts,
// with tiny dimensions. Target history/features are explicitly seeded here;
// this is not a target-capture, checkpoint-parity, or serving qualification.
// fak-test:runtime medium est=3s lane=default
func TestV41DSparkNativeReference(t *testing.T) {
	ctx := context.Background()
	t.Run("three stages shared head Markov confidence and read-only proposal", func(t *testing.T) {
		f, shard, rows := v41DSparkReferenceFixture(t, nil)
		if err := f.AppendCommitted(ctx, rows); err != nil {
			t.Fatal(err)
		}
		before := v41DSparkRingCopy(f)
		got, err := f.Propose(ctx, 1)
		if err != nil {
			t.Fatal(err)
		}
		if got.Anchor != 1 || got.AnchorPosition != 2 || got.Proposed != [5]int{2, 3, 0, 1, 2} {
			t.Fatalf("anchor/proposal accounting: %+v", got)
		}
		if f.next != 2 || !reflect.DeepEqual(before, v41DSparkRingCopy(f)) {
			t.Fatal("proposal mutated committed rings")
		}
		for i := range got.Proposed {
			if len(got.Logits[i]) != 8 || len(got.hidden[i]) != f.cfg.HiddenSize {
				t.Fatal("incomplete five-row head")
			}
			confWeight, err := f.w.row(ctx, "mtp.2.confidence_head.proj.weight", 0)
			if err != nil {
				t.Fatal(err)
			}
			prev := 1
			if i != 0 {
				prev = got.Proposed[i-1]
			}
			markov, err := f.w.row(ctx, "mtp.2.markov_head.embed.weight", prev)
			if err != nil {
				t.Fatal(err)
			}
			var want float32
			for j, v := range got.hidden[i] {
				want += float32(v * confWeight[j])
			}
			for j, v := range markov {
				want += float32(v * confWeight[f.cfg.HiddenSize+j])
			}
			if math.Abs(float64(got.ConfidenceLogits[i]-want)) > 1e-5 ||
				math.Abs(float64(got.ConditionalConfidence[i])-1/(1+math.Exp(-float64(want)))) > 1e-6 {
				t.Fatalf("confidence row%d does not use pre-norm hidden and previous-token embedding", i)
			}
		}
		again, err := f.Propose(ctx, 1)
		if err != nil || !reflect.DeepEqual(got, again) {
			t.Fatalf("read-only replay: %v", err)
		}
		got.hidden[0][0] = 999
		got.Logits[0][0] = 999
		third, err := f.Propose(ctx, 1)
		if err != nil || !reflect.DeepEqual(again, third) {
			t.Fatal("proposal result aliases persistent state")
		}

		// A target-owned head change must affect draft logits; there is no
		// invented mtp.embed/head or Qwen-specific head in the draft source.
		head := f.model.tensor("lm_head.weight")
		for i := range head {
			head[i] = 0
		}
		for i := 0; i < f.cfg.HiddenSize; i++ {
			head[6*f.cfg.HiddenSize+i] = 100
		}
		shared, err := f.Propose(ctx, 1)
		if err != nil {
			t.Fatal(err)
		}
		if reflect.DeepEqual(shared.Logits, again.Logits) {
			t.Fatal("shared target head was not consumed")
		}
		if _, err := shard.Reference("mtp.2.head.weight"); err == nil {
			t.Fatal("draft exposed a separate target-owned head")
		}
		if !errors.Is(f.cfg.RefuseDeepSeekV41DSpark(), ErrV41DSparkUnsupported) {
			t.Fatal("reference enabled serving admission")
		}
	})
	t.Run("actual packed row formats and row scale geometry", func(t *testing.T) {
		f, _, _ := v41DSparkReferenceFixture(t, func(ts map[string]tinySTTensor) {
			name := "mtp.0.ffn.experts.127.w1.weight"
			w := ts[name]
			w.data[0] = 0x92
			ts[name] = w
			s := ts[strings.TrimSuffix(name, ".weight")+".scale"]
			s.data[0] = 128
			ts[strings.TrimSuffix(name, ".weight")+".scale"] = s
			name = "mtp.0.main_proj.weight"
			w = ts[name]
			for i := range w.data {
				w.data[i] = 0x38
			}
			ts[name] = w
			s = ts["mtp.0.main_proj.scale"]
			for i := range s.data {
				s.data[i] = 127
				if i >= s.shape[1] {
					s.data[i] = 128
				}
			}
			ts["mtp.0.main_proj.scale"] = s
		})
		got, err := f.w.row(ctx, "mtp.0.ffn.experts.127.w1.weight", 0)
		if err != nil || got[0] != 2 || got[1] != -1 {
			t.Fatalf("MXFP4 low/high nibble or group32 scale: %v %v", got, err)
		}
		a, err := f.w.row(ctx, "mtp.0.main_proj.weight", 0)
		if err != nil {
			t.Fatal(err)
		}
		b, err := f.w.row(ctx, "mtp.0.main_proj.weight", 40)
		if err != nil {
			t.Fatal(err)
		}
		if a[0] != 1 || b[0] != 2 {
			t.Fatal("FP8 weight scale is not per32x32 tile")
		}
		m, err := f.w.row(ctx, "mtp.2.markov_head.embed.weight", 2)
		if err != nil || !reflect.DeepEqual(m, []float32{0, 0, 1, 0}) {
			t.Fatalf("BF16 Markov row: %v %v", m, err)
		}
		for _, bad := range []int{-1, 128} {
			if _, err := f.w.row(ctx, "mtp.0.ffn.experts.127.w1.weight", bad); err == nil {
				t.Fatal("invalid row admitted")
			}
		}
	})
	t.Run("partial payload scale and cancellation publish no row", func(t *testing.T) {
		f, shard, _ := v41DSparkReferenceFixture(t, nil)
		name := "mtp.0.ffn.experts.127.w1.weight"
		ref, err := shard.Reference(name)
		if err != nil {
			t.Fatal(err)
		}
		original := shard.packed.file.r
		injected := errors.New("partial draft row read")
		for _, start := range []int64{ref.Start, ref.ScaleStart} {
			for _, failWithError := range []bool{false, true} {
				shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
					if off == start {
						n, err := original.ReadAt(p[:len(p)/2], off)
						if err != nil {
							return n, err
						}
						if failWithError {
							return n, injected
						}
						return n, nil
					}
					return original.ReadAt(p, off)
				})
				got, err := f.w.row(ctx, name, 0)
				if got != nil || err == nil || failWithError && !errors.Is(err, injected) {
					t.Fatalf("partial read emitted a row: %v %v", got, err)
				}
			}
		}
		cancelled, cancel := context.WithCancel(ctx)
		defer cancel()
		shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
			n, err := original.ReadAt(p, off)
			if off == ref.ScaleStart {
				cancel()
			}
			return n, err
		})
		if got, err := f.w.row(cancelled, name, 0); got != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled scale read published a row: %v %v", got, err)
		}
		shard.packed.file.r = original
	})
	t.Run("late append and head failures are atomic", func(t *testing.T) {
		f, shard, rows := v41DSparkReferenceFixture(t, nil)
		if err := f.AppendCommitted(ctx, rows[:1]); err != nil {
			t.Fatal(err)
		}
		before := v41DSparkRingCopy(f)
		original := shard.packed.file.r
		ref, err := shard.Reference("mtp.2.attn.wkv.weight")
		if err != nil {
			t.Fatal(err)
		}
		injected := errors.New("injected third-stage context read")
		shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
			if off >= ref.Start && off < ref.End {
				return 0, injected
			}
			return original.ReadAt(p, off)
		})
		if err := f.AppendCommitted(ctx, rows[1:]); !errors.Is(err, injected) {
			t.Fatalf("late append error lost: %v", err)
		}
		if f.next != 1 || !reflect.DeepEqual(before, v41DSparkRingCopy(f)) {
			t.Fatal("late stage committed a partial context")
		}
		shard.packed.file.r = original
		if err := f.AppendCommitted(ctx, rows[1:]); err != nil {
			t.Fatal(err)
		}
		before = v41DSparkRingCopy(f)
		ref, err = shard.Reference("mtp.2.confidence_head.proj.weight")
		if err != nil {
			t.Fatal(err)
		}
		shard.packed.file.r = v41DSparkTestReaderAt(func(p []byte, off int64) (int, error) {
			if off >= ref.Start && off < ref.End {
				return 0, injected
			}
			return original.ReadAt(p, off)
		})
		if got, err := f.Propose(ctx, 1); got != nil || !errors.Is(err, injected) {
			t.Fatalf("late head emitted partial proposal: %v", err)
		}
		if f.next != 2 || !reflect.DeepEqual(before, v41DSparkRingCopy(f)) {
			t.Fatal("failed proposal mutated context")
		}
		shard.packed.file.r = original
		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		if got, err := f.Propose(cancelled, 1); got != nil || !errors.Is(err, context.Canceled) {
			t.Fatal("cancelled proposal ran")
		}
	})
	t.Run("prefix lineage restore and uncommitted rows refuse", func(t *testing.T) {
		f, _, rows := v41DSparkReferenceFixture(t, nil)
		if err := f.AppendCommitted(ctx, rows); err != nil {
			t.Fatal(err)
		}
		f.state.history[0] = 6
		if _, err := f.Propose(ctx, 1); err == nil {
			t.Fatal("same-length changed prefix admitted")
		}
		f.state.history[0] = 1
		old := f.target.v41Forward
		f.target.v41Forward = &v41ForwardState{history: append([]int(nil), old.history...)}
		if _, err := f.Propose(ctx, 1); err == nil {
			t.Fatal("restored target state admitted")
		}
		f.target.v41Forward = old
		old.activeStepRound = &v41StepRound{}
		if _, err := f.Propose(ctx, 1); err == nil {
			t.Fatal("speculative target state admitted")
		}
		old.activeStepRound = nil
		old.history = append(old.history, 3)
		if _, err := f.Propose(ctx, 1); err == nil {
			t.Fatal("unsynchronized target context admitted")
		}
		row := v41DSparkReferenceRows(f.cfg, []int{3}, 2)[0]
		row.Token = 4
		if err := f.AppendCommitted(ctx, []v41DSparkContextRow{row}); err == nil {
			t.Fatal("wrong-token feature admitted")
		}
		row.Token = 3
		row.Layers[0] = 1
		if err := f.AppendCommitted(ctx, []v41DSparkContextRow{row}); err == nil {
			t.Fatal("wrong tap admitted")
		}
	})
	t.Run("mean streams ownership and finite checks", func(t *testing.T) {
		streams := [][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
		got, err := v41DSparkMeanStreams(streams, 2)
		if err != nil || !reflect.DeepEqual(got, []float32{4, 5}) {
			t.Fatalf("mHC mean: %v %v", got, err)
		}
		streams[0][0] = 100
		if got[0] != 4 {
			t.Fatal("target feature aliases a stream")
		}
		streams[0][0] = float32(math.NaN())
		if _, err := v41DSparkMeanStreams(streams, 2); err == nil {
			t.Fatal("nonfinite target feature admitted")
		}
	})
	t.Run("original activation FP8 ties and scale", func(t *testing.T) {
		x := make([]float32, 32)
		x[0], x[1], x[2], x[3] = 448, 1.0625, 1.1875, -1.0625
		q, err := v41DSparkActFP8(x)
		if err != nil || q[0] != 448 || q[1] != 1 || q[2] != 1.25 || q[3] != -1 {
			t.Fatalf("FP8 ties-to-even: %v %v", q, err)
		}
		x[0] = 449
		q, err = v41DSparkActFP8(x)
		if err != nil || q[0] != 448 {
			t.Fatalf("E8M0 scale round-up: %v %v", q, err)
		}
		if _, err := v41DSparkActFP8(x[:31]); err == nil {
			t.Fatal("non-group32 activation admitted")
		}
	})
}

// fak-test:runtime medium est=2s lane=default
func TestV41DSparkPreMixLagAndRingWrap(t *testing.T) {
	ctx := context.Background()
	// If the first FFN emits zero, changing only the first attention's PRE
	// coefficients cannot affect its attention or any later stage. The PRE
	// belongs to that FFN input, never the same attention input.
	makeArm := func(changePre, zeroFFN bool) (*v41DSparkReference, []v41DSparkContextRow) {
		f, _, rows := v41DSparkReferenceFixture(t, func(ts map[string]tinySTTensor) {
			if changePre {
				w := ts["mtp.0.hc_attn_base"]
				for i := 0; i < 4; i++ {
					binary.LittleEndian.PutUint32(w.data[i*4:], math.Float32bits(float32(2+i)))
				}
				ts["mtp.0.hc_attn_base"] = w
			}
			if zeroFFN {
				for name, w := range ts {
					if strings.HasPrefix(name, "mtp.0.ffn.") && strings.HasSuffix(name, ".w2.weight") {
						clear(w.data)
						ts[name] = w
					}
				}
			}
		})
		return f, rows
	}
	a, ar := makeArm(false, true)
	b, br := makeArm(true, true)
	if err := a.AppendCommitted(ctx, ar); err != nil {
		t.Fatal(err)
	}
	if err := b.AppendCommitted(ctx, br); err != nil {
		t.Fatal(err)
	}
	pa, err := a.Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	pb, err := b.Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pa.hidden, pb.hidden) {
		t.Fatal("attention consumed its own PRE instead of incoming carry")
	}
	c, cr := makeArm(false, false)
	d, dr := makeArm(true, false)
	if err := c.AppendCommitted(ctx, cr); err != nil {
		t.Fatal(err)
	}
	if err := d.AppendCommitted(ctx, dr); err != nil {
		t.Fatal(err)
	}
	pc, err := c.Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	pd, err := d.Propose(ctx, 1)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(pc.hidden, pd.hidden) {
		t.Fatal("PRE negative control is vacuous")
	}

	// Ring wrap has a fixed128-row capacity independently in each stage.
	ids := make([]int, 130)
	for i := range ids {
		ids[i] = 1 + i%6
	}
	a.state.history = ids
	a.next = 0
	a.prefix = v41DSparkPrefix(nil)
	a.ring = [3][v41WindowSize][]float32{}
	all := v41DSparkReferenceRows(a.cfg, ids, 0)
	if err := a.AppendCommitted(ctx, all[:128]); err != nil {
		t.Fatal(err)
	}
	first := append([]float32(nil), a.ring[0][0]...)
	if err := a.AppendCommitted(ctx, all[128:]); err != nil {
		t.Fatal(err)
	}
	if a.next != 130 || reflect.DeepEqual(first, a.ring[0][0]) {
		t.Fatal("ring did not replace position zero at128")
	}
	for stage := range a.ring {
		for _, row := range a.ring[stage] {
			if len(row) != a.cfg.HeadDim {
				t.Fatal("ring row lost")
			}
		}
	}
	before := v41DSparkRingCopy(a)
	if _, err := a.Propose(ctx, 1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, v41DSparkRingCopy(a)) {
		t.Fatal("noise KV persisted across wrap")
	}
}

func v41DSparkRingCopy(f *v41DSparkReference) [3][][]float32 {
	var out [3][][]float32
	for stage := range out {
		out[stage] = make([][]float32, v41WindowSize)
		for i, row := range f.ring[stage] {
			out[stage][i] = append([]float32(nil), row...)
		}
	}
	return out
}

func v41DSparkReferenceRows(cfg Config, ids []int, start int) []v41DSparkContextRow {
	rows := make([]v41DSparkContextRow, len(ids))
	for i, id := range ids {
		row := v41DSparkContextRow{Position: start + i, Token: id, Layers: append([]int(nil), cfg.DeepSeekV41.DSpark.TargetLayerIDs...)}
		for tap := range row.Layers {
			values := make([]float32, cfg.HiddenSize)
			for j := range values {
				values[j] = float32(1+((start+i)*(j%7+1)+tap*3+j)%31) / 32
			}
			row.Hidden = append(row.Hidden, values)
		}
		rows[i] = row
	}
	return rows
}

func v41DSparkReferenceFixture(t *testing.T, edit func(map[string]tinySTTensor)) (*v41DSparkReference, *V41DSparkPackedShard, []v41DSparkContextRow) {
	t.Helper()
	cfg := v41TestReducedConfig(t, 3, V41RouterExperts)
	cfg.DeepSeekV41.DSpark.TargetLayerIDs = []int{0, 1, 2}
	cfg.DeepSeekV41.DSpark.NumRoutedExperts = 128
	cfg.DeepSeekV41.DSpark.NumExpertsPerTok = 3
	cfg.DeepSeekV41.DSpark.MarkovRank = 4
	cfg.DeepSeekV41.DSpark.NoiseTokenID = 7
	cfg.DeepSeekV41.CompressRatios = make([]int, cfg.NumLayers+3)
	cfg.NSharedExperts = 1
	cfg.NormTopKProb = true
	cfg.ScoringFunc = "sqrtsoftplus"
	cfg.RoutedScalingFactor = 1.5
	cfg.RopeTheta = 10000
	inv, err := DeepSeekV41DSparkInventory(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ts := make(map[string]tinySTTensor)
	for _, name := range inv.Names() {
		spec, _ := inv.Lookup(name)
		dtype := spec.AllowedDtypes[0]
		n, ok := checkedShapeProduct(spec.Shape...)
		if !ok {
			t.Fatal("fixture size")
		}
		bytesPer := 1
		if dtype == "BF16" {
			bytesPer = 2
		}
		if dtype == "F32" {
			bytesPer = 4
		}
		data := make([]byte, n*bytesPer)
		for i := 0; i < n; i++ {
			var value float32
			switch {
			case strings.Contains(name, "markov_head.embed"):
				if i%4 == (i/4)%4 {
					value = 1
				}
			case strings.Contains(name, "markov_head.head"):
				row, col := i/4, i%4
				if row < 4 && col == (row+3)%4 {
					value = 10
				}
			case strings.Contains(name, "confidence_head"):
				value = float32(1+i%3) / 64
			case strings.HasSuffix(name, "norm.weight"):
				value = 1
			case strings.HasSuffix(name, "e_score_correction_bias"):
				value = float32(i) / 128
			case strings.HasSuffix(name, "ffn.gate.weight"):
				value = 0
			case strings.HasSuffix(name, "_scale"):
				value = 1
			case strings.HasSuffix(name, "_base"):
				value = float32(i%7-3) / 16
				if i >= 4 && i < 8 {
					value = float32(i-5) * 2
				}
			case strings.HasSuffix(name, "_fn"):
				value = float32(i%9-4) / 4096
			default:
				value = float32(i%11-5) / 256
			}
			switch dtype {
			case "BF16":
				binary.LittleEndian.PutUint16(data[i*2:], uint16(math.Float32bits(v41RoundBF16(value))>>16))
			case "F32":
				binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
			case "I8":
				data[i] = 0x21
			case "F8_E4M3":
				data[i] = 0x20
				if i%5 == 0 {
					data[i] = 0xA0
				}
				if i%17 == 0 {
					data[i] = 0x38
				}
			}
		}
		ts[name] = tinySTTensor{dtype: dtype, shape: append([]int(nil), spec.Shape...), data: data}
		if len(spec.ScaleShape) != 0 {
			count, _ := checkedShapeProduct(spec.ScaleShape...)
			scales := make([]byte, count)
			code := byte(123)
			if dtype == "I8" {
				code = 121
			}
			for i := range scales {
				scales[i] = code
			}
			ts[strings.TrimSuffix(name, ".weight")+".scale"] = tinySTTensor{dtype: "F8_E8M0", shape: append([]int(nil), spec.ScaleShape...), data: scales}
		}
	}
	if edit != nil {
		edit(ts)
	}
	shard, err := OpenV41DSparkPackedShard(writeTinySafetensors(t, ts), cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := shard.Close(); err != nil {
			t.Error(err)
		}
	})
	w, err := newV41DSparkRows(cfg, []*V41DSparkPackedShard{shard})
	if err != nil {
		t.Fatal(err)
	}
	manifest, raw := synthBuildRaw([]synthTensor{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize}},
		{"lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize}},
	}, func(name string, next func() float32) float32 {
		if name == "lm_head.weight" {
			return 0
		}
		return 0.5 + next()/8
	})
	m := &Model{Cfg: cfg, manifest: manifest, raw: raw}
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	s := m.NewSession()
	s.v41Forward = &v41ForwardState{history: []int{1, 2}}
	t.Cleanup(s.Close)
	f, err := newV41DSparkReference(s, w)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := f.Close(); err != nil {
			t.Error(err)
		}
	})
	return f, shard, v41DSparkReferenceRows(cfg, []int{1, 2}, 0)
}
