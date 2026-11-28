package ggufload

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeInk2DraftBaseBackend struct {
	*fakeInk2Budget
	compute.Inkling2VerifyDevice
	q8Uploads int
	qkF32     int
	ringF32   int
	attendF32 int
	fuses     int
	silus     int
	topms     int
	mkIns     int
	mkGates   int
	mkPicks   int
	nextTok   int32
}

func newFakeInk2DraftBaseBackend(capBytes int64) *fakeInk2DraftBaseBackend {
	return &fakeInk2DraftBaseBackend{
		fakeInk2Budget: newFakeInk2Budget(capBytes),
		nextTok:        7,
	}
}

func (b *fakeInk2DraftBaseBackend) Upload(t compute.Tensor, _ compute.Dtype) compute.Tensor {
	b.q8Uploads++
	return t
}

func (b *fakeInk2DraftBaseBackend) Ink2Add(a, c, out compute.Ink2Ref, n int) error {
	aa, cc, oo := b.rows(a), b.rows(c), b.rows(out)
	for i := 0; i < n; i++ {
		oo[i] = aa[i] + cc[i]
	}
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2ReadInts(src compute.Ink2Ref, n int) ([]int32, error) {
	out := make([]int32, n)
	for i, v := range b.rows(src)[:n] {
		out[i] = int32(math.Float32bits(v))
	}
	return out, nil
}

func (b *fakeInk2DraftBaseBackend) Ink2SoftcapArgmax(_ compute.Ink2Ref, _ int, _ float32, outIdx compute.Ink2Ref) error {
	b.rows(outIdx)[0] = math.Float32frombits(uint32(b.nextTok))
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2VRows(_ compute.Tensor, _, _ compute.Ink2Ref, _ int) (bool, error) {
	return false, nil
}
func (b *fakeInk2DraftBaseBackend) Ink2VQ8Rows(_ compute.Tensor, _, _, _, _ compute.Ink2Ref, _ int, _ bool) (bool, error) {
	return false, nil
}
func (b *fakeInk2DraftBaseBackend) Ink2VQ8Segs(_ []compute.Tensor, _, _ []compute.Ink2Ref, _ compute.Ink2Ref, _ int) (bool, error) {
	return false, nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DFuse(_ []compute.Ink2Ref, _ []float32, _, _ compute.Ink2Ref, _, _ int, _ float32) error {
	b.fuses++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DQK(_, _, dst, posDev compute.Ink2Ref, rows, heads, dim, cap int, ring bool, _ float32, _ float64) error {
	b.qkF32++
	if ring {
		pos := int(int32(math.Float32bits(b.rows(posDev)[0])))
		need := ((pos+rows-1)%cap + 1) * heads * dim
		if cap*heads*dim < need {
			return errors.New("fake qk ring bounds")
		}
		_ = dst
	}
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DRingStore(_, _ compute.Ink2Ref, _ compute.Ink2Ref, _, _, _ int) error {
	b.ringF32++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DAttend(_, _, _, _, _, _ compute.Ink2Ref, _ compute.Ink2Ref, _, _, _, _, _, _ int) error {
	b.attendF32++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DSiluMul(_, _, _ compute.Ink2Ref, _ int) error {
	b.silus++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DTopM(_ compute.Ink2Ref, _, _, _, _ int, _, _, _ compute.Ink2Ref) error {
	b.topms++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DMarkovIn(_, _, _, _ compute.Ink2Ref, _ int, _, _, _ compute.Ink2Ref, _, _, _ int, _ compute.Ink2Ref) error {
	b.mkIns++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DMarkovGate(_, _, _, _ compute.Ink2Ref, _, _ int) error {
	b.mkGates++
	return nil
}

func (b *fakeInk2DraftBaseBackend) Ink2DMarkovPick(_ compute.Tensor, _, _ compute.Ink2Ref, _ int, _, _, idx compute.Ink2Ref, step int) error {
	b.mkPicks++
	out := b.rows(idx)
	out[4*step] = math.Float32frombits(uint32(b.nextTok + int32(step)))
	out[4*step+1] = inkling2SpecMinMargin + 1.0
	return nil
}

type fakeInk2DraftFP8Backend struct {
	*fakeInk2DraftBaseBackend
	qkFP8         int
	ringFP8       int
	attendFP8     int
	lastAttendKC  compute.Ink2Ref
	lastAttendVC  compute.Ink2Ref
	lastAttendKS  compute.Ink2Ref
	lastAttendVS  compute.Ink2Ref
	forceOverflow int32
}

func (b *fakeInk2DraftFP8Backend) Ink2DQKFP8(_, _, dst, sc, overflow, _ compute.Ink2Ref, _, heads, dim, cap int, _ bool, _ float32, _ float64) error {
	b.qkFP8++
	wantCodes := compute.Ink2FP8KVCodeFloats(cap, heads, dim)
	wantScales := compute.Ink2FP8KVScaleFloats(cap, heads)
	if len(b.rows(dst)) < wantCodes || len(b.rows(sc)) < wantScales {
		return errors.New("fake qk fp8 backing too small")
	}
	if b.forceOverflow != 0 && overflow.T.Buf() != nil {
		b.rows(overflow)[0] = math.Float32frombits(uint32(b.forceOverflow))
	}
	return nil
}

func (b *fakeInk2DraftFP8Backend) Ink2DRingStoreFP8(_, ring, sc, overflow, _ compute.Ink2Ref, _, kvHeads, dim, cap int) error {
	b.ringFP8++
	wantCodes := compute.Ink2FP8KVCodeFloats(cap, kvHeads, dim)
	wantScales := compute.Ink2FP8KVScaleFloats(cap, kvHeads)
	if len(b.rows(ring)) < wantCodes || len(b.rows(sc)) < wantScales {
		return errors.New("fake ring store fp8 backing too small")
	}
	if b.forceOverflow != 0 && overflow.T.Buf() != nil {
		b.rows(overflow)[0] = math.Float32frombits(uint32(b.forceOverflow))
	}
	return nil
}

func (b *fakeInk2DraftFP8Backend) Ink2DAttendFP8(_, _, _, kc, vc, ks, vs, _, _ compute.Ink2Ref, _, _, kvHeads, dim, _, cap int) error {
	b.attendFP8++
	b.lastAttendKC, b.lastAttendVC = kc, vc
	b.lastAttendKS, b.lastAttendVS = ks, vs
	wantCodes := compute.Ink2FP8KVCodeFloats(cap, kvHeads, dim)
	wantScales := compute.Ink2FP8KVScaleFloats(cap, kvHeads)
	if len(b.rows(kc)) < wantCodes || len(b.rows(vc)) < wantCodes || len(b.rows(ks)) < wantScales || len(b.rows(vs)) < wantScales {
		return errors.New("fake attend fp8 backing too small")
	}
	return nil
}

func newTestInkling2DSparkModel() *Inkling2DSparkModel {
	hp := model.Inkling2DSparkHParams{
		Hidden: 64, Layers: 3, Heads: 4, KVHeads: 2, HeadDim: 32, Intermediate: 128,
		Block: 4, MarkovRank: 32, SlidingWindow: 16, VocabUnpadded: 32, Taps: 2,
		LocalLayer: []bool{true, false, true}, NormEps: 1e-6, RopeTheta: 10000,
	}
	bf16Mat := func(rows, cols int) *inkling2BF16Matrix {
		raw := make([]byte, 2*rows*cols)
		for i := 0; i < rows*cols; i++ {
			binary.LittleEndian.PutUint16(raw[2*i:], 0x3F80)
		}
		return &inkling2BF16Matrix{rows: rows, cols: cols, raw: raw}
	}
	ones := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = 1
		}
		return v
	}
	layers := make([]model.Inkling2DSparkLayerWeights, hp.Layers)
	fuse := make([][]float32, hp.Layers+1)
	qDim := hp.Heads * hp.HeadDim
	kvDim := hp.KVHeads * hp.HeadDim
	for l := range layers {
		layers[l] = model.Inkling2DSparkLayerWeights{
			InputNorm: ones(hp.Hidden), PostAttnNorm: ones(hp.Hidden), DFlareNorm: ones(hp.Hidden),
			QNorm: ones(hp.HeadDim), KNorm: ones(hp.HeadDim), KNormCtx: ones(hp.HeadDim),
			Wq: bf16Mat(qDim, hp.Hidden), Wk: bf16Mat(kvDim, hp.Hidden), Wv: bf16Mat(kvDim, hp.Hidden),
			Wo: bf16Mat(hp.Hidden, qDim), WkCtx: bf16Mat(kvDim, hp.Hidden), WvCtx: bf16Mat(kvDim, hp.Hidden),
			Gate: bf16Mat(hp.Intermediate, hp.Hidden), Up: bf16Mat(hp.Intermediate, hp.Hidden),
			Down: bf16Mat(hp.Hidden, hp.Intermediate),
		}
		fuse[l] = ones(hp.Taps)
	}
	fuse[hp.Layers] = ones(hp.Taps)
	w := &model.Inkling2DSparkWeights{
		Layers:        layers,
		Norm:          ones(hp.Hidden),
		MaskEmbed:     ones(hp.Hidden),
		FuseWeight:    fuse,
		MarkovW1:      inkling2BF16Embedding{m: bf16Mat(hp.VocabUnpadded, hp.MarkovRank)},
		RnnProj:       bf16Mat(3*hp.MarkovRank, 2*hp.MarkovRank+2*hp.Hidden),
		RnnBias:       ones(3 * hp.MarkovRank),
		HiddenOut:     bf16Mat(hp.Hidden, hp.MarkovRank),
		HiddenOutBias: ones(hp.Hidden),
		MarkovNorm:    ones(hp.Hidden),
	}
	return &Inkling2DSparkModel{HP: hp, Weights: w, TargetVisits: []int{2, 6}}
}

func newTestDSparkResident(be compute.Backend, hp model.Inkling2DSparkHParams, ctx, chunk int) (*inkling2Resident, *fakeInk2Budget) {
	var budget *fakeInk2Budget
	switch b := be.(type) {
	case *fakeInk2DraftBaseBackend:
		budget = b.fakeInk2Budget
	case *fakeInk2DraftFP8Backend:
		budget = b.fakeInk2Budget
	}
	dev := be.(compute.Inkling2Device)
	ones := make([]float32, hp.Hidden)
	for i := range ones {
		ones[i] = 1
	}
	s := &inkling2Resident{
		hp:       model.Inkling2LoopHParams{Hidden: hp.Hidden},
		w:        &model.Inkling2LoopWeights{TokEmbd: fakeInk2Embedding{}, TokNorm: ones},
		be:       be,
		dev:      dev,
		ctx:      ctx,
		chunkMax: chunk,
		output: ink2Mat{
			t:    compute.Tensor{Dtype: compute.Q5_K, Shape: []int{hp.VocabUnpadded, hp.Hidden}},
			rows: hp.VocabUnpadded,
			cols: hp.Hidden,
		},
	}
	s.verifyPos = s.alloc(1)
	return s, budget
}

func TestInkling2DSparkStateBytesFP8CompressionAndBudget(t *testing.T) {
	mini := inkling2MiniKVShape()
	mini.Hidden = 2304
	mini.ExpertsUsed = 8
	mini.ExpertWidth = 768

	const (
		ctx128k             = 131072
		chunk               = 128
		dsparkQ8WeightBytes = 278 << 20
		outputVocabRows     = 151936
	)
	draftHP3 := model.Inkling2DSparkHParams{
		Hidden: 2304, Layers: 3, Heads: 16, KVHeads: 8, HeadDim: 128,
		Intermediate: 6144, Block: 8, MarkovRank: 256, SlidingWindow: 1024,
		VocabUnpadded: outputVocabRows, Taps: 4, LocalLayer: []bool{true, false, true},
	}

	kvRingBytes := func(hp model.Inkling2DSparkHParams, fp8 bool) int64 {
		kvDim := hp.KVHeads * hp.HeadDim
		var floats int64
		for l := 0; l < hp.Layers; l++ {
			local := l < len(hp.LocalLayer) && hp.LocalLayer[l]
			cap, _ := inkling2DraftLayerRing(ctx128k, chunk, hp.SlidingWindow, local)
			if fp8 {
				floats += 2 * (int64(compute.Ink2FP8KVCodeFloats(cap, hp.KVHeads, hp.HeadDim)) +
					int64(compute.Ink2FP8KVScaleFloats(cap, hp.KVHeads)))
			} else {
				floats += 2 * int64(cap) * int64(kvDim)
			}
		}
		return 4 * floats
	}

	ringF32 := kvRingBytes(draftHP3, false)
	ringFP8 := kvRingBytes(draftHP3, true)
	if ratio := float64(ringF32) / float64(ringFP8); ratio < 3.8 {
		t.Fatalf("DSpark L=3 KV ring compression ratio = %.3fx, want >= 3.8x (f32=%d, fp8=%d)", ratio, ringF32, ringFP8)
	}

	stateM32FP32 := inkling2DSparkStateBytesFor(draftHP3, ctx128k, chunk, outputVocabRows, 32, false)
	stateM32FP8 := inkling2DSparkStateBytesFor(draftHP3, ctx128k, chunk, outputVocabRows, 32, true)
	stateM0FP8 := inkling2DSparkStateBytesFor(draftHP3, ctx128k, chunk, outputVocabRows, 0, true)

	if wantDiff := ringF32 - ringFP8 - 4; stateM32FP32-stateM32FP8 != wantDiff {
		t.Fatalf("state delta FP32-FP8 = %d, want %d", stateM32FP32-stateM32FP8, wantDiff)
	}
	if stateM0FP8 >= stateM32FP8 {
		t.Fatalf("shortlist=0 FP8 state (%d) should be strictly smaller than shortlist=32 (%d)", stateM0FP8, stateM32FP8)
	}

	targetNVFP4KV := inkling2ResidentKVBytesFor(mini, ctx128k, true, chunk, Inkling2KVNVFP4)
	targetOther := inkling2ResidentOtherBytes(mini, inkling2ArenaFloats, chunk, 0)
	const rtx5080FreeAfterWeights = int64(245) * (1 << 30) / 100
	for _, tc := range []struct {
		name      string
		state     int64
		shortlist int
	}{
		{"L3_shortlist32_fp8", stateM32FP8, 32},
		{"L3_shortlist0_fp8", stateM0FP8, 0},
	} {
		total := targetNVFP4KV + targetOther + dsparkQ8WeightBytes + tc.state
		t.Logf("%s: kv_rings=%.2f MiB->%.2f MiB, dspark_state=%.2f MiB, total_post_weights=%.3f GiB / %.2f GiB",
			tc.name, float64(ringF32)/(1<<20), float64(ringFP8)/(1<<20),
			float64(tc.state)/(1<<20), float64(total)/(1<<30), float64(rtx5080FreeAfterWeights)/(1<<30))
		if total > rtx5080FreeAfterWeights {
			t.Fatalf("%s total %.3f GiB exceeds 2.45 GiB headroom", tc.name, float64(total)/(1<<30))
		}
	}
}

func TestInkling2DSparkAttachDraftFP8RoutingAndOverflow(t *testing.T) {
	dm := newTestInkling2DSparkModel()
	const (
		ctxTok = 64
		chunk  = 8
	)

	for _, tc := range []struct {
		name      string
		useFP8    bool
		shortlist string
		wantSL    int
	}{
		{name: "fp32_fallback_shortlist32", useFP8: false, shortlist: "32", wantSL: 32},
		{name: "fp8_default_shortlist32", useFP8: true, shortlist: "32", wantSL: 32},
		{name: "fp8_shortlist0", useFP8: true, shortlist: "0", wantSL: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_INKLING2_DSPARK_SHORTLIST", tc.shortlist)
			base := newFakeInk2DraftBaseBackend(1 << 28)
			var be compute.Backend = base
			var fp8be *fakeInk2DraftFP8Backend
			if tc.useFP8 {
				fp8be = &fakeInk2DraftFP8Backend{fakeInk2DraftBaseBackend: base}
				be = fp8be
			}
			s, budget := newTestDSparkResident(be, dm.HP, ctxTok, chunk)
			preDraftLive := 4 * int64(budget.live)

			d, err := s.attachDraft(dm)
			if err != nil {
				t.Fatalf("attachDraft: %v", err)
			}
			if d.fp8KV != tc.useFP8 {
				t.Fatalf("d.fp8KV = %v, want %v", d.fp8KV, tc.useFP8)
			}
			if d.shortlist != tc.wantSL {
				t.Fatalf("d.shortlist = %d, want %d", d.shortlist, tc.wantSL)
			}
			if tc.wantSL == 0 && (d.frnn.T.Buf() != nil || d.w1.T.Buf() != nil || d.state.T.Buf() != nil || d.cand.T.Buf() != nil) {
				t.Fatal("shortlist=0 allocated Markov fRNN/state buffers")
			}

			uploadFloats := int64(dm.HP.Layers*(3*dm.HP.Hidden+3*dm.HP.HeadDim) + dm.HP.Hidden)
			if tc.wantSL > 0 {
				uploadFloats += int64(dm.HP.VocabUnpadded*dm.HP.MarkovRank/2 + 3*dm.HP.MarkovRank + 2*dm.HP.Hidden)
			}
			wantState := inkling2DSparkStateBytesFor(dm.HP, ctxTok, chunk, dm.HP.VocabUnpadded, tc.wantSL, tc.useFP8)
			gotAllocDelta := 4*int64(budget.live) - preDraftLive
			if wantAllocDelta := wantState + 4*uploadFloats; gotAllocDelta != wantAllocDelta {
				t.Fatalf("attachDraft allocated %d state+upload bytes, want %d (state=%d, upload=%d)",
					gotAllocDelta, wantAllocDelta, wantState, 4*uploadFloats)
			}

			for l, layer := range d.layers {
				if layer.kc.T.Buf() == nil || layer.vc.T.Buf() == nil {
					t.Fatalf("layer %d missing kc/vc ring", l)
				}
				if tc.useFP8 {
					if layer.ks.T.Buf() == nil || layer.vs.T.Buf() == nil {
						t.Fatalf("layer %d missing ks/vs scale ring", l)
					}
					wantCodeLen := compute.Ink2FP8KVCodeFloats(layer.cap, dm.HP.KVHeads, dm.HP.HeadDim)
					wantScaleLen := compute.Ink2FP8KVScaleFloats(layer.cap, dm.HP.KVHeads)
					if got := len(base.rows(layer.kc)); got != wantCodeLen {
						t.Fatalf("layer %d kc len=%d, want %d", l, got, wantCodeLen)
					}
					if got := len(base.rows(layer.ks)); got != wantScaleLen {
						t.Fatalf("layer %d ks len=%d, want %d", l, got, wantScaleLen)
					}
				} else if layer.ks.T.Buf() != nil || layer.vs.T.Buf() != nil {
					t.Fatalf("layer %d unexpectedly allocated FP8 scale buffers", l)
				}
			}

			scr := s.alloc(16384)
			p := &ink2Prefill{
				s:         s,
				pd:        be.(compute.Inkling2PrefillDevice),
				T:         2,
				scr:       scr,
				scrFloats: 16384,
			}
			taps := make([]ink2Ref, dm.HP.Taps)
			for i := range taps {
				taps[i] = s.alloc(p.T * dm.HP.Hidden)
			}
			d.commitTaps(p, taps)
			if s.err != nil {
				t.Fatalf("commitTaps: %v", s.err)
			}
			props, err := d.propose(3, 2, dm.HP.Block-1)
			if err != nil {
				t.Fatalf("propose: %v", err)
			}
			if len(props) != dm.HP.Block-1 {
				t.Fatalf("len(props) = %d, want %d", len(props), dm.HP.Block-1)
			}

			if tc.useFP8 {
				if fp8be.qkFP8 != dm.HP.Layers || fp8be.ringFP8 != dm.HP.Layers || fp8be.attendFP8 != dm.HP.Layers {
					t.Fatalf("FP8 calls (qk=%d, ring=%d, attend=%d), want all == %d",
						fp8be.qkFP8, fp8be.ringFP8, fp8be.attendFP8, dm.HP.Layers)
				}
				wantRingF32 := 0
				if tc.wantSL > 0 {
					wantRingF32 = 1 // Markov fRNN feature ring stays FP32
				}
				// q/k intra-block RoPE uses FP32 Ink2DQK(ring=false) = 2*Layers, while context Attend uses Ink2DAttendFP8.
				if base.qkF32 != 2*dm.HP.Layers || base.ringF32 != wantRingF32 || base.attendF32 != 0 {
					t.Fatalf("FP32 calls during FP8 run (qk=%d, ring=%d, attend=%d), want (%d, %d, 0)",
						base.qkF32, base.ringF32, base.attendF32, 2*dm.HP.Layers, wantRingF32)
				}
				if fp8be.lastAttendKS.T.Buf() != d.layers[dm.HP.Layers-1].ks.T.Buf() || fp8be.lastAttendVS.T.Buf() != d.layers[dm.HP.Layers-1].vs.T.Buf() {
					t.Fatal("Ink2DAttendFP8 did not receive the layer ks/vs scale buffers")
				}

				// Inject overflow during next commitTaps and verify propose refuses cleanly.
				fp8be.forceOverflow = 1
				d.commitTaps(p, taps)
				if _, err := d.propose(4, 4, dm.HP.Block-1); !errors.Is(err, ErrInkling2DSparkFP8Overflow) {
					t.Fatalf("overflow propose err = %v, want ErrInkling2DSparkFP8Overflow", err)
				}
				if fp8be.lastAttendKC.T.Buf() == nil || fp8be.lastAttendVC.T.Buf() == nil {
					t.Fatal("Ink2DAttendFP8 missing kc/vc ring buffers")
				}
			} else {
				if base.qkF32 != 3*dm.HP.Layers || base.attendF32 != dm.HP.Layers {
					t.Fatalf("FP32 calls (qk=%d, attend=%d), want (%d, %d)",
						base.qkF32, base.attendF32, 3*dm.HP.Layers, dm.HP.Layers)
				}
			}
			s.close()
			if budget.live != 0 || budget.frees != budget.allocs {
				t.Fatalf("leak after close: live=%d allocs=%d frees=%d", budget.live, budget.allocs, budget.frees)
			}
		})
	}
}

func BenchmarkInkling2DSparkStateBytes(b *testing.B) {
	hp := model.Inkling2DSparkHParams{
		Hidden: 2304, Layers: 3, Heads: 16, KVHeads: 8, HeadDim: 128,
		Intermediate: 6144, Block: 8, MarkovRank: 256, SlidingWindow: 1024,
		VocabUnpadded: 151936, Taps: 4, LocalLayer: []bool{true, false, true},
	}
	var sink int64
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink += inkling2DSparkStateBytesFor(hp, 131072, 128, 151936, 0, true)
	}
	_ = sink
}
