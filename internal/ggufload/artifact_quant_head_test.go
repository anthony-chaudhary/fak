package ggufload

import (
	"bytes"
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func almostEqualFloat(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

// fak-test:runtime fast est=1s
func TestAnalyzeOutputHeadQuantTiedAndUntied(t *testing.T) {
	t.Parallel()

	t.Run("untied output.weight", func(t *testing.T) {
		t.Parallel()
		tensors := []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{4096, 1024}, Type: TensorF16},
			{Name: "blk.0.attn_q.weight", Dims: []uint64{1024, 1024}, Type: TensorQ4_K},
			{Name: "output.weight", Dims: []uint64{4096, 1024}, Type: TensorQ6_K},
		}
		got, err := AnalyzeOutputHeadQuant(model.Config{TieWordEmbeddings: false}, tensors)
		if err != nil {
			t.Fatalf("AnalyzeOutputHeadQuant: %v", err)
		}
		wantElems := uint64(4096 * 1024)
		wantBytes := (wantElems / qkK) * blockQ6KBytes
		if got.TensorName != "output.weight" || got.Tied {
			t.Fatalf("resolved = (%q, tied=%v), want (output.weight, tied=false)", got.TensorName, got.Tied)
		}
		if got.Type != TensorQ6_K || got.TypeName != "Q6_K" || !got.IsQuantized {
			t.Fatalf("type/quant = (%v, %q, %v), want (Q6_K, Q6_K, true)", got.Type, got.TypeName, got.IsQuantized)
		}
		if got.Elements != wantElems || got.PayloadBytes != wantBytes {
			t.Fatalf("size = (%d elems, %d bytes), want (%d, %d)", got.Elements, got.PayloadBytes, wantElems, wantBytes)
		}
		if !almostEqualFloat(got.BitsPerWeight, 6.5625, 1e-9) {
			t.Fatalf("BitsPerWeight = %v, want 6.5625", got.BitsPerWeight)
		}
	})

	t.Run("untied lm_head.weight unquantized BF16", func(t *testing.T) {
		t.Parallel()
		tensors := []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{2048, 512}, Type: TensorQ8_0},
			{Name: "lm_head.weight", Dims: []uint64{2048, 512}, Type: TensorBF16},
		}
		got, err := AnalyzeOutputHeadQuant(model.Config{TieWordEmbeddings: false}, tensors)
		if err != nil {
			t.Fatalf("AnalyzeOutputHeadQuant: %v", err)
		}
		if got.TensorName != "lm_head.weight" || got.Tied || got.IsQuantized {
			t.Fatalf("got %+v, want untied unquantized lm_head.weight", got)
		}
		if got.Type != TensorBF16 || got.PayloadBytes != 2048*512*2 || !almostEqualFloat(got.BitsPerWeight, 16.0, 1e-9) {
			t.Fatalf("got %+v, want BF16 16.0 bpw", got)
		}
	})

	t.Run("tied token_embd.weight", func(t *testing.T) {
		t.Parallel()
		tensors := []TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{4096, 1024}, Type: TensorQ4_K},
			{Name: "blk.0.ffn_gate.weight", Dims: []uint64{1024, 1024}, Type: TensorQ4_K},
			{Name: "output_norm.weight", Dims: []uint64{1024}, Type: TensorF32},
		}
		got, err := AnalyzeOutputHeadQuant(model.Config{TieWordEmbeddings: true}, tensors)
		if err != nil {
			t.Fatalf("AnalyzeOutputHeadQuant: %v", err)
		}
		wantElems := uint64(4096 * 1024)
		wantBytes := (wantElems / qkK) * blockQ4KBytes
		if got.TensorName != "token_embd.weight" || !got.Tied || !got.IsQuantized {
			t.Fatalf("got %+v, want tied quantized token_embd.weight", got)
		}
		if got.PayloadBytes != wantBytes || !almostEqualFloat(got.BitsPerWeight, 4.5, 1e-9) {
			t.Fatalf("payload/bpw = (%d, %v), want (%d, 4.5)", got.PayloadBytes, got.BitsPerWeight, wantBytes)
		}
	})

	t.Run("fail closed on invalid head inventories", func(t *testing.T) {
		t.Parallel()
		cases := []struct {
			name    string
			cfg     model.Config
			tensors []TensorInfo
		}{
			{name: "empty inventory", cfg: model.Config{}, tensors: nil},
			{
				name: "untied missing output head",
				cfg:  model.Config{TieWordEmbeddings: false},
				tensors: []TensorInfo{
					{Name: "token_embd.weight", Dims: []uint64{256, 256}, Type: TensorF16},
				},
			},
			{
				name: "tied missing token_embd",
				cfg:  model.Config{TieWordEmbeddings: true},
				tensors: []TensorInfo{
					{Name: "blk.0.attn_q.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
				},
			},
			{
				name: "tied conflicts with explicit output.weight",
				cfg:  model.Config{TieWordEmbeddings: true},
				tensors: []TensorInfo{
					{Name: "token_embd.weight", Dims: []uint64{256, 256}, Type: TensorF16},
					{Name: "output.weight", Dims: []uint64{256, 256}, Type: TensorF16},
				},
			},
			{
				name: "duplicate explicit heads",
				cfg:  model.Config{TieWordEmbeddings: false},
				tensors: []TensorInfo{
					{Name: "output.weight", Dims: []uint64{256, 256}, Type: TensorF16},
					{Name: "lm_head.weight", Dims: []uint64{256, 256}, Type: TensorF16},
				},
			},
			{
				name: "zero dimension head",
				cfg:  model.Config{TieWordEmbeddings: false},
				tensors: []TensorInfo{
					{Name: "output.weight", Dims: []uint64{256, 0}, Type: TensorF16},
				},
			},
			{
				name: "unknown tensor type",
				cfg:  model.Config{TieWordEmbeddings: false},
				tensors: []TensorInfo{
					{Name: "output.weight", Dims: []uint64{256, 256}, Type: TensorType(0xffff)},
				},
			},
		}
		for _, tc := range cases {
			tc := tc
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				if got, err := AnalyzeOutputHeadQuant(tc.cfg, tc.tensors); err == nil {
					t.Fatalf("expected error, got %+v", got)
				}
			})
		}
	})
}

// fak-test:runtime fast est=1s
func TestClassifyModuleSensitivityBreakdown(t *testing.T) {
	t.Parallel()

	tensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{4096, 256}, Type: TensorQ8_0},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
		{Name: "blk.0.ssm_out.weight", Dims: []uint64{256, 256}, Type: TensorQ5_K},
		{Name: "blk.0.attn_norm.weight", Dims: []uint64{256}, Type: TensorF32},
		{Name: "blk.0.ffn_gate.weight", Dims: []uint64{512, 256}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_gate_shexp.weight", Dims: []uint64{512, 256}, Type: TensorQ4_K},
		{Name: "blk.0.ffn_norm.weight", Dims: []uint64{256}, Type: TensorF32},
		{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 512, 4}, Type: TensorQ2_K},
		{Name: "output_norm.weight", Dims: []uint64{256}, Type: TensorF32},
		{Name: "output.weight", Dims: []uint64{4096, 256}, Type: TensorF16},
		{Name: "rope_freqs.weight", Dims: []uint64{128}, Type: TensorF32},
	}

	report, err := ClassifyModuleSensitivity(model.Config{TieWordEmbeddings: false}, tensors)
	if err != nil {
		t.Fatalf("ClassifyModuleSensitivity: %v", err)
	}

	if report.TotalTensors != len(tensors) {
		t.Fatalf("TotalTensors = %d, want %d", report.TotalTensors, len(tensors))
	}
	if len(report.Ordered) != 7 {
		t.Fatalf("len(Ordered) = %d, want all 7 module sensitivity classes", len(report.Ordered))
	}

	// Verify norms were not folded into attention or mlp despite "attn"/"ffn" substrings.
	norm := report.Class(SensitivityNorm)
	if norm.TensorCount != 3 || norm.Elements != 768 || norm.PayloadBytes != 768*4 {
		t.Fatalf("norm stats = %+v, want 3 tensors / 768 elems / 3072 bytes", norm)
	}
	if !almostEqualFloat(norm.UnquantizedShare, 1.0, 1e-9) || !almostEqualFloat(norm.EffectiveBPW, 32.0, 1e-9) {
		t.Fatalf("norm share/bpw = (%v, %v), want (1.0, 32.0)", norm.UnquantizedShare, norm.EffectiveBPW)
	}

	// Verify batched routed expert separated from dense MLP + shared expert.
	routed := report.Class(SensitivityRoutedExpert)
	if routed.TensorCount != 1 || routed.Elements != 256*512*4 {
		t.Fatalf("routed stats = %+v, want 1 tensor / %d elems", routed, 256*512*4)
	}
	mlp := report.Class(SensitivityMLP)
	if mlp.TensorCount != 2 || mlp.Elements != 2*512*256 {
		t.Fatalf("mlp stats = %+v, want 2 tensors (dense + shared expert) / %d elems", mlp, 2*512*256)
	}
	attn := report.Class(SensitivityAttention)
	if attn.TensorCount != 2 || attn.Elements != 2*256*256 {
		t.Fatalf("attention stats = %+v, want 2 tensors (attn_q + ssm_out) / %d elems", attn, 2*256*256)
	}
	other := report.Class(SensitivityOther)
	if other.TensorCount != 1 || other.Elements != 128 {
		t.Fatalf("other stats = %+v, want 1 tensor / 128 elems", other)
	}

	// Output head is F16 while inner layers are quantized: it holds almost all unquantized bytes.
	head := report.Class(SensitivityOutputHead)
	if head.TensorCount != 1 || head.PayloadBytes != 4096*256*2 {
		t.Fatalf("head stats = %+v", head)
	}
	if !almostEqualFloat(head.UnquantizedShare, 1.0, 1e-9) || !almostEqualFloat(head.EffectiveBPW, 16.0, 1e-9) {
		t.Fatalf("head share/bpw = (%v, %v), want (1.0, 16.0)", head.UnquantizedShare, head.EffectiveBPW)
	}
	if head.UnquantizedByteShareOfTotal <= 0.99 {
		t.Fatalf("head UnquantizedByteShareOfTotal = %v, want > 0.99", head.UnquantizedByteShareOfTotal)
	}

	// Verify class sums reconcile exactly with report totals.
	var sumElems, sumBytes, sumUnqBytes uint64
	for _, st := range report.Ordered {
		sumElems += st.Elements
		sumBytes += st.PayloadBytes
		sumUnqBytes += st.UnquantizedBytes
	}
	if sumElems != report.TotalElements || sumBytes != report.TotalPayloadBytes || sumUnqBytes != report.TotalUnquantizedBytes {
		t.Fatalf("sum mismatch: got (%d,%d,%d), want (%d,%d,%d)",
			sumElems, sumBytes, sumUnqBytes, report.TotalElements, report.TotalPayloadBytes, report.TotalUnquantizedBytes)
	}

	// Tied model classifies token_embd.weight as SensitivityOutputHead.
	tiedTensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{4096, 256}, Type: TensorF16},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{256, 256}, Type: TensorQ4_K},
	}
	tiedReport, err := ClassifyModuleSensitivity(model.Config{TieWordEmbeddings: true}, tiedTensors)
	if err != nil {
		t.Fatalf("tied ClassifyModuleSensitivity: %v", err)
	}
	if tiedReport.Class(SensitivityOutputHead).PayloadBytes != 4096*256*2 {
		t.Fatalf("tied output_head payload = %d, want %d", tiedReport.Class(SensitivityOutputHead).PayloadBytes, 4096*256*2)
	}
	if tiedReport.Class(SensitivityEmbedding).TensorCount != 0 {
		t.Fatalf("tied embedding count = %d, want 0", tiedReport.Class(SensitivityEmbedding).TensorCount)
	}
}

// fak-test:runtime fast est=1s
func TestSweepWeightByteFloorMonotonicAndHeadLever(t *testing.T) {
	t.Parallel()

	// Unquantized F16 model with untied output.weight and F32 norms.
	tensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{8192, 512}, Type: TensorF16}, // untied gather: excluded from swept bytes/token
		{Name: "blk.0.attn_q.weight", Dims: []uint64{512, 512}, Type: TensorF16},
		{Name: "blk.0.attn_k.weight", Dims: []uint64{512, 512}, Type: TensorF16},
		{Name: "blk.0.attn_v.weight", Dims: []uint64{512, 512}, Type: TensorF16},
		{Name: "blk.0.attn_output.weight", Dims: []uint64{512, 512}, Type: TensorF16},
		{Name: "blk.0.attn_norm.weight", Dims: []uint64{512}, Type: TensorF32},
		{Name: "blk.0.ffn_gate.weight", Dims: []uint64{1024, 512}, Type: TensorF16},
		{Name: "blk.0.ffn_up.weight", Dims: []uint64{1024, 512}, Type: TensorF16},
		{Name: "blk.0.ffn_down.weight", Dims: []uint64{512, 1024}, Type: TensorF16},
		{Name: "blk.0.ffn_norm.weight", Dims: []uint64{512}, Type: TensorF32},
		{Name: "output_norm.weight", Dims: []uint64{512}, Type: TensorF32},
		{Name: "output.weight", Dims: []uint64{8192, 512}, Type: TensorF16},
	}

	report, err := SweepWeightByteFloor(model.Config{TieWordEmbeddings: false}, tensors, DefaultWeightByteFloorSpec())
	if err != nil {
		t.Fatalf("SweepWeightByteFloor: %v", err)
	}

	wantEmbedGather := uint64(8192 * 512 * 2)
	wantHeadBytes := uint64(8192 * 512 * 2)
	wantDenseBytes := uint64((4*512*512+3*1024*512)*2 + 3*512*4)
	if report.BaselineEmbedGatherBytes != wantEmbedGather {
		t.Fatalf("BaselineEmbedGatherBytes = %d, want %d", report.BaselineEmbedGatherBytes, wantEmbedGather)
	}
	if report.BaselineHeadBytes != wantHeadBytes || report.BaselineDenseBytes != wantDenseBytes {
		t.Fatalf("baseline dense/head = (%d, %d), want (%d, %d)",
			report.BaselineDenseBytes, report.BaselineHeadBytes, wantDenseBytes, wantHeadBytes)
	}
	if report.BaselineBytesPerToken != wantDenseBytes+wantHeadBytes {
		t.Fatalf("BaselineBytesPerToken = %d, want %d", report.BaselineBytesPerToken, wantDenseBytes+wantHeadBytes)
	}

	// Verify explicit HeadQuantLever shows immediate bytes/token reduction before touching dense layers.
	lever := report.HeadLever
	wantQuantHeadG128 := uint64(8192*512/2 + (8192*512/128)*2) // 4.125 bpw
	if lever.QuantizedHeadBytes != wantQuantHeadG128 {
		t.Fatalf("HeadLever.QuantizedHeadBytes = %d, want %d", lever.QuantizedHeadBytes, wantQuantHeadG128)
	}
	if lever.SavedBytes != wantHeadBytes-wantQuantHeadG128 || lever.SavedBytes == 0 {
		t.Fatalf("HeadLever.SavedBytes = %d, want %d", lever.SavedBytes, wantHeadBytes-wantQuantHeadG128)
	}
	if lever.WithHeadQuantBytesPerToken != report.BaselineBytesPerToken-lever.SavedBytes {
		t.Fatalf("WithHeadQuantBytesPerToken = %d, want %d",
			lever.WithHeadQuantBytesPerToken, report.BaselineBytesPerToken-lever.SavedBytes)
	}

	// Verify monotonic decrease across group sizes 32 -> 64 -> 128 for both INT4 and INT8.
	int4Pts := report.PointsForBits(4)
	int8Pts := report.PointsForBits(8)
	if len(int4Pts) != 3 || len(int8Pts) != 3 {
		t.Fatalf("sweep point counts = (int4=%d, int8=%d), want (3, 3)", len(int4Pts), len(int8Pts))
	}
	for i := range int4Pts {
		p4, p8 := int4Pts[i], int8Pts[i]
		if p4.GroupSize != p8.GroupSize {
			t.Fatalf("group size mismatch at %d: %d vs %d", i, p4.GroupSize, p8.GroupSize)
		}
		if p4.BytesPerTokenWithQuantHead >= p4.BytesPerTokenWithBaseHead {
			t.Fatalf("INT4 g=%d: quantized head bytes/tok %d must be strictly less than base head %d",
				p4.GroupSize, p4.BytesPerTokenWithQuantHead, p4.BytesPerTokenWithBaseHead)
		}
		if p8.BytesPerTokenWithQuantHead >= p8.BytesPerTokenWithBaseHead {
			t.Fatalf("INT8 g=%d: quantized head bytes/tok %d must be strictly less than base head %d",
				p8.GroupSize, p8.BytesPerTokenWithQuantHead, p8.BytesPerTokenWithBaseHead)
		}
		if p4.BytesPerTokenWithQuantHead >= p8.BytesPerTokenWithQuantHead {
			t.Fatalf("g=%d: INT4 bytes/tok %d must be strictly less than INT8 %d",
				p4.GroupSize, p4.BytesPerTokenWithQuantHead, p8.BytesPerTokenWithQuantHead)
		}
		if i > 0 {
			if int4Pts[i].BytesPerTokenWithQuantHead >= int4Pts[i-1].BytesPerTokenWithQuantHead {
				t.Fatalf("INT4 floor not strictly decreasing: g=%d (%d) >= g=%d (%d)",
					int4Pts[i].GroupSize, int4Pts[i].BytesPerTokenWithQuantHead,
					int4Pts[i-1].GroupSize, int4Pts[i-1].BytesPerTokenWithQuantHead)
			}
			if int8Pts[i].BytesPerTokenWithQuantHead >= int8Pts[i-1].BytesPerTokenWithQuantHead {
				t.Fatalf("INT8 floor not strictly decreasing: g=%d (%d) >= g=%d (%d)",
					int8Pts[i].GroupSize, int8Pts[i].BytesPerTokenWithQuantHead,
					int8Pts[i-1].GroupSize, int8Pts[i-1].BytesPerTokenWithQuantHead)
			}
		}
		if p4.HeadShareWithoutHeadQuant <= lever.HeadShareOfBaseline {
			t.Fatalf("g=%d: unquantized head share after dense quant (%v) should exceed baseline share (%v)",
				p4.GroupSize, p4.HeadShareWithoutHeadQuant, lever.HeadShareOfBaseline)
		}
	}

	// Tied embedding case: token_embd.weight is NOT subtracted as gather, it IS the swept head.
	tiedTensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{8192, 512}, Type: TensorF16},
		{Name: "blk.0.attn_q.weight", Dims: []uint64{512, 512}, Type: TensorQ4_K},
		{Name: "output_norm.weight", Dims: []uint64{512}, Type: TensorF32},
	}
	tiedReport, err := SweepWeightByteFloor(model.Config{TieWordEmbeddings: true}, tiedTensors, WeightByteFloorSpec{
		BitWidths:          []int{4},
		GroupSizes:         []int{128, 32, 64}, // unsorted input must be normalized ascending
		ScaleBytesPerGroup: 4,                  // e.g. scale + zero-point (4.5 bpw at g=64)
	})
	if err != nil {
		t.Fatalf("tied SweepWeightByteFloor: %v", err)
	}
	if tiedReport.BaselineEmbedGatherBytes != 0 {
		t.Fatalf("tied BaselineEmbedGatherBytes = %d, want 0", tiedReport.BaselineEmbedGatherBytes)
	}
	if !tiedReport.Head.Tied || tiedReport.HeadLever.SavedBytes == 0 {
		t.Fatalf("tied HeadLever = %+v, want Tied=true and SavedBytes > 0", tiedReport.HeadLever)
	}
	if len(tiedReport.Points) != 3 || tiedReport.Points[0].GroupSize != 32 || tiedReport.Points[1].GroupSize != 64 || tiedReport.Points[2].GroupSize != 128 {
		t.Fatalf("sorted group sizes = %+v, want [32, 64, 128]", tiedReport.Points)
	}
	if !almostEqualFloat(tiedReport.Points[1].HeadEffectiveBPW, 4.5, 1e-9) {
		t.Fatalf("g=64 scaleBytes=4 HeadEffectiveBPW = %v, want 4.5", tiedReport.Points[1].HeadEffectiveBPW)
	}

	// MoE active routed-expert scaling: K=2 out of E=8 charges 2/8 of routed expert elements per token.
	moeTensors := []TensorInfo{
		{Name: "token_embd.weight", Dims: []uint64{1024, 256}, Type: TensorF16},
		{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorF16},
		{Name: "output.weight", Dims: []uint64{1024, 256}, Type: TensorF16},
	}
	moeReport, err := SweepWeightByteFloor(model.Config{NumExperts: 8, NumExpertsPerTok: 2}, moeTensors, WeightByteFloorSpec{})
	if err != nil {
		t.Fatalf("MoE SweepWeightByteFloor: %v", err)
	}
	wantActiveRoutedBytes := uint64(256 * 256 * 2 * 2) // 2 active experts * 256*256 * 2 bytes
	if moeReport.BaselineDenseBytes != wantActiveRoutedBytes {
		t.Fatalf("MoE BaselineDenseBytes = %d, want %d (2/8 active experts)", moeReport.BaselineDenseBytes, wantActiveRoutedBytes)
	}
}

// fak-test:runtime fast est=1s
func TestSweepWeightByteFloorFailClosedInputs(t *testing.T) {
	t.Parallel()

	valid := []TensorInfo{
		{Name: "blk.0.attn_q.weight", Dims: []uint64{256, 256}, Type: TensorF16},
		{Name: "output.weight", Dims: []uint64{512, 256}, Type: TensorF16},
	}

	cases := []struct {
		name    string
		cfg     model.Config
		tensors []TensorInfo
		spec    WeightByteFloorSpec
	}{
		{name: "empty tensors", tensors: nil, spec: DefaultWeightByteFloorSpec()},
		{name: "zero bit width", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{0}, GroupSizes: []int{32}}},
		{name: "invalid bit width 3", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{3}, GroupSizes: []int{32}}},
		{name: "explicit empty bit widths", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{}, GroupSizes: []int{32}}},
		{name: "duplicate bit widths", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4, 4}, GroupSizes: []int{32}}},
		{name: "zero group size", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{0}}},
		{name: "negative group size", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{-32}}},
		{name: "non power of two group size", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{24}}},
		{name: "too small group size", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{4}}},
		{name: "explicit empty group sizes", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{}}},
		{name: "duplicate group sizes", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{64, 64}}},
		{name: "negative scale bytes", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{32}, ScaleBytesPerGroup: -1}},
		{name: "invalid head bits", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{32}, HeadBits: 16}},
		{name: "invalid head group size", tensors: valid, spec: WeightByteFloorSpec{BitWidths: []int{4}, GroupSizes: []int{32}, HeadGroupSize: 12}},
		{
			name: "zero dim tensor",
			tensors: []TensorInfo{
				{Name: "blk.0.attn_q.weight", Dims: []uint64{256, 0}, Type: TensorF16},
				{Name: "output.weight", Dims: []uint64{512, 256}, Type: TensorF16},
			},
			spec: DefaultWeightByteFloorSpec(),
		},
		{
			name: "invalid moe active expert count",
			cfg:  model.Config{NumExperts: 8, NumExpertsPerTok: 0},
			tensors: []TensorInfo{
				{Name: "blk.0.ffn_gate_exps.weight", Dims: []uint64{256, 256, 8}, Type: TensorQ4_K},
				{Name: "output.weight", Dims: []uint64{512, 256}, Type: TensorF16},
			},
			spec: DefaultWeightByteFloorSpec(),
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got, err := SweepWeightByteFloor(tc.cfg, tc.tensors, tc.spec); err == nil {
				t.Fatalf("expected error, got %+v", got)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestConfirmQuantizedHeadPath(t *testing.T) {
	t.Parallel()

	q4kHead := OutputHeadQuant{TensorName: "output.weight", Type: TensorQ4_K, TypeName: "Q4_K", IsQuantized: true}
	q6kHead := OutputHeadQuant{TensorName: "output.weight", Type: TensorQ6_K, TypeName: "Q6_K", IsQuantized: true}
	q8Head := OutputHeadQuant{TensorName: "output.weight", Type: TensorQ8_0, TypeName: "Q8_0", IsQuantized: true}
	q2Head := OutputHeadQuant{TensorName: "output.weight", Type: TensorPQ2_0, TypeName: "PQ2_0", IsQuantized: true}
	f16Head := OutputHeadQuant{TensorName: "output.weight", Type: TensorF16, TypeName: "F16", IsQuantized: false}

	// Accepted quantized paths.
	okCases := []struct {
		name   string
		expect HeadKernelExpectation
		path   string
	}{
		{name: "q4k resident head", expect: HeadKernelExpectation{Head: q4kHead}, path: "lm_head_q4k"},
		{name: "q6k resident kquant head", expect: HeadKernelExpectation{Head: q6kHead}, path: "lm_head_kquant"},
		{name: "q6k named path", expect: HeadKernelExpectation{Head: q6kHead}, path: "metal_q6_k_head"},
		{name: "q8 resident head", expect: HeadKernelExpectation{Head: q8Head}, path: "lm_head_q8"},
		{name: "pq2 resident head", expect: HeadKernelExpectation{Head: q2Head}, path: "lm_head_q2"},
		{name: "qa int4 target on f16 baseline", expect: HeadKernelExpectation{Head: f16Head, TargetBits: 4}, path: "fused_dequant_int4"},
		{name: "qa int4 gptq path", expect: HeadKernelExpectation{RequireQuantized: true, TargetBits: 4}, path: "lm_head_gptq"},
		{name: "qa int8 target", expect: HeadKernelExpectation{RequireQuantized: true, TargetBits: 8}, path: "lm_head_q8"},
		{name: "unquantized head allows float when not required", expect: HeadKernelExpectation{Head: f16Head}, path: "lm_head_f32"},
		{name: "explicit expected path", expect: HeadKernelExpectation{RequireQuantized: true, ExpectedPath: "lm_head_q4k"}, path: "lm_head_q4k"},
	}
	for _, tc := range okCases {
		tc := tc
		t.Run("accept/"+tc.name, func(t *testing.T) {
			t.Parallel()
			if err := ConfirmQuantizedHeadPath(tc.expect, tc.path); err != nil {
				t.Fatalf("ConfirmQuantizedHeadPath(%+v, %q) = %v, want nil", tc.expect, tc.path, err)
			}
		})
	}

	// Fail-closed rejection cases.
	failCases := []struct {
		name    string
		expect  HeadKernelExpectation
		path    string
		wantErr error
	}{
		{
			name:    "missing reported path",
			expect:  HeadKernelExpectation{RequireQuantized: true},
			path:    "   ",
			wantErr: ErrHeadKernelPathMissing,
		},
		{
			name:    "q4k head fell back to lm_head_f32",
			expect:  HeadKernelExpectation{Head: q4kHead},
			path:    "lm_head_f32",
			wantErr: ErrHeadKernelPathFloatFallback,
		},
		{
			name:    "require quantized rejects bf16",
			expect:  HeadKernelExpectation{RequireQuantized: true},
			path:    "bf16",
			wantErr: ErrHeadKernelPathFloatFallback,
		},
		{
			name:    "int4 target rejects f32 dequant fallback",
			expect:  HeadKernelExpectation{Head: f16Head, TargetBits: 4},
			path:    "q4k_dequant_f32",
			wantErr: ErrHeadKernelPathFloatFallback,
		},
		{
			name:    "q4k head silently rerouted to q8",
			expect:  HeadKernelExpectation{Head: q4kHead},
			path:    "lm_head_q8",
			wantErr: ErrHeadKernelPathMismatch,
		},
		{
			name:    "int4 target received int8 path",
			expect:  HeadKernelExpectation{RequireQuantized: true, TargetBits: 4},
			path:    "lm_head_q8",
			wantErr: ErrHeadKernelPathMismatch,
		},
		{
			name:    "int8 target received q4k path",
			expect:  HeadKernelExpectation{RequireQuantized: true, TargetBits: 8},
			path:    "lm_head_q4k",
			wantErr: ErrHeadKernelPathMismatch,
		},
		{
			name:    "q6k head received q5k path",
			expect:  HeadKernelExpectation{Head: q6kHead},
			path:    "lm_head_q5k",
			wantErr: ErrHeadKernelPathMismatch,
		},
		{
			name:    "unrecognized kernel path",
			expect:  HeadKernelExpectation{RequireQuantized: true},
			path:    "custom_unknown_op",
			wantErr: ErrHeadKernelPathMismatch,
		},
		{
			name:    "empty expectation rejected",
			expect:  HeadKernelExpectation{},
			path:    "lm_head_f32",
			wantErr: ErrHeadKernelPathMismatch,
		},
		{
			name:    "invalid target bits rejected",
			expect:  HeadKernelExpectation{TargetBits: 3},
			path:    "lm_head_q4k",
			wantErr: ErrHeadKernelPathMismatch,
		},
	}
	for _, tc := range failCases {
		tc := tc
		t.Run("reject/"+tc.name, func(t *testing.T) {
			t.Parallel()
			err := ConfirmQuantizedHeadPath(tc.expect, tc.path)
			if err == nil {
				t.Fatalf("ConfirmQuantizedHeadPath(%+v, %q) = nil, want %v", tc.expect, tc.path, tc.wantErr)
			}
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("ConfirmQuantizedHeadPath(%+v, %q) error = %v, want errors.Is(%v)", tc.expect, tc.path, err, tc.wantErr)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestQwen38RealHeaderHeadAndSensitivity(t *testing.T) {
	t.Parallel()
	raw := qwen38Q4KMHeaderRaw(t)
	gg, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("Read pinned header: %v", err)
	}
	cfg, err := gg.Config()
	if err != nil {
		t.Fatalf("Config pinned header: %v", err)
	}

	head, err := AnalyzeOutputHeadQuant(cfg, gg.Tensors)
	if err != nil {
		t.Fatalf("AnalyzeOutputHeadQuant: %v", err)
	}
	if head.TensorName != "output.weight" || head.Tied || !head.IsQuantized {
		t.Fatalf("unexpected real header head = %+v", head)
	}

	sens, err := ClassifyModuleSensitivity(cfg, gg.Tensors)
	if err != nil {
		t.Fatalf("ClassifyModuleSensitivity: %v", err)
	}
	if sens.Class(SensitivityOutputHead).PayloadBytes != head.PayloadBytes {
		t.Fatalf("sensitivity output_head bytes %d != head bytes %d",
			sens.Class(SensitivityOutputHead).PayloadBytes, head.PayloadBytes)
	}
	if sens.Class(SensitivityAttention).PayloadBytes == 0 || sens.Class(SensitivityMLP).PayloadBytes == 0 {
		t.Fatalf("expected non-zero attention and mlp classes in %+v", sens.Ordered)
	}

	floor, err := SweepWeightByteFloor(cfg, gg.Tensors, DefaultWeightByteFloorSpec())
	if err != nil {
		t.Fatalf("SweepWeightByteFloor: %v", err)
	}
	if floor.BaselineEmbedGatherBytes == 0 || floor.BaselineBytesPerToken <= floor.BaselineHeadBytes {
		t.Fatalf("unexpected real header floor baseline = %+v", floor)
	}
	if err := ConfirmQuantizedHeadPath(HeadKernelExpectation{Head: head}, "lm_head_kquant"); err != nil {
		t.Fatalf("ConfirmQuantizedHeadPath(real head, lm_head_kquant): %v", err)
	}
	if err := ConfirmQuantizedHeadPath(HeadKernelExpectation{Head: head}, "lm_head_f32"); !errors.Is(err, ErrHeadKernelPathFloatFallback) {
		t.Fatalf("ConfirmQuantizedHeadPath(real head, lm_head_f32) = %v, want ErrHeadKernelPathFloatFallback", err)
	}
}
