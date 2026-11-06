package compute

import (
	"errors"
	"testing"
)

func splitPlanTensor(dtype Dtype, out, in int) Tensor {
	return Tensor{Dtype: dtype, Shape: []int{out, in}}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDASequenceSplitScratchPlan(t *testing.T) {
	q8 := splitPlanTensor(Q8_0, 4, 256)
	q4 := splitPlanTensor(Q4_K, 8, 256)
	q2 := splitPlanTensor(Q2_K, 7, 512)
	q3 := splitPlanTensor(Q3_K, 8, 512)
	badArm := errors.New("latched Q4 arm refusal")
	gdn := Qwen35SequenceLayer{Linear: true,
		GDNInQKV: splitPlanTensor(Q8_0, 40, 256), GDNInZ: splitPlanTensor(Q8_0, 24, 256),
		GDNInB: q8, GDNInA: q8, GDNOut: splitPlanTensor(Q8_0, 16, 512)}
	mixedGDN := gdn
	mixedGDN.GDNInA.Dtype = F32
	for _, tc := range []struct {
		name    string
		tokens  int
		layer   Qwen35SequenceLayer
		arm     CUDAQ4KPrefillArm
		armErr  error
		w, x    int
		wantErr bool
	}{
		{name: "127 rows no loan", tokens: 127, layer: Qwen35SequenceLayer{Gate: q4}, armErr: badArm},
		{name: "Q8 threshold", tokens: 128, layer: Qwen35SequenceLayer{Gate: q8}, w: 2048, x: 65536},
		{name: "Q2", tokens: 128, layer: Qwen35SequenceLayer{Down: q2}, w: 7168, x: 131072},
		{name: "Q3 only ignores unrelated Q4 refusal", tokens: 128, layer: Qwen35SequenceLayer{Gate: q3}, armErr: badArm, w: 8192, x: 131072},
		{name: "Q3 exceeds mixed Q8 loan", tokens: 128, layer: Qwen35SequenceLayer{Gate: q8, Down: q3}, w: 8192, x: 131072},
		{name: "Q3 weight and Q8 activation maxima", tokens: 128, layer: Qwen35SequenceLayer{
			Gate: splitPlanTensor(Q3_K, 1024, 256), Down: splitPlanTensor(Q8_0, 128, 512)}, w: 524288, x: 131072},
		{name: "Q8 weight and Q3 activation maxima", tokens: 128, layer: Qwen35SequenceLayer{
			Gate: splitPlanTensor(Q8_0, 1024, 256), Up: splitPlanTensor(Q3_K, 128, 512)}, w: 524288, x: 131072},
		{name: "Q3 sub-128 stays outside loan", tokens: 127, layer: Qwen35SequenceLayer{Gate: q3}},
		{name: "Q3 attention Q", tokens: 128, layer: Qwen35SequenceLayer{Q: q3}, w: 8192, x: 131072},
		{name: "Q3 attention K", tokens: 128, layer: Qwen35SequenceLayer{K: q3}, w: 8192, x: 131072},
		{name: "Q3 attention V", tokens: 128, layer: Qwen35SequenceLayer{V: q3}, w: 8192, x: 131072},
		{name: "Q3 attention O", tokens: 128, layer: Qwen35SequenceLayer{O: q3}, w: 8192, x: 131072},
		{name: "unsupported GDN Q3 remains excluded", tokens: 128, layer: Qwen35SequenceLayer{
			Linear: true, GDNInQKV: q3, GDNInZ: q3, GDNInB: q3, GDNInA: q3, GDNOut: q3}},
		{name: "Q3 zero dimension", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q3_K, 0, 256)}, wantErr: true},
		{name: "Q3 negative dimension", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q3_K, 8, -256)}, wantErr: true},
		{name: "Q3 wrong rank", tokens: 128, layer: Qwen35SequenceLayer{Gate: Tensor{Dtype: Q3_K, Shape: []int{8, 256, 1}}}, wantErr: true},
		{name: "Q3 weight product overflow", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q3_K, 65536, 65536)}, wantErr: true},
		{name: "Q3 activation product overflow", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q3_K, 1, 1<<25)}, wantErr: true},
		{name: "IQ4_XS exceeds mixed GDN Q8 loan", tokens: 128, layer: Qwen35SequenceLayer{
			Linear: true, Gate: splitPlanTensor(IQ4_XS, 1024, 256),
			GDNOut: splitPlanTensor(Q8_0, 256, 128)}, w: 524288, x: 65536},
		{name: "IQ4_XS only", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(IQ4_XS, 1024, 256)}, w: 524288, x: 65536},
		{name: "IQ4_XS sub-128 stays outside loan", tokens: 127, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(IQ4_XS, 1024, 256)}},
		{name: "attention Q K V O", tokens: 128, layer: Qwen35SequenceLayer{
			Q: q8, K: q4, V: q2, O: splitPlanTensor(Q8_0, 16, 256)},
			arm: CUDAQ4KPrefillCompensated, w: 8192, x: 131072},
		{name: "Q4 compensated", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4}, arm: CUDAQ4KPrefillCompensated, w: 4096, x: 65536},
		{name: "Q4 mmq excluded", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4}, arm: CUDAQ4KPrefillMMQ},
		{name: "Q4 fp8lt excluded", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4}, arm: CUDAQ4KPrefillFP8LT},
		{name: "mixed mmq Q8 remains", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4, Up: q8}, arm: CUDAQ4KPrefillMMQ, w: 2048, x: 65536},
		{name: "mixed fp8lt Q2 remains", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4, Down: q2}, arm: CUDAQ4KPrefillFP8LT, w: 7168, x: 131072},
		{name: "irrelevant invalid Q4 arm", tokens: 128, layer: Qwen35SequenceLayer{Gate: q8}, armErr: badArm, w: 2048, x: 65536},
		{name: "large Q4 arm refused", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4}, armErr: badArm, wantErr: true},
		{name: "unknown Q4 arm refused", tokens: 128, layer: Qwen35SequenceLayer{Gate: q4}, arm: "unknown", wantErr: true},
		{name: "GDN quartet and output", tokens: 128, layer: gdn, w: 20480, x: 131072},
		{name: "mixed GDN input but independent Q8 output", tokens: 128, layer: mixedGDN, w: 16384, x: 131072},
		{name: "excluded routes", tokens: 128, layer: Qwen35SequenceLayer{
			Gate: splitPlanTensor(F32, 1<<30, 1<<30), Up: splitPlanTensor(F16, 1<<30, 1<<30),
			Down: splitPlanTensor(Q5_K, 1<<30, 1<<30), Q: splitPlanTensor(Q6_K, 1<<30, 1<<30)}},
		{name: "crossed maxima", tokens: 128, layer: Qwen35SequenceLayer{
			Gate: splitPlanTensor(Q8_0, 1024, 256), Down: splitPlanTensor(Q2_K, 128, 512)}, w: 524288, x: 131072},
		{name: "zero dimension", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q8_0, 0, 256)}, wantErr: true},
		{name: "negative dimension", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q8_0, 8, -256)}, wantErr: true},
		{name: "weight product overflow", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q8_0, 65536, 65536)}, wantErr: true},
		{name: "activation product overflow", tokens: 128, layer: Qwen35SequenceLayer{Gate: splitPlanTensor(Q8_0, 1, 1<<25)}, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := Qwen35SequencePrefillRequest{
				TokenIDs: make([]int, tc.tokens), Layers: []Qwen35SequenceLayer{tc.layer},
				// A huge last-row head is deliberately outside the panel loan.
				Output: splitPlanTensor(Q8_0, 1<<30, 1<<30),
			}
			got, err := planCUDASequenceSplitScratch(req, tc.arm, tc.armErr)
			if tc.wantErr {
				if err == nil {
					t.Fatal("invalid plan was admitted")
				}
				if tc.armErr != nil && !errors.Is(err, tc.armErr) {
					t.Fatalf("arm refusal cause lost: %v", err)
				}
				return
			}
			if err != nil || got.weightHalfBytes != tc.w || got.activationHalfBytes != tc.x {
				t.Fatalf("plan=%+v error=%v, want weight/activation half bytes=%d/%d", got, err, tc.w, tc.x)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestCUDASequenceSplitScratchByteOverflow(t *testing.T) {
	maxInt := uint64(^uint(0) >> 1)
	for _, tc := range [][2]uint64{{0, 1}, {1, 0}, {maxInt/2 + 1, 1}, {1, maxInt/2 + 1}, {maxInt / 4, maxInt / 4}} {
		if _, err := cudaSequenceSplitScratchBytes(tc[0], tc[1]); err == nil {
			t.Fatalf("unsafe element capacities accepted: %v", tc)
		}
	}
	if got, err := cudaSequenceSplitScratchBytes(1, 1); err != nil || got.weightHalfBytes != 2 || got.activationHalfBytes != 2 {
		t.Fatalf("small valid capacities rejected: %+v %v", got, err)
	}
}
