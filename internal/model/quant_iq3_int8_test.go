package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"testing"
)

func w3TestConfig(layers int) Config {
	types := make([]string, layers)
	for i := range types {
		types[i] = "linear_attention"
	}
	if layers > 0 {
		types[layers-1] = "full_attention"
	}
	return Config{ModelType: "qwen35", NumLayers: layers, LayerTypes: types}
}

func w3ProjectionName(layer int, projection string) string {
	return fmt.Sprintf("model.layers.%d.mlp.%s_proj.weight", layer, projection)
}

func TestW3MLPRequestedDefaultOff(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{{"", false}, {"0", false}, {"false", false}, {"yes", false}, {"1", true}, {"on", true}, {"true", true}, {" TRUE ", true}} {
		t.Run(fmt.Sprintf("value_%q", tc.value), func(t *testing.T) {
			t.Setenv("FAK_W3_MLP", tc.value)
			if got := W3MLPRequested(); got != tc.want {
				t.Fatalf("W3MLPRequested()=%v for %q, want %v", got, tc.value, tc.want)
			}
		})
	}
}

func TestW3MLPProjectionGate(t *testing.T) {
	cfg := w3TestConfig(4)
	for _, name := range []string{
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.2.mlp.up_proj.weight",
		"model.layers.3.mlp.down_proj.weight",
	} {
		if !ResidentW3MLPEligible(cfg, name) {
			t.Errorf("ResidentW3MLPEligible(%q)=false, want true", name)
		}
	}
	for _, name := range []string{
		"model.layers.00.mlp.gate_proj.weight",
		"model.layers.-1.mlp.gate_proj.weight",
		"model.layers.4.mlp.gate_proj.weight",
		"model.layers.0.mlp.gate_proj.bias",
		"model.layers.0.mlp.gate_proj.weight.extra",
		"model.layers.0.mlp.gate_up_proj.weight",
		"model.layers.0.mlp.experts.0.gate_proj.weight",
		"model.layers.0.mlp.shared_experts.up_proj.weight",
		"model.layers.0.mlp.router.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.linear_attn.in_proj_qkv.weight",
		"lm_head.weight",
		"model.visual.blocks.0.mlp.up_proj.weight",
		"model.layers.4.nextn.mlp.up_proj.weight",
	} {
		if ResidentW3MLPEligible(cfg, name) {
			t.Errorf("ResidentW3MLPEligible(%q)=true, want false", name)
		}
	}
	if ResidentW3MLPEligible(Config{ModelType: "llama", NumLayers: 4}, "model.layers.0.mlp.up_proj.weight") {
		t.Error("non-hybrid config must be refused")
	}
	moe := cfg
	moe.NumExperts = 8
	if ResidentW3MLPEligible(moe, "model.layers.0.mlp.up_proj.weight") {
		t.Error("MoE config must be refused")
	}
}

func completeW3ModelForTest(layers int) *Model {
	m := &Model{Cfg: w3TestConfig(layers), kqw: map[string]*kQuantTensor{}}
	for l := 0; l < layers; l++ {
		for _, projection := range []string{"gate", "up", "down"} {
			name := w3ProjectionName(l, projection)
			m.kqw[name] = &kQuantTensor{kind: kindIQ3XXS, w3MLP: true}
		}
	}
	return m
}

func cloneW3ModelForTest(src *Model) *Model {
	dst := &Model{Cfg: src.Cfg, kqw: make(map[string]*kQuantTensor, len(src.kqw))}
	for name, qt := range src.kqw {
		copyQT := *qt
		dst.kqw[name] = &copyQT
	}
	return dst
}

func TestValidateResidentW3MLP(t *testing.T) {
	complete := completeW3ModelForTest(2)
	if err := complete.ValidateResidentW3MLP(); err != nil {
		t.Fatalf("complete W3 band rejected: %v", err)
	}
	if got := complete.ResidentW3MLPCount(); got != 6 {
		t.Fatalf("ResidentW3MLPCount()=%d, want 6", got)
	}
	if !complete.HasResidentW3MLP(w3ProjectionName(1, "down")) {
		t.Fatal("HasResidentW3MLP rejected a tagged IQ3_XXS projection")
	}

	missing := cloneW3ModelForTest(complete)
	missingName := w3ProjectionName(1, "down")
	delete(missing.kqw, missingName)
	if err := missing.ValidateResidentW3MLP(); err == nil || !strings.Contains(err.Error(), missingName) || !strings.Contains(err.Error(), "5/6") {
		t.Fatalf("missing projection error=%v, want name and 5/6 count", err)
	}

	wrongKind := cloneW3ModelForTest(complete)
	wrongName := w3ProjectionName(0, "up")
	wrongKind.kqw[wrongName].kind = kindQ6K
	if err := wrongKind.ValidateResidentW3MLP(); err == nil || !strings.Contains(err.Error(), wrongName) || !strings.Contains(err.Error(), "Q6_K") {
		t.Fatalf("wrong-kind error=%v, want tensor name and Q6_K", err)
	}

	unexpected := cloneW3ModelForTest(complete)
	unexpectedName := "model.layers.0.self_attn.v_proj.weight"
	unexpected.kqw[unexpectedName] = &kQuantTensor{kind: kindIQ3XXS, w3MLP: true}
	if err := unexpected.ValidateResidentW3MLP(); err == nil || !strings.Contains(err.Error(), unexpectedName) || !strings.Contains(err.Error(), "7/6") {
		t.Fatalf("unexpected-tag error=%v, want name and 7/6 count", err)
	}

	empty := &Model{Cfg: w3TestConfig(2), kqw: map[string]*kQuantTensor{}}
	if err := empty.ValidateResidentW3MLP(); err == nil || !strings.Contains(err.Error(), w3ProjectionName(0, "gate")) || !strings.Contains(err.Error(), "0/6") {
		t.Fatalf("empty-band error=%v, want first missing name and 0/6 count", err)
	}

	wrongArch := &Model{Cfg: Config{ModelType: "llama", NumLayers: 2}, kqw: map[string]*kQuantTensor{}}
	if err := wrongArch.ValidateResidentW3MLP(); err == nil || !strings.Contains(err.Error(), "dense Qwen3.5-family hybrid") {
		t.Fatalf("wrong-arch error=%v, want dense hybrid refusal", err)
	}
}

func deterministicIQ3Rows(out, nblk int, seed uint64) []byte {
	raw := make([]byte, out*nblk*iq3xxsBlockBytes)
	s := seed
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*iq3xxsBlockBytes:]
			binary.LittleEndian.PutUint16(blk, f16One)
			for i := 2; i < 2+qkK/4; i++ {
				s = s*6364136223846793005 + 1442695040888963407
				blk[i] = byte(s >> 32)
			}
			for sub := 0; sub < 8; sub++ {
				var aux uint32
				for group := 0; group < 4; group++ {
					sel := uint32((o*29 + b*17 + sub*11 + group*7) & 127)
					aux |= sel << (7 * uint(group))
				}
				aux |= uint32((o+b+sub)%16) << 28
				binary.LittleEndian.PutUint32(blk[2+qkK/4+4*sub:], aux)
			}
		}
	}
	return raw
}

func TestIQ3XXSReduceScalarMatchesDequantizedIntegerOracle(t *testing.T) {
	cycle := [...]int8{-127, -64, -1, 0, 1, 63, 127}
	for _, nblk := range []int{1, 20, 68} {
		t.Run(fmt.Sprintf("in_%d", nblk*qkK), func(t *testing.T) {
			raw := deterministicIQ3Rows(1, nblk, 0x4628c5a5)
			qx := make([]int8, nblk*qkK)
			for i := range qx {
				qx[i] = cycle[i%len(cycle)]
			}
			got := make([]int32, nblk*8)
			iq3xxsReduceRowScalar(raw, nblk, qx, got)
			dequant := make([]float32, qkK)
			for b := 0; b < nblk; b++ {
				blk := raw[b*iq3xxsBlockBytes : (b+1)*iq3xxsBlockBytes]
				iq3xxsDequantSuperBlock(dequant, blk)
				sas := blk[2+qkK/4:]
				for sub := 0; sub < 8; sub++ {
					aux := binary.LittleEndian.Uint32(sas[4*sub:])
					db := (0.5 + float32(aux>>28)) * 0.5
					var want int32
					for i := 0; i < 32; i++ {
						weightInteger := int32(dequant[sub*32+i] / db)
						want += weightInteger * int32(qx[b*qkK+sub*32+i])
					}
					if got[b*8+sub] != want {
						t.Fatalf("block=%d sub=%d reducer=%d dequant-oracle=%d", b, sub, got[b*8+sub], want)
					}
				}
			}
		})
	}
}

func TestW3MLPDispatchRealReductionDimensions(t *testing.T) {
	for _, in := range []int{5120, 17408} {
		t.Run(fmt.Sprintf("in_%d", in), func(t *testing.T) {
			const out = 4
			nblk := in / qkK
			raw := deterministicIQ3Rows(out, nblk, uint64(in))
			qt := quantizeKQuantFromRaw(raw, out, in, kindIQ3XXS)
			qt.w3MLP = true
			x := make([]float32, in)
			for i := range x {
				x[i] = float32(math.Sin(float64(i)*0.017) + 0.25*math.Cos(float64(i)*0.0031))
			}

			got := make([]float32, out)
			kQuantMatRowsInto(qt, x, got)
			qv := quantizeVecQ8(x)
			wantDispatch := make([]float32, out)
			iq3xxsMatRowsRangeInt8(qt, qv, wantDispatch, 0, out)
			for row := range got {
				if got[row] != wantDispatch[row] {
					t.Fatalf("row=%d dispatch=%v direct-int8=%v", row, got[row], wantDispatch[row])
				}
			}

			f32 := make([]float32, out)
			untagged := *qt
			untagged.w3MLP = false
			kQuantMatRowsRange(&untagged, x, f32, 0, out)
			var sumSq, maxAbs float64
			for row := range got {
				delta := math.Abs(float64(got[row] - f32[row]))
				if delta > maxAbs {
					maxAbs = delta
				}
				sumSq += float64(f32[row]) * float64(f32[row])
			}
			rms := math.Sqrt(sumSq / out)
			if rms == 0 {
				t.Fatal("f32 reference RMS is zero")
			}
			if rel := maxAbs / rms; rel > 0.05 {
				t.Fatalf("max|int8-f32|/referenceRMS=%g, want <= 0.05", rel)
			}
		})
	}
}
