package model

import (
	"errors"
	"strings"
	"testing"
)

// TestV41QuantSensitivityFormatClosedSet pins the closed re-quant format set and
// its effective-bits ranking: every declared format has a rank, unknown tags
// have none (fail closed), and the ordering separates FP4 from IQ3_XXS from
// IQ2_XXS exactly where the floors depend on it.
func TestV41QuantSensitivityFormatClosedSet(t *testing.T) {
	formats := []V41RequantFormat{
		V41FormatFP8, V41FormatQ8_0, V41FormatQ6_K, V41FormatQ5_K, V41FormatQ4_K,
		V41FormatFP4, V41FormatIQ3_XXS, V41FormatQ3_K, V41FormatIQ2_XXS, V41FormatQ2_K,
	}
	seen := map[V41RequantFormat]bool{}
	for _, f := range formats {
		if seen[f] {
			t.Fatalf("duplicate format %q in the closed set", f)
		}
		seen[f] = true
		if _, ok := v41RequantFormatBits(f); !ok {
			t.Fatalf("declared format %q has no effective-bits rank", f)
		}
	}
	// The floor reasoning only works if these inequalities hold.
	fp4, _ := v41RequantFormatBits(V41FormatFP4)
	iq3, _ := v41RequantFormatBits(V41FormatIQ3_XXS)
	iq2, _ := v41RequantFormatBits(V41FormatIQ2_XXS)
	if !(fp4 > iq3 && iq3 > iq2) {
		t.Fatalf("rank ordering broken: FP4=%v IQ3_XXS=%v IQ2_XXS=%v", fp4, iq3, iq2)
	}
	if _, ok := v41RequantFormatBits("NOPE"); ok {
		t.Fatal("unknown format tag must have no rank (fail closed)")
	}
}

// TestV41QuantFloorEveryClassAndGroup asserts the floor map is total: every
// V4TensorClass resolves a floor, and the four v41-only groups (Engram tables,
// Engram gate, hyper-connection, ViT) are keyed explicitly.
func TestV41QuantFloorEveryClassAndGroup(t *testing.T) {
	classes := []V4TensorClass{
		V4ClassRoutedExpert, V4ClassIndexerQK, V4ClassSharedExpert, V4ClassAttention,
		V4ClassRouter, V4ClassDenseFFN, V4ClassHead, V4ClassEmbedding, V4ClassNorm,
		V4ClassMTP, V4ClassVision,
	}
	for _, c := range classes {
		if _, ok := v41ClassFloors(c); !ok {
			t.Fatalf("class %q has no declared floor", c)
		}
	}
	for _, g := range []v41SensitiveGroup{v41GroupEngramTable, v41GroupEngramGate, v41GroupHyperConn, v41GroupVision} {
		fl, why, ok := V41SensitivityFloor(g)
		if !ok {
			t.Fatalf("group %q has no declared floor", g)
		}
		if fl == "" || why == "" {
			t.Fatalf("group %q floor is empty: format=%q why=%q", g, fl, why)
		}
	}
	// Fail closed on an unplaced group.
	if _, _, ok := V41SensitivityFloor("not_a_group"); ok {
		t.Fatal("unplaced group must refuse closed")
	}
}

// TestV41QuantFloorExpertFamily pins the class-specific floors the issue calls
// out: expert up/gate floor at IQ3_XXS (never IQ2_XXS), expert down-proj floors
// at the trained FP4, and attn/indexer floor at the checkpoint FP8/FP4 family.
func TestV41QuantFloorExpertFamily(t *testing.T) {
	cases := []struct {
		name string
		want V41RequantFormat
	}{
		{"model.layers.0.mlp.experts.0.gate_proj.weight", V41FormatIQ3_XXS},
		{"model.layers.0.mlp.experts.0.up_proj.weight", V41FormatIQ3_XXS},
		{"model.layers.0.mlp.experts.0.down_proj.weight", V41FormatFP4},
		{"model.layers.0.self_attn.indexer.wq_b.weight", V41FormatFP4},
		{"model.layers.0.self_attn.q_proj.weight", V41FormatFP8},
	}
	for _, tc := range cases {
		_, _, fl, ok := v41FloorFor(tc.name)
		if !ok {
			t.Fatalf("%s: no floor resolved", tc.name)
		}
		if fl.Format != tc.want {
			t.Fatalf("%s: floor = %q, want %q", tc.name, fl.Format, tc.want)
		}
	}
}

// TestV41QuantFloorEngramAndNewGroups proves the four previously-unmapped groups
// classify and floor: Engram n-gram tables, Engram gate rows, hyper-connection
// combine matrices, and the ViT tower.
func TestV41QuantFloorEngramAndNewGroups(t *testing.T) {
	cases := []struct {
		name  string
		group v41SensitiveGroup
		want  V41RequantFormat
	}{
		{"model.layers.1.engram.table.weight", v41GroupEngramTable, V41FormatFP8},
		{"model.layers.14.engram.gate_proj.weight", v41GroupEngramGate, V41FormatFP8},
		{"model.layers.0.hc.combine.weight", v41GroupHyperConn, V41FormatFP8},
		{"model.visual.blocks.0.attn.qkv.weight", v41GroupVision, V41FormatFP8},
	}
	for _, tc := range cases {
		group, _, fl, ok := v41FloorFor(tc.name)
		if !ok {
			t.Fatalf("%s: no floor resolved", tc.name)
		}
		if group != tc.group {
			t.Fatalf("%s: group = %q, want %q", tc.name, group, tc.group)
		}
		if fl.Format != tc.want {
			t.Fatalf("%s: floor = %q, want %q", tc.name, fl.Format, tc.want)
		}
	}
}

// TestV41QuantAdmitBelowFloorRefuses is the headline guard: a below-floor
// proposal is refused with a typed error naming the tensor, class, floor, and
// proposed format; an at-or-above-floor proposal is admitted.
func TestV41QuantAdmitBelowFloorRefuses(t *testing.T) {
	refuse := []struct {
		name     string
		proposed V41RequantFormat
		floor    V41RequantFormat
	}{
		{"model.layers.0.mlp.experts.0.down_proj.weight", V41FormatIQ2_XXS, V41FormatFP4},
		{"model.layers.0.mlp.experts.0.up_proj.weight", V41FormatIQ2_XXS, V41FormatIQ3_XXS},
		{"model.layers.0.self_attn.q_proj.weight", V41FormatQ2_K, V41FormatFP8},
		{"model.layers.0.self_attn.indexer.wq_b.weight", V41FormatIQ3_XXS, V41FormatFP4},
		{"model.layers.1.engram.table.weight", V41FormatIQ2_XXS, V41FormatFP8},
		{"model.layers.14.engram.gate_proj.weight", V41FormatQ2_K, V41FormatFP8},
		{"model.layers.0.hc.combine.weight", V41FormatIQ2_XXS, V41FormatFP8},
		{"model.embed_tokens.weight", V41FormatIQ2_XXS, V41FormatFP8},
		{"lm_head.weight", V41FormatQ2_K, V41FormatFP8},
	}
	for _, tc := range refuse {
		v, err := admitRequantTensor(V4TensorMeta{Name: tc.name, Dtype: "FP4"}, tc.proposed)
		if err == nil {
			t.Fatalf("%s @ %s should refuse below floor", tc.name, tc.proposed)
		}
		var ue *UnsupportedRequantError
		if !errors.As(err, &ue) {
			t.Fatalf("%s: error is %T, want *UnsupportedRequantError", tc.name, err)
		}
		if v.Disposition != V4Refuse {
			t.Fatalf("%s: disposition = %q, want REFUSE", tc.name, v.Disposition)
		}
		if ue.Floor != tc.floor {
			t.Fatalf("%s: error floor = %q, want %q", tc.name, ue.Floor, tc.floor)
		}
		if ue.Proposed != tc.proposed {
			t.Fatalf("%s: error proposed = %q, want %q", tc.name, ue.Proposed, tc.proposed)
		}
		if ue.Class == "" || ue.Group == "" {
			t.Fatalf("%s: error must name class and group, got class=%q group=%q", tc.name, ue.Class, ue.Group)
		}
		msg := err.Error()
		for _, want := range []string{tc.name, string(tc.proposed), string(tc.floor)} {
			if !strings.Contains(msg, want) {
				t.Fatalf("%s: error message %q missing %q", tc.name, msg, want)
			}
		}
	}

	admit := []struct {
		name     string
		proposed V41RequantFormat
	}{
		{"model.layers.0.mlp.experts.0.down_proj.weight", V41FormatFP4},
		{"model.layers.0.mlp.experts.0.down_proj.weight", V41FormatQ5_K},
		{"model.layers.0.mlp.experts.0.up_proj.weight", V41FormatIQ3_XXS},
		{"model.layers.0.mlp.experts.0.up_proj.weight", V41FormatFP8},
		{"model.layers.0.self_attn.q_proj.weight", V41FormatQ8_0},
		{"model.layers.1.engram.table.weight", V41FormatFP8},
	}
	for _, tc := range admit {
		v, err := admitRequantTensor(V4TensorMeta{Name: tc.name, Dtype: "FP4"}, tc.proposed)
		if err != nil {
			t.Fatalf("%s @ %s should admit at/above floor, got %v", tc.name, tc.proposed, err)
		}
		if v.Disposition != V4Admit {
			t.Fatalf("%s: disposition = %q, want ADMIT", tc.name, v.Disposition)
		}
	}
}

// TestV41QuantAdmitFailClosed proves the guard refuses closed on an unknown
// format tag and an unclassifiable tensor, with no default-admit.
func TestV41QuantAdmitFailClosed(t *testing.T) {
	// Unknown format tag.
	v, err := admitRequantTensor(V4TensorMeta{Name: "model.layers.0.self_attn.q_proj.weight"}, "SUPER_2BIT")
	if err == nil {
		t.Fatal("unknown format tag must refuse closed")
	}
	if v.Disposition != V4Refuse {
		t.Fatalf("unknown format: disposition = %q, want REFUSE", v.Disposition)
	}
	// Unclassifiable tensor name.
	v, err = admitRequantTensor(V4TensorMeta{Name: "totally.unknown.tensor"}, V41FormatFP8)
	if err == nil {
		t.Fatal("unclassifiable tensor must refuse closed")
	}
	if v.Disposition != V4Refuse {
		t.Fatalf("unclassifiable tensor: disposition = %q, want REFUSE", v.Disposition)
	}
	// Vision tower is SKIP, not REFUSE and not ADMIT.
	v, err = admitRequantTensor(V4TensorMeta{Name: "model.visual.blocks.0.attn.qkv.weight"}, V41FormatIQ2_XXS)
	if err != nil {
		t.Fatalf("vision tower should SKIP: %v", err)
	}
	if v.Disposition != V4Skip {
		t.Fatalf("vision tower: disposition = %q, want SKIP", v.Disposition)
	}
}

// TestV41QuantSensitivityExistingAdmissionUnchanged pins that adding the floor
// guard did not perturb the existing precision-family admission verdicts.
func TestV41QuantSensitivityExistingAdmissionUnchanged(t *testing.T) {
	// The existing gate still admits an FP4 routed expert at its trained dtype...
	v, err := admitV4Tensor(V4TensorMeta{Name: "model.layers.0.mlp.experts.0.gate_proj.weight", Dtype: "F4_E2M1"})
	if err != nil || v.Disposition != V4Admit {
		t.Fatalf("existing FP4 expert admission regressed: v=%+v err=%v", v, err)
	}
	// ...and still fails closed on an unrecognized FP4 class.
	if _, err := admitV4Tensor(V4TensorMeta{Name: "mystery.fp4.tensor", Dtype: "FP4"}); err == nil {
		t.Fatal("existing fail-closed FP4 admission regressed")
	}
}
