package ggufinterop

import (
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"testing"
)

func fixture(typ uint32, meta map[string]any) *ggufload.File {
	m := map[string]ggufload.Value{}
	for k, v := range meta {
		m[k] = ggufload.Value{Value: v}
	}
	return &ggufload.File{Metadata: m, Tensors: []ggufload.TensorInfo{{Type: ggufload.TensorType(typ)}}}
}
func TestMapFamilies(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  uint32
		want string
	}{{"k", 12, "k-quant"}, {"iq", 20, "iq"}, {"ternary", 28, "ternary"}} {
		t.Run(tc.name, func(t *testing.T) {
			r := Map(fixture(tc.typ, map[string]any{"general.architecture": "qwen2"}))
			if r.Outcome != OutcomeSupported || string(r.Descriptor.Extra["gguf_quant_family"]) != "\""+tc.want+"\"" {
				t.Fatalf("%+v", r)
			}
		})
	}
}
func TestMapUnknownAbstains(t *testing.T) {
	if r := Map(fixture(999, map[string]any{"general.architecture": "qwen2"})); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
}
func TestMapSplitDelegates(t *testing.T) {
	r := Map(fixture(12, map[string]any{"general.architecture": "qwen2", "split.count": uint32(3)}))
	if r.Outcome != OutcomeDelegate || r.SplitCount != 3 {
		t.Fatalf("%+v", r)
	}
}
func TestMapMissingArchitectureAbstains(t *testing.T) {
	if r := Map(fixture(12, map[string]any{})); r.Outcome != OutcomeAbstain {
		t.Fatalf("%+v", r)
	}
}
