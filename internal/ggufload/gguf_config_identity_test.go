package ggufload

import "testing"

func TestStringMetaOrPreservesExactCheckpointName(t *testing.T) {
	meta := map[string]Value{
		"general.name": {Type: TypeString, Value: "Qwen3.8-27B-Q4_K_M"},
	}
	if got := stringMetaOr(meta, "general.name", ""); got != "Qwen3.8-27B-Q4_K_M" {
		t.Fatalf("checkpoint name = %q, want exact GGUF general.name", got)
	}
}

func TestStringMetaOrFailsSafeForAbsentOrInvalidName(t *testing.T) {
	for name, meta := range map[string]map[string]Value{
		"absent":     {},
		"blank":      {"general.name": {Type: TypeString, Value: "  "}},
		"wrong type": {"general.name": {Type: TypeUint32, Value: uint32(38)}},
	} {
		t.Run(name, func(t *testing.T) {
			if got := stringMetaOr(meta, "general.name", "fallback"); got != "fallback" {
				t.Fatalf("checkpoint name = %q, want fallback", got)
			}
		})
	}
}
