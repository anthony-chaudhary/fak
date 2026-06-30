package ggufload

import "testing"

func TestIssue7TensorNameMappings(t *testing.T) {
	got, ok := CanonicalTensorNameArch("blk.0.post_ffw_norm.weight", "gemma3")
	if !ok || got != "model.layers.0.post_feedforward_layernorm.weight" {
		t.Fatalf("Gemma3 post_ffw_norm mapping = (%q,%v), want post_feedforward_layernorm", got, ok)
	}

	for _, tc := range []struct {
		name string
		proj string
	}{
		{name: "blk.0.ffn_gate_exps.weight", proj: "gate_proj"},
		{name: "blk.0.ffn_up_exps.weight", proj: "up_proj"},
		{name: "blk.0.ffn_down_exps.weight", proj: "down_proj"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			layer, proj, ok := glmMoeDsaBatchedExpert(tc.name)
			if !ok || layer != 0 || proj != tc.proj {
				t.Fatalf("batched expert classifier = layer:%d proj:%q ok:%v, want layer:0 proj:%q ok:true", layer, proj, ok, tc.proj)
			}
			host, err := tensorCPUOffloadExpert(tc.name, "qwen3moe")
			if err != nil || !host {
				t.Fatalf("tensorCPUOffloadExpert(%q,qwen3moe) = host:%v err:%v, want host:true nil", tc.name, host, err)
			}
		})
	}
}
