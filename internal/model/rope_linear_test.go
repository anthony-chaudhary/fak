package model

import (
	"encoding/json"
	"math"
	"testing"
)

// fak-test:runtime fast est=1s
func TestLinearRopeModelLoading(t *testing.T) {
	const base = `"model_type":"qwen3_moe","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":4,"head_dim":16,"vocab_size":32,"intermediate_size":128,"max_position_embeddings":4096`
	cases := []struct {
		name, fields  string
		theta, factor float64
	}{
		{"flat", `"rope_theta":10000,"rope_scaling_type":"linear","rope_scaling_factor":4`, 10000, 4},
		{"classic", `"rope_theta":10000,"rope_scaling":{"rope_type":"linear","factor":4}`, 10000, 4},
		{"transformers5", `"rope_parameters":{"rope_theta":10000,"rope_type":"linear","factor":4}`, 10000, 4},
		{"nested_default", `"rope_parameters":{"default":{"rope_theta":10000,"rope_type":"linear","factor":2}}`, 10000, 2},
		{"classic_precedence", `"rope_theta":20000,"rope_scaling":{"type":"linear","factor":2},"rope_parameters":{"rope_theta":10000,"rope_type":"linear","factor":4}`, 20000, 2},
		{"zero_nested_fallback", `"rope_theta":10000,"rope_scaling_type":"linear","rope_scaling_factor":0,"rope_parameters":{"rope_type":"linear","factor":4}`, 10000, 4},
		{"zero_without_fallback", `"rope_theta":10000,"rope_scaling_type":"linear","rope_scaling_factor":0`, 10000, 1},
		{"negative_factor", `"rope_theta":10000,"rope_scaling_type":"linear","rope_scaling_factor":-2`, 10000, 1},
		{"unscaled", `"rope_theta":10000`, 10000, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal([]byte(`{`+base+`,`+tc.fields+`}`), &cfg); err != nil {
				t.Fatal(err)
			}
			for _, got := range [][]float64{invFreq(cfg, 0), cachedInvFreq(cfg, 0)} {
				if len(got) != cfg.rotaryDim()/2 {
					t.Fatalf("frequency count=%d", len(got))
				}
				for j, f := range got {
					want := math.Pow(tc.theta, -float64(2*j)/16) / tc.factor
					if math.Abs(f-want) > 1e-14 {
						t.Fatalf("frequency[%d]=%.17g want %.17g", j, f, want)
					}
				}
			}
		})
	}
}
