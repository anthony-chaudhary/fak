package model

import (
	"encoding/json"
	"math"
	"testing"
)

// yarn arrives in EITHER HF shape and both must reach the same scaled inv_freq table.
// The newer checkpoints nest it under rope_parameters.default; the long-context Qwen line
// still ships it under the CLASSIC rope_scaling key, which decodes into Config.LongRope.
// Only the llama3 kind was promoted off LongRope, so a classic-key yarn config left
// RopeScaling=="" and applyRopeScaling returned a BARE inv_freq -- at a long context the
// rotation was unscaled and long-offset positions diverged silently, with nothing to notice
// it (#4874). These tests are that witness: they assert the promotion AND that it actually
// changes the numbers, so the promotion cannot be reverted or no-op'd unobserved.

func decodeRopeCfg(t *testing.T, js string) Config {
	t.Helper()
	var cfg Config
	if err := json.Unmarshal([]byte(js), &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return cfg
}

const yarnBaseFields = `"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
	"intermediate_size": 128, "vocab_size": 32, "rms_norm_eps": 1e-6, "model_type": "qwen2"`

func TestClassicRopeScalingYarnReachesTheYarnPath(t *testing.T) {
	classic := decodeRopeCfg(t, `{`+yarnBaseFields+`, "rope_theta": 10000,
		"rope_scaling": {
			"rope_type": "yarn", "factor": 4.0,
			"original_max_position_embeddings": 32768
		}
	}`)

	if classic.RopeScaling != "yarn" {
		t.Fatalf("classic-key yarn left RopeScaling=%q, want %q: applyRopeScaling would return a bare inv_freq", classic.RopeScaling, "yarn")
	}
	if classic.RopeFactor != 4.0 {
		t.Fatalf("RopeFactor = %v, want 4.0", classic.RopeFactor)
	}
	if classic.RopeOrigContext != 32768 {
		t.Fatalf("RopeOrigContext = %d, want 32768", classic.RopeOrigContext)
	}

	// The promotion must MOVE THE NUMBERS, not just set a string. Same config with the
	// rope_scaling block removed is the unscaled reference; if the two tables agree, the
	// yarn path is not running and the assertions above are decorative.
	bare := decodeRopeCfg(t, `{`+yarnBaseFields+`, "rope_theta": 10000}`)
	if bare.RopeScaling != "" {
		t.Fatalf("reference config picked up RopeScaling=%q; it must be unscaled", bare.RopeScaling)
	}

	got, want := invFreq(classic, 0), invFreq(bare, 0)
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("inv_freq lengths %d vs %d (want equal and non-empty)", len(got), len(want))
	}
	var maxDelta float64
	for j := range got {
		if d := math.Abs(got[j] - want[j]); d > maxDelta {
			maxDelta = d
		}
	}
	if maxDelta == 0 {
		t.Fatal("classic-key yarn produced a BARE inv_freq identical to the unscaled table: the promotion is a no-op")
	}
	t.Logf("classic-key yarn shifts inv_freq by max|delta| = %g vs unscaled", maxDelta)
}

func TestEffectiveRopeParametersPrefersTheNestedShape(t *testing.T) {
	// Both shapes present and DISAGREEING, so whichever wins is observable in the factor.
	both := decodeRopeCfg(t, `{`+yarnBaseFields+`, "rope_theta": 10000,
		"rope_parameters": {
			"default": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 8192}
		},
		"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
	}`)
	if rp := both.effectiveRopeParameters(); rp.Factor != 2.0 {
		t.Fatalf("effectiveRopeParameters().Factor = %v, want 2.0 (the NESTED rope_parameters.default block wins)", rp.Factor)
	}

	// Classic key alone still resolves -- that is the whole point of the dual shape.
	classicOnly := decodeRopeCfg(t, `{`+yarnBaseFields+`, "rope_theta": 10000,
		"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
	}`)
	if rp := classicOnly.effectiveRopeParameters(); rp.Factor != 4.0 {
		t.Fatalf("effectiveRopeParameters().Factor = %v, want 4.0 (classic rope_scaling must resolve when it is the only shape)", rp.Factor)
	}

	// Neither shape present degrades to the zero value, NOT to an early return: every
	// caller falls its fields through to the flat Config twins.
	none := decodeRopeCfg(t, `{`+yarnBaseFields+`, "rope_theta": 10000}`)
	if rp := none.effectiveRopeParameters(); rp.Factor != 0 {
		t.Fatalf("effectiveRopeParameters() on a config with neither shape = %+v, want the zero value", rp)
	}
}
