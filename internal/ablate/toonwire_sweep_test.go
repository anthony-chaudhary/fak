package ablate

import "testing"

// TestToonWireSweepArm proves the #3067 registration end-to-end at the harness
// boundary: `--sweep toon_wire` yields an arm whose child environment carries
// FAK_TOON_WIRE=1 (and the all-off baseline pins it to 0), which is exactly the
// flag the gateway's mcpToolResult TOON wire reads per call.
func TestToonWireSweepArm(t *testing.T) {
	configs, err := BuildSweep([]string{FeatureToonWire})
	if err != nil {
		t.Fatal(err)
	}
	if len(configs) != 2 {
		t.Fatalf("want all-off + toon_wire arms, got %d", len(configs))
	}
	off, on := configs[0], configs[1]
	if env := off.childEnv(); len(env) != 1 || env[0] != "FAK_TOON_WIRE=0" {
		t.Fatalf("all-off child env = %v, want [FAK_TOON_WIRE=0]", env)
	}
	if on.EnvFeatures[FeatureToonWire] != "on" {
		t.Fatalf("toon_wire arm not on: %#v", on)
	}
	if env := on.childEnv(); len(env) != 1 || env[0] != "FAK_TOON_WIRE=1" {
		t.Fatalf("toon_wire child env = %v, want [FAK_TOON_WIRE=1]", env)
	}
	if !EnvGated(FeatureToonWire) {
		t.Fatal("toon_wire must be env-gated (subprocess rung)")
	}
}
