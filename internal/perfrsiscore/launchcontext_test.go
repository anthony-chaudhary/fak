package perfrsiscore

import (
	"encoding/json"
	"strings"
	"testing"
)

func populatedLaunchContext() LaunchContext {
	c := LaunchContext{Schema: Schema}
	c.Hardware = HardwareContext{
		Platform:      "strix-halo",
		Backend:       "vulkan",
		DriverVersion: "24.0.0",
		KernelVersion: "6.12.0",
		Governor:      "performance",
		ClocksMHz:     2400,
		ThermalC:      61.5,
		PowerW:        95.0,
	}
	c.CodeVersion = CodeVersion{SHA: "0123456789abcdef0123456789abcdef01234567", Dirty: false}
	c.LaunchParams = LaunchParams{
		Argv:          []string{"fak", "serve", "--model", "deepseek-v4"},
		EnvKeys:       []string{"FAK_ENGINE", "FAK_KV"},
		ResolvedFlags: map[string]string{"ctx": "4096", "batch": "8"},
	}
	c.ChangeSet = ChangeSet{Base: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Head: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
	c.Kernels = []string{"flash_attn_fwd", "rmsnorm_fwd"}
	c.InputConditions = InputConditions{
		Shapes:       []string{"1x4096"},
		DTypes:       []string{"bf16"},
		BatchSize:    8,
		Concurrency:  4,
		PromptTokens: 2048,
		PrefixTokens: 512,
		DecodeTokens: 256,
	}
	c.Threads = ThreadContext{Workers: 8, Observers: 1, Engine: 6}
	c.Intervals = []Interval{{Name: "queue", NS: 1000}, {Name: "load", NS: 2000}}
	c.CodePaths = []string{"internal/model/v41_forward.go", "internal/compute/vulkan.go"}
	return c
}

func allSentinelLaunchContext() LaunchContext {
	c := LaunchContext{Schema: Schema}
	c.Hardware = HardwareContext{Platform: Unmeasured, Backend: Unmeasured, DriverVersion: Unmeasured, KernelVersion: Unmeasured, Governor: Unmeasured}
	c.CodeVersion = CodeVersion{SHA: Unmeasured}
	c.LaunchParams = LaunchParams{}
	c.ChangeSet = ChangeSet{Base: Unmeasured, Head: Unmeasured}
	return c
}

func TestDecodeLaunchContextRejectsUnknownField(t *testing.T) {
	_, err := DecodeLaunchContext([]byte(`{"schema":"fak.launch-context.v1","surprise":true}`))
	if err == nil || !strings.Contains(err.Error(), "decode launch context:") {
		t.Fatalf("DecodeLaunchContext() error = %v, want unknown-field error", err)
	}
}

func TestDecodeLaunchContextRejectsSchemaMismatch(t *testing.T) {
	_, err := DecodeLaunchContext([]byte(`{"schema":"fak.launch-context.v2"}`))
	if err == nil || !strings.Contains(err.Error(), "schema") {
		t.Fatalf("DecodeLaunchContext() error = %v, want schema mismatch error", err)
	}
}

func TestFixedFixtureDigestIsStable(t *testing.T) {
	c := populatedLaunchContext()
	const want = "dc7d6888ab21fac8dfbc487161084f3dec7cbda74996cf1dfc426f419d916c50"
	if got := c.ComputeDigest(); got != want {
		t.Fatalf("ComputeDigest() = %q, want %q", got, want)
	}
	if got := c.ComputeDigest(); got != want {
		t.Fatalf("ComputeDigest() second call = %q, want %q (nondeterministic)", got, want)
	}

	c.Digest = c.ComputeDigest()
	encoded, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeLaunchContext(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Digest != want {
		t.Fatalf("round-trip digest = %q, want %q", decoded.Digest, want)
	}
	if got := decoded.ComputeDigest(); got != want {
		t.Fatalf("round-trip ComputeDigest() = %q, want %q", got, want)
	}
}

func TestWitnessedCountsSentinelAndPopulated(t *testing.T) {
	empty := allSentinelLaunchContext()
	if got := empty.WitnessedCount(); got != 0 {
		t.Fatalf("all-sentinel WitnessedCount() = %d, want 0: %+v", got, empty.Witnessed())
	}
	if got := empty.AxisCount(); got == 0 {
		t.Fatalf("AxisCount() = 0, want the declared axis set")
	}

	full := populatedLaunchContext()
	if got, want := full.WitnessedCount(), full.AxisCount(); got != want {
		t.Fatalf("populated WitnessedCount() = %d, want all %d axes witnessed: %+v", got, want, full.Witnessed())
	}
	for axis, ok := range full.Witnessed() {
		if !ok {
			t.Fatalf("populated axis %q reported unwitnessed", axis)
		}
	}
}

func TestDecodeLaunchContextRejectsDigestMismatch(t *testing.T) {
	c := populatedLaunchContext()
	c.Digest = strings.Repeat("0", 64)
	encoded, err := json.Marshal(c)
	if err != nil {
		t.Fatal(err)
	}
	_, err = DecodeLaunchContext(encoded)
	if err == nil || !strings.Contains(err.Error(), "does not match recomputed digest") {
		t.Fatalf("DecodeLaunchContext() error = %v, want digest mismatch error", err)
	}
}

// TestDigestCanonicalAcrossNilAndEmptySlices pins the canonicalization contract:
// two LaunchContext values that differ only in whether a slice (or map) field is
// nil or a non-nil empty must digest to the SAME SHA-256. Each case mutates one
// field of an otherwise-populated context to empty.
func TestDigestCanonicalAcrossNilAndEmptySlices(t *testing.T) {
	cases := []struct {
		name  string
		apply func(*LaunchContext)
	}{
		{"kernels", func(c *LaunchContext) { c.Kernels = nil }},
		{"code_paths", func(c *LaunchContext) { c.CodePaths = nil }},
		{"launch_params.argv", func(c *LaunchContext) { c.LaunchParams.Argv = nil }},
		{"launch_params.env_keys", func(c *LaunchContext) { c.LaunchParams.EnvKeys = nil }},
		{"launch_params.resolved_flags", func(c *LaunchContext) { c.LaunchParams.ResolvedFlags = nil }},
		{"input_conditions.shapes", func(c *LaunchContext) { c.InputConditions.Shapes = nil }},
		{"input_conditions.dtypes", func(c *LaunchContext) { c.InputConditions.DTypes = nil }},
		{"intervals", func(c *LaunchContext) { c.Intervals = nil }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			nilCtx := populatedLaunchContext()
			tc.apply(&nilCtx)

			emptyCtx := populatedLaunchContext()
			tc.apply(&emptyCtx)
			setEmpty(&emptyCtx, tc.name)

			nilDigest := nilCtx.ComputeDigest()
			emptyDigest := emptyCtx.ComputeDigest()
			if nilDigest != emptyDigest {
				t.Fatalf("nil vs empty %s digest differ: nil=%q empty=%q", tc.name, nilDigest, emptyDigest)
			}

			// A marshal/unmarshal round-trip must preserve the digest too.
			emptyCtx.Digest = emptyDigest
			encoded, err := json.Marshal(emptyCtx)
			if err != nil {
				t.Fatal(err)
			}
			decoded, err := DecodeLaunchContext(encoded)
			if err != nil {
				t.Fatal(err)
			}
			if got := decoded.ComputeDigest(); got != emptyDigest {
				t.Fatalf("round-trip %s digest = %q, want %q", tc.name, got, emptyDigest)
			}
		})
	}
}

// setEmpty replaces the named field with a non-nil zero-length value (slice or
// map), the representation the canonicalizer must make indistinguishable from nil.
func setEmpty(c *LaunchContext, name string) {
	switch name {
	case "kernels":
		c.Kernels = []string{}
	case "code_paths":
		c.CodePaths = []string{}
	case "launch_params.argv":
		c.LaunchParams.Argv = []string{}
	case "launch_params.env_keys":
		c.LaunchParams.EnvKeys = []string{}
	case "launch_params.resolved_flags":
		c.LaunchParams.ResolvedFlags = map[string]string{}
	case "input_conditions.shapes":
		c.InputConditions.Shapes = []string{}
	case "input_conditions.dtypes":
		c.InputConditions.DTypes = []string{}
	case "intervals":
		c.Intervals = []Interval{}
	default:
		panic("unknown field " + name)
	}
}

// TestWitnessedNilAndEmptySlices pins the Witnessed() invariant: neither a nil
// slice field nor an empty one reads as witnessed.
func TestWitnessedNilAndEmptySlices(t *testing.T) {
	for _, empty := range []bool{false, true} {
		c := populatedLaunchContext()
		c.Kernels = nil
		c.CodePaths = nil
		c.LaunchParams.Argv = nil
		c.LaunchParams.EnvKeys = nil
		c.InputConditions.Shapes = nil
		c.InputConditions.DTypes = nil
		c.Intervals = nil
		if empty {
			c.Kernels = []string{}
			c.CodePaths = []string{}
			c.LaunchParams.Argv = []string{}
			c.LaunchParams.EnvKeys = []string{}
			c.InputConditions.Shapes = []string{}
			c.InputConditions.DTypes = []string{}
			c.Intervals = []Interval{}
		}
		w := c.Witnessed()
		for _, axis := range []string{
			"kernels", "code_paths", "launch_params.argv", "launch_params.env_keys",
			"input_conditions.shapes", "input_conditions.dtypes", "intervals",
		} {
			if w[axis] {
				t.Fatalf("empty=%v axis %q reported witnessed, want unwitnessed", empty, axis)
			}
		}
	}
}
