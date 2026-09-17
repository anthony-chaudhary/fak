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
