package devcmd

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestHostSetupBenchShapeNormalize(t *testing.T) {
	s, err := (HostSetupBenchShape{Name: "x", Layers: 4}).normalize()
	if err != nil {
		t.Fatalf("normalize: %v", err)
	}
	if s.Block != 32 || s.Out != 1024 || s.In != 1024 {
		t.Fatalf("defaults not filled: %+v", s)
	}
	for _, bad := range []HostSetupBenchShape{
		{Layers: 0, Out: 1, In: 32},
		{Layers: 1, Out: -1, In: 32},
		{Layers: 1, Out: 1, In: -1},
		{Layers: 1, Out: 1, In: 32, Block: 7},   // not divisible by 4
		{Layers: 1, Out: 1, In: 100, Block: 32}, // in not a multiple of block
		{Layers: 1, Out: 1, In: 16, Block: 32},  // in < block
	} {
		if _, err := bad.normalize(); err == nil {
			t.Errorf("normalize(%+v) must fail", bad)
		}
	}
}

// runHostSetupBenchJSON runs the verb with --json and decodes the payload.
func runHostSetupBenchJSON(t *testing.T, args ...string) (*HostSetupBenchResult, int, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := RunHostSetupBench(&out, &errb, append(append([]string{}, args...), "--json"))
	var r HostSetupBenchResult
	if code == 0 || code == 1 {
		if err := json.Unmarshal(out.Bytes(), &r); err != nil {
			t.Fatalf("decode result: %v\nstdout=%s stderr=%s", err, out.String(), errb.String())
		}
	}
	return &r, code, errb.String()
}

// TestHostSetupBenchRunCPURefArithmetic runs the full bench on the cpu-ref
// fallback and checks the report's internal arithmetic: kernel = launch arm
// minus setup arm, host_fraction = setup/total, and the verdict matches the
// threshold reading of that fraction.
func TestHostSetupBenchRunCPURefArithmetic(t *testing.T) {
	r, code, errOut := runHostSetupBenchJSON(t, "--preset", "micro", "--warmup", "1", "--iters", "4")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr=%q)", code, errOut)
	}
	if r.Schema != hostSetupBenchSchema {
		t.Fatalf("schema = %q", r.Schema)
	}
	if r.BackendMeasured != "cpu-ref" || r.Verification != verificationSWVerified {
		t.Fatalf("measured backend/verification = %s/%s, want cpu-ref/sw-verified", r.BackendMeasured, r.Verification)
	}
	if r.Shape.Layers != 4 || r.Shape.Out != 128 || r.Shape.In != 128 {
		t.Fatalf("shape = %+v, want the micro preset", r.Shape)
	}
	total := r.SetupPlusLaunch.MeanUsPerLayer
	if total <= 0 {
		t.Fatalf("setup+launch mean = %f us/layer, want > 0", total)
	}
	if got, want := r.KernelUsPerLayer, total-r.SetupOnly.MeanUsPerLayer; math.Abs(got-want) > 1e-9 {
		t.Fatalf("kernel_us_per_layer = %f, want total-setup = %f", got, want)
	}
	if r.KernelUsPerLayer < 0 {
		t.Fatalf("kernel_us_per_layer = %f, want >= 0", r.KernelUsPerLayer)
	}
	if got, want := r.HostFraction, r.SetupOnly.MeanUsPerLayer/total; math.Abs(got-want) > 1e-9 {
		t.Fatalf("host_fraction = %f, want setup/total = %f", got, want)
	}
	wantVerdict := hostSetupVerdictBalanced
	if r.HostFraction > r.Threshold {
		wantVerdict = hostSetupVerdictDominant
	}
	if r.Verdict != wantVerdict {
		t.Fatalf("verdict = %q, want %q for fraction %f vs threshold %f", r.Verdict, wantVerdict, r.HostFraction, r.Threshold)
	}
}

// TestHostSetupBenchFlagOverridesApply pins that the numeric shape flags
// actually override the preset values (the parser once marked them explicit
// without applying them).
func TestHostSetupBenchFlagOverridesApply(t *testing.T) {
	r, code, errOut := runHostSetupBenchJSON(t, "--layers", "3", "--out", "64", "--in", "64", "--block", "32", "--warmup", "1", "--iters", "2")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr=%q)", code, errOut)
	}
	if r.Shape.Layers != 3 || r.Shape.Out != 64 || r.Shape.In != 64 || r.Shape.Block != 32 {
		t.Fatalf("shape = %+v, want layers=3 out=64 in=64 block=32", r.Shape)
	}
}

// TestHostSetupBenchGateExitsOneOnDominant drives the regression gate
// deterministically: at a near-zero threshold any nonzero per-layer setup cost
// is LAUNCH_SETUP_DOMINANT, so --gate must exit 1.
func TestHostSetupBenchGateExitsOneOnDominant(t *testing.T) {
	r, code, errOut := runHostSetupBenchJSON(t, "--preset", "micro", "--warmup", "1", "--iters", "2", "--threshold", "0.001", "--gate")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 under --gate (stderr=%q)", code, errOut)
	}
	if r.Verdict != hostSetupVerdictDominant {
		t.Fatalf("verdict = %q, want %q", r.Verdict, hostSetupVerdictDominant)
	}
	if !r.Gate {
		t.Fatal("gate flag not recorded in the result")
	}
}

// TestHostSetupBenchTextLabelSWVerified pins the [SW-VERIFIED] label on the
// text report whenever the measured path is the CPU reference fallback.
func TestHostSetupBenchTextLabelSWVerified(t *testing.T) {
	var out, errb bytes.Buffer
	code := RunHostSetupBench(&out, &errb, []string{"--preset", "micro", "--warmup", "1", "--iters", "2"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr=%q)", code, errb.String())
	}
	for _, want := range []string{"[SW-VERIFIED]", "setup_us_per_layer", "kernel_us_per_layer", "host_fraction", "verdict"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("text report missing %q:\n%s", want, out.String())
		}
	}
}

// TestHostSetupBenchUnregisteredBackendFallsBackToCPU exercises the wired-but-
// skipped CUDA path: requesting cuda in a build where it is not registered must
// report the skip with a reason and measure on cpu-ref instead of failing.
func TestHostSetupBenchUnregisteredBackendFallsBackToCPU(t *testing.T) {
	if _, ok := compute.Lookup("cuda"); ok {
		t.Skip("cuda is registered in this build; the fallback path is not exercisable")
	}
	r, code, errOut := runHostSetupBenchJSON(t, "--backend", "cuda", "--preset", "micro", "--warmup", "1", "--iters", "2")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr=%q)", code, errOut)
	}
	if r.BackendRequested != "cuda" || r.BackendMeasured != "cpu-ref" {
		t.Fatalf("requested/measured = %s/%s, want cuda/cpu-ref", r.BackendRequested, r.BackendMeasured)
	}
	found := false
	for _, s := range r.Skipped {
		if s.Backend == "cuda" && strings.Contains(s.Reason, "not registered") {
			found = true
		}
	}
	if !found {
		t.Fatalf("skipped entries = %+v, want a cuda 'not registered' row", r.Skipped)
	}
}

func TestHostSetupBenchUsageErrors(t *testing.T) {
	cases := [][]string{
		{"--preset", "nope"},
		{"--threshold", "0"},
		{"--threshold", "1.5"},
		{"--iters", "0"},
		{"--warmup", "-1"},
		{"--layers", "0"},
		{"--bogus"},
		{"--model-shape", "{bad json"},
		{"--block", "7"}, // quant block not divisible by 4
		{"--in", "100"},  // in not a multiple of the default block
		{"--layers", "abc"},
	}
	for _, args := range cases {
		var out, errb bytes.Buffer
		if code := RunHostSetupBench(&out, &errb, args); code != 2 {
			t.Errorf("args=%v exit=%d, want 2 (stderr=%q)", args, code, errb.String())
		}
	}
}
