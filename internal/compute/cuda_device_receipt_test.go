package compute

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// TestCUDADeviceReceipt witnesses the typed sm_120 (RTX 5090) device receipt on a host with
// NO GPU: the receipt carries the captured identity (name, VRAM, driver version, tier), the
// coverage verdict wired from ClassifyCUDAArch's rule, and — the point of the ticket — the
// per-arm admissions reported as EXPLICIT refusals with reasons, never silent downgrades.
func TestCUDADeviceReceipt(t *testing.T) {
	const vram32GiB = int64(32) << 30
	const capturedDriver = "12.9" // fixture fact the receipt must echo verbatim
	r := BuildCUDADeviceReceipt(CUDADeviceInitFacts{
		DeviceName:    "NVIDIA GeForce RTX 5090",
		Major:         12,
		Minor:         0,
		TotalMemBytes: vram32GiB,
		DriverVersion: capturedDriver,
	}, CUDABuildArches{})

	if r.Backend != "cuda" {
		t.Fatalf("receipt backend = %q, want cuda", r.Backend)
	}
	if r.DeviceName != "NVIDIA GeForce RTX 5090" {
		t.Errorf("receipt device name = %q, want the captured name", r.DeviceName)
	}
	if r.TotalMemBytes != vram32GiB {
		t.Errorf("receipt total_mem_bytes = %d, want %d", r.TotalMemBytes, vram32GiB)
	}
	if r.DriverVersion != capturedDriver {
		t.Errorf("receipt driver_version = %q, want the captured fixture fact %q", r.DriverVersion, capturedDriver)
	}
	if r.Tier != "sm_120" {
		t.Errorf("receipt tier = %q, want sm_120", r.Tier)
	}
	if r.Coverage != CUDANative {
		t.Errorf("sm_120 coverage = %v, want native (the shipped full build embeds an sm_120 cubin)", r.Coverage)
	}

	// Both tracked arms must be present and explicitly refused-with-reason on sm_120.
	for _, arm := range []CUDADeviceArm{CUDADeviceArmMxFP8Cast, CUDADeviceArmQ4KPrefillFP8LT} {
		ad, ok := r.Arms[arm]
		if !ok {
			t.Fatalf("receipt has no admission for %s", arm)
		}
		if ad.Enabled {
			t.Errorf("%s enabled on sm_120, want explicit refusal", arm)
		}
		if ad.Reason == "" {
			t.Errorf("%s refused on sm_120 without a reason (silent downgrade)", arm)
		}
	}

	// Positive controls: the same builder admits each arm exactly where its kernel allows it.
	r100 := BuildCUDADeviceReceipt(CUDADeviceInitFacts{Major: 10, Minor: 0}, CUDABuildArches{})
	if !r100.Arms[CUDADeviceArmMxFP8Cast].Enabled {
		t.Errorf("mxfp8-cast not enabled on sm_100: %+v", r100.Arms[CUDADeviceArmMxFP8Cast])
	}
	if r100.Arms[CUDADeviceArmQ4KPrefillFP8LT].Enabled {
		t.Errorf("fp8lt unexpectedly enabled on sm_100 (admitted interval is sm_89..sm_90)")
	}
	r89 := BuildCUDADeviceReceipt(CUDADeviceInitFacts{Major: 8, Minor: 9}, CUDABuildArches{})
	if !r89.Arms[CUDADeviceArmQ4KPrefillFP8LT].Enabled {
		t.Errorf("fp8lt not enabled on sm_89: %+v", r89.Arms[CUDADeviceArmQ4KPrefillFP8LT])
	}
	if r89.Arms[CUDADeviceArmMxFP8Cast].Enabled {
		t.Errorf("mxfp8-cast unexpectedly enabled on sm_89")
	}

	// Unknown driver version degrades gracefully: the receipt stays complete.
	rUnknown := BuildCUDADeviceReceipt(CUDADeviceInitFacts{Major: 12, Minor: 0, TotalMemBytes: vram32GiB}, CUDABuildArches{})
	if rUnknown.DriverVersion != "" {
		t.Errorf("unknown driver rendered as %q, want empty", rUnknown.DriverVersion)
	}
	if rUnknown.Coverage != CUDANative || len(rUnknown.Arms) != 2 {
		t.Errorf("degraded receipt lost content: %+v", rUnknown)
	}

	// Single-arch build seam: an sm_89-only build on an sm_120 card is uncovered, and the
	// receipt says so instead of a confident bare tier.
	rSingle := BuildCUDADeviceReceipt(CUDADeviceInitFacts{Major: 12, Minor: 0}, CUDABuildArches{Cubins: []cudaCC{{8, 9}}})
	if rSingle.Coverage != CUDAUncovered {
		t.Errorf("single-arch sm_89 build on sm_120 = %v, want uncovered", rSingle.Coverage)
	}
}

// TestCUDAArchArmAdmissions pins the per-arch admission table for both tracked arm families:
// every refusal carries a reason, and unknown arches fail closed.
func TestCUDAArchArmAdmissions(t *testing.T) {
	cases := []struct {
		major, minor int
		mxfp8        bool
		fp8lt        bool
	}{
		{10, 0, true, false},  // B200: mxfp8-cast admitted; fp8lt interval is sm_89..sm_90
		{10, 3, true, false},  // B300
		{12, 0, false, false}, // RTX 5090: both refused with reasons
		{12, 1, false, false}, // consumer Blackwell forward-minor: neither arm
		{9, 0, false, true},   // H100: fp8lt admitted
		{8, 9, false, true},   // Ada: fp8lt admitted
		{8, 0, false, false},  // Ampere: below the fp8lt floor
		{11, 0, false, false}, // hypothetical 11.x: neither arm
		{0, 0, false, false},  // unknown arch fails closed
	}
	for _, c := range cases {
		got := CUDAArchArmAdmissions(c.major, c.minor)
		if got[CUDADeviceArmMxFP8Cast].Enabled != c.mxfp8 {
			t.Errorf("mxfp8-cast on sm_%d%d = enabled=%t reason=%q, want enabled=%t",
				c.major, c.minor, got[CUDADeviceArmMxFP8Cast].Enabled, got[CUDADeviceArmMxFP8Cast].Reason, c.mxfp8)
		}
		if got[CUDADeviceArmQ4KPrefillFP8LT].Enabled != c.fp8lt {
			t.Errorf("fp8lt on sm_%d%d = enabled=%t reason=%q, want enabled=%t",
				c.major, c.minor, got[CUDADeviceArmQ4KPrefillFP8LT].Enabled, got[CUDADeviceArmQ4KPrefillFP8LT].Reason, c.fp8lt)
		}
		for arm, ad := range got {
			if !ad.Enabled && ad.Reason == "" {
				t.Errorf("%s refused on sm_%d%d without a reason (silent downgrade)", arm, c.major, c.minor)
			}
		}
	}
}

// TestCUDAMxFP8CastGateMatchesKernel is the anti-drift guard between the Go-side arch gate
// constants and the C++ kernel they mirror: fcuda_mxfp8_cast_supported in cuda/mxfp8_cast.cuh
// must keep admitting exactly the arches CUDAArchArmAdmissions admit. If the kernel gate
// changes shape, update the kernel and cudaMxFP8CastMajor/cudaMxFP8CastMinors together.
func TestCUDAMxFP8CastGateMatchesKernel(t *testing.T) {
	raw, err := os.ReadFile("cuda/mxfp8_cast.cuh")
	if err != nil {
		t.Fatalf("read cuda/mxfp8_cast.cuh: %v", err)
	}
	re := regexp.MustCompile(`p\.major\s*==\s*(\d+)\s*&&\s*\(\s*p\.minor\s*==\s*(\d+)\s*\|\|\s*p\.minor\s*==\s*(\d+)\s*\)`)
	m := re.FindSubmatch(raw)
	if m == nil {
		t.Fatalf("could not find the mxfp8-cast arch gate in cuda/mxfp8_cast.cuh; if the gate changed shape, update the kernel and the Go constants together")
	}
	if len(cudaMxFP8CastMinors) != 2 {
		t.Fatalf("cudaMxFP8CastMinors has %d entries, the drift guard expects exactly two", len(cudaMxFP8CastMinors))
	}
	major, _ := strconv.Atoi(string(m[1]))
	minorA, _ := strconv.Atoi(string(m[2]))
	minorB, _ := strconv.Atoi(string(m[3]))
	if major != cudaMxFP8CastMajor || minorA != cudaMxFP8CastMinors[0] || minorB != cudaMxFP8CastMinors[1] {
		t.Fatalf("kernel gate sm_%d{,%d,%d} drifted from Go constants sm_%d{,%v}",
			major, minorA, minorB, cudaMxFP8CastMajor, cudaMxFP8CastMinors)
	}
}

// TestFormatCUDADriverVersion pins the CUDA_VERSION integer -> "major.minor[.patch]" rendering
// the receipt uses for cuDriverGetVersion results.
func TestFormatCUDADriverVersion(t *testing.T) {
	for in, want := range map[int]string{
		0: "", 999: "", 12090: "12.9", 13000: "13.0", 12091: "12.9.1", 8004: "8.0.4",
	} {
		if got := formatCUDADriverVersion(in); got != want {
			t.Errorf("formatCUDADriverVersion(%d) = %q, want %q", in, got, want)
		}
	}
}

// TestCUDADeviceReceiptProviderSeam witnesses the consumer path: CUDADeviceReceiptOf reads
// the receipt off a Backend through the optional provider interface, and reports absence
// (not a zero-value lie) for backends without one.
func TestCUDADeviceReceiptProviderSeam(t *testing.T) {
	fake := &receiptFakeBackend{receipt: BuildCUDADeviceReceipt(CUDADeviceInitFacts{Major: 12, Minor: 0}, CUDABuildArches{})}
	got, ok := CUDADeviceReceiptOf(fake)
	if !ok {
		t.Fatal("provider-backed backend did not report a receipt")
	}
	if got.Tier != "sm_120" || got.Coverage != CUDANative {
		t.Errorf("receipt through the seam = %+v, want sm_120 native", got)
	}
	if _, ok := CUDADeviceReceiptOf(nil); ok {
		t.Error("nil backend reported a receipt, want absent")
	}
}

// receiptFakeBackend stands in for a device backend behind the provider seam: it embeds the
// (nil) Backend interface so it satisfies the parameter type without implementing the op
// surface, which the seam test never calls.
type receiptFakeBackend struct {
	Backend
	receipt CUDADeviceReceipt
}

func (f *receiptFakeBackend) CUDADeviceReceipt() CUDADeviceReceipt { return f.receipt }
