package ctxmmu

import (
	"fmt"
	"reflect"
	"testing"
)

// meminfoText renders a minimal /proc/meminfo sample with the given
// MemAvailable value in kB.
func meminfoText(memAvailableKiB int64) string {
	return fmt.Sprintf("MemTotal:       32768000 kB\nMemFree:         1024000 kB\nMemAvailable:   %d kB\nBuffers:          512000 kB\nCached:          2048000 kB\n", memAvailableKiB)
}

// igpuSizingHarness pins the detector side and installs an injected meminfo
// reader so the sizing path never depends on a real /proc/meminfo.
type igpuSizingHarness struct {
	t          *testing.T
	integrated bool
	probed     bool
}

func (h *igpuSizingHarness) install(memAvailableKiB int64) {
	oldDetector := integratedDetector
	oldReader := memInfoReader
	integratedDetector = func() bool { return h.integrated }
	memInfoReader = func() (string, error) {
		h.probed = true
		return meminfoText(memAvailableKiB), nil
	}
	h.t.Cleanup(func() {
		integratedDetector = oldDetector
		memInfoReader = oldReader
	})
}

func TestParseMemAvailable(t *testing.T) {
	const kiB = int64(1024)
	cases := []struct {
		name string
		text string
		want int64
		ok   bool
	}{
		{name: "present", text: "MemTotal: 32768000 kB\nMemAvailable: 16000000 kB\n", want: 16000000 * kiB, ok: true},
		{name: "absent", text: "MemTotal: 32768000 kB\n", ok: false},
		{name: "unparseable value", text: "MemAvailable: unknown kB\n", ok: false},
		{name: "empty value", text: "MemAvailable:\n", ok: false},
		{name: "overflow guard", text: "MemAvailable: 18446744073709551615 kB\n", ok: false},
		{name: "padded key", text: "  MemAvailable:   512 kB\n", want: 512 * kiB, ok: true},
		{name: "similar key ignored", text: "MemAvailableX: 512 kB\nMemAvailable: 256 kB\n", want: 256 * kiB, ok: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ParseMemAvailable(tc.text)
			if ok != tc.ok || got != tc.want {
				t.Fatalf("ParseMemAvailable(%q) = (%d, %v), want (%d, %v)", tc.text, got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestDetectIntegratedDeviceOverride(t *testing.T) {
	cases := []struct {
		env  string
		want bool
	}{
		{env: "1", want: true},
		{env: "true", want: true},
		{env: "on", want: true},
		{env: "yes", want: true},
		{env: "0", want: false},
		{env: "false", want: false},
		{env: "off", want: false},
		{env: "no", want: false},
	}
	for _, tc := range cases {
		t.Run(tc.env, func(t *testing.T) {
			t.Setenv(EnvHalogenIGPU, tc.env)
			if got := detectIntegratedDevice(); got != tc.want {
				t.Fatalf("detectIntegratedDevice with %s=%s = %v, want %v", EnvHalogenIGPU, tc.env, got, tc.want)
			}
		})
	}
}

func TestIGPUMemAvailableKVSizing(t *testing.T) {
	const (
		miB    = int64(1) << 20
		bpt    = int64(1024) // 1 KiB per token
		total  = int64(8) * miB
		maxTok = 1_000_000 // nominal ceiling above the byte limit
	)

	// Invariants (a) and (b): the estimate never lowers raw cudaFree and is
	// capped at total for every physically possible input (cudaFree <= total).
	estCases := []struct {
		name     string
		cudaFree int64
		memAvail int64
		total    int64
		want     int64
	}{
		{name: "memavail below cudafree takes cudafree", cudaFree: 4 * miB, memAvail: 2 * miB, total: 8 * miB, want: 4 * miB},
		{name: "memavail above cudafree takes min of memavail and total", cudaFree: 2 * miB, memAvail: 6 * miB, total: 8 * miB, want: 6 * miB},
		{name: "total cap binds when memavail exceeds total", cudaFree: 2 * miB, memAvail: 64 * miB, total: 8 * miB, want: 8 * miB},
		{name: "zero memavail keeps cudafree", cudaFree: 3 * miB, memAvail: 0, total: 8 * miB, want: 3 * miB},
		{name: "zero cudafree takes capped memavail", cudaFree: 0, memAvail: 5 * miB, total: 8 * miB, want: 5 * miB},
		{name: "equal values", cudaFree: 4 * miB, memAvail: 4 * miB, total: 8 * miB, want: 4 * miB},
	}
	for _, tc := range estCases {
		t.Run("estimate/"+tc.name, func(t *testing.T) {
			got := EffectiveFreeForIntegratedDevice(tc.cudaFree, tc.memAvail, tc.total)
			if got < tc.cudaFree {
				t.Fatalf("estimate %d lowered raw cudaFree %d", got, tc.cudaFree)
			}
			if got > tc.total {
				t.Fatalf("estimate %d exceeded total %d", got, tc.total)
			}
			if got != tc.want {
				t.Fatalf("estimate = %d, want %d", got, tc.want)
			}
		})
	}

	// End-to-end init on a pinned-integrated device: the pool fits the
	// effective estimate through FitPoolToAvailableMemory.
	scenarios := []struct {
		name          string
		cudaFree      int64
		memAvailKiB   int64
		wantTotal     int64
		wantMaxTokens int
	}{
		{name: "memavail below cudafree keeps cudafree", cudaFree: 4 * miB, memAvailKiB: 2048, wantTotal: 4 * miB, wantMaxTokens: 4096},
		{name: "memavail above cudafree takes min of memavail and total", cudaFree: 2 * miB, memAvailKiB: 6144, wantTotal: 6 * miB, wantMaxTokens: 6144},
		{name: "reclaimable cache large raises estimate above raw free", cudaFree: 1 * miB, memAvailKiB: 7168, wantTotal: 7 * miB, wantMaxTokens: 7168},
		{name: "total cap binds when memavail exceeds total", cudaFree: 2 * miB, memAvailKiB: 16384, wantTotal: 8 * miB, wantMaxTokens: 8192},
	}
	for _, sc := range scenarios {
		t.Run("init/"+sc.name, func(t *testing.T) {
			h := &igpuSizingHarness{t: t, integrated: true}
			h.install(sc.memAvailKiB)
			pool, err := NewSharedTokenPoolFromDevice(maxTok, total, bpt, 1.0, sc.cudaFree)
			if err != nil {
				t.Fatalf("NewSharedTokenPoolFromDevice: %v", err)
			}
			if !h.probed {
				t.Fatalf("kernel probe was not consulted on the integrated path")
			}
			if pool.TotalGTTBytes() != sc.wantTotal {
				t.Fatalf("TotalGTTBytes = %d, want %d", pool.TotalGTTBytes(), sc.wantTotal)
			}
			if pool.MaxTokens() != sc.wantMaxTokens {
				t.Fatalf("MaxTokens = %d, want %d", pool.MaxTokens(), sc.wantMaxTokens)
			}
			if pool.EffectiveMaxTokens() != sc.wantMaxTokens {
				t.Fatalf("EffectiveMaxTokens = %d, want %d", pool.EffectiveMaxTokens(), sc.wantMaxTokens)
			}
		})
	}

	// Invariant (c): on a pinned-discrete device the init path is
	// byte-identical to the unfitted baseline and the kernel probe is never
	// consulted.
	t.Run("init/discrete path byte identical baseline", func(t *testing.T) {
		baseline, err := NewSharedTokenPoolWithCapacity(maxTok, total, bpt, 1.0)
		if err != nil {
			t.Fatalf("NewSharedTokenPoolWithCapacity: %v", err)
		}
		h := &igpuSizingHarness{t: t, integrated: false}
		h.install(0)
		fitted, err := NewSharedTokenPoolFromDevice(maxTok, total, bpt, 1.0, 4*miB)
		if err != nil {
			t.Fatalf("NewSharedTokenPoolFromDevice: %v", err)
		}
		if h.probed {
			t.Fatalf("kernel probe consulted on the discrete path")
		}
		if !reflect.DeepEqual(baseline, fitted) {
			t.Fatalf("discrete path diverged from baseline:\nbasis = %#v\nfitted = %#v", baseline, fitted)
		}
	})
}
