package compute

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestCUDABlockScaledWKVExplicitConfiguration(t *testing.T) {
	t.Setenv("FAK_CUDA_BLOCKSCALED_WKV", "1") // Retired ambient consent is inert.
	var config cudaBlockScaledWKVConfiguration
	for _, enabled := range []bool{false, true, false} {
		config.enabled = enabled
		for _, rows := range []int{0, 127, 128, 512, 4096, 4097} {
			err := config.admit(rows)
			want := enabled && (rows == 128 || rows == 512 || rows == 4096)
			if (err == nil) != want || err != nil && !errors.Is(err, ErrBlockScaledGEMMUnsupported) {
				t.Fatalf("enabled=%v rows=%d error=%v", enabled, rows, err)
			}
		}
	}
	for _, backend := range []Backend{nil, struct{ Backend }{}} {
		for _, enabled := range []bool{false, true} {
			if err := ConfigureCUDABlockScaledWKV(backend, enabled); !errors.Is(err, ErrBlockScaledGEMMUnsupported) {
				t.Fatalf("unsupported backend accepted: %v", err)
			}
		}
	}
}

// This is a host source-contract witness, not CUDA compilation or execution.
// fak-test:runtime fast est=1ms
func TestCUDABlockScaledWKVConsentBoundary(t *testing.T) {
	for _, path := range []string{"cuda_blockscaled_wkv.go", "cuda/blockscaled_wkv.cuh"} {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		src := string(data)
		for _, op := range []string{"support", "upload", "run"} {
			if strings.Contains(src, "fcuda_blockscaled_wkv_"+op+"(") {
				t.Fatalf("%s retains incompatible legacy native symbol for %s", path, op)
			}
			if path == "cuda/blockscaled_wkv.cuh" && strings.Count(src, `extern "C" int fcuda_blockscaled_wkv_`+op+"_configured(") != 2 {
				t.Fatalf("%s must migrate both toolkit branches for %s", path, op)
			}
		}
		if strings.Contains(src, "FAK_CUDA_BLOCKSCALED_WKV") {
			t.Fatalf("%s retains ambient runtime consent", path)
		}
		if path == "cuda_blockscaled_wkv.go" {
			setter := `func (c *cudaBackend) configureCUDABlockScaledWKV(enabled bool) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if c == nil || c != cudaDev {
		return ErrBlockScaledGEMMUnsupported
	}
	if c.capturing {
		return fmt.Errorf("%w: capture is not supported", ErrBlockScaledGEMMUnsupported)
	}
	c.blockScaledWKVConfig.enabled = enabled
	return nil
}`
			if !strings.Contains(src, setter) {
				t.Fatal("setter must lock and refuse nil/foreign root and capture before mutating consent")
			}

			for _, required := range []string{
				"if c == nil || c != cudaDev {",
				"cudaMu.Lock()\n\tdefer cudaMu.Unlock()\n\tif c == nil || c != cudaDev {",
				"c.blockScaledWKVConfig.enabled = enabled",
				"c.blockScaledWKVConfig.admit(rows)",
				"C.fcuda_blockscaled_wkv_support_configured(c.blockScaledWKVConsent(),",
				"C.fcuda_blockscaled_wkv_upload_configured(c.blockScaledWKVConsent(),",
				"C.fcuda_blockscaled_wkv_run_configured(c.blockScaledWKVConsent(),",
			} {
				if !strings.Contains(src, required) {
					t.Fatalf("%s missing %q", path, required)
				}
			}
		} else if !strings.Contains(src, "if (enabled != 1) return BSW_UNSUPPORTED;") || strings.Count(src, "capability(enabled,") != 3 {
			t.Fatal("native probe/upload/run must independently require explicit consent")
		}
	}
}

// A public-dispatch test double only, never a CUDA execution substitute.
type blockScaledWKVDispatchBackend struct {
	Backend
	seen  bool
	calls int
	err   error
}

func (b *blockScaledWKVDispatchBackend) configureCUDABlockScaledWKV(enabled bool) error {
	b.seen = enabled
	b.calls++
	return b.err
}

// fak-test:runtime fast est=1ms
func TestCUDABlockScaledWKVConfigurationDispatch(t *testing.T) {
	b := &blockScaledWKVDispatchBackend{}
	if err := ConfigureCUDABlockScaledWKV(b, true); err != nil || !b.seen || b.calls != 1 {
		t.Fatalf("explicit ON dispatch: backend=%+v error=%v", b, err)
	}
	b.err = errors.New("configuration refused")
	if err := ConfigureCUDABlockScaledWKV(b, false); err != b.err || b.seen || b.calls != 2 {
		t.Fatalf("explicit OFF refusal propagation: backend=%+v error=%v", b, err)
	}
}
