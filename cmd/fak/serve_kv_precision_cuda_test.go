package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=5ms lane=default
func TestServeKVPrecisionCUDARealizes16Bit(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	if got := serveKVBackendName("cuda", false); got != kvBackendCUDA {
		t.Fatalf("serveKVBackendName(cuda) = %q, want %q", got, kvBackendCUDA)
	}
	for _, flag := range []string{"fp16", "bf16"} {
		res, err := resolveServeKVPrecision(flag, kvBackendCUDA, false, realizedServeKVPrecisions)
		if err != nil || string(res.Realized) != flag || res.Source != "flag" {
			t.Fatalf("--kv-precision %s on cuda = %+v, %v; want realized from flag", flag, res, err)
		}
	}
	if _, err := resolveServeKVPrecision("q8_0", kvBackendCUDA, false, realizedServeKVPrecisions); !errors.Is(err, ErrKVPrecisionUnrealized) {
		t.Fatalf("q8_0 on cuda: err=%v, want ErrKVPrecisionUnrealized", err)
	}
	res, err := resolveServeKVPrecision("", kvBackendCUDA, false, realizedServeKVPrecisions)
	if err != nil || res.Realized != model.KVPrecisionFP32 || res.Source != "default" {
		t.Fatalf("unset on cuda = %+v, %v; want the exact f32 default", res, err)
	}
	if _, err := resolveServeKVPrecision("bf16", kvBackendVulkan, false, realizedServeKVPrecisions); !errors.Is(err, ErrKVPrecisionUnrealized) {
		t.Fatalf("bf16 on vulkan: err=%v, want ErrKVPrecisionUnrealized", err)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestServeKVPrecision16BitFitTier(t *testing.T) {
	for raw, want := range map[string]compute.KVPrecision{
		"fp16": compute.KVPrecisionF16,
		"bf16": compute.KVPrecisionBF16,
		"q8_0": compute.KVPrecisionQ8,
		"":     compute.KVPrecisionF32,
	} {
		t.Setenv("FAK_UP_KV_PRECISION", raw)
		if got := serveKVPrecision(); got != want {
			t.Fatalf("serveKVPrecision(%q) = %v, want %v", raw, got, want)
		}
	}
	t.Setenv("FAK_UP_KV_PRECISION", "")
	if err := validateServeKVPrecision("bf16"); err != nil {
		t.Fatalf("bf16 must be a valid token: %v", err)
	}
}
