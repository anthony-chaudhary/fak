package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=10ms lane=default
func TestServeKVPrecisionDefault(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	fp16Device := func(string) []model.KVPrecision {
		return []model.KVPrecision{model.KVPrecisionFP32, model.KVPrecisionFP16}
	}

	res, err := resolveServeKVPrecision("", kvBackendVulkan, true, fp16Device)
	if err != nil || res.Realized != model.KVPrecisionFP16 || res.Source != "halo-default" || res.Note != "" {
		t.Fatalf("halo profile on an fp16-capable backend = %+v, %v; want fp16 halo-default", res, err)
	}

	res, err = resolveServeKVPrecision("", kvBackendHost, true, realizedServeKVPrecisions)
	if err != nil || res.Realized != model.KVPrecisionQ8_0 {
		t.Fatalf("halo profile on host KV = %+v, %v; want q8_0", res, err)
	}

	res, err = resolveServeKVPrecision("", kvBackendVulkan, true, realizedServeKVPrecisions)
	if err != nil || res.Realized != model.KVPrecisionFP32 || res.Note == "" {
		t.Fatalf("halo profile on f32-only vulkan KV = %+v, %v; want f32 with a recorded reason", res, err)
	}

	res, err = resolveServeKVPrecision("", kvBackendVulkan, false, realizedServeKVPrecisions)
	if err != nil || res.Realized != model.KVPrecisionFP32 || res.Source != "default" {
		t.Fatalf("non-halo unset = %+v, %v; want f32 default", res, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestServeKVPrecisionDefaultRefusesUnrealized(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	for _, tc := range []struct{ flag, backend string }{
		{"q8_0", kvBackendVulkan},
		{"fp16", kvBackendVulkan},
		{"q8_0", kvBackendMetal},
		{"fp16", kvBackendHost},
	} {
		if _, err := resolveServeKVPrecision(tc.flag, tc.backend, true, realizedServeKVPrecisions); !errors.Is(err, ErrKVPrecisionUnrealized) {
			t.Fatalf("--kv-precision %s on %s: err=%v, want ErrKVPrecisionUnrealized", tc.flag, tc.backend, err)
		}
	}

	t.Setenv("FAK_UP_KV_PRECISION", "q8_0")
	if _, err := resolveServeKVPrecision("", kvBackendVulkan, true, realizedServeKVPrecisions); !errors.Is(err, ErrKVPrecisionUnrealized) {
		t.Fatalf("env q8_0 on vulkan: err=%v, want ErrKVPrecisionUnrealized", err)
	}
	res, err := resolveServeKVPrecision("", kvBackendHost, false, realizedServeKVPrecisions)
	if err != nil || res.Realized != model.KVPrecisionQ8_0 || res.Source != "env" {
		t.Fatalf("env q8_0 on host = %+v, %v; want q8_0 from env", res, err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeKVPrecisionDefaultBackendName(t *testing.T) {
	for _, tc := range []struct {
		device string
		metal  bool
		want   string
	}{
		{"", false, kvBackendHost},
		{"", true, kvBackendMetal},
		{"Vulkan", false, kvBackendVulkan},
	} {
		if got := serveKVBackendName(tc.device, tc.metal); got != tc.want {
			t.Fatalf("serveKVBackendName(%q, %v) = %q, want %q", tc.device, tc.metal, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeKVPrecisionDefaultPublishedBeforeLoad(t *testing.T) {
	t.Setenv("FAK_UP_KV_PRECISION", "")
	flagValue := ""
	res, err := resolveServeKVPrecisionForLoad(&flagValue, kvBackendHost, true)
	if err != nil || res.Realized != model.KVPrecisionQ8_0 {
		t.Fatalf("Halo host default = %+v, %v; want q8_0", res, err)
	}
	if flagValue != string(res.Realized) || serveKVPrecision() != compute.KVPrecisionQ8 {
		t.Fatalf("flag=%q sizing=%v; want q8_0 flag and Q8 sizing before model load", flagValue, serveKVPrecision())
	}
}
