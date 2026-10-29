package main

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// ErrKVPrecisionUnrealized is the closed refusal for a KV tier the serving backend
// cannot hold where attention reads it. Serving it anyway would charge the fit
// estimator the requested density while the device keeps f32 rows, or drop device KV.
var ErrKVPrecisionUnrealized = errors.New("kv precision not realized by the serving backend")

const (
	kvBackendHost   = "host"
	kvBackendMetal  = "metal"
	kvBackendVulkan = "vulkan"
	kvBackendCUDA   = "cuda"
)

// haloKVPrecisionPreference is the Halo profile's default order: densest exact-enough
// tier first. q8_0 joins the device order once a quantized device KVStore is qualified.
var haloKVPrecisionPreference = []model.KVPrecision{model.KVPrecisionFP16, model.KVPrecisionQ8_0, model.KVPrecisionFP32}

// realizedServeKVPrecisions is the tier set each serving backend actually stores.
// Host KVCache realizes f32 and q8_0 (fp16 is declared, unrealized: kvcache_q8.go).
// Metal's q8_0 declines the device decode mirror (prepareDecodeMirror), so only f32
// keeps device KV. The CUDA HAL store holds fp16/bf16 attended K/V (compute/cuda_kv.go,
// f32 pre-RoPE shadow); the planner falls those back to f32 for a forward path that
// bypasses the typed store. Other HAL stores (vulkan, cpu-ref) hold only f32.
func realizedServeKVPrecisions(backend string) []model.KVPrecision {
	switch backend {
	case kvBackendHost:
		return []model.KVPrecision{model.KVPrecisionFP32, model.KVPrecisionQ8_0}
	case kvBackendCUDA:
		return []model.KVPrecision{model.KVPrecisionFP32, model.KVPrecisionFP16, model.KVPrecisionBF16}
	}
	return []model.KVPrecision{model.KVPrecisionFP32}
}

type serveKVPrecisionResolution struct {
	Requested model.KVPrecision
	Realized  model.KVPrecision
	Backend   string
	Source    string
	Note      string
}

// serveKVBackendName names the KV storage owner for a serve run.
func serveKVBackendName(deviceBackend string, useMetal bool) string {
	switch {
	case useMetal:
		return kvBackendMetal
	case strings.TrimSpace(deviceBackend) == "":
		return kvBackendHost
	}
	return strings.ToLower(strings.TrimSpace(deviceBackend))
}

// resolveServeKVPrecision picks the KV tier a serve run realizes. An explicit tier
// (flag, else FAK_UP_KV_PRECISION) the backend cannot store refuses. Unset on Halo
// takes the first preferred tier the backend realizes and says why when that is f32.
// Unset elsewhere keeps the historical f32 zero value.
func resolveServeKVPrecision(flagValue, backend string, halo bool, realized func(string) []model.KVPrecision) (serveKVPrecisionResolution, error) {
	res := serveKVPrecisionResolution{Backend: backend}
	raw, source := strings.TrimSpace(flagValue), "flag"
	if raw == "" {
		raw, source = strings.TrimSpace(os.Getenv("FAK_UP_KV_PRECISION")), "env"
	}
	have := realized(backend)
	if raw == "" {
		if !halo {
			res.Realized, res.Source = model.KVPrecisionFP32, "default"
			return res, nil
		}
		res.Source = "halo-default"
		for _, want := range haloKVPrecisionPreference {
			if kvPrecisionIn(want, have) {
				res.Realized = want
				break
			}
		}
		if res.Realized == "" {
			return res, fmt.Errorf("%w: backend %q realizes none of %v", ErrKVPrecisionUnrealized, backend, haloKVPrecisionPreference)
		}
		if res.Realized != haloKVPrecisionPreference[0] {
			res.Note = fmt.Sprintf("halo default prefers %s; backend %q stores only %v", haloKVPrecisionPreference[0], backend, have)
		}
		return res, nil
	}
	prec, err := model.ParseKVPrecision(raw)
	if err != nil {
		return res, err
	}
	res.Requested, res.Source = prec, source
	if !kvPrecisionIn(prec, have) {
		return res, fmt.Errorf("%w: --kv-precision %s on backend %q (realized: %v)", ErrKVPrecisionUnrealized, prec, backend, have)
	}
	res.Realized = prec
	return res, nil
}

func kvPrecisionIn(p model.KVPrecision, set []model.KVPrecision) bool {
	for _, s := range set {
		if s == p {
			return true
		}
	}
	return false
}
