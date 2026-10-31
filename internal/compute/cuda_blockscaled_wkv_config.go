package compute

import (
	"errors"
	"fmt"
)

// ErrBlockScaledGEMMUnsupported is a capability refusal, not a request to silently
// fall back after execution. No serving caller selects this experimental route.
var ErrBlockScaledGEMMUnsupported = errors.New("experimental SM120 blockscaled WKV GEMM unsupported")

type cudaBlockScaledWKVConfigurer interface {
	configureCUDABlockScaledWKV(enabled bool) error
}

// ConfigureCUDABlockScaledWKV explicitly selects the experimental SM120 WKV
// baseline. It defaults OFF for each backend lifetime; environment variables do
// not enable it. Only the registered CUDA backend accepts configuration, and an
// active capture refuses changes. Unsupported backends/builds return a capability
// refusal rather than silently accepting an inert setting.
//
// Configuration is serialized with operations on the shared CUDA stream. This
// selects consent only: each probe, upload and run still checks hardware, shape,
// capture and liveness independently. It neither qualifies hardware nor enables
// serving dispatch. Disabling prevents new operations but does not strand valid
// weight owners: Release remains available under its existing lifetime guards.
// Re-enabling the same backend permits reuse of still-live weights; it never
// clears faults, changes ownership epochs or transfers consent to a new backend.
func ConfigureCUDABlockScaledWKV(backend Backend, enabled bool) error {
	configured, ok := backend.(cudaBlockScaledWKVConfigurer)
	if !ok {
		return ErrBlockScaledGEMMUnsupported
	}
	return configured.configureCUDABlockScaledWKV(enabled)
}

// CUDA readers and writers hold cudaMu. Keeping policy separate makes default-off
// and exact geometry testable on hosts without inventing a CUDA backend.
type cudaBlockScaledWKVConfiguration struct{ enabled bool }

func (s cudaBlockScaledWKVConfiguration) admit(rows int) error {
	if !s.enabled || (rows != 128 && rows != 512 && rows != 4096) {
		return fmt.Errorf("%w: require explicit blockscaled WKV configuration and P=128,512,4096", ErrBlockScaledGEMMUnsupported)
	}
	return nil
}
