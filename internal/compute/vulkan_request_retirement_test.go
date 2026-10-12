package compute

import (
	"errors"
	"fmt"
	"testing"
)

type vulkanRetirementFaultBackend struct {
	Backend
	calls int
	fault error
}

func (b *vulkanRetirementFaultBackend) RetireRequestResources() {
	b.calls++
	if b.fault != nil {
		panic(b.fault)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestVulkanRequestRetirementFailure(t *testing.T) {
	primary := &BackendError{Backend: "vulkan", Site: "request", Class: VulkanClassInvalidGeometry, Err: ErrVulkanInvalidGeometry}
	cleanup := fmt.Errorf("retirement failed: %w", ErrVulkanDeviceLost)
	for _, tc := range []struct {
		name       string
		requestErr error
		panicBody  bool
		cleanupErr error
	}{
		{"success", nil, false, nil},
		{"request-error", primary, false, nil},
		{"retirement-panic", nil, false, cleanup},
		{"error-and-retirement-panic", primary, false, cleanup},
		{"panic-and-retirement-panic", primary, true, cleanup},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := &vulkanRetirementFaultBackend{fault: tc.cleanupErr}
			_, err := ExecuteVulkanRequest(backend, "request", func() (int, error) {
				if tc.panicBody {
					panic(tc.requestErr)
				}
				return 7, tc.requestErr
			})
			if backend.calls != 1 {
				t.Fatalf("retirement calls = %d, want 1", backend.calls)
			}
			if tc.cleanupErr == nil {
				if err != tc.requestErr {
					t.Fatalf("request error identity changed: got %v, want %v", err, tc.requestErr)
				}
			} else {
				if !errors.Is(err, cleanup) || !errors.Is(err, ErrVulkanDeviceLost) {
					t.Fatalf("retirement cause or device-loss classification missing: %v", err)
				}
				var typed *BackendError
				if !errors.As(err, &typed) {
					t.Fatalf("retirement panic was not typed: %v", err)
				}
				if tc.requestErr != nil {
					if typed != primary || !errors.Is(err, primary) || !errors.Is(err, ErrVulkanInvalidGeometry) {
						t.Fatalf("primary error identity or classification lost: %v", err)
					}
				} else if typed.Site != "request.RetireRequestResources" {
					t.Fatalf("cleanup site = %q", typed.Site)
				}
			}
			backend.fault = nil
			result, err := ExecuteVulkanRequest(backend, "next", func() (int, error) { return 42, nil })
			if result != 42 || err != nil || backend.calls != 2 {
				t.Fatalf("subsequent request: result=%d err=%v retirement calls=%d", result, err, backend.calls)
			}
		})
	}
}
