//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41EngramRowDecodeDeviceOnlyVulkan$' -count=1
// Synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"errors"
	"fmt"
	"slices"
	"testing"
)

// On the ordinary Vulkan build every device-only seam an Engram model uses
// binds except the host Engram row decode, so device-only admission names
// exactly that op and refuses typed before any native dispatch.
// fak-test:justify why=integration when=changed:internal/model/v41_device_only_binding*
// fak-test:runtime slow est=10s lane=optin
func TestV41EngramRowDecodeDeviceOnlyVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("engram_compressed_%t", compressed), func(t *testing.T) {
			m, _ := v41IncrementalEngramFixture(t, compressed)
			defer func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			}()
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			unbound := v41HostOpNames(s.v41DeviceOnlyUnboundOps())
			if !slices.Equal(unbound, []string{"engram-row-decode"}) {
				t.Fatalf("device-only unbound ops = %v, want exactly [engram-row-decode]", unbound)
			}
			var refusal error
			n := v41SeamDispatches(t, be, func() {
				refusal = recoverError(func() { _ = s.Prefill([]int{1, 2, 3}) })
			})
			var closed *BackendForwardOperationError
			var typed *v41DeviceOnlyHostOpError
			if !errors.Is(refusal, errV41DeviceOnlyHostOp) || !errors.As(refusal, &closed) || closed.Path != "device-only" ||
				!errors.As(refusal, &typed) || !slices.Contains(typed.ops, v41HostOp{name: "engram-row-decode", site: "decodeV41EngramRow"}) {
				t.Fatalf("device-only prefill = %v, want the typed engram-row-decode admission refusal", refusal)
			}
			if n != 0 || !s.BackendSessionClosed() {
				t.Fatalf("refusal issued %d native dispatches closed=%t, want none and a closed session", n, s.BackendSessionClosed())
			}
			t.Logf("V41_ENGRAM_ROW_DECODE_DEVICE_ONLY compressed=%t unbound=%v dispatches=%d", compressed, unbound, n)
		})
	}
}
