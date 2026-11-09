//go:build vulkan && (windows || linux) && cgo

package model

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime integration est=5s lane=optin
func TestVulkanHeadlessPrefixConsumerAndContinuation(t *testing.T) {
	backend, sequence := requiredVulkanSequenceBackend(t)
	headless, ok := backend.(compute.Qwen35SequenceHeadlessPrefixBackend)
	if !ok || headless.Qwen35SequenceHeadlessPrefixPath() != compute.Qwen35SequenceHeadlessPrefixPath {
		t.Fatal("Vulkan does not advertise the exact headless-prefix capability")
	}
	m := NewSynthetic(qwen35HybridTestCfg())
	// A distinct name makes output staging distinguishable from input embedding.
	m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
	device, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(device.Close)
	serial := m.NewSession()
	t.Cleanup(serial.Close)
	prefix := []int{3, 7, 11}
	serial.Prefill(prefix)
	if err := device.PrefillNoLogitsContext(context.Background(), prefix); err != nil {
		t.Fatal(err)
	}
	if route := device.qwen35HAL.prefillRoute; route.EffectivePath != compute.Qwen35SequencePrefillPath || route.FallbackActive {
		t.Fatalf("prefix did not use the actual sequence consumer: %+v", route)
	}
	for key := range device.halW {
		if strings.Contains(key, "lm_head.weight") {
			t.Fatalf("prefix staged its unused output head under %q", key)
		}
	}
	assertVulkanDeviceStateMatchesSerial(t, device, serial, prefix)
	got, want := device.Step(17), serial.Step(17)
	compareVulkanSequenceVector(t, "headless_prefix_continuation", got, want)
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatal("headless prefix changed the continuation token")
	}
	assertVulkanDeviceStateMatchesSerial(t, device, serial, append(prefix, 17))

	for _, name := range []string{"missing_last_head", "missing_all_head", "malformed_supplied_head"} {
		t.Run(name, func(t *testing.T) {
			control, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(control.Close)
			req := control.qwen35SequencePrefillRequest([]int{3, 7}, false)
			switch name {
			case "missing_last_head":
				req.Output, req.NeedLogits = compute.Tensor{}, true
			case "missing_all_head":
				req.Output, req.NeedAllLogits = compute.Tensor{}, true
			case "malformed_supplied_head":
				req.Output.Shape = []int{1}
			}
			_, err = control.qwen35HAL.mutateSequence(backend, func(states []compute.Qwen35SequenceState) (compute.Qwen35SequencePrefillResult, error) {
				req.States = states
				return sequence.Qwen35SequencePrefill(req)
			})
			var failure *compute.Qwen35SequenceError
			if !errors.As(err, &failure) || failure.Stage != "tensor-preflight" || !strings.HasPrefix(failure.Reason, "output:") || control.halKV.Len() != 0 {
				t.Fatalf("invalid head must fail before mutation: error=%v kv=%d", err, control.halKV.Len())
			}
		})
	}
}
