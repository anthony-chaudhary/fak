//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"
	"testing"
)

// TestVulkanQwen35PartialCommitMatchesOrdinaryWithoutTargetReplay is the
// required-device counterpart to the injected-backend contract. Every proper
// partial cut must retain the serial target's KV, recurrent state, lineage, and
// next-token behavior without entering the ordinary full-model Step path.
func TestVulkanQwen35PartialCommitMatchesOrdinaryWithoutTargetReplay(t *testing.T) {
	backend, _ := requiredVulkanSequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}

	for accepted := 1; accepted < len(draft); accepted++ {
		t.Run(fmt.Sprintf("accepted_%d", accepted), func(t *testing.T) {
			device, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			defer device.Close()
			boundary := device.Prefill(prefix)
			beforeStep, beforeWarm := device.halStep, device.halLogitsWarm

			tx, err := beginQwen35MTPTargetTransaction(device, boundary)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := tx.Verify(draft)
			if err != nil {
				t.Fatal(err)
			}
			verified := m.NewSession()
			verified.Prefill(prefix)
			for row, token := range draft {
				compareVulkanSequenceVector(t, fmt.Sprintf("verified_row_%d", row), rows[row], verified.Step(token))
			}
			verified.Close()

			steps := 0
			tx.step = func(token int) []float32 {
				steps++
				return device.Step(token)
			}
			gotBoundary, err := tx.Commit(accepted)
			if err != nil {
				t.Fatal(err)
			}
			if steps != 0 {
				t.Fatalf("full target replay steps=%d, want 0", steps)
			}
			if device.halStep != beforeStep+accepted || device.halLogitsWarm != (beforeWarm || accepted > 0) {
				t.Fatalf("HAL metadata step/warm=%d/%t, want %d/%t", device.halStep, device.halLogitsWarm, beforeStep+accepted, beforeWarm || accepted > 0)
			}
			receipt := tx.VerificationReceipt()
			if receipt.FullTargetReplaySteps != 0 || receipt.RecurrentRepairTokens != accepted {
				t.Fatalf("replay/repair receipt=%d/%d, want 0/%d", receipt.FullTargetReplaySteps, receipt.RecurrentRepairTokens, accepted)
			}

			serial := m.NewSession()
			defer serial.Close()
			serial.Prefill(prefix)
			var wantBoundary []float32
			for _, token := range draft[:accepted] {
				wantBoundary = serial.Step(token)
			}
			compareVulkanSequenceVector(t, "committed_boundary", gotBoundary, wantBoundary)
			committed := append(append([]int(nil), prefix...), draft[:accepted]...)
			assertVulkanDeviceStateMatchesSerial(t, device, serial, committed)

			continuation := (accepted*7 + 3) % m.Cfg.VocabSize
			compareVulkanSequenceVector(t, "continuation", device.Step(continuation), serial.Step(continuation))
		})
	}
}
