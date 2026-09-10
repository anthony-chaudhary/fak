//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"
	"reflect"
	"testing"
)

func assertVulkanDeviceStateMatchesSerial(t *testing.T, target, serial *Session, tokens []int) {
	t.Helper()
	if target.halKV == nil || target.qwen35HAL == nil {
		t.Fatal("device session is missing hybrid KV or recurrent state")
	}
	if target.halKV.Len() != serial.Cache.Len() {
		t.Fatalf("device/serial cache lengths=%d/%d", target.halKV.Len(), serial.Cache.Len())
	}
	if !reflect.DeepEqual(target.halKV.Pos(), serial.Cache.pos) {
		t.Fatalf("device positions=%v want=%v", target.halKV.Pos(), serial.Cache.pos)
	}
	for layer := 0; layer < serial.M.Cfg.NumLayers; layer++ {
		if serial.M.Cfg.isLinearAttnLayer(layer) {
			got := target.qwen35HAL.layers[layer]
			want := serial.Cache.linear.layers[layer]
			compareVulkanSequenceVector(t, fmt.Sprintf("layer_%d_conv", layer), target.Backend.Read(got.conv), flattenRows(want.conv))
			compareVulkanSequenceVector(t, fmt.Sprintf("layer_%d_recurrent", layer), target.Backend.Read(got.recurrent), flattenRows(want.recurrent))
			continue
		}
		compact := qwen35HALKVLayer(serial.M.Cfg, layer)
		compareVulkanSequenceVector(t, fmt.Sprintf("layer_%d_k", layer), target.Backend.Read(target.halKV.KeysView(compact)), serial.Cache.K[layer])
		compareVulkanSequenceVector(t, fmt.Sprintf("layer_%d_v", layer), target.Backend.Read(target.halKV.ValuesView(compact)), serial.Cache.V[layer])
	}
	if _, err := target.VerifyTokenLineage(tokens); err != nil {
		t.Fatalf("device token lineage: %v", err)
	}
}

func TestVulkanQwen35DeviceVerificationAllRowsAndContinuationMatchCPU(t *testing.T) {
	backend, _ := requiredVulkanSequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	device, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	device.captureTargetHidden = true
	serial := m.NewSession()
	serial.captureTargetHidden = true
	defer serial.Close()

	prefix := []int{3, 7, 11, 5, 17, 19}
	boundary := device.Prefill(prefix)
	serial.Prefill(prefix)
	panels := [][]int{
		{23, 2, 29, 31},
		{37, 41, 43, 47},
		{53, 59, 61, 3},
		{7, 13, 17, 23},
		{29, 31, 37, 41},
	}
	committed := append([]int(nil), prefix...)
	for sample, draft := range panels {
		rows, receipt, err := device.verifyQwen35DevicePanel(draft, boundary)
		if err != nil {
			t.Fatalf("sample %d: %v", sample, err)
		}
		if len(rows) != len(draft) || receipt.Path != targetVerificationQwen38DevicePanelPath || !receipt.OneOperation || receipt.TargetVerificationOperations != 1 || receipt.TargetDecodeSteps != 0 {
			t.Fatalf("sample %d rows=%d receipt=%+v", sample, len(rows), receipt)
		}
		for row, token := range draft {
			want := serial.Step(token)
			compareVulkanSequenceVector(t, fmt.Sprintf("sample_%d_row_%d", sample, row), rows[row], want)
		}
		committed = append(committed, draft...)
		assertVulkanDeviceStateMatchesSerial(t, device, serial, committed)

		continuation := (sample*11 + 2) % m.Cfg.VocabSize
		want := serial.Step(continuation)
		got := device.Step(continuation)
		compareVulkanSequenceVector(t, fmt.Sprintf("sample_%d_continuation", sample), got, want)
		committed = append(committed, continuation)
		assertVulkanDeviceStateMatchesSerial(t, device, serial, committed)
		boundary = got
	}
	t.Logf("engine=fak-native backend=vulkan device=%s samples=%d draft_depth=4 per_row_logits=true continuation_parity=true", backend.Tier(), len(panels))
}

func TestVulkanQwen35DeviceTransactionAcceptsOnlyCommittedPrefix(t *testing.T) {
	backend, _ := requiredVulkanSequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	prefix := []int{3, 7, 11, 5, 17, 19}
	draft := []int{23, 2, 29, 31}
	for _, accepted := range []int{len(draft), 2, 0} {
		t.Run(fmt.Sprintf("accepted_%d", accepted), func(t *testing.T) {
			device, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			defer device.Close()
			device.captureTargetHidden = true
			serial := m.NewSession()
			serial.captureTargetHidden = true
			defer func() { serial.Close() }()
			boundary := device.Prefill(prefix)
			serial.Prefill(prefix)

			tx, err := beginQwen35MTPTargetTransaction(device, boundary)
			if err != nil {
				t.Fatal(err)
			}
			rows, err := tx.Verify(draft)
			if err != nil {
				t.Fatal(err)
			}
			for row, token := range draft {
				compareVulkanSequenceVector(t, fmt.Sprintf("verified_row_%d", row), rows[row], serial.Step(token))
			}
			serial.Close()
			serial = m.NewSession()
			serial.captureTargetHidden = true
			serial.Prefill(prefix)
			for _, token := range draft[:accepted] {
				serial.Step(token)
			}
			steps := 0
			tx.step = func(token int) []float32 {
				steps++
				return device.Step(token)
			}
			if _, err := tx.Commit(accepted); err != nil {
				t.Fatal(err)
			}
			wantSteps := accepted
			if accepted == len(draft) {
				wantSteps = 0
			}
			if steps != wantSteps {
				t.Fatalf("Step replay=%d want=%d", steps, wantSteps)
			}
			committed := append(append([]int(nil), prefix...), draft[:accepted]...)
			assertVulkanDeviceStateMatchesSerial(t, device, serial, committed)
			continuation := 41
			compareVulkanSequenceVector(t, "continuation", device.Step(continuation), serial.Step(continuation))
		})
	}
}
