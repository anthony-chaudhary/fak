//go:build vulkan && (windows || linux) && cgo

package model

import (
	"context"
	"os"

	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
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
	device.captureTargetHidden = false
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
			device.captureTargetHidden = false
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
			wantSteps := 0
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

func TestVulkanQwen35FeaturePanelsAndAcceptedPrefix(t *testing.T) {
	backend, _ := requiredVulkanSequenceBackend(t)
	if _, ok := backend.(compute.V41DeferredStatusBackend); !ok {
		t.Fatal("Vulkan target must expose the selected grouped tap reader")
	}
	profile := backend.(interface {
		VulkanDebugResetDispatchProfile()
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	cfg := qwen35HybridTestCfg()
	cfg.NumLayers = 8
	cfg.LayerTypes = append(append([]string(nil), cfg.LayerTypes...), cfg.LayerTypes...)
	m := NewSynthetic(cfg)
	prefix, layers := []int{3, 7, 11}, []int{0, 1, 3, 5, 7}
	for _, width := range []int{4, 8} {
		for _, accepted := range []int{0, 2, width} {
			t.Run(fmt.Sprintf("rows_%d_accept_%d", width, accepted), func(t *testing.T) {
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
				draft := []int{23, 2, 29, 31, 37, 41, 43, 47}[:width]
				f, err := beginQwen35TargetFeaturePanelTransaction(device, boundary, layers, width)
				if err != nil {
					t.Fatal(err)
				}
				defer func() { _ = f.Abort() }()
				profile.VulkanDebugResetDispatchProfile()
				rows, err := f.Verify(context.Background(), draft)
				readback := profile.VulkanDebugDispatchProfileSnapshot()
				if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" && readback.OneShotD2HSubmits != 3 {
					// One grouped tap read, one raw-hidden read, one logits read.
					t.Fatalf("feature panel D2H submissions=%d want=3", readback.OneShotD2HSubmits)
				}
				if err != nil {
					t.Fatal(err)
				}
				wantFeatures := make([]qwen35TargetFeatureRow, width)
				for row, token := range draft {
					wantFeatures[row] = qwen35TargetFeaturesManualStep(serial, token, layers)
					wantLogits := serial.head(serial.M.finalNorm(wantFeatures[row].Hidden[len(layers)-1]))
					compareVulkanSequenceVector(t, fmt.Sprintf("logits_%d", row), rows[row], wantLogits)
				}
				_, got, err := f.Commit(context.Background(), accepted)
				if err != nil || len(got) != accepted {
					t.Fatalf("commit: rows=%d err=%v", len(got), err)
				}
				r := f.tx.VerificationReceipt()
				if !r.OneOperation || r.TargetVerificationOperations != 1 || r.TargetDecodeSteps != 0 || r.FullTargetReplaySteps != 0 || (accepted > 0 && accepted < width && r.RecurrentRepairTokens != accepted) {
					t.Fatalf("panel/replay accounting: %+v", r)
				}
				for row, feature := range got {
					want := wantFeatures[row]
					if feature.Position != want.Position || feature.Token != want.Token || !reflect.DeepEqual(feature.Layers, want.Layers) {
						t.Fatal("feature identity changed")
					}
					for i, values := range feature.Hidden {
						compareVulkanSequenceVector(t, fmt.Sprintf("tap_%d_%d", row, i), values, want.Hidden[i])
					}
				}
				serial.Close()
				serial = m.NewSession()
				serial.captureTargetHidden = true
				committed := append(append([]int(nil), prefix...), draft[:accepted]...)
				serial.Prefill(committed)
				assertVulkanDeviceStateMatchesSerial(t, device, serial, committed)
				compareVulkanSequenceVector(t, "continuation", device.Step(53), serial.Step(53))
				assertVulkanDeviceStateMatchesSerial(t, device, serial, append(committed, 53))
				if len(device.targetHidden) != len(serial.targetHidden) {
					t.Fatal("raw-hidden history length changed")
				}
			})
		}
	}
}

func TestQwen35FeaturePanelRefusesQSACrossing(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	base := compute.QSADynamicGatingThreshold - 8
	if err := qwen35TargetFeaturePanelPosition(cfg, base-1, 8); err != nil {
		t.Fatal(err)
	}
	if err := qwen35TargetFeaturePanelPosition(cfg, base, 8); err == nil {
		t.Fatal("last-row QSA crossing admitted")
	}
	if err := qwen35TargetFeaturePanelPosition(cfg, base, 4); err != nil {
		t.Fatal(err)
	}
}

// Preserve the existing operation interfaces while hiding only feature capture.
type qwen35PanelNoFeatures struct {
	compute.Backend
	compute.ForwardExecutionOwnerBackend
	Qwen35SequencePrefillBackend
	compute.Qwen35SequenceAllLogitsBackend
	compute.Qwen35SequencePrefixReplayBackend
	compute.Qwen35SequenceRawHiddenBackend
}

type qwen35PanelWrongFeatures struct{ *qwen35PanelNoFeatures }

func (*qwen35PanelWrongFeatures) Qwen35SequenceFeatureRowsPath() string { return "wrong-feature-path" }

func TestVulkanQwen35FeaturePanelAdmissionBeforeMutation(t *testing.T) {
	backend, seq := requiredVulkanSequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	s, err := m.NewBackendSessionChecked(backend)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.captureTargetHidden = true
	boundary := s.Prefill([]int{3, 7, 11})
	base := s.halKV.Len()
	masked := &qwen35PanelNoFeatures{
		Backend: backend, ForwardExecutionOwnerBackend: backend.(compute.ForwardExecutionOwnerBackend),
		Qwen35SequencePrefillBackend: seq, Qwen35SequenceAllLogitsBackend: backend.(compute.Qwen35SequenceAllLogitsBackend),
		Qwen35SequencePrefixReplayBackend: backend.(compute.Qwen35SequencePrefixReplayBackend),
		Qwen35SequenceRawHiddenBackend:    backend.(compute.Qwen35SequenceRawHiddenBackend),
	}
	for _, tc := range []struct {
		name    string
		backend compute.Backend
		layers  []int
		width   int
	}{
		{"missing_capture", masked, []int{0, 3}, 8},
		{"wrong_capture", &qwen35PanelWrongFeatures{masked}, []int{0, 3}, 8},
		{"duplicate_taps", backend, []int{0, 0}, 8},
		{"oversized_taps", backend, []int{0, 1, 2, 3, 4, 5}, 8},
		{"invalid_width", backend, []int{0, 3}, 9},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s.Backend = tc.backend
			defer func() { s.Backend = backend }()
			if f, err := beginQwen35TargetFeaturePanelTransaction(s, boundary, tc.layers, tc.width); err == nil || f != nil {
				t.Fatal("invalid explicit panel admitted")
			}
			if s.qwen35TargetFeatureTxn != nil || s.halKV.Len() != base {
				t.Fatal("refusal mutated target")
			}
		})
	}
}

// Reuse the real selected path and its private identity check as the fault seam;
// do not substitute an emulated sequence backend for result/readback ownership.
func TestVulkanQwen35FeaturePanelInterruptedPublication(t *testing.T) {
	t.Setenv("GODEBUG", "panicnil=1")
	backend, _ := requiredVulkanSequenceBackend(t)
	m := NewSynthetic(qwen35HybridTestCfg())
	for _, phase := range []string{"result", "grouped-read-error", "tap-read", "scope-release"} {
		t.Run(phase, func(t *testing.T) {
			device, err := m.NewBackendSessionChecked(backend)
			if err != nil {
				t.Fatal(err)
			}
			defer device.Close()
			serial := m.NewSession()
			defer serial.Close()
			prefix := []int{3, 7, 11}
			boundary := device.Prefill(prefix)
			serial.Prefill(prefix)
			f, err := beginQwen35TargetFeaturePanelTransaction(device, boundary, []int{0, 3}, 4)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = f.Abort() }()
			debug := backend.(interface{ VulkanDebugSetD2HStagingFailureOnce(bool) })
			defer debug.VulkanDebugSetD2HStagingFailureOnce(false)
			valid := f.panelValid
			checks, injected := 0, false
			f.panelValid = func() bool {
				if f.tx.prefixReplay != nil {
					checks++
					// Checks 1/2/3 are result owner, tap reader entry, then
					// completed grouped read before copying its first tap.
					if phase == "grouped-read-error" && checks == 2 {
						injected = true
						debug.VulkanDebugSetD2HStagingFailureOnce(true)
					}
					if (phase == "result" && checks == 1) || (phase == "tap-read" && checks == 3) {
						injected = true
						panic(nil)
					}
					if phase == "scope-release" && !injected {
						injected = true
						device.backendExecutionScope.afterRelease(func() { panic(nil) })
					}
				}
				return valid()
			}
			var rows [][]float32
			var verifyErr error
			out := siblingRun(func() { rows, verifyErr = f.Verify(context.Background(), []int{23, 2, 29, 31}) })
			if !injected || rows != nil || (out.returned && verifyErr == nil) || !f.tx.closed || f.active() {
				t.Fatalf("interrupted selected panel published: injected=%t outcome=%+v rows=%d err=%v closed=%t", injected, out, len(rows), verifyErr, f.tx.closed)
			}
			assertVulkanDeviceStateMatchesSerial(t, device, serial, prefix)
			compareVulkanSequenceVector(t, "continuation", device.Step(37), serial.Step(37))
		})
	}
}
