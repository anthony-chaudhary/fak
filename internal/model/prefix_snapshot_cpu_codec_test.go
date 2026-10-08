package model

import (
	"encoding/binary"
	"errors"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=500ms lane=default
func TestCPUPrefixSnapshotCodecContract(t *testing.T) {
	cfg := llamaArchConfig()
	cfg.DenseMLP = true
	cfg.HeadDim = 16
	cfg.HiddenSize = 64
	m := NewSynthetic(cfg)
	prompt := []int{3, 7, 11, 5}

	var validWire []byte
	for _, precision := range []KVPrecision{KVPrecisionFP32, KVPrecisionQ8_0} {
		t.Run(string(precision), func(t *testing.T) {
			live := m.NewSessionWithKVPrecision(precision)
			live.Prefill(prompt)
			snapshot, err := live.PrefixSnapshot()
			if err != nil {
				t.Fatalf("PrefixSnapshot: %v", err)
			}
			defer snapshot.Close()
			wire, err := snapshot.MarshalCPU()
			if err != nil {
				t.Fatalf("MarshalCPU: %v", err)
			}
			if precision == KVPrecisionFP32 {
				validWire = append([]byte(nil), wire...)
			}

			recovered, err := DecodeCPUPrefixSnapshot(wire, cfg)
			if err != nil {
				t.Fatalf("DecodeCPUPrefixSnapshot: %v", err)
			}
			defer recovered.Close()
			restored := m.NewSessionWithKVPrecision(precision)
			if err := recovered.Restore(restored); err != nil {
				t.Fatalf("Restore: %v", err)
			}
			if restored.Cache.Precision() != precision || restored.Cache.Len() != len(prompt) {
				t.Fatalf("restored cache precision/positions=%s/%d, want %s/%d",
					restored.Cache.Precision(), restored.Cache.Len(), precision, len(prompt))
			}
			if precision == KVPrecisionQ8_0 && !restored.Cache.quantized() {
				t.Fatal("q8 snapshot restored through an f32 fallback")
			}

			got := restored.Step(13)
			want := live.Step(13)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("restored next-step logits changed for %s", precision)
			}
		})
	}
	if len(validWire) == 0 {
		t.Fatal("f32 arm produced no rejection fixture")
	}

	t.Run("qwen35 recurrent continuation", func(t *testing.T) {
		hybridCfg := qwen35HybridTestCfg()
		hybrid := NewSynthetic(hybridCfg)
		live := hybrid.NewSession()
		live.Prefill(prompt)
		snapshot, err := live.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		defer snapshot.Close()
		wire, err := snapshot.MarshalCPU()
		if err != nil {
			t.Fatalf("MarshalCPU: %v", err)
		}
		recovered, err := DecodeCPUPrefixSnapshot(wire, hybridCfg)
		if err != nil {
			t.Fatalf("DecodeCPUPrefixSnapshot: %v", err)
		}
		defer recovered.Close()
		restored := hybrid.NewSession()
		if err := recovered.Restore(restored); err != nil {
			t.Fatalf("Restore: %v", err)
		}
		if got, want := restored.Step(13), live.Step(13); !reflect.DeepEqual(got, want) {
			t.Fatal("restored Qwen3.5 recurrent continuation changed next-step logits")
		}
	})

	tampered := append([]byte(nil), validWire...)
	tampered[len(tampered)-1] ^= 0x80
	wrongVersion := append([]byte(nil), validWire...)
	binary.BigEndian.PutUint16(wrongVersion[len(cpuPrefixSnapshotMagic):], cpuPrefixSnapshotVersion+1)
	wrongCfg := cfg
	wrongCfg.VocabSize++

	baseSession := m.NewSession()
	baseSession.Prefill(prompt)
	base, err := baseSession.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer base.Close()
	if !base.HostDiskSerializable() {
		t.Fatal("complete CPU snapshot rejected by host-disk admission")
	}
	placement, err := base.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer placement.Close()
	placement.DenseGPULayers = 1
	placement.GPULayers = 1
	placement.ExecutionPolicy = ExecutionPolicyDeviceOnly
	if !placement.HostDiskSerializable() {
		t.Fatal("portable envelope placement metadata rejected by host-disk admission")
	}
	placementWire, err := placement.MarshalCPU()
	if err != nil {
		t.Fatalf("MarshalCPU placement envelope: %v", err)
	}
	placementRecovered, err := DecodeCPUPrefixSnapshot(placementWire, cfg)
	if err != nil {
		t.Fatalf("DecodeCPUPrefixSnapshot placement envelope: %v", err)
	}
	defer placementRecovered.Close()
	if placementRecovered.DenseGPULayers != 1 || placementRecovered.GPULayers != 1 || placementRecovered.ExecutionPolicy != ExecutionPolicyDeviceOnly {
		t.Fatalf("placement envelope=%d/%d/%d, want 1/1/device-only", placementRecovered.DenseGPULayers, placementRecovered.GPULayers, placementRecovered.ExecutionPolicy)
	}
	v41, err := base.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer v41.Close()
	v41.v41 = &v41ForwardSnapshot{hadState: true}
	hidden, err := base.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer hidden.Close()
	hidden.captureTargetHidden = true
	hidden.targetHidden = [][]float32{{1}}
	hidden.targetHiddenTokens = []int{prompt[len(prompt)-1]}
	privateHAL, err := base.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer privateHAL.Close()
	backend := newRecordingQwen35Backend(m)
	v41.Backend = backend
	hidden.Backend = backend
	privateHAL.Backend = backend
	privateHAL.qwen35 = &qwen35HALState{}
	if v41.HostDiskSerializable() || hidden.HostDiskSerializable() || privateHAL.HostDiskSerializable() {
		t.Fatal("incomplete continuation state admitted to host disk")
	}

	rejections := []struct {
		name string
		err  error
		run  func() error
	}{
		{name: "tamper", err: ErrCPUPrefixSnapshotIntegrity, run: func() error { _, err := DecodeCPUPrefixSnapshot(tampered, cfg); return err }},
		{name: "version", err: ErrCPUPrefixSnapshotVersion, run: func() error { _, err := DecodeCPUPrefixSnapshot(wrongVersion, cfg); return err }},
		{name: "configuration", err: ErrCPUPrefixSnapshotScope, run: func() error { _, err := DecodeCPUPrefixSnapshot(validWire, wrongCfg); return err }},
		{name: "v41 continuation", err: ErrCPUPrefixSnapshotState, run: func() error { _, err := v41.MarshalCPU(); return err }},
		{name: "target hidden history", err: ErrCPUPrefixSnapshotState, run: func() error { _, err := hidden.MarshalCPU(); return err }},
	}
	for _, tc := range rejections {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); !errors.Is(err, tc.err) {
				t.Fatalf("error=%v, want errors.Is(_, %v)", err, tc.err)
			}
		})
	}
}
