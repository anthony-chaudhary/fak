package model

import (
	"math"
	"strings"
	"testing"
)

func TestAdmitQwen38MTPBoundaryOutcomes(t *testing.T) {
	base := Qwen38MTPAdmissionInput{
		TargetTensorBytes:          100,
		RetainedMTPTensorBytes:     40,
		TransactionalStateBytes:    20,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             200,
		HeadroomBytes:              20,
		ReservedBytes:              10,
		Pressure:                   Qwen38MTPPressureNominal,
		HostAssistedSupported:      true,
	}
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionResident || got.PlannedBytes != 170 || got.DeviceBytes != 170 || got.UsableBytes != 170 {
		t.Fatalf("exact resident boundary=%+v", got)
	}

	oneByteShort := base
	oneByteShort.AvailableBytes--
	if got := AdmitQwen38MTP(oneByteShort); got.Outcome != Qwen38MTPAdmissionHostAssisted || got.DeviceBytes != 130 || got.PlannedBytes != 170 || got.UsableBytes != 169 {
		t.Fatalf("one-byte-short host assistance=%+v", got)
	}

	targetOnly := oneByteShort
	targetOnly.AvailableBytes = 159 // usable 129, one byte below target + transaction + verifier.
	if got := AdmitQwen38MTP(targetOnly); got.Outcome != Qwen38MTPAdmissionTargetOnly || got.DeviceBytes != 0 || got.UsableBytes != 129 {
		t.Fatalf("one-byte-short target-only downgrade=%+v", got)
	}
}

func TestAdmitQwen38MTPAccountsForEveryComponentAndReservation(t *testing.T) {
	base := Qwen38MTPAdmissionInput{
		TargetTensorBytes:          10,
		RetainedMTPTensorBytes:     10,
		TransactionalStateBytes:    10,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             40,
		Pressure:                   Qwen38MTPPressureNominal,
	}
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionResident {
		t.Fatalf("baseline=%+v", got)
	}
	for _, tc := range []struct {
		name   string
		mutate func(*Qwen38MTPAdmissionInput)
	}{
		{"target", func(in *Qwen38MTPAdmissionInput) { in.TargetTensorBytes++ }},
		{"retained_mtp", func(in *Qwen38MTPAdmissionInput) { in.RetainedMTPTensorBytes++ }},
		{"transaction", func(in *Qwen38MTPAdmissionInput) { in.TransactionalStateBytes++ }},
		{"verification", func(in *Qwen38MTPAdmissionInput) { in.VerificationWorkspaceBytes++ }},
		{"headroom", func(in *Qwen38MTPAdmissionInput) { in.HeadroomBytes++ }},
		{"reserved", func(in *Qwen38MTPAdmissionInput) { in.ReservedBytes++ }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := base
			tc.mutate(&in)
			if got := AdmitQwen38MTP(in); got.Outcome == Qwen38MTPAdmissionResident {
				t.Fatalf("component did not alter boundary decision: %+v", got)
			}
		})
	}
}

func TestAdmitQwen38MTPPressureAndOverflowFailClosed(t *testing.T) {
	base := Qwen38MTPAdmissionInput{
		TargetTensorBytes:          10,
		RetainedMTPTensorBytes:     10,
		TransactionalStateBytes:    10,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             100,
		Pressure:                   Qwen38MTPPressureElevated,
		HostAssistedSupported:      true,
	}
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionResident {
		t.Fatalf("elevated pressure with resident fit=%+v", got)
	}
	base.AvailableBytes = 35
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionTargetOnly {
		t.Fatalf("elevated pressure must not host-assist=%+v", got)
	}
	base.AvailableBytes = 100
	base.Pressure = Qwen38MTPPressureCritical
	if got := AdmitQwen38MTP(base); got.Outcome != Qwen38MTPAdmissionTargetOnly {
		t.Fatalf("critical pressure=%+v", got)
	}

	overflow := base
	overflow.Pressure = Qwen38MTPPressureNominal
	overflow.TargetTensorBytes = math.MaxUint64
	if got := AdmitQwen38MTP(overflow); got.Outcome != Qwen38MTPAdmissionTargetOnly || got.PlannedBytes != 0 {
		t.Fatalf("component overflow=%+v", got)
	}
	overflow = base
	overflow.Pressure = Qwen38MTPPressureNominal
	overflow.HeadroomBytes = math.MaxUint64
	overflow.ReservedBytes = 1
	if got := AdmitQwen38MTP(overflow); got.Outcome != Qwen38MTPAdmissionTargetOnly || got.UsableBytes != 0 {
		t.Fatalf("reservation overflow=%+v", got)
	}
}

func TestQwen38MTPAdmissionValidatesReceiptAccounting(t *testing.T) {
	admission := AdmitQwen38MTP(Qwen38MTPAdmissionInput{
		TargetTensorBytes:          100,
		RetainedMTPTensorBytes:     40,
		TransactionalStateBytes:    20,
		VerificationWorkspaceBytes: 10,
		AvailableBytes:             169,
		Pressure:                   Qwen38MTPPressureNominal,
		HostAssistedSupported:      true,
	})
	if err := admission.validate(); err != nil {
		t.Fatalf("valid admission: %v", err)
	}
	admission.PlannedBytes++
	if err := admission.validate(); err == nil {
		t.Fatal("tampered component accounting validated")
	}
}

func TestQwen38MTPMetalWeightAdmission(t *testing.T) {
	t.Run("AppleSiliconPreflight", func(t *testing.T) {
		const (
			gib16 = 16 * 1024 * 1024 * 1024
			gib36 = 36 * 1024 * 1024 * 1024
			gib8  = 8 * 1024 * 1024 * 1024
			mtp06 = 600 * 1024 * 1024
		)

		// 1. 36GB host with standard base (~16GB) + MTP (~0.6GB) -> Approved=true, Outcome=Resident.
		res36 := admitQwen38MTPDevice(gib16, mtp06, gib36)
		if !res36.Approved || res36.Outcome != Qwen38MTPAdmissionResident {
			t.Fatalf("36GB host standard base: got Approved=%v Outcome=%v, want Approved=true Outcome=resident", res36.Approved, res36.Outcome)
		}

		in36 := Qwen38MTPAdmissionInput{
			TargetTensorBytes:      gib16,
			RetainedMTPTensorBytes: mtp06,
			AvailableBytes:         gib36,
			HeadroomBytes:          4 * 1024 * 1024 * 1024,
			ReservedBytes:          2 * 1024 * 1024 * 1024,
			Pressure:               Qwen38MTPPressureNominal,
		}
		resDirect := admitQwen38MTPAppleSilicon(in36)
		if !resDirect.Approved || resDirect.Outcome != Qwen38MTPAdmissionResident {
			t.Fatalf("direct 36GB preflight: got Approved=%v Outcome=%v, want Approved=true Outcome=resident", resDirect.Approved, resDirect.Outcome)
		}

		// 2. 16GB host (or 8GB) -> Approved=false, Outcome=TargetOnly (swap thrashing prevented).
		for _, mem := range []uint64{gib16, gib8} {
			res16 := admitQwen38MTPDevice(gib16/2, mtp06, mem)
			if res16.Approved || res16.Outcome != Qwen38MTPAdmissionTargetOnly {
				t.Fatalf("mem=%d: got Approved=%v Outcome=%v, want Approved=false Outcome=native_target_only", mem, res16.Approved, res16.Outcome)
			}
			if !strings.Contains(res16.Reason, "prevent swap thrashing") {
				t.Fatalf("mem=%d: Reason %q does not contain 'prevent swap thrashing'", mem, res16.Reason)
			}
		}

		// 3. Critical memory pressure -> Approved=false, Outcome=TargetOnly.
		inCrit := in36
		inCrit.Pressure = Qwen38MTPPressureCritical
		resCrit := admitQwen38MTPAppleSilicon(inCrit)
		if resCrit.Approved || resCrit.Outcome != Qwen38MTPAdmissionTargetOnly {
			t.Fatalf("critical pressure: got Approved=%v Outcome=%v, want Approved=false Outcome=native_target_only", resCrit.Approved, resCrit.Outcome)
		}
		if !strings.Contains(resCrit.Reason, "memory pressure critical") {
			t.Fatalf("critical pressure: Reason %q does not contain 'memory pressure critical'", resCrit.Reason)
		}

		// 4. Memory saturated (planned > usable) -> Approved=false.
		resSat := admitQwen38MTPDevice(32*1024*1024*1024, mtp06, gib36)
		if resSat.Approved || resSat.Outcome != Qwen38MTPAdmissionTargetOnly {
			t.Fatalf("saturated: got Approved=%v Outcome=%v, want Approved=false Outcome=native_target_only", resSat.Approved, resSat.Outcome)
		}
	})

	t.Run("SafeTensorsMTPRetention", func(t *testing.T) {
		cfg := qwen35MTPLoadConfig()
		shapes, err := qwen35MTPExpectedShapes(cfg)
		if err != nil {
			t.Fatalf("derive MTP fixture shapes: %v", err)
		}
		_, safe := qwen35MTPLoadTensors(t, cfg)
		// Add a vision tensor to verify it is always dropped:
		safe["model.visual.patch_embed.proj.weight"] = tinySTTensor{
			dtype: "F32",
			shape: []int{4, 4},
			data:  f32TestBytes(make([]float32, 16)),
		}
		stPath := writeTinySafetensors(t, safe)

		// 1. When RetainMTP=false and no env var: MTP tensors and visual tensors dropped by default.
		t.Run("DefaultDrop", func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", "")
			t.Setenv("SPECULATIVE", "")
			orig := RetainMTP
			SetRetainMTP(false)
			defer SetRetainMTP(orig)

			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("load safetensors: %v", err)
			}
			for _, name := range qwen35MTPRequiredTensors {
				if _, ok := m.manifest[name]; ok {
					t.Fatalf("tensor %s should be dropped when RetainMTP=false", name)
				}
			}
			if _, ok := m.manifest["model.visual.patch_embed.proj.weight"]; ok {
				t.Fatalf("vision tensor should be dropped")
			}
		})

		// 2. When RetainMTP=true: retains all 15 MTP tensors with exact shapes and dtypes; visual still dropped.
		t.Run("RetainMTPFlag", func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", "")
			t.Setenv("SPECULATIVE", "")
			orig := RetainMTP
			SetRetainMTP(true)
			defer SetRetainMTP(orig)

			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("load safetensors: %v", err)
			}
			if len(qwen35MTPRequiredTensors) != 15 {
				t.Fatalf("expected 15 MTP required tensors, got %d", len(qwen35MTPRequiredTensors))
			}
			for _, name := range qwen35MTPRequiredTensors {
				meta, ok := m.manifest[name]
				if !ok {
					t.Fatalf("manifest missing retained tensor %s", name)
				}
				if !strings.EqualFold(meta.Dtype, "f32") {
					t.Fatalf("tensor %s dtype=%q, want f32", name, meta.Dtype)
				}
				expectedShape := shapes[name]
				if !sameShape(meta.Shape, expectedShape) {
					t.Fatalf("tensor %s shape=%v, want %v", name, meta.Shape, expectedShape)
				}
			}
			if _, ok := m.manifest["model.visual.patch_embed.proj.weight"]; ok {
				t.Fatalf("vision tensor should be dropped even when RetainMTP=true")
			}
		})

		// 3. When FAK_SPECULATIVE=mtp (and RetainMTP=false): retains all 15 MTP tensors; visual still dropped.
		t.Run("EnvFAKSpeculative", func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", "mtp")
			t.Setenv("SPECULATIVE", "")
			orig := RetainMTP
			SetRetainMTP(false)
			defer SetRetainMTP(orig)

			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("load safetensors: %v", err)
			}
			for _, name := range qwen35MTPRequiredTensors {
				meta, ok := m.manifest[name]
				if !ok {
					t.Fatalf("manifest missing retained tensor %s with FAK_SPECULATIVE=mtp", name)
				}
				if !strings.EqualFold(meta.Dtype, "f32") {
					t.Fatalf("tensor %s dtype=%q, want f32", name, meta.Dtype)
				}
				expectedShape := shapes[name]
				if !sameShape(meta.Shape, expectedShape) {
					t.Fatalf("tensor %s shape=%v, want %v", name, meta.Shape, expectedShape)
				}
			}
			if _, ok := m.manifest["model.visual.patch_embed.proj.weight"]; ok {
				t.Fatalf("vision tensor should be dropped")
			}
		})

		// 4. When SPECULATIVE=mtp (and RetainMTP=false): retains all 15 MTP tensors; visual still dropped.
		t.Run("EnvSpeculative", func(t *testing.T) {
			t.Setenv("FAK_SPECULATIVE", "")
			t.Setenv("SPECULATIVE", "mtp")
			orig := RetainMTP
			SetRetainMTP(false)
			defer SetRetainMTP(orig)

			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("load safetensors: %v", err)
			}
			for _, name := range qwen35MTPRequiredTensors {
				meta, ok := m.manifest[name]
				if !ok {
					t.Fatalf("manifest missing retained tensor %s with SPECULATIVE=mtp", name)
				}
				if !strings.EqualFold(meta.Dtype, "f32") {
					t.Fatalf("tensor %s dtype=%q, want f32", name, meta.Dtype)
				}
				expectedShape := shapes[name]
				if !sameShape(meta.Shape, expectedShape) {
					t.Fatalf("tensor %s shape=%v, want %v", name, meta.Shape, expectedShape)
				}
			}
			if _, ok := m.manifest["model.visual.patch_embed.proj.weight"]; ok {
				t.Fatalf("vision tensor should be dropped")
			}
		})
	})
}
