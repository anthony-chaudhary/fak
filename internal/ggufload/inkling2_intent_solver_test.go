package ggufload

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s lane=default

func TestResolveInkling2ServeIntent(t *testing.T) {
	t.Parallel()

	const gib = int64(1) << 30
	hp := inkling2MiniKVShape()

	// Realistic post-weight free VRAM envelopes for Inkling-2-Mini NVFP4 (~12 GiB weights):
	//   - 16 GB RTX 5080 (sm_120): ~3.5 GiB free after weights
	//   - 24 GB RTX 5090 (sm_120): ~11.5 GiB free after weights
	//   - 16 GB Ada (sm_89, no NVFP4): ~3.5 GiB free after weights
	//   - 80 GB datacenter GPU (sm_120): ~65 GiB free after weights
	hw5080 := HardwareEnvelope{
		FreeVRAMBytes:  3*gib + 512<<20,
		SMArch:         120,
		HasNVFP4:       true,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}
	hw5080Tight := HardwareEnvelope{
		FreeVRAMBytes:  2*gib + 512<<20,
		SMArch:         120,
		HasNVFP4:       true,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}
	hw5080UltraTight := HardwareEnvelope{
		FreeVRAMBytes:  2 * gib,
		SMArch:         120,
		HasNVFP4:       true,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}
	hw5090 := HardwareEnvelope{
		FreeVRAMBytes:  11*gib + 512<<20,
		SMArch:         120,
		HasNVFP4:       true,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}
	hwAda16 := HardwareEnvelope{
		FreeVRAMBytes:  3*gib + 512<<20,
		SMArch:         89,
		HasNVFP4:       false,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}
	hwAda16Tight := HardwareEnvelope{
		FreeVRAMBytes:  2*gib + 512<<20,
		SMArch:         89,
		HasNVFP4:       false,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}
	hw80GB := HardwareEnvelope{
		FreeVRAMBytes:  65 * gib,
		SMArch:         120,
		HasNVFP4:       true,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}

	cases := []struct {
		name            string
		intent          ServeIntent
		hw              HardwareEnvelope
		wantProfile     ServeProfile
		wantCtx         int
		wantMinCtx      int
		wantConcurrency int
		wantDType       Inkling2KVDType
		wantChunk       int
		wantSpec        bool
		wantSpillZero   bool
	}{
		{
			name:            "rtx5080_coding_experimental_hits_128k_fp8",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: true},
			hw:              hw5080,
			wantProfile:     ServeProfileCoding,
			wantCtx:         131072,
			wantConcurrency: 1,
			wantDType:       Inkling2KVFP8,
			wantChunk:       512,
			wantSpec:        false, // 2.97 GiB KV+other in 2.975 GiB budget leaves <384 MiB for draft
			wantSpillZero:   true,
		},
		{
			name:            "rtx5080_tight_coding_steps_down_to_nvfp4_128k",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: true},
			hw:              hw5080Tight,
			wantProfile:     ServeProfileCoding,
			wantCtx:         131072,
			wantConcurrency: 1,
			wantDType:       Inkling2KVNVFP4,
			wantChunk:       512,
			wantSpec:        false,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5080_ultra_tight_coding_walks_chunk_ladder_nvfp4_128k",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: true},
			hw:              hw5080UltraTight,
			wantProfile:     ServeProfileCoding,
			wantCtx:         131072,
			wantConcurrency: 1,
			wantDType:       Inkling2KVNVFP4,
			wantChunk:       128,
			wantSpec:        false,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5080_coding_no_experimental_stays_f16_and_fits_min_ctx",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: false},
			hw:              hw5080,
			wantProfile:     ServeProfileCoding,
			wantMinCtx:      65536,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5080_chat_prefers_f16_with_spec",
			intent:          ServeIntent{Profile: ServeProfileChat, AllowExperimentalKV: true},
			hw:              hw5080,
			wantProfile:     ServeProfileChat,
			wantCtx:         32768,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        true,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5080_demo_prefers_f16_with_spec",
			intent:          ServeIntent{Profile: ServeProfileDemo, AllowExperimentalKV: true},
			hw:              hw5080,
			wantProfile:     ServeProfileDemo,
			wantCtx:         32768,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        true,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5090_coding_stays_f16_128k_with_spec",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: true},
			hw:              hw5090,
			wantProfile:     ServeProfileCoding,
			wantCtx:         131072,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        true,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5090_long_doc_256k_all_resident",
			intent:          ServeIntent{Profile: ServeProfileLongDoc, AllowExperimentalKV: true},
			hw:              hw5090,
			wantProfile:     ServeProfileLongDoc,
			wantCtx:         262144,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        true,
			wantSpillZero:   true,
		},
		{
			name:            "rtx5090_batch_4_slots_f16_32k",
			intent:          ServeIntent{Profile: ServeProfileBatch, AllowExperimentalKV: true},
			hw:              hw5090,
			wantProfile:     ServeProfileBatch,
			wantCtx:         32768,
			wantConcurrency: 4,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        false,
			wantSpillZero:   true,
		},
		{
			name:            "ada16_coding_uses_fp8_never_nvfp4",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: true},
			hw:              hwAda16,
			wantProfile:     ServeProfileCoding,
			wantCtx:         131072,
			wantConcurrency: 1,
			wantDType:       Inkling2KVFP8,
			wantChunk:       512,
			wantSpec:        false,
			wantSpillZero:   true,
		},
		{
			name:            "ada16_tight_coding_degrades_ctx_not_nvfp4",
			intent:          ServeIntent{Profile: ServeProfileCoding, AllowExperimentalKV: true},
			hw:              hwAda16Tight,
			wantProfile:     ServeProfileCoding,
			wantMinCtx:      65536,
			wantConcurrency: 1,
			wantDType:       Inkling2KVFP8,
			wantSpillZero:   true,
		},
		{
			name:            "dc80gb_long_doc_256k_f16_spec",
			intent:          ServeIntent{Profile: ServeProfileLongDoc, AllowExperimentalKV: true},
			hw:              hw80GB,
			wantProfile:     ServeProfileLongDoc,
			wantCtx:         262144,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        true,
			wantSpillZero:   true,
		},
		{
			name:            "dc80gb_batch_8_slots_f16_32k",
			intent:          ServeIntent{Profile: ServeProfileBatch, Concurrency: 8, AllowExperimentalKV: true},
			hw:              hw80GB,
			wantProfile:     ServeProfileBatch,
			wantCtx:         32768,
			wantConcurrency: 8,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        false,
			wantSpillZero:   true,
		},
		{
			name:            "default_empty_profile_is_coding",
			intent:          ServeIntent{AllowExperimentalKV: true},
			hw:              hw5090,
			wantProfile:     ServeProfileCoding,
			wantCtx:         131072,
			wantConcurrency: 1,
			wantDType:       Inkling2KVF16,
			wantChunk:       512,
			wantSpec:        true,
			wantSpillZero:   true,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			plan, err := ResolveInkling2ServeIntent(hp, tc.intent, tc.hw)
			if err != nil {
				t.Fatalf("ResolveInkling2ServeIntent() error = %v", err)
			}
			t.Logf("resolved: %s (est=%.2f GiB headroom=%.2f GiB)",
				plan.Summary, float64(plan.EstimatedBytes)/float64(gib), float64(plan.HeadroomBytes)/float64(gib))

			if plan.Profile != tc.wantProfile {
				t.Errorf("Profile = %q, want %q", plan.Profile, tc.wantProfile)
			}
			if tc.wantCtx > 0 && plan.Ctx != tc.wantCtx {
				t.Errorf("Ctx = %d, want %d", plan.Ctx, tc.wantCtx)
			}
			if tc.wantMinCtx > 0 && plan.Ctx < tc.wantMinCtx {
				t.Errorf("Ctx = %d, want >= %d", plan.Ctx, tc.wantMinCtx)
			}
			if plan.Concurrency != tc.wantConcurrency {
				t.Errorf("Concurrency = %d, want %d", plan.Concurrency, tc.wantConcurrency)
			}
			if plan.KVDType != tc.wantDType {
				t.Errorf("KVDType = %s, want %s", plan.KVDType, tc.wantDType)
			}
			if tc.wantChunk > 0 && plan.PrefillChunk != tc.wantChunk {
				t.Errorf("PrefillChunk = %d, want %d", plan.PrefillChunk, tc.wantChunk)
			}
			if tc.wantCtx > 0 && plan.EnableSpec != tc.wantSpec {
				t.Errorf("EnableSpec = %t, want %t", plan.EnableSpec, tc.wantSpec)
			}
			if tc.wantSpillZero && plan.SpillLayers != 0 {
				t.Errorf("SpillLayers = %d, want 0", plan.SpillLayers)
			}
			if plan.SpillLayers == 0 {
				budget := tc.hw.FreeVRAMBytes - inkling2DeviceHeadroom(tc.hw.FreeVRAMBytes)
				if plan.EstimatedBytes > budget {
					t.Errorf("0-spill plan EstimatedBytes=%d exceeds budget=%d", plan.EstimatedBytes, budget)
				}
			}
			if plan.HeadroomBytes != tc.hw.FreeVRAMBytes-plan.EstimatedBytes {
				t.Errorf("HeadroomBytes = %d, want %d", plan.HeadroomBytes, tc.hw.FreeVRAMBytes-plan.EstimatedBytes)
			}
			if plan.Summary == "" {
				t.Error("Summary must not be empty")
			}
		})
	}
}

func TestResolveInkling2ServeIntentExpertOverrides(t *testing.T) {
	t.Parallel()

	const gib = int64(1) << 30
	hp := inkling2MiniKVShape()
	hw5090 := HardwareEnvelope{
		FreeVRAMBytes:  11*gib + 512<<20,
		SMArch:         120,
		HasNVFP4:       true,
		HasFP8:         true,
		HasFP16:        true,
		HasDSparkDraft: true,
	}

	nvfp4 := Inkling2KVNVFP4
	specOff := false
	plan, err := ResolveInkling2ServeIntent(hp, ServeIntent{
		Profile:             ServeProfileCoding,
		AllowExperimentalKV: false, // explicit expert override still wins on supported hardware
		ExpertOverrides: ExpertOverrides{
			Ctx:          49152,
			KVDType:      &nvfp4,
			PrefillChunk: 128,
			Spec:         &specOff,
		},
	}, hw5090)
	if err != nil {
		t.Fatalf("ResolveInkling2ServeIntent() error = %v", err)
	}
	if plan.Ctx != 49152 || plan.KVDType != Inkling2KVNVFP4 || plan.PrefillChunk != 128 || plan.EnableSpec {
		t.Fatalf("overrides not honored: %+v", plan)
	}
	if len(plan.OverridesApplied) != 4 {
		t.Fatalf("OverridesApplied = %v, want 4 entries", plan.OverridesApplied)
	}
	if !strings.Contains(plan.Summary, "overrides:") {
		t.Errorf("Summary %q should note applied overrides", plan.Summary)
	}

	// Unsupported hardware KV dtype override must return an error.
	hwAda := HardwareEnvelope{FreeVRAMBytes: 4 * gib, SMArch: 89, HasFP8: true, HasFP16: true}
	if _, err := ResolveInkling2ServeIntent(hp, ServeIntent{
		Profile:         ServeProfileCoding,
		ExpertOverrides: ExpertOverrides{KVDType: &nvfp4},
	}, hwAda); err == nil {
		t.Fatal("expected error when overriding nvfp4 on sm_89 without HasNVFP4")
	}

	// Over-budget expert ctx override must report honest SpillLayers > 0.
	hwTiny := HardwareEnvelope{FreeVRAMBytes: 1 * gib, SMArch: 120, HasFP16: true, HasFP8: true, HasNVFP4: true}
	spilled, err := ResolveInkling2ServeIntent(hp, ServeIntent{
		Profile:         ServeProfileCoding,
		ExpertOverrides: ExpertOverrides{Ctx: 131072},
	}, hwTiny)
	if err != nil {
		t.Fatalf("unexpected error on spilled override: %v", err)
	}
	if spilled.SpillLayers <= 0 {
		t.Errorf("expected SpillLayers > 0 for 128k f16 on 1 GiB free, got %d", spilled.SpillLayers)
	}
	if !strings.Contains(spilled.Summary, "spilled") {
		t.Errorf("Summary %q should report spilled residency", spilled.Summary)
	}
}

func TestResolveInkling2ServeIntentErrors(t *testing.T) {
	t.Parallel()

	hp := inkling2MiniKVShape()
	hw := HardwareEnvelope{FreeVRAMBytes: 4 << 30, SMArch: 120, HasFP16: true}

	if _, err := ResolveInkling2ServeIntent(hp, ServeIntent{Profile: "unknown"}, hw); err == nil {
		t.Error("expected error for unknown profile")
	}
	if _, err := ResolveInkling2ServeIntent(hp, ServeIntent{}, HardwareEnvelope{FreeVRAMBytes: 0}); err == nil {
		t.Error("expected error for zero FreeVRAMBytes")
	}
	if _, err := ResolveInkling2ServeIntent(model.Inkling2LoopHParams{}, ServeIntent{}, hw); err == nil {
		t.Error("expected error for invalid HParams")
	}
}
