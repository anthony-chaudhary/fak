package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A sharded expert-parallel rank must only ever reach a load arm that threads WithExpertShard.
// The pre-fix guard asked "--cpu-offload-experts OR FAK_Q4K?", which is not the question
// loadServeInKernelModel's switch answers: two configurations passed it and then fell through to
// an arm with no shard seam, silently loading the FULL model on every rank.
func TestServeShardSeamRefusalMatchesTheArmThatWouldBeSelected(t *testing.T) {
	quantized := serveCapBackend{Backend: compute.Default(), uploadDtype: true}
	plain := serveCapBackend{Backend: compute.Default(), uploadDtype: false}

	for _, tc := range []struct {
		name              string
		backend           compute.Backend
		cpuOffloadExperts bool
		q4k               bool
		wantRefuse        bool
		wantMentions      string
	}{
		{
			name:    "device cpu-offload arm carries the seam",
			backend: quantized, cpuOffloadExperts: true, q4k: false,
			wantRefuse: false,
		},
		{
			// That arm raises its own UploadDtype message naming --cpu-offload-experts, so this
			// guard must not pre-empt it with a different one.
			name:    "device cpu-offload arm defers its own upload-capability message",
			backend: plain, cpuOffloadExperts: true, q4k: false,
			wantRefuse: false,
		},
		{
			name:    "device FAK_Q4K arm carries the seam when the backend can take quantized uploads",
			backend: quantized, cpuOffloadExperts: false, q4k: true,
			wantRefuse: false,
		},
		{
			// THE BUG: FAK_Q4K passed the old predicate, but the FAK_Q4K arm is gated on
			// UploadDtype, so this rank skipped it, skipped the Q8 arm (same gate), and landed on
			// the f32-resident arm — no shard seam, full model on every rank.
			name:    "device FAK_Q4K without quantized upload falls through to the seamless f32 arm",
			backend: plain, cpuOffloadExperts: false, q4k: true,
			wantRefuse: true, wantMentions: "f32-resident",
		},
		{
			name:    "device with neither switch",
			backend: quantized, cpuOffloadExperts: false, q4k: false,
			wantRefuse: true, wantMentions: "FAK_Q4K=1",
		},
		{
			name:    "CPU path with FAK_Q4K carries the seam",
			backend: nil, cpuOffloadExperts: false, q4k: true,
			wantRefuse: false,
		},
		{
			// THE OTHER HOLE: --cpu-offload-experts selects nothing without a device backend (its
			// arm is gated on backend != nil), so this rank fell through to the lean CPU arm.
			name:    "CPU path with only --cpu-offload-experts selects no seam-carrying arm",
			backend: nil, cpuOffloadExperts: true, q4k: false,
			wantRefuse: true, wantMentions: "FAK_Q4K=1",
		},
		{
			name:    "CPU path with neither switch",
			backend: nil, cpuOffloadExperts: false, q4k: false,
			wantRefuse: true, wantMentions: "FAK_Q4K=1",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := serveShardSeamRefusal(tc.backend, tc.cpuOffloadExperts, tc.q4k)
			if tc.wantRefuse && err == nil {
				t.Fatal("want a refusal: this configuration reaches a load arm with no expert-shard seam, which loads the full model on every rank")
			}
			if !tc.wantRefuse && err != nil {
				t.Fatalf("want nil (this arm carries the shard seam), got %v", err)
			}
			if err != nil && !strings.Contains(err.Error(), tc.wantMentions) {
				t.Fatalf("refusal %q must name the operator's next step (%q)", err.Error(), tc.wantMentions)
			}
		})
	}
}

// A nil backend is the CPU path, not "no opinion": the guard must still decide, because the CPU
// arms differ in whether they carry the seam.
func TestServeShardSeamRefusalNeverFailsOpenOnANilBackend(t *testing.T) {
	if err := serveShardSeamRefusal(nil, true, false); err == nil {
		t.Fatal("nil backend + --cpu-offload-experts must refuse, not fail open into the lean CPU loader")
	}
}
