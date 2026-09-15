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
		cpuOffloadArm     bool
		q4k               bool
		wantRefuse        bool
		wantMentions      string
	}{
		{
			name:    "device cpu-offload arm carries the seam",
			backend: quantized, cpuOffloadExperts: true, cpuOffloadArm: true, q4k: false,
			wantRefuse: false,
		},
		{
			// That arm raises its own UploadDtype message naming --cpu-offload-experts, so this
			// guard must not pre-empt it with a different one.
			name:    "device cpu-offload arm defers its own upload-capability message",
			backend: plain, cpuOffloadExperts: true, cpuOffloadArm: true, q4k: false,
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
			// #13116 closed the other hole: the host offload arm is now reachable with no device
			// backend, and it carries the expert-shard seam, so a qualified artifact is admitted.
			name:    "CPU path with a qualified --cpu-offload-experts arm carries the seam",
			backend: nil, cpuOffloadExperts: true, cpuOffloadArm: true, q4k: false,
			wantRefuse: false,
		},
		{
			// A nil backend + --cpu-offload-experts on an UNQUALIFIED artifact selects no seam arm.
			name:    "CPU path with an unqualified --cpu-offload-experts artifact selects no seam-carrying arm",
			backend: nil, cpuOffloadExperts: true, cpuOffloadArm: false, q4k: false,
			wantRefuse: true, wantMentions: "FAK_Q4K=1",
		},
		{
			name:    "CPU path with neither switch",
			backend: nil, cpuOffloadExperts: false, q4k: false,
			wantRefuse: true, wantMentions: "FAK_Q4K=1",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := serveShardSeamRefusal(tc.backend, tc.cpuOffloadExperts, tc.cpuOffloadArm, tc.q4k)
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
	if err := serveShardSeamRefusal(nil, true, false, false); err == nil {
		t.Fatal("nil backend + an UNQUALIFIED --cpu-offload-experts artifact must refuse, not fail open into the lean CPU loader")
	}
	// #13116: a nil backend + a QUALIFIED offload arm carries the seam and must be admitted.
	if err := serveShardSeamRefusal(nil, true, true, false); err != nil {
		t.Fatalf("nil backend + a qualified --cpu-offload-experts arm now carries the seam, want nil, got %v", err)
	}
}
