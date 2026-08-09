package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// serve_shard_seam.go — the guard that keeps a SHARDED expert-parallel rank off a load arm with
// no expert-shard seam.
//
// A rank handed a band (--expert-parallel N, FAK_EP_RANK set) is sized for its band alone. Only
// the resident-Q4_K arms of loadServeInKernelModel thread ggufload.WithExpertShard into the
// loader; the Q8-resident and f32-resident arms have no such seam and load every tensor in the
// file. Landing a sharded rank on one of those does not fail loudly — it quietly loads the FULL
// model on every rank, which either OOMs the box or defeats the shard while reporting success.
//
// The guard therefore has to answer the question the arm switch actually answers, not a
// convenient proxy for it. The proxy it used to ask — "is --cpu-offload-experts set, or is
// FAK_Q4K in the environment?" — admitted two configurations that then fell through to a
// seamless arm:
//
//   - FAK_Q4K=1 on a device backend that advertises no quantized UploadDtype. The FAK_Q4K arm is
//     gated on that capability, so the rank skips it, skips the Q8 arm (same capability), and
//     lands on the f32-resident arm.
//   - --cpu-offload-experts with NO device backend. That arm is gated on `backend != nil`, so the
//     rank falls past it to the plain CPU arm — which, without FAK_Q4K, is the lean loader.
//
// serveShardSeamRefusal mirrors the switch's first three cases in the same order, so the two stay
// legible against each other: whatever the switch would select is what this decides about.

// serveShardSeamRefusal returns nil when a sharded rank would land on an arm carrying the
// expert-shard seam, and a typed refusal naming the arm it would otherwise fall through to.
func serveShardSeamRefusal(backend compute.Backend, cpuOffloadExperts, q4k bool) error {
	switch {
	case backend != nil && cpuOffloadExperts:
		// The direct-resident-Q4_K + host-expert arm. It carries the seam, and it raises its own
		// message about a backend with no quantized upload, so there is nothing to add here.
		return nil
	case backend != nil:
		if !q4k {
			return fmt.Errorf("fak serve: --expert-parallel sharded load requires the resident-Q4K path; set FAK_Q4K=1 (or --cpu-offload-experts) so this rank admits only its expert band")
		}
		if !backend.Caps().UploadDtype {
			return fmt.Errorf("fak serve: --expert-parallel sharded load cannot run on backend %q: it advertises no quantized UploadDtype, so the FAK_Q4K resident arm is unavailable and this rank would fall through to the f32-resident arm — which has no expert-shard seam and would load the FULL model on every rank. Use a quantized-upload backend, or --cpu-offload-experts", backend.Name())
		}
		return nil
	default:
		if !q4k {
			return fmt.Errorf("fak serve: --expert-parallel sharded load on the CPU path requires FAK_Q4K=1 — the pure-CPU resident-Q4K loader is the only CPU arm with an expert-shard seam, and --cpu-offload-experts selects nothing without a device backend")
		}
		return nil
	}
}

// serveShardSeamEnvQ4K reads the FAK_Q4K switch the arm selection keys on, so the guard and the
// switch cannot disagree about what "FAK_Q4K is set" means.
func serveShardSeamEnvQ4K() bool { return os.Getenv("FAK_Q4K") != "" }
