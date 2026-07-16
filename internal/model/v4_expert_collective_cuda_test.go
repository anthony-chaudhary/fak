package model

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const v4CUDACollectiveRequiredEnv = "FAK_V4_CUDA_COLLECTIVE_REQUIRED"

// errV4CollectiveCapacityRefused is the TYPED capacity refusal this rung preserves when the
// node cannot host a true multi-rank device collective. It is deliberately an error value, not
// a t.Skip string: a caller (and the acceptance harness) can errors.Is it and tell "this box
// lacks 2 GPUs" apart from "the collective path is broken". Those two must never collapse into
// one green.
var errV4CollectiveCapacityRefused = errors.New("CAPACITY_REFUSED")

// admitV4DeviceCollective admits the production device-collective path over `world` ranks, or
// returns a typed CAPACITY_REFUSED when the hardware cannot provide it.
//
// WHY world >= 2 IS THE ONLY EXPRESSIBLE DEVICE RUNG. The CUDA backend's honesty line (#971)
// advertises Caps().Collective ONLY once a real NCCL communicator is up over MORE THAN ONE
// device (Caps() reads cudaNCCLWorld > 1), and ensureNCCL refuses world < 2 outright — "a
// 1-rank collective is the identity and does not need a device communicator". Since
// newV4CollectiveExpertTransport requires Caps().Collective, a world=1 CUDA collective
// transport is not constructible by design. A "world=1 device collective witness" would
// therefore have to either bypass the capability gate or fall back to cpu-ref, and either one
// would be a false multi-GPU witness — exactly what #971 exists to prevent. So the single-GPU
// case is reported as a typed CAPACITY_REFUSED rather than downgraded to a 1-rank green.
func admitV4DeviceCollective(world int) (compute.Backend, error) {
	if world < 2 {
		return nil, fmt.Errorf("device collective rung needs world >= 2, got %d", world)
	}
	be, ok := compute.Lookup("cuda")
	if !ok {
		return nil, fmt.Errorf("%w: exact cuda backend not registered on this host", errV4CollectiveCapacityRefused)
	}
	init, ok := be.(compute.CollectiveInitializer)
	if !ok {
		return nil, fmt.Errorf("backend %q lacks the CollectiveInitializer seam", be.Name())
	}
	if err := init.InitCollective(world); err != nil {
		return nil, fmt.Errorf("%w: world=%d NCCL admission refused: %v", errV4CollectiveCapacityRefused, world, err)
	}
	if !be.Caps().Collective {
		return nil, fmt.Errorf("%w: backend %q did not advertise Caps().Collective after world=%d admission", errV4CollectiveCapacityRefused, be.Name(), world)
	}
	return be, nil
}

// TestV4CollectiveExpertTransportCUDAMultiRank is the sanctioned-hardware acceptance rung for
// the production device-collective path: it proves a real NCCL communicator admits over >= 2
// GPUs, that each rank's partial is placed on ITS OWN device, and that the reduced selected-
// expert output reads back in deterministic rank order.
//
// Ordinary CPU hosts skip with the typed CAPACITY_REFUSED. An acceptance run on a sanctioned
// GPU node sets FAK_V4_CUDA_COLLECTIVE_REQUIRED=1, under which a refusal or a cpu-ref fallback
// is a HARD RED — that env var is what makes this rung un-skippable where it is supposed to run.
//
// NON-CLAIM: this proves device-collective admission, rank placement, and read-back. It is not
// a full-weight parity, capacity, or NCCL throughput claim.
func TestV4CollectiveExpertTransportCUDAMultiRank(t *testing.T) {
	const world = 2
	be, err := admitV4DeviceCollective(world)
	if err != nil {
		if errors.Is(err, errV4CollectiveCapacityRefused) {
			if os.Getenv(v4CUDACollectiveRequiredEnv) == "1" {
				t.Fatalf("acceptance run required a real multi-rank device collective: %v", err)
			}
			t.Skipf("%v", err)
		}
		t.Fatal(err)
	}
	if be.Class() != compute.Approx {
		t.Fatalf("device collective backend class = %v, want Approx (a Reference class here means cpu-ref, not a GPU)", be.Class())
	}

	placement, err := NewV4ExpertPlacement(pinnedV4RuntimeConfig(), world, 0)
	if err != nil {
		t.Fatal(err)
	}
	transport, err := newV4CollectiveExpertTransport(be, placement)
	if err != nil {
		t.Fatal(err)
	}
	// One owned pick per rank: rank 0 evaluates expert 0, rank 1 evaluates expert 192. The
	// collective must sum the rank-local partials in rank order.
	got, partials, err := transport.Forward(
		map[int][]V4ExpertDispatch{
			0: {{Rank: 0, Expert: 0, Weight: .25}},
			1: {{Rank: 1, Expert: 192, Weight: .5}},
		},
		func(picks []routePick) ([]float32, error) {
			return []float32{float32(picks[0].expert), picks[0].weight}, nil
		},
	)
	if err != nil {
		t.Fatalf("device collective forward: %v", err)
	}
	if partials != world {
		t.Fatalf("device collective partials = %d, want %d", partials, world)
	}
	if len(got) != 2 || got[0] != 192 || got[1] != .75 {
		t.Fatalf("device collective read-back = %v, want [192 0.75] (rank-order sum)", got)
	}
}
