package model

import (
	"errors"
	"fmt"
	"os"
	"strings"
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

// v4CollectiveBuildFix is the COMPLETE recipe both build-gate misses below point at. It names
// the NATIVE step on purpose: `-tags cuda,nccl` alone does not link, because the NCCL objects
// (cuda_nccl.cu / cuda_nccl_pg.cu) only enter libfakcuda.a when build_cuda.sh runs with
// FAK_CUDA_NCCL=1, which is also what adds -lnccl to the cgo link. A refusal that stops at the
// Go tag therefore hands the operator a command that dies in a cgo link error instead of the
// rung — the same misdirection this file already fixed one gate earlier, just one rung later,
// and it burns the scarce multi-GPU window that is the whole cost of this acceptance run.
const v4CollectiveBuildFix = "rebuild with FAK_CUDA_NCCL=1 bash internal/compute/build_cuda.sh build, then go test -tags cuda,nccl"

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
	return admitV4DeviceCollectiveVia(world, compute.Lookup)
}

// admitV4DeviceCollectiveVia is admitV4DeviceCollective over an INJECTABLE backend lookup, so
// the refusal classification below is witnessable on an ordinary CPU host. Without the seam the
// build-tag rungs are only reachable on hardware we cannot get to, which is precisely how they
// stayed mis-typed.
func admitV4DeviceCollectiveVia(world int, lookup func(string) (compute.Backend, bool)) (compute.Backend, error) {
	if world < 2 {
		return nil, fmt.Errorf("device collective rung needs world >= 2, got %d", world)
	}
	// WHY BOTH MISSES BELOW ARE TYPED CAPACITY REFUSALS, AND WHY THEY NAME BUILD TAGS.
	// Neither miss means "the collective path is broken" — it means "this BINARY cannot express
	// the rung", which is the same class as "this box has < 2 GPUs" and must stay errors.Is-able
	// as CAPACITY_REFUSED. The messages name the missing build tag rather than the host because
	// the admission is gated by THREE different compile gates, not by hardware:
	// internal/compute/cuda.go is //go:build cuda (a default untagged build registers no cuda
	// backend on ANY node, GPU or not), and InitCollective is defined ONLY in
	// cuda_collective.go, //go:build cuda && nccl (so -tags cuda alone still has no seam).
	// An "on this host" message sends the operator hunting the node; the fix is the build.
	be, ok := lookup("cuda")
	if !ok {
		return nil, fmt.Errorf("%w: no cuda backend registered — internal/compute/cuda.go is //go:build cuda, so an untagged build registers none on ANY node, GPU or not; %s", errV4CollectiveCapacityRefused, v4CollectiveBuildFix)
	}
	init, ok := be.(compute.CollectiveInitializer)
	if !ok {
		return nil, fmt.Errorf("%w: backend %q lacks the CollectiveInitializer seam — InitCollective exists only in internal/compute/cuda_collective.go (//go:build cuda && nccl), so a -tags cuda build WITHOUT nccl cannot express this rung; %s", errV4CollectiveCapacityRefused, be.Name(), v4CollectiveBuildFix)
	}
	if err := init.InitCollective(world); err != nil {
		return nil, fmt.Errorf("%w: world=%d NCCL admission refused: %v", errV4CollectiveCapacityRefused, world, err)
	}
	if !be.Caps().Collective {
		return nil, fmt.Errorf("%w: backend %q did not advertise Caps().Collective after world=%d admission", errV4CollectiveCapacityRefused, be.Name(), world)
	}
	return be, nil
}

// seamlessCUDABackend is a stand-in for a `-tags cuda`-WITHOUT-`nccl` build: the cuda backend
// is registered, but no InitCollective is compiled in, so it does not satisfy
// compute.CollectiveInitializer. Only Name() is reachable on the path under test, so the
// embedded nil Backend is never called.
type seamlessCUDABackend struct{ compute.Backend }

func (seamlessCUDABackend) Name() string { return "cuda" }

// TestV4DeviceCollectiveAdmissionTypesBuildTagMisses pins the two BUILD-GATE misses as typed
// CAPACITY_REFUSALS that name the missing tag. This is the CPU-host-runnable half of #5106: the
// device witness needs >= 2 GPUs, but the acceptance command's failure MODES do not, and they
// were the trap — a `-tags cuda` build without `nccl` used to hard-t.Fatal with an untyped
// "lacks the CollectiveInitializer seam", and an untagged build blamed "this host". Both read as
// a broken collective or bad hardware when the real fix is the build command. Getting this wrong
// costs a scarce multi-GPU window, so it is pinned here where it can actually be run.
func TestV4DeviceCollectiveAdmissionTypesBuildTagMisses(t *testing.T) {
	for _, tc := range []struct {
		name   string
		lookup func(string) (compute.Backend, bool)
		want   string
	}{
		{
			name:   "untagged build registers no cuda backend",
			lookup: func(string) (compute.Backend, bool) { return nil, false },
			want:   "-tags cuda,nccl",
		},
		{
			name:   "tags cuda without nccl has no CollectiveInitializer",
			lookup: func(string) (compute.Backend, bool) { return seamlessCUDABackend{}, true },
			want:   "-tags cuda,nccl",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := admitV4DeviceCollectiveVia(2, tc.lookup)
			if err == nil {
				t.Fatal("admission succeeded without a real device collective")
			}
			// The load-bearing assertion: errors.Is must hold, or the acceptance harness cannot
			// tell "this binary/box cannot express the rung" from "the collective path broke".
			if !errors.Is(err, errV4CollectiveCapacityRefused) {
				t.Fatalf("error is not a typed CAPACITY_REFUSED: %v", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("refusal does not name the build fix %q: %v", tc.want, err)
			}
			// The Go tag alone is NOT a runnable fix: without the FAK_CUDA_NCCL=1 native build
			// there is no libfakcuda.a carrying the NCCL objects (and no -lnccl), so the tagged
			// command dies in a cgo link error on the very node this rung is scheduled for. The
			// refusal has to carry the whole recipe or it just relocates the trap.
			if !strings.Contains(err.Error(), "build_cuda.sh") {
				t.Fatalf("refusal names the Go tag but not the native NCCL build step, so following it verbatim cannot link: %v", err)
			}
			// It must NOT blame the node: the miss is a compile gate, reproducible on an 8-GPU box.
			if strings.Contains(err.Error(), "on this host") {
				t.Fatalf("refusal blames the host for a build-tag miss: %v", err)
			}
		})
	}
}

// refusingCollectiveBackend is a CORRECTLY built (-tags cuda,nccl) cuda backend whose NCCL
// admission fails — the single-GPU sanctioned-node shape, where ensureNCCL refuses world < 2.
type refusingCollectiveBackend struct{ compute.Backend }

func (refusingCollectiveBackend) Name() string { return "cuda" }
func (refusingCollectiveBackend) InitCollective(int) error {
	return errors.New("a 1-rank collective is the identity and does not need a device communicator")
}

// silentCollectiveBackend admits without error and then does NOT advertise Caps().Collective —
// the #971 honesty line holding: a cpu-ref fallback, or a communicator that never really came up.
type silentCollectiveBackend struct{ compute.Backend }

func (silentCollectiveBackend) Name() string             { return "cuda" }
func (silentCollectiveBackend) InitCollective(int) error { return nil }
func (silentCollectiveBackend) Caps() compute.Caps       { return compute.Caps{} }

// TestV4DeviceCollectiveAdmissionTypesHardwareRefusals pins the two refusal modes that fire on
// a CORRECTLY BUILT binary — i.e. the ones the sanctioned GPU node itself can hit, which the
// build-tag table above cannot reach. Both must stay typed CAPACITY_REFUSALS, and (the dual of
// the test above) neither may misdirect to the native build recipe: the binary is already right,
// so sending the operator back to build_cuda.sh burns the scarce multi-GPU window on a rebuild
// that cannot change the outcome. The node's GPU count is the cause.
//
// This is the load-bearing guard behind #5106's explicit non-claim — "a one-GPU run is explicitly
// NOT a substitute and must stay CAPACITY_REFUSED". A single-GPU node lands in the first case
// below; a silent cpu-ref fallback lands in the second. If either ever degraded to an untyped
// error, FAK_V4_CUDA_COLLECTIVE_REQUIRED=1 would still red — but a run WITHOUT it would skip on
// an untyped error it was never meant to swallow, and a cpu-ref green could be read as a device
// witness. Neither is reachable from this CPU host except through the injectable lookup seam.
func TestV4DeviceCollectiveAdmissionTypesHardwareRefusals(t *testing.T) {
	for _, tc := range []struct {
		name   string
		lookup func(string) (compute.Backend, bool)
		want   string
	}{
		{
			name:   "nccl admission refused on a single-GPU node",
			lookup: func(string) (compute.Backend, bool) { return refusingCollectiveBackend{}, true },
			want:   "NCCL admission refused",
		},
		{
			name:   "admitted but never advertised Caps().Collective",
			lookup: func(string) (compute.Backend, bool) { return silentCollectiveBackend{}, true },
			want:   "did not advertise Caps().Collective",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := admitV4DeviceCollectiveVia(2, tc.lookup)
			if err == nil {
				t.Fatal("admission succeeded without a real device collective")
			}
			// The load-bearing assertion, same as the build-tag table: errors.Is must hold, or
			// the acceptance harness cannot tell "this box has < 2 GPUs" from "the collective
			// path broke".
			if !errors.Is(err, errV4CollectiveCapacityRefused) {
				t.Fatalf("error is not a typed CAPACITY_REFUSED: %v", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("refusal does not name the hardware miss %q: %v", tc.want, err)
			}
			// The DUAL of the build-tag assertion above: a build-gate miss must name the build,
			// and a hardware miss must NOT — otherwise both classes collapse into one message
			// and the operator cannot tell which one they are holding.
			if strings.Contains(err.Error(), "build_cuda.sh") {
				t.Fatalf("hardware refusal misdirects to the native build recipe: %v", err)
			}
		})
	}
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
