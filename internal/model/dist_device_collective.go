package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// dist_device_collective.go — devicePGCollective bridges the multi-PROCESS NCCL process group
// (compute.ProcessGroupBackend, cuda_collective_pg.go) to the model.Collective seam, so a
// sharded EP serve process can reduce its routed-expert partial through a REAL cross-GPU
// device-tensor reduce instead of the host DistComm reduce (distCommCollective in
// dist_collective.go). It is the device-process-group twin of BackendCollective
// (collective_bridge.go), which bridges the single-process compute.CollectiveBackend instead.
//
// THE JOIN ITSELF (ProcessGroupUniqueID -> DistComm.BroadcastFromRoot -> InitProcessGroup) is
// NOT this file's job — it is a one-time rendezvous cmd/fak/serve.go performs once, over the
// already-open DistComm group from dialEPGroup, before constructing this adapter. This file is
// only the per-collective-call bridge, the same split BackendCollective/NewBackendCollective
// already draws between "join the seam" and "reduce through it".
//
// SHAPE. Like distCommCollective (and unlike BackendCollective), this process holds only ITS
// OWN part — a genuine distributed collective cannot see a peer's data — so AllReduceSum here
// accepts exactly one part, uploads it as a device tensor, reduces through
// ProcessGroupBackend.AllReduceSumPG, and reads the result back. AllGather refuses: the
// rank-local EP forward only ever all-reduces its [H] routed partial (mirrors
// distCommCollective's AllGather refusal, same reason).
//
// HONESTY. This adapter is unverified end-to-end: compute.ProcessGroupBackend has no
// implementation reachable from a GPU-free host (cuda_collective_pg.go builds only under
// -tags cuda,nccl on a real CUDA+NCCL toolchain), so the actual cross-process device reduce
// this file drives has never been witnessed on real GPUs. What IS verified here is that the
// adapter's Go-level shape/error contract matches distCommCollective's.
type devicePGCollective struct {
	be compute.Backend
	pg compute.ProcessGroupBackend
}

// NewDevicePGCollective wraps a backend already joined to an NCCL process group
// (ProcessGroupBackend.InitProcessGroup already called by the caller) as a model.Collective for
// the sharded EP decode path (SetExpertParallelCollective). Fails closed if be is nil or does
// not implement ProcessGroupBackend — the caller is expected to have already gated on this via
// a type-assert before performing the join rendezvous, so this is defense-in-depth, not the
// primary gate.
func NewDevicePGCollective(be compute.Backend) (Collective, error) {
	if be == nil {
		return nil, fmt.Errorf("model: NewDevicePGCollective got a nil backend")
	}
	pg, ok := be.(compute.ProcessGroupBackend)
	if !ok {
		return nil, fmt.Errorf("model: backend %q does not implement the ProcessGroupBackend seam", be.Name())
	}
	return devicePGCollective{be: be, pg: pg}, nil
}

func (d devicePGCollective) AllReduceSum(parts [][]float32) ([]float32, error) {
	if len(parts) != 1 {
		return nil, fmt.Errorf("model: devicePGCollective.AllReduceSum expects this rank's single part, got %d (a distributed collective holds only its own part)", len(parts))
	}
	t := compute.NewF32(d.be, []int{len(parts[0])}, parts[0])
	up := d.be.Upload(t, compute.F32)
	out, err := d.pg.AllReduceSumPG(up)
	if err != nil {
		return nil, err
	}
	return d.be.Read(out), nil
}

func (d devicePGCollective) AllGather(parts [][]float32, p TPPlan) ([]float32, error) {
	return nil, fmt.Errorf("model: devicePGCollective does not implement AllGather (expert-parallel reduces [H] partials through AllReduceSum, it never gathers expert bands)")
}

// devicePGCollective is a drop-in for the model Collective seam.
var _ Collective = devicePGCollective{}
