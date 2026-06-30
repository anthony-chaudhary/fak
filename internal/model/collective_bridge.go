package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// collective_bridge.go — BackendCollective bridges the model-side Collective seam (host
// []float32 parts, the shape ColumnParallelMatMul / RowParallelMatMul / TensorParallelFFN /
// ForwardTP reduce) to the compute.CollectiveBackend HAL seam (device []Tensor parts, the
// AllReduce/AllGather/ReduceScatter/AllToAll contract a real NCCL/RCCL communicator
// implements). It is the de-risk-now step of the native-753B Pillar-3 multi-GPU track:
// today the ONLY backend behind compute.CollectiveBackend is the cpu-ref one, so wiring the
// in-process TP primitives through it does not gain a single GPU — what it gains is that the
// model→HAL collective path is exercised and PINNED (BackendCollective == LocalCollective at
// max|Δ|=0, ForwardTP equal both ways) long before a real cross-device backend exists. When
// one lands, it is a swap behind compute.CollectiveBackend, not a rewrite of the matmul
// decomposition — and it is correct exactly when it reproduces these bytes.
//
// Why a bridge rather than reusing the host seam: the model package reduces host []float32
// row-slices (the natural shape for the sharded matRows partials), while the HAL collective
// reduces compute.Tensor values so a CUDA all-reduce never round-trips through host memory.
// BackendCollective is the adapter — upload each rank's slice as an F32 tensor, run the HAL
// collective, read the result back — and it deliberately re-does the model-side fail-closed
// width checks so its error contract matches LocalCollective's, not the HAL's (the HAL
// AllGather takes uneven bands by design and would not catch a TPPlan width mismatch). On the
// cpu-ref backend Upload is identity and Read returns the same f32, so the bridge is a thin,
// allocation-light pass-through that is byte-identical to LocalCollective; on a device backend
// the SAME calls move the bytes across the wire.
//
// Honesty: "multi-GPU" is NOT claimed by this file. compute.CollectiveBackend has only the
// cpu-ref implementation today; a non-cpu-ref (NCCL/RCCL or a TCP transport) CollectiveBackend
// is the separate Pillar-3 milestone after which a 2-process all-reduce of a device tensor may
// be claimed. This bridge only proves the seam is wired and exact.
type BackendCollective struct {
	be   compute.Backend
	coll compute.CollectiveBackend
}

// NewBackendCollective wraps a backend that advertises AND implements the optional
// compute.CollectiveBackend seam (the HAL discovery idiom: Caps().Collective + type-assert).
// It fails closed if the backend is nil or lacks the seam, so a caller learns at construction
// — not at first reduce — that this backend cannot drive tensor-parallel collectives.
// compute.Default() (cpu-ref) satisfies it today.
func NewBackendCollective(be compute.Backend) (*BackendCollective, error) {
	if be == nil {
		return nil, fmt.Errorf("model: NewBackendCollective got a nil backend")
	}
	coll, ok := be.(compute.CollectiveBackend)
	if !ok || !be.Caps().Collective {
		return nil, fmt.Errorf("model: backend %q does not implement the CollectiveBackend seam (Caps().Collective=%v)", be.Name(), be.Caps().Collective)
	}
	return &BackendCollective{be: be, coll: coll}, nil
}

func (b *BackendCollective) uploadRankF32(rank int, shape []int, data []float32) (compute.Tensor, error) {
	t := compute.NewF32(b.be, shape, data)
	if up, ok := b.be.(compute.RankUploader); ok {
		return up.UploadRank(t, compute.F32, rank)
	}
	return b.be.Upload(t, compute.F32), nil
}

// AllGather concatenates the per-rank output bands in rank order through the HAL AllGather,
// after the SAME width validation LocalCollective.AllGather performs — each parts[r] must be
// exactly p.Shards[r].Width() long and the result exactly p.Dim — so a mis-sized rank is
// rejected at the boundary with the local seam's contract instead of silently shifting every
// downstream feature. The bytes are identical to LocalCollective's: the HAL concatenates the
// same slices in the same rank order.
func (b *BackendCollective) AllGather(parts [][]float32, p TPPlan) ([]float32, error) {
	if len(parts) != len(p.Shards) {
		return nil, fmt.Errorf("model: AllGather got %d parts, plan has %d shards", len(parts), len(p.Shards))
	}
	ts := make([]compute.Tensor, len(parts))
	for r, s := range p.Shards {
		if len(parts[r]) != s.Width() {
			return nil, fmt.Errorf("model: AllGather rank %d part len = %d, want shard width %d", r, len(parts[r]), s.Width())
		}
		t, err := b.uploadRankF32(r, []int{len(parts[r])}, parts[r])
		if err != nil {
			return nil, err
		}
		ts[r] = t
	}
	out, err := b.coll.AllGather(ts)
	if err != nil {
		return nil, err
	}
	got := b.be.Read(out)
	if len(got) != p.Dim {
		return nil, fmt.Errorf("model: AllGather produced %d elements, want dim %d", len(got), p.Dim)
	}
	return got, nil
}

// AllReduceSum sums the equal-length per-rank partials in rank order through the HAL
// AllReduceSum. The HAL reduces parts[0] then += parts[r] for r=1.. over the validated
// equal-length f32 views — the IDENTICAL order LocalCollective.AllReduceSum and
// sumPartialsRankOrder use — so the result is byte-for-byte the same. It fails closed on no
// parts (mirroring the local seam) before touching the backend; ragged partials fail closed at
// the HAL boundary (collectF32's equal-length rule).
func (b *BackendCollective) AllReduceSum(parts [][]float32) ([]float32, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("model: AllReduceSum has no parts")
	}
	ts := make([]compute.Tensor, len(parts))
	for r := range parts {
		t, err := b.uploadRankF32(r, []int{len(parts[r])}, parts[r])
		if err != nil {
			return nil, err
		}
		ts[r] = t
	}
	out, err := b.coll.AllReduceSum(ts)
	if err != nil {
		return nil, err
	}
	return b.be.Read(out), nil
}

// BackendCollective is a drop-in for the model Collective seam.
var _ Collective = (*BackendCollective)(nil)

// SetExpertParallelDeviceCollective wires the device CollectiveBackend behind `be` as the
// Collective the live decode EP path reduces through (Model.epColl), so `--expert-parallel N>1`
// on a multi-GPU box all-reduces the routed-expert partials across the GPUs serve.go required
// Caps().Collective for — instead of the hardcoded single-box LocalCollective glmMoeEPFFN reduced
// through before. It is the gateway's single call (the `model` package name is shadowed there by
// the model-id string, so the BackendCollective construction lives here). Fails closed exactly
// like NewBackendCollective: a backend lacking the seam returns an error and leaves the bit-exact
// LocalCollective default in place, so the EP output stays correct (just reduced host-side).
func (m *Model) SetExpertParallelDeviceCollective(be compute.Backend) error {
	bc, err := NewBackendCollective(be)
	if err != nil {
		return err
	}
	m.SetExpertParallelCollective(bc)
	return nil
}
