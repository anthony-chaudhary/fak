package compute

// moe_linear_layout.go — linearized MoE expert weight layout: resident expert
// rows packed contiguously behind a per-expert offset table, with a launch
// planner that consumes those offsets instead of runtime gathers.
//
// ADAPT of the CUDA MoE layout linearization in ggml-org/llama.cpp PR #31047
// (`ggml/src/ggml-cuda/mmf.cu` and `ggml/src/ggml-cuda/mmf.cuh` at pin
// 69ee2ecc0150ba7d1204a7f7393dfd651569db8c, MIT): there the quantized mixed-
// input matmul and indexer dot-product kernels read contiguous per-expert row
// blocks instead of gathering across a strided expert pool, and kernel launches
// receive per-expert contiguous offsets rather than runtime gathers. This file
// ports the LAYOUT CONTRACT ONLY, re-expressed in idiomatic Go: the compacted
// pool, the offset table, the residency seam, and the planner plus its strided
// oracle. No upstream code is copied, no kernel is tuned, and nothing here is a
// speed claim — the witness is that offset-table addressing returns byte-
// identical tiles to the strided reference on fixed fixtures, and that the
// planner issues no gather step for a fully resident layout.
//
// It composes with the token-side seam MoeAlignBlockSizes (moe_align_sort.go):
// align-and-sort produces the per-expert token lists, this layout produces the
// per-expert weight spans a grouped GEMM dispatch consumes. Neither calls the
// other yet; wiring the two into one grouped launch is follow-on work outside
// this ticket's fence.

import (
	"errors"
	"fmt"
)

// MoeLinearMaxPoolBytes bounds the compacted pool extent in signed 32-bit
// bytes, keeping host int arithmetic safe on supported 32-bit software-test
// hosts, following this package's bounded byte-extent convention for geometry
// planners.
const MoeLinearMaxPoolBytes = 0x7fffffff

var (
	// ErrMoeLinearGeometry is the closed refusal for pool/selection geometry
	// that violates the uniform expert-row contract or the byte-extent bound.
	ErrMoeLinearGeometry = errors.New("compute: moe linear-layout geometry refused")
	// ErrMoeLinearUnknownExpert is the closed refusal for an expert id outside
	// [0, NumExperts).
	ErrMoeLinearUnknownExpert = errors.New("compute: moe linear-layout expert id out of range")
	// ErrMoeLinearDuplicateExpert is the closed refusal for a selection that
	// lists one expert twice.
	ErrMoeLinearDuplicateExpert = errors.New("compute: moe linear-layout selection repeats an expert")
	// ErrMoeLinearTileRange is the closed refusal for a row range that leaves
	// the expert's [0, RowsPerExpert) block.
	ErrMoeLinearTileRange = errors.New("compute: moe linear-layout tile row range refused")
	// ErrMoeLinearNotResident is the closed refusal for addressing an expert
	// the layout does not hold contiguously.
	ErrMoeLinearNotResident = errors.New("compute: moe linear-layout expert not resident")
)

// MoeLinearLayout is the compacted expert-pool storage plus its per-expert
// offset table. Active holds the resident expert ids in launch order; Pool
// holds their rows back to back, active expert i occupying bytes
// [Offsets[i], Offsets[i]+RowsPerExpert*RowBytes); Offsets carries one entry
// per active expert plus the terminating sentinel, so a consumer can bound-
// check a span without any multiplication.
type MoeLinearLayout struct {
	NumExperts    int
	RowsPerExpert int
	RowBytes      int
	Active        []int
	Offsets       []uint64
	Pool          []byte
}

// NewMoeLinearLayout compacts the selected experts' row blocks out of the
// strided pool (expert e at row e*RowsPerExpert) into one contiguous pool and
// builds the offset table. The input pool is left untouched. An empty
// selection is admitted: it yields an empty pool and the single-entry offset
// table [0].
func NewMoeLinearLayout(pool []byte, numExperts, rowsPerExpert, rowBytes int, active []int) (*MoeLinearLayout, error) {
	if numExperts <= 0 || rowsPerExpert <= 0 || rowBytes <= 0 {
		return nil, fmt.Errorf("%w: numExperts=%d rowsPerExpert=%d rowBytes=%d", ErrMoeLinearGeometry, numExperts, rowsPerExpert, rowBytes)
	}
	total := uint64(numExperts) * uint64(rowsPerExpert) * uint64(rowBytes)
	if total > MoeLinearMaxPoolBytes || int64(total) != int64(len(pool)) {
		return nil, fmt.Errorf("%w: pool %d bytes != numExperts*rowsPerExpert*rowBytes=%d", ErrMoeLinearGeometry, len(pool), total)
	}
	for _, id := range active {
		if id < 0 || id >= numExperts {
			return nil, fmt.Errorf("%w: expert %d of %d", ErrMoeLinearUnknownExpert, id, numExperts)
		}
	}
	seen := make(map[int]bool, len(active))
	for _, id := range active {
		if seen[id] {
			return nil, fmt.Errorf("%w: expert %d listed twice", ErrMoeLinearDuplicateExpert, id)
		}
		seen[id] = true
	}
	block := rowsPerExpert * rowBytes // fits int: bounded by MoeLinearMaxPoolBytes above
	compact := make([]byte, len(active)*block)
	offsets := make([]uint64, len(active)+1)
	for i, id := range active {
		src := pool[id*block : (id+1)*block]
		dst := compact[offsets[i] : offsets[i]+uint64(block)]
		copy(dst, src)
		offsets[i+1] = offsets[i] + uint64(block)
	}
	return &MoeLinearLayout{
		NumExperts:    numExperts,
		RowsPerExpert: rowsPerExpert,
		RowBytes:      rowBytes,
		Active:        append([]int(nil), active...),
		Offsets:       offsets,
		Pool:          compact,
	}, nil
}

// ActiveIndex reports the launch-order position of expertID in Active, or -1
// when the layout does not hold it.
func (l *MoeLinearLayout) ActiveIndex(expertID int) int {
	for i, id := range l.Active {
		if id == expertID {
			return i
		}
	}
	return -1
}

// Tile returns the contiguous bytes of rows [rowStart, rowEnd) of one resident
// expert, addressed through the offset table. A valid expert id the layout does
// not hold is refused as not resident; an id outside [0, NumExperts) is refused
// as unknown.
func (l *MoeLinearLayout) Tile(expertID, rowStart, rowEnd int) ([]byte, error) {
	idx := l.ActiveIndex(expertID)
	if idx < 0 {
		if expertID < 0 || expertID >= l.NumExperts {
			return nil, fmt.Errorf("%w: expert %d of %d", ErrMoeLinearUnknownExpert, expertID, l.NumExperts)
		}
		return nil, fmt.Errorf("%w: expert %d", ErrMoeLinearNotResident, expertID)
	}
	start, end, err := l.tileBounds(idx, rowStart, rowEnd)
	if err != nil {
		return nil, err
	}
	return l.Pool[start:end], nil
}

// tileBounds maps a row range of active slot idx onto byte bounds of Pool.
func (l *MoeLinearLayout) tileBounds(idx, rowStart, rowEnd int) (int, int, error) {
	if rowStart < 0 || rowEnd <= rowStart || rowEnd > l.RowsPerExpert {
		return 0, 0, fmt.Errorf("%w: rows [%d,%d) of %d", ErrMoeLinearTileRange, rowStart, rowEnd, l.RowsPerExpert)
	}
	base := int(l.Offsets[idx])
	return base + rowStart*l.RowBytes, base + rowEnd*l.RowBytes, nil
}

// MoeLinearResidency answers where one expert's rows already sit contiguously
// in target storage. It is the seam a paged or spilled expert pool plugs into;
// a MoeLinearLayout is the fully-resident case for its Active set.
type MoeLinearResidency interface {
	ResidentSpan(expertID int) (offsetBytes, lengthBytes uint64, ok bool)
}

// ResidentSpan implements MoeLinearResidency: a resident expert owns its whole
// contiguous block; anything else is not resident.
func (l *MoeLinearLayout) ResidentSpan(expertID int) (uint64, uint64, bool) {
	idx := l.ActiveIndex(expertID)
	if idx < 0 {
		return 0, 0, false
	}
	return l.Offsets[idx], uint64(l.RowsPerExpert) * uint64(l.RowBytes), true
}

// StridedExpertTile is the pre-linearization oracle: it addresses the same row
// tile in the original strided pool, where expert e sits at row
// e*RowsPerExpert and a consumer jumping between experts crosses the pool by
// strides. It exists so the offset-table addressing is held to byte identity
// against the layout it replaces, on fixed fixtures.
func StridedExpertTile(pool []byte, expertID, rowStart, rowEnd, rowsPerExpert, rowBytes int) ([]byte, error) {
	if rowsPerExpert <= 0 || rowBytes <= 0 {
		return nil, fmt.Errorf("%w: rowsPerExpert=%d rowBytes=%d", ErrMoeLinearGeometry, rowsPerExpert, rowBytes)
	}
	if expertID < 0 {
		return nil, fmt.Errorf("%w: expert %d", ErrMoeLinearUnknownExpert, expertID)
	}
	block := rowsPerExpert * rowBytes
	if uint64(block) > uint64(len(pool)) {
		return nil, fmt.Errorf("%w: pool %d bytes holds no %d-byte expert block", ErrMoeLinearGeometry, len(pool), block)
	}
	numExperts := len(pool) / block
	if expertID >= numExperts {
		return nil, fmt.Errorf("%w: expert %d of %d", ErrMoeLinearUnknownExpert, expertID, numExperts)
	}
	if rowStart < 0 || rowEnd <= rowStart || rowEnd > rowsPerExpert {
		return nil, fmt.Errorf("%w: rows [%d,%d) of %d", ErrMoeLinearTileRange, rowStart, rowEnd, rowsPerExpert)
	}
	base := expertID * block
	return pool[base+rowStart*rowBytes : base+rowEnd*rowBytes], nil
}

// MoeLinearRequest asks for one kernel launch over a contiguous row tile of
// one expert's matrix.
type MoeLinearRequest struct {
	ExpertID int
	RowStart int
	RowEnd   int
}

// MoeLinearLaunch is one planned launch: a single contiguous span of target
// storage named by the offset table, with no gather pass in front of it.
type MoeLinearLaunch struct {
	ExpertID    int
	OffsetBytes uint64
	LengthBytes uint64
}

// MoeLinearLaunchPlan is the planner output. Launches holds one entry per
// request the residency served contiguously, in request order; GatherSteps
// counts the requests it could NOT serve contiguously. The planner never
// fabricates an address for bytes it does not know to be contiguous: an
// unserved request is counted, not guessed.
type MoeLinearLaunchPlan struct {
	Launches    []MoeLinearLaunch
	GatherSteps int
}

// PlanMoeLinearLaunches turns launch requests into contiguous-span launches
// against a residency. A request is served iff its expert is resident with a
// block covering the whole expert and the tile fits inside that block; every
// other request is counted as a gather step and gets no launch. Invalid shapes
// (negative expert id, out-of-block row range) are hard refusals, not gathers.
func PlanMoeLinearLaunches(res MoeLinearResidency, rowsPerExpert, rowBytes int, requests []MoeLinearRequest) (*MoeLinearLaunchPlan, error) {
	if rowsPerExpert <= 0 || rowBytes <= 0 {
		return nil, fmt.Errorf("%w: rowsPerExpert=%d rowBytes=%d", ErrMoeLinearGeometry, rowsPerExpert, rowBytes)
	}
	fullBlock := uint64(rowsPerExpert) * uint64(rowBytes)
	plan := &MoeLinearLaunchPlan{}
	for _, req := range requests {
		if req.ExpertID < 0 {
			return nil, fmt.Errorf("%w: expert %d", ErrMoeLinearUnknownExpert, req.ExpertID)
		}
		if req.RowStart < 0 || req.RowEnd <= req.RowStart || req.RowEnd > rowsPerExpert {
			return nil, fmt.Errorf("%w: expert %d rows [%d,%d) of %d", ErrMoeLinearTileRange, req.ExpertID, req.RowStart, req.RowEnd, rowsPerExpert)
		}
		offset, length, ok := res.ResidentSpan(req.ExpertID)
		if !ok || length < fullBlock {
			plan.GatherSteps++
			continue
		}
		plan.Launches = append(plan.Launches, MoeLinearLaunch{
			ExpertID:    req.ExpertID,
			OffsetBytes: offset + uint64(req.RowStart)*uint64(rowBytes),
			LengthBytes: uint64(req.RowEnd-req.RowStart) * uint64(rowBytes),
		})
	}
	return plan, nil
}

// PlanLaunches plans launch requests against this layout's own residency:
// every admitted request is served from the offset table, so a green plan
// carries one contiguous launch per request and zero gather steps.
func (l *MoeLinearLayout) PlanLaunches(requests []MoeLinearRequest) (*MoeLinearLaunchPlan, error) {
	return PlanMoeLinearLaunches(l, l.RowsPerExpert, l.RowBytes, requests)
}
