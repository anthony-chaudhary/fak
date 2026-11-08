package compute

import (
	"errors"
	"math"
)

// This file adapts the segment-mark recurrent-state discipline of ashhart/TensorFold's
// DeltaNet kernels — markList/slotKernel at zig/src/core/deltanet_kernels.zig:9 and the
// GDN recurrence at zig/src/core/gdn.zig:117, both at revision
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209 (Apache-2.0; see THIRD_PARTY_NOTICES.md,
// "TensorFold GDN segment-mark recurrent state").
//
// Adapted contract:
//   - Recurrent state stores at segment marks: the GDN recurrent state matrix is
//     snapshotted into a bounded store exactly at the marked rows, so a consumer can
//     resume the recurrence from any mark without replaying earlier rows.
//   - A mark list keeps at most GDNMaxMarks (=4) ascending non-zero rows, each <= the
//     segment's total row count. It refuses empty (zero-row) marks, late marks (past the
//     segment end), repeated marks, out-of-order marks, and the fifth mark.
//   - Tree state selects the narrowest slot kernel holding its peak: given the peak
//     magnitude of a state matrix, pick the narrowest storage format in the ladder whose
//     representable range contains that peak.
//
// The types here sit beside the existing GDN dispatch seam (the DeltaNet step
// dispatch and its SIMD fast path in wave32_kda_dispatch.go and deltanet_amd64.go):
// the recurrence kernels themselves are unchanged, and this file adds the bounded
// checkpoint/selection layer around them.

// GDNMaxMarks bounds the segment mark list. Four marks give a checkpoint every quarter of
// a long segment while keeping the snapshot store fixed-size.
const GDNMaxMarks = 4

// Sentinel errors for mark-list admission, state-store access, and slot-kernel selection.
var (
	// ErrGDNTotalRowsInvalid is returned by NewGDNMarkList when the segment has no rows.
	ErrGDNTotalRowsInvalid = errors.New("gdn: total rows must be >= 1")
	// ErrGDNMarkZero is returned for an empty (row 0) mark: no token has been processed at
	// row 0, so there is no recurrent state to store there.
	ErrGDNMarkZero = errors.New("gdn: mark row must be non-zero")
	// ErrGDNMarkLate is returned for a mark past the segment end: it names a row that does
	// not exist in this segment.
	ErrGDNMarkLate = errors.New("gdn: mark row is late: it exceeds the segment's total rows")
	// ErrGDNMarkOutOfOrder is returned for a mark behind the stream position: marks must
	// ascend strictly.
	ErrGDNMarkOutOfOrder = errors.New("gdn: mark row arrived out of order: marks must ascend")
	// ErrGDNMarkRepeated is returned for a second mark on the same row.
	ErrGDNMarkRepeated = errors.New("gdn: repeated mark row")
	// ErrGDNMarkCapacity is returned for the fifth mark: the list holds at most four.
	ErrGDNMarkCapacity = errors.New("gdn: fifth mark exceeds the 4-slot mark list")
	// ErrGDNNoPendingMark is returned when Snapshot is called after every accepted mark has
	// already been stored.
	ErrGDNNoPendingMark = errors.New("gdn: no pending segment mark left to snapshot")
	// ErrGDNNoStoredState is returned when Restore asks for a row before the first stored
	// mark (or none is stored yet).
	ErrGDNNoStoredState = errors.New("gdn: no recurrent state stored at or before the requested row")
	// ErrGDNStateShape is returned when the restore target buffer has a different length
	// than the stored snapshot.
	ErrGDNStateShape = errors.New("gdn: recurrent state shape does not match the stored snapshot")
	// ErrGDNSlotNaN is returned when the peak magnitude is NaN: no slot format holds NaN.
	ErrGDNSlotNaN = errors.New("gdn: slot selection peak is NaN")
	// ErrGDNSlotOverflow is returned when the peak exceeds every slot kernel's capacity.
	ErrGDNSlotOverflow = errors.New("gdn: slot selection peak exceeds every slot kernel's capacity")
	// ErrGDNSlotEmptyLadder is returned when the kernel ladder itself is empty.
	ErrGDNSlotEmptyLadder = errors.New("gdn: slot kernel ladder is empty")
)

// GDNMarkList holds the accepted segment marks: at most GDNMaxMarks ascending non-zero
// rows, each <= TotalRows. Admission is one-way; there is no removal because the
// recurrence only moves forward.
type GDNMarkList struct {
	marks     [GDNMaxMarks]int
	n         int
	totalRows int
}

// NewGDNMarkList creates an empty mark list for a segment of totalRows rows. totalRows
// must be >= 1: a zero-row segment cannot hold any valid (non-zero, in-range) mark.
func NewGDNMarkList(totalRows int) (*GDNMarkList, error) {
	if totalRows < 1 {
		return nil, ErrGDNTotalRowsInvalid
	}
	return &GDNMarkList{totalRows: totalRows}, nil
}

// Add admits row as the next mark. It returns ErrGDNMarkZero for row <= 0,
// ErrGDNMarkLate for row > TotalRows, ErrGDNMarkRepeated for a second mark on the last
// admitted row, ErrGDNMarkOutOfOrder for a row behind the last admitted one, and
// ErrGDNMarkCapacity once four marks are held. On success the row becomes the new last
// mark.
func (m *GDNMarkList) Add(row int) error {
	if m.n == GDNMaxMarks {
		return ErrGDNMarkCapacity
	}
	if row <= 0 {
		return ErrGDNMarkZero
	}
	if row > m.totalRows {
		return ErrGDNMarkLate
	}
	if m.n > 0 {
		switch {
		case row == m.marks[m.n-1]:
			return ErrGDNMarkRepeated
		case row < m.marks[m.n-1]:
			return ErrGDNMarkOutOfOrder
		}
	}
	m.marks[m.n] = row
	m.n++
	return nil
}

// Len reports how many marks have been admitted.
func (m *GDNMarkList) Len() int { return m.n }

// At returns the i-th admitted mark. It panics for i outside [0, Len()).
func (m *GDNMarkList) At(i int) int {
	if i < 0 || i >= m.n {
		panic("gdn: mark index out of range")
	}
	return m.marks[i]
}

// Last returns the most recently admitted mark, or -1 when the list is empty.
func (m *GDNMarkList) Last() int {
	if m.n == 0 {
		return -1
	}
	return m.marks[m.n-1]
}

// Contains reports whether row is one of the admitted marks.
func (m *GDNMarkList) Contains(row int) bool {
	for i := 0; i < m.n; i++ {
		if m.marks[i] == row {
			return true
		}
	}
	return false
}

// TotalRows returns the segment length the list was created for.
func (m *GDNMarkList) TotalRows() int { return m.totalRows }

// Validate re-checks the full invariant set (capacity, non-zero, in-range, strictly
// ascending) and returns the first violation. It exists so callers that reconstruct or
// deserialize a list can prove it well-formed instead of trusting construction.
func (m *GDNMarkList) Validate() error {
	if m.totalRows < 1 {
		return ErrGDNTotalRowsInvalid
	}
	if m.n > GDNMaxMarks {
		return ErrGDNMarkCapacity
	}
	prev := 0
	for i := 0; i < m.n; i++ {
		row := m.marks[i]
		if row <= 0 {
			return ErrGDNMarkZero
		}
		if row > m.totalRows {
			return ErrGDNMarkLate
		}
		if row <= prev {
			if row == prev {
				return ErrGDNMarkRepeated
			}
			return ErrGDNMarkOutOfOrder
		}
		prev = row
	}
	return nil
}

// GDNSlotKernel names one storage format for a GDN tree-state slot. PeakCap is the
// largest |value| the format represents without overflow.
type GDNSlotKernel struct {
	Name    string
	Bits    uint8
	PeakCap float32
}

// GDNDefaultSlotKernels is the narrow-to-wide ladder tree state selects from. The caps are
// the exact maxima of the IEEE formats: E4M3 448, binary16 65504, bfloat16 ~3.39e38,
// binary32 MaxFloat32.
var GDNDefaultSlotKernels = []GDNSlotKernel{
	{Name: "f8e4m3", Bits: 8, PeakCap: 448},
	{Name: "f16", Bits: 16, PeakCap: 65504},
	{Name: "bf16", Bits: 16, PeakCap: 3.3895313892515355e+38},
	{Name: "f32", Bits: 32, PeakCap: math.MaxFloat32},
}

// SelectGDNSlotKernel returns the narrowest kernel in the ladder whose PeakCap holds
// |peak|, i.e. the first entry in narrow-to-wide order with PeakCap >= |peak|. A zero
// peak selects the narrowest kernel. NaN peaks return ErrGDNSlotNaN; a peak beyond every
// cap (including ±Inf) returns ErrGDNSlotOverflow; an empty ladder returns
// ErrGDNSlotEmptyLadder.
func SelectGDNSlotKernel(peak float32, kernels []GDNSlotKernel) (GDNSlotKernel, error) {
	if len(kernels) == 0 {
		return GDNSlotKernel{}, ErrGDNSlotEmptyLadder
	}
	p := math.Abs(float64(peak))
	if math.IsNaN(p) {
		return GDNSlotKernel{}, ErrGDNSlotNaN
	}
	// A caller-provided infinite cap must not qualify non-finite state.
	if math.IsInf(p, 0) {
		return GDNSlotKernel{}, ErrGDNSlotOverflow
	}
	for _, k := range kernels {
		if p <= float64(k.PeakCap) {
			return k, nil
		}
	}
	return GDNSlotKernel{}, ErrGDNSlotOverflow
}

// GDNPeakOf returns max|x| over st. An empty slice yields 0; any NaN element yields NaN
// so that downstream selection fails closed with ErrGDNSlotNaN.
func GDNPeakOf(st []float32) float32 {
	var peak float64
	sawNaN := false
	for _, x := range st {
		ax := math.Abs(float64(x))
		if math.IsNaN(ax) {
			sawNaN = true
			continue
		}
		if ax > peak {
			peak = ax
		}
	}
	if sawNaN {
		return float32(math.NaN())
	}
	return float32(peak)
}

// GDNSelectSlotForState computes the peak of a recurrent-state matrix and selects the
// narrowest slot kernel that holds it.
func GDNSelectSlotForState(st []float32, kernels []GDNSlotKernel) (GDNSlotKernel, error) {
	return SelectGDNSlotKernel(GDNPeakOf(st), kernels)
}

// GDNMarkedRecurrence snapshots the GDN recurrent state at each accepted segment mark, in
// ascending mark order, and restores the latest snapshot at or before a requested row.
// Snapshots are defensive copies: mutating the source buffer after Snapshot never corrupts
// the store.
type GDNMarkedRecurrence struct {
	marks *GDNMarkList
	snaps [GDNMaxMarks][]float32
	next  int
}

// NewGDNMarkedRecurrence binds a state store to a mark list. The mark list must already
// hold every mark the segment will use; marks added after binding are visible but their
// slots start empty until Snapshot fills them in order.
func NewGDNMarkedRecurrence(marks *GDNMarkList) *GDNMarkedRecurrence {
	return &GDNMarkedRecurrence{marks: marks}
}

// Snapshot stores a copy of st in the slot for the next pending mark and returns that
// mark's row. Call it exactly once per mark, in ascending mark order, as the recurrence
// passes each marked row. After all marks are stored it returns ErrGDNNoPendingMark.
func (r *GDNMarkedRecurrence) Snapshot(st []float32) (int, error) {
	if r.next >= r.marks.Len() {
		return 0, ErrGDNNoPendingMark
	}
	row := r.marks.At(r.next)
	snap := make([]float32, len(st))
	copy(snap, st)
	r.snaps[r.next] = snap
	r.next++
	return row, nil
}

// Restore copies the snapshot taken at the greatest marked row <= targetRow into st and
// returns that mark's row. It returns ErrGDNNoStoredState when no stored mark is at or
// before targetRow, and ErrGDNStateShape when len(st) differs from the snapshot length.
func (r *GDNMarkedRecurrence) Restore(st []float32, targetRow int) (int, error) {
	best := -1
	for i := 0; i < r.next; i++ {
		if row := r.marks.At(i); row <= targetRow {
			best = i
		} else {
			break // marks ascend: nothing further can qualify
		}
	}
	if best < 0 {
		return 0, ErrGDNNoStoredState
	}
	if len(st) != len(r.snaps[best]) {
		return 0, ErrGDNStateShape
	}
	copy(st, r.snaps[best])
	return r.marks.At(best), nil
}

// Stored reports how many marks currently hold a snapshot.
func (r *GDNMarkedRecurrence) Stored() int { return r.next }

// SlotForLatest selects the narrowest slot kernel that holds the peak of the most
// recently stored snapshot and returns that kernel with the snapshot's mark row. It
// returns ErrGDNNoStoredState when nothing is stored yet.
func (r *GDNMarkedRecurrence) SlotForLatest(kernels []GDNSlotKernel) (GDNSlotKernel, int, error) {
	if r.next == 0 {
		return GDNSlotKernel{}, 0, ErrGDNNoStoredState
	}
	row := r.marks.At(r.next - 1)
	k, err := GDNSelectSlotForState(r.snaps[r.next-1], kernels)
	return k, row, err
}
