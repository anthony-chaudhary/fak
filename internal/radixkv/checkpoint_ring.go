package radixkv

import (
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// CheckpointRing is a bounded per-lane ring of KV context checkpoints taken at
// checkpoints (tools/server --ctx-checkpoints N, default 32, and
// --checkpoint-min-step, default 8192 tokens; save/restore/erase slot endpoints,
//
// A lane (session) keeps a bounded ring of checkpoints. Each checkpoint is a
// cheap page-set descriptor: the list of physical block IDs that back the lane's
// KV up to a token boundary, plus a retained reference on each block so the
// snapshot's pages stay resident even after the lane advances past them. No KV
// bytes are copied at save time — only the descriptor and refcounts move.
//
// Restore re-pins a checkpoint's page set (retains its blocks) and truncates the
// lane back to that boundary (releasing the lane's current, longer page set), so
// a branch/rollback is O(checkpoint) instead of O(re-prefill).
//
// The ring is opt-in and behavior-neutral by default: a zero-value ring is
// inert, and nothing here runs unless a caller constructs one and drives it.
var (
	// ErrCheckpointRingNil is returned when an operation is called on a nil ring.
	ErrCheckpointRingNil = errors.New("radixkv: nil checkpoint ring")
	// ErrCheckpointUnknown is the fail-closed refusal for restoring or erasing a
	// checkpoint id that is not present in the lane's ring.
	ErrCheckpointUnknown = errors.New("radixkv: unknown checkpoint")
	// ErrCheckpointTooClose is returned when a save would land closer than the
	// configured minimum step to the lane's most recent checkpoint.
	ErrCheckpointTooClose = errors.New("radixkv: checkpoint closer than min step")
	// ErrLaneNotAdvanced is returned when a save is requested for a lane that has
	// not yet advanced to a token boundary with a page set.
	ErrLaneNotAdvanced = errors.New("radixkv: lane has no current page set to checkpoint")
	// ErrLaneRegression is returned when Advance moves a lane backwards.
	ErrLaneRegression = errors.New("radixkv: lane token position regressed")
	// ErrCheckpointBranchSameLane is returned when Branch targets the source lane.
	ErrCheckpointBranchSameLane = errors.New("radixkv: branch target lane must differ from source lane")
)

// CheckpointID is a ring-scoped, monotonically increasing handle for one saved
// checkpoint. It is opaque to callers and only valid within the ring that minted
// it; a handle for an evicted or erased checkpoint no longer validates.
type CheckpointID uint64

// DefaultCheckpointRingCapacity is the default per-lane checkpoint ring size,
// matching llama.cpp's --ctx-checkpoints default of 32.
const DefaultCheckpointRingCapacity = 32

// DefaultCheckpointMinStep is the default minimum token distance between
// consecutive checkpoints, matching llama.cpp's --checkpoint-min-step default of
// 8192 tokens.
const DefaultCheckpointMinStep = 8192

// checkpoint is one saved page-set descriptor at a token boundary.
type checkpoint struct {
	id       CheckpointID
	tokenPos int
	pageSet  []int // physical block IDs backing the prefix up to tokenPos
}

// laneState is the ring's view of one lane's current (un-checkpointed) position
// and page set.
type laneState struct {
	pos     int
	pageSet []int // retained by the ring while the lane is at this position
}

// CheckpointRing is a bounded, per-lane ring of KV context checkpoints.
type CheckpointRing struct {
	mu       sync.Mutex
	alloc    *ctxmmu.PhysicalBlockAllocator
	capacity int
	minStep  int

	nextID   uint64
	lanes    map[string]*laneState
	checkpts map[string]map[CheckpointID]*checkpoint
	// order tracks, per lane, checkpoint ids in save order (oldest first) so the
	// ring can evict the oldest when it exceeds capacity.
	order map[string][]CheckpointID
}

// NewCheckpointRing constructs a checkpoint ring over the given physical block
// allocator. capacity <= 0 falls back to DefaultCheckpointRingCapacity and
// minStep <= 0 to DefaultCheckpointMinStep.
func NewCheckpointRing(alloc *ctxmmu.PhysicalBlockAllocator, capacity, minStep int) *CheckpointRing {
	if capacity <= 0 {
		capacity = DefaultCheckpointRingCapacity
	}
	if minStep <= 0 {
		minStep = DefaultCheckpointMinStep
	}
	return &CheckpointRing{
		alloc:    alloc,
		capacity: capacity,
		minStep:  minStep,
		lanes:    make(map[string]*laneState),
		checkpts: make(map[string]map[CheckpointID]*checkpoint),
		order:    make(map[string][]CheckpointID),
	}
}

// Capacity returns the configured per-lane ring capacity.
func (r *CheckpointRing) Capacity() int {
	if r == nil {
		return 0
	}
	return r.capacity
}

// MinStep returns the configured minimum token distance between checkpoints.
func (r *CheckpointRing) MinStep() int {
	if r == nil {
		return 0
	}
	return r.minStep
}

// Advance moves a lane to a new token boundary, adopting pageSet as its current
// page set. The ring retains the new page set and releases the lane's previous
// one. tokenPos must be >= the lane's current position (a lane only grows here;
// moving backwards is a Restore, not an Advance).
func (r *CheckpointRing) Advance(lane string, tokenPos int, pageSet []int) error {
	if r == nil {
		return ErrCheckpointRingNil
	}
	if tokenPos < 0 {
		return ErrLaneRegression
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	cur, ok := r.lanes[lane]
	if ok && tokenPos < cur.pos {
		return ErrLaneRegression
	}

	// Retain the new page set before releasing the old, so a block shared between
	// the two stays resident across the swap.
	if _, err := retainBlocks(r.alloc, pageSet); err != nil {
		return err
	}
	if ok {
		r.releasePageSet(cur.pageSet)
	}
	r.lanes[lane] = &laneState{pos: tokenPos, pageSet: append([]int(nil), pageSet...)}
	return nil
}

// Save snapshots the lane's current page-set descriptor at its current token
// boundary and appends it to the lane's ring. It is a cheap descriptor copy plus
// a retained reference on each block — no KV bytes move.
//
// Save refuses (fail-closed) when the lane has not advanced to a boundary
// (ErrLaneNotAdvanced) or when the boundary is closer than MinStep to the lane's
// most recent checkpoint (ErrCheckpointTooClose). When the ring is full the
// oldest checkpoint is evicted to make room.
func (r *CheckpointRing) Save(lane string) (CheckpointID, error) {
	if r == nil {
		return 0, ErrCheckpointRingNil
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	cur, ok := r.lanes[lane]
	if !ok || len(cur.pageSet) == 0 {
		return 0, ErrLaneNotAdvanced
	}

	// Enforce the minimum step against the most recent checkpoint in the ring.
	if ids := r.order[lane]; len(ids) > 0 {
		last := r.checkpts[lane][ids[len(ids)-1]]
		if cur.pos-last.tokenPos < r.minStep {
			return 0, ErrCheckpointTooClose
		}
	}

	// Retain the snapshot's page set independently of the lane's current hold so
	// the checkpoint survives the lane advancing past it.
	if _, err := retainBlocks(r.alloc, cur.pageSet); err != nil {
		return 0, err
	}

	r.nextID++
	id := CheckpointID(r.nextID)
	cp := &checkpoint{
		id:       id,
		tokenPos: cur.pos,
		pageSet:  append([]int(nil), cur.pageSet...),
	}
	if r.checkpts[lane] == nil {
		r.checkpts[lane] = make(map[CheckpointID]*checkpoint)
	}
	r.checkpts[lane][id] = cp
	r.order[lane] = append(r.order[lane], id)

	// Bounded ring: evict the oldest checkpoint(s) until within capacity.
	for len(r.order[lane]) > r.capacity {
		r.evictLocked(lane, r.order[lane][0])
	}
	return id, nil
}

// Restore re-pins the checkpoint's page set and truncates the lane back to that
// boundary: the lane's current (longer) page set is released and the checkpoint's
// page set becomes the lane's current state. It returns the lane's new token
// position. Restoring an unknown checkpoint fails closed with ErrCheckpointUnknown.
func (r *CheckpointRing) Restore(lane string, id CheckpointID) (int, error) {
	if r == nil {
		return 0, ErrCheckpointRingNil
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	cp, ok := r.checkpts[lane][id]
	if !ok {
		return 0, ErrCheckpointUnknown
	}

	// Re-pin the snapshot's page set for the lane, then release the lane's
	// current page set. Retain before release so shared blocks stay resident.
	if _, err := retainBlocks(r.alloc, cp.pageSet); err != nil {
		return 0, err
	}
	if cur, ok := r.lanes[lane]; ok {
		r.releasePageSet(cur.pageSet)
	}
	r.lanes[lane] = &laneState{pos: cp.tokenPos, pageSet: append([]int(nil), cp.pageSet...)}
	return cp.tokenPos, nil
}

// RestoreLatest restores the most recent checkpoint in lane's ring (O(checkpoint)
// undo/retry) and returns its id and token position. Fails closed with
// ErrCheckpointUnknown when the lane has no checkpoints.
func (r *CheckpointRing) RestoreLatest(lane string) (CheckpointID, int, error) {
	if r == nil {
		return 0, 0, ErrCheckpointRingNil
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	ids := r.order[lane]
	if len(ids) == 0 {
		return 0, 0, ErrCheckpointUnknown
	}
	id := ids[len(ids)-1]
	cp := r.checkpts[lane][id]
	if _, err := retainBlocks(r.alloc, cp.pageSet); err != nil {
		return 0, 0, err
	}
	if cur, ok := r.lanes[lane]; ok {
		r.releasePageSet(cur.pageSet)
	}
	r.lanes[lane] = &laneState{pos: cp.tokenPos, pageSet: append([]int(nil), cp.pageSet...)}
	return id, cp.tokenPos, nil
}

// Branch forks dstLane from srcLane's checkpoint id in O(checkpoint) without
// mutating srcLane or copying KV bytes: dstLane adopts and retains cp.pageSet at
// cp.tokenPos. Fails closed on nil ring, empty/identical lanes, or unknown id.
func (r *CheckpointRing) Branch(srcLane, dstLane string, id CheckpointID) (int, error) {
	if r == nil {
		return 0, ErrCheckpointRingNil
	}
	if srcLane == "" || dstLane == "" || srcLane == dstLane {
		return 0, ErrCheckpointBranchSameLane
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	cp, ok := r.checkpts[srcLane][id]
	if !ok {
		return 0, ErrCheckpointUnknown
	}
	if _, err := retainBlocks(r.alloc, cp.pageSet); err != nil {
		return 0, err
	}
	if cur, ok := r.lanes[dstLane]; ok {
		r.releasePageSet(cur.pageSet)
	}
	r.lanes[dstLane] = &laneState{pos: cp.tokenPos, pageSet: append([]int(nil), cp.pageSet...)}
	return cp.tokenPos, nil
}

// Erase removes a single checkpoint from the lane's ring, releasing its retained
// page-set references. Erasing an unknown checkpoint fails closed with
// ErrCheckpointUnknown.
func (r *CheckpointRing) Erase(lane string, id CheckpointID) error {
	if r == nil {
		return ErrCheckpointRingNil
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, ok := r.checkpts[lane][id]; !ok {
		return ErrCheckpointUnknown
	}
	r.evictLocked(lane, id)
	return nil
}

// EraseLane drops every checkpoint and the current state for a lane, releasing
// all retained references. It is idempotent.
func (r *CheckpointRing) EraseLane(lane string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if cur, ok := r.lanes[lane]; ok {
		r.releasePageSet(cur.pageSet)
		delete(r.lanes, lane)
	}
	for _, id := range r.order[lane] {
		if cp := r.checkpts[lane][id]; cp != nil {
			r.releasePageSet(cp.pageSet)
		}
	}
	delete(r.checkpts, lane)
	delete(r.order, lane)
}

// Len returns the number of checkpoints currently held for a lane.
func (r *CheckpointRing) Len(lane string) int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.order[lane])
}

// Latest reports the lane's most recent checkpoint id and its token position.
// ok is false when the lane has no checkpoints.
func (r *CheckpointRing) Latest(lane string) (CheckpointID, int, bool) {
	if r == nil {
		return 0, 0, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	ids := r.order[lane]
	if len(ids) == 0 {
		return 0, 0, false
	}
	cp := r.checkpts[lane][ids[len(ids)-1]]
	return cp.id, cp.tokenPos, true
}

// TokenPosition returns the lane's current (post-restore/advance) token position.
// ok is false when the lane has not advanced.
func (r *CheckpointRing) TokenPosition(lane string) (int, bool) {
	if r == nil {
		return 0, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	cur, ok := r.lanes[lane]
	if !ok {
		return 0, false
	}
	return cur.pos, true
}

// releasePageSet releases the ring's retained references on a page set given as
// physical block IDs. A block is freed back to the allocator only when its
// reference count reaches zero (i.e. the tree and any other holders have also
// released it). Callers must hold r.mu.
func (r *CheckpointRing) releasePageSet(pageSet []int) {
	if r.alloc == nil {
		return
	}
	for _, bID := range pageSet {
		if blk, err := r.alloc.GetBlock(bID); err == nil {
			if blk.Release() <= 0 {
				r.alloc.Free(bID)
			}
		}
	}
}

// evictLocked removes one checkpoint from a lane's ring and releases its
// retained page-set references. Callers must hold r.mu.
func (r *CheckpointRing) evictLocked(lane string, id CheckpointID) {
	if cp, ok := r.checkpts[lane][id]; ok {
		r.releasePageSet(cp.pageSet)
		delete(r.checkpts[lane], id)
	}
	// Remove id from the order slice.
	ids := r.order[lane]
	for i, v := range ids {
		if v == id {
			r.order[lane] = append(ids[:i], ids[i+1:]...)
			break
		}
	}
	if len(r.checkpts[lane]) == 0 {
		delete(r.checkpts, lane)
	}
}

// String implements fmt.Stringer for a checkpoint id.
func (id CheckpointID) String() string {
	return fmt.Sprintf("ckpt-%d", uint64(id))
}
