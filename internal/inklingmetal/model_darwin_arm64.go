//go:build darwin && arm64 && cgo

package inklingmetal

/*
#cgo CXXFLAGS: -std=c++20 -x objective-c++ -fobjc-arc -O2 -Wno-unguarded-availability-new -I${SRCDIR}/../../third_party/inklingmetal/runtime
#cgo LDFLAGS: -framework Foundation -framework Metal -framework IOKit -framework CoreML -framework IOSurface -framework Accelerate
#include <stdlib.h>
#include "bridge.h"
*/
import "C"
import (
	"cmp"
	"errors"
	_ "github.com/anthony-chaudhary/fak/third_party/inklingmetal"
	"math"
	"slices"
	"sync"
	"unsafe"
)

func Available() bool { return C.fak_ink_available() != 0 }

type Model struct {
	mu           sync.Mutex
	ptr          *C.FakInkModel
	idle         *C.FakInkSession
	idleCapacity int
	hasDraft     bool
}
type Session struct {
	mu       sync.Mutex
	ptr      *C.FakInkSession
	failure  error
	model    *Model
	capacity int
}

func nativeError(msg *C.char) error {
	if msg == nil {
		return nil
	}
	defer C.free(unsafe.Pointer(msg))
	return errors.New(C.GoString(msg))
}
func openModel(path, draft string) (*Model, error) {
	cp, cd := C.CString(path), C.CString(draft)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(cd))
	m := &Model{hasDraft: draft != ""}
	if err := nativeError(C.fak_ink_model(cp, cd, &m.ptr)); err != nil {
		return nil, err
	}
	return m, nil
}
func (m *Model) NewSession(capacity int) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ptr == nil {
		return nil, errors.New("closed Inkling model")
	}
	if capacity <= 0 || uint64(capacity) > math.MaxUint32 {
		return nil, errors.New("invalid Inkling context capacity")
	}
	s := &Session{model: m, capacity: capacity}
	if m.idle != nil {
		ptr, cap := m.idle, m.idleCapacity
		m.idle = nil
		m.idleCapacity = 0
		if cap >= capacity {
			if err := nativeError(C.fak_ink_reset(ptr, C.uint32_t(capacity))); err != nil {
				C.fak_ink_free_session(ptr)
				return nil, err
			}
			s.ptr = ptr
			s.capacity = cap
			return s, nil
		}
		C.fak_ink_free_session(ptr)
	}
	if err := nativeError(C.fak_ink_session(m.ptr, C.uint32_t(capacity), &s.ptr)); err != nil {
		return nil, err
	}
	// The native page table rounds up to its extent alignment. The logical
	// request still stops at capacity. Reuse only sizes we have admitted.
	return s, nil
}
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	C.fak_ink_free_session(m.idle)
	m.idle = nil
	m.idleCapacity = 0
	C.fak_ink_free_model(m.ptr)
	m.ptr = nil
	return nil
}
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ptr == nil {
		return nil
	}
	m := s.model
	m.mu.Lock()
	defer m.mu.Unlock()
	if s.failure == nil && m.ptr != nil && (m.idle == nil || s.capacity > m.idleCapacity) {
		C.fak_ink_free_session(m.idle)
		m.idle = s.ptr
		m.idleCapacity = s.capacity
	} else {
		C.fak_ink_free_session(s.ptr)
	}
	s.ptr = nil
	return nil
}
func (s *Session) ready() error {
	if s.ptr == nil {
		return errors.New("closed Inkling session")
	}
	return s.failure
}
func (s *Session) step(ids []int32, prefill bool, scores []float32, argmax []int32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return err
	}
	if len(ids) == 0 || uint64(len(ids)) > math.MaxUint32 || (!prefill && len(ids) > MaxBlock) {
		return errors.New("invalid Inkling input length")
	}
	var pre C.int
	if prefill {
		pre = 1
	}
	s.failure = nativeError(C.fak_ink_step(s.ptr, (*C.int32_t)(unsafe.Pointer(unsafe.SliceData(ids))), C.uint32_t(len(ids)), pre, (*C.float)(unsafe.Pointer(unsafe.SliceData(scores))), (*C.int32_t)(unsafe.Pointer(unsafe.SliceData(argmax)))))
	return s.failure
}
func (s *Session) Prefill(ids []int32) ([]float32, error) {
	scores := make([]float32, Vocabulary)
	if err := s.step(ids, true, scores, nil); err != nil {
		return nil, err
	}
	return scores, nil
}

// PrefillID returns a greedy target choice without transferring the vocabulary.
func (s *Session) PrefillID(ids []int32) (int32, error) {
	var out [1]int32
	if err := s.step(ids, true, nil, out[:]); err != nil {
		return 0, err
	}
	return out[0], nil
}
func (s *Session) BeginBlock(ids []int32) ([][]float32, error) {
	if len(ids) == 0 || len(ids) > MaxBlock {
		return nil, errors.New("invalid Inkling verification block")
	}
	scores := make([]float32, len(ids)*Vocabulary)
	if err := s.step(ids, false, scores, nil); err != nil {
		return nil, err
	}
	out := make([][]float32, len(ids))
	for row := range out {
		out[row] = scores[row*Vocabulary : (row+1)*Vocabulary : (row+1)*Vocabulary]
	}
	return out, nil
}
func (s *Session) BeginBlockIDs(ids []int32) ([]int32, error) {
	if len(ids) == 0 || len(ids) > MaxBlock {
		return nil, errors.New("invalid Inkling verification block")
	}
	out := make([]int32, len(ids))
	if err := s.step(ids, false, nil, out); err != nil {
		return nil, err
	}
	return out, nil
}
func (s *Session) Commit(keep int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return err
	}
	if keep < 0 || keep > MaxBlock {
		return errors.New("invalid Inkling commit length")
	}
	s.failure = nativeError(C.fak_ink_commit(s.ptr, C.uint32_t(keep)))
	return s.failure
}

// StepBatch commits one token to each session and returns each session's
// greedy next token, all in one GPU submission. Every session's result is the
// one its own single steps give. The sessions need a model opened without a
// draft and no uncommitted block. One session takes the single-session calls.
func (m *Model) StepBatch(sessions []*Session, ids []int32) ([]int32, error) {
	if len(sessions) == 1 && len(ids) == 1 && sessions[0] != nil && sessions[0].model == m {
		out, err := sessions[0].BeginBlockIDs(ids)
		if err != nil {
			return nil, err
		}
		if err := sessions[0].Commit(1); err != nil {
			return nil, err
		}
		return out, nil
	}
	return m.stepBatch(sessions, ids, false, nil)
}

// StepBatchWide is StepBatch with every session in one target pass. From three
// sessions on the kernels round a row together with the rows beside it, so a
// session's scores vary with what it is batched with and a near tie may pick
// another token than the session alone would.
func (m *Model) StepBatchWide(sessions []*Session, ids []int32) ([]int32, error) {
	return m.stepBatch(sessions, ids, true, nil)
}

// A non-nil scores receives each session's vocabulary row, in session order.
func (m *Model) stepBatch(sessions []*Session, ids []int32, wide bool, scores []float32) ([]int32, error) {
	if len(sessions) == 0 || len(sessions) > MaxLanes || len(ids) != len(sessions) ||
		(scores != nil && len(scores) != len(sessions)*Vocabulary) {
		return nil, errors.New("invalid Inkling batch width")
	}
	if m.hasDraft {
		return nil, errors.New("batched Inkling decode requires a model opened without a draft")
	}
	order := slices.Clone(sessions)
	if slices.Contains(order, nil) {
		return nil, errors.New("nil Inkling batch session")
	}
	// Overlapping batches lock their sessions in one order.
	slices.SortFunc(order, func(a, b *Session) int {
		return cmp.Compare(uintptr(unsafe.Pointer(a)), uintptr(unsafe.Pointer(b)))
	})
	if len(slices.Compact(slices.Clone(order))) != len(order) {
		return nil, errors.New("repeated Inkling batch session")
	}
	for _, s := range order {
		s.mu.Lock()
		defer s.mu.Unlock()
	}
	ptrs := make([]*C.FakInkSession, len(sessions))
	for i, s := range sessions {
		if s.model != m {
			return nil, errors.New("Inkling batch session of another model")
		}
		if err := s.ready(); err != nil {
			return nil, err
		}
		ptrs[i] = s.ptr
	}
	var onePass C.int
	if wide {
		onePass = 1
	}
	out := make([]int32, len(sessions))
	err := nativeError(C.fak_ink_step_batch(&ptrs[0], (*C.int32_t)(unsafe.Pointer(&ids[0])), C.uint32_t(len(ids)), onePass, (*C.float)(unsafe.Pointer(unsafe.SliceData(scores))), (*C.int32_t)(unsafe.Pointer(&out[0]))))
	if err != nil {
		for _, s := range sessions {
			s.failure = err
		}
		return nil, err
	}
	return out, nil
}

// batchBytes is the scratch the model holds for batched steps so far.
func (m *Model) batchBytes() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ptr == nil {
		return 0
	}
	return uint64(C.fak_ink_batch_bytes(m.ptr))
}
func (s *Session) Draft(anchor int32, depth int) ([]int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return nil, err
	}
	if anchor < 0 || anchor >= Vocabulary || depth < 1 || depth > 7 {
		return nil, errors.New("invalid Inkling draft request")
	}
	out := make([]int32, depth)
	var n C.uint32_t
	s.failure = nativeError(C.fak_ink_propose(s.ptr, C.int32_t(anchor), C.uint32_t(depth), (*C.int32_t)(unsafe.Pointer(&out[0])), &n))
	if s.failure != nil {
		return nil, s.failure
	}
	return out[:n], nil
}

// DraftTree returns the anchor and its candidate descendants in parent-before-child order.
// Each row holds token, parent index (-1 for root), and depth.
func (s *Session) DraftTree(anchor int32, depth, width int) ([][3]int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return nil, err
	}
	if anchor < 0 || anchor >= Vocabulary || depth < 1 || depth > 7 || width < 1 || width > 4 {
		return nil, errors.New("invalid Inkling draft tree")
	}
	out := make([][3]int32, 8)
	var n C.uint32_t
	s.failure = nativeError(C.fak_ink_draft_tree(s.ptr, C.int32_t(anchor), C.uint32_t(depth), C.uint32_t(width), (*C.int32_t)(unsafe.Pointer(&out[0][0])), &n))
	if s.failure != nil {
		return nil, s.failure
	}
	return out[:n], nil
}

func (s *Session) BeginTreeIDs(rows [][3]int32) ([]int32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return nil, err
	}
	if len(rows) < 1 || len(rows) > 16 {
		return nil, errors.New("invalid Inkling verification tree size")
	}
	out := make([]int32, len(rows))
	s.failure = nativeError(C.fak_ink_tree(s.ptr, (*C.int32_t)(unsafe.Pointer(&rows[0][0])), C.uint32_t(len(rows)), (*C.int32_t)(unsafe.Pointer(&out[0]))))
	if s.failure != nil {
		return nil, s.failure
	}
	return out, nil
}

// CommitTree retains the root-to-leaf path. A leaf of -1 discards the staged tree.
func (s *Session) CommitTree(leaf int) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return err
	}
	if leaf < -1 || leaf >= 16 {
		return errors.New("invalid Inkling tree leaf index")
	}
	s.failure = nativeError(C.fak_ink_commit_tree(s.ptr, C.int32_t(leaf)))
	return s.failure
}

// ProposeVerify evaluates the draft tree and the target's greedy choices in one GPU command.
// CommitTree applies the path selected by the caller. A model opened without a draft returns empty results.
func (s *Session) ProposeVerify(anchor int32, depth, width int) ([][3]int32, []int32, error) {
	rows, ids, _, err := s.proposeVerify(anchor, depth, width, 0, -1)
	return rows, ids, err
}

// ProposeVerifyCommit verifies the tree and retains the greedy input path in
// one submission. The mask includes its anchor and excludes EOS. Maximum bounds
// retained input rows. A model without a draft returns an empty tree and mask.
func (s *Session) ProposeVerifyCommit(anchor int32, depth, width, maximum int, eos int32) ([][3]int32, []int32, uint32, error) {
	if maximum < 1 || maximum > depth+1 || eos < -1 || eos >= Vocabulary || anchor == eos {
		return nil, nil, 0, errors.New("invalid Inkling terminal token or retention bound")
	}
	return s.proposeVerify(anchor, depth, width, maximum, eos)
}

func (s *Session) proposeVerify(anchor int32, depth, width, maximum int, eos int32) ([][3]int32, []int32, uint32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ready(); err != nil {
		return nil, nil, 0, err
	}
	if anchor < 0 || anchor >= Vocabulary || depth < 1 || depth > 15 || width < 1 || width > 4 {
		return nil, nil, 0, errors.New("invalid Inkling fused draft request")
	}
	rows := make([][3]int32, depth+1)
	ids := make([]int32, depth+1)
	tree := (*C.int32_t)(unsafe.Pointer(&rows[0][0]))
	target := (*C.int32_t)(unsafe.Pointer(&ids[0]))
	var n, retained C.uint32_t
	if maximum == 0 {
		s.failure = nativeError(C.fak_ink_propose_verify(s.ptr, C.int32_t(anchor), C.uint32_t(depth), C.uint32_t(width), tree, target, &n))
	} else {
		s.failure = nativeError(C.fak_ink_cycle(s.ptr, C.int32_t(anchor), C.uint32_t(depth), C.uint32_t(width), C.uint32_t(maximum), C.int32_t(eos), tree, target, &n, &retained))
	}
	if s.failure != nil {
		return nil, nil, 0, s.failure
	}
	return rows[:n], ids[:n], uint32(retained), nil
}
func estimateMemory(path, draft string, contextTokens int) (MemoryPlan, error) {
	if contextTokens < 1 || uint64(contextTokens) > math.MaxUint32 {
		return MemoryPlan{}, errors.New("invalid Inkling memory plan context")
	}
	cp, cd := C.CString(path), C.CString(draft)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(cd))
	var p C.FakInkMemory
	if err := nativeError(C.fak_ink_plan(cp, cd, C.uint32_t(contextTokens), &p)); err != nil {
		return MemoryPlan{}, err
	}
	return MemoryPlan{TargetImages: uint64(p.target_images), DraftImages: uint64(p.draft_images), PrefillScratch: uint64(p.prefill_scratch), DecodeScratch: uint64(p.decode_scratch), State: uint64(p.state), KV: uint64(p.kv)}, nil
}
