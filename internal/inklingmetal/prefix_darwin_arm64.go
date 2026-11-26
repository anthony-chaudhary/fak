//go:build darwin && arm64 && cgo

package inklingmetal

/*
#include "bridge.h"
*/
import "C"
import (
	"errors"
	"math"
	"slices"
	"unsafe"
)

// prefixBoundaries bounds the chunk states a session keeps, 2 MiB each.
const prefixBoundaries = 32

type prefixIdentity struct {
	ready bool
	key   PrefixKey
	limit int
}

func (m *Model) prefixKey(scope string) (PrefixKey, int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ptr == nil {
		return PrefixKey{}, 0, errors.New("closed Inkling model")
	}
	if !m.prefix.ready {
		key, err := prefixFiles(m.path, m.draft)
		if err != nil {
			return PrefixKey{}, 0, err
		}
		var k C.FakInkKernels
		C.fak_ink_kernels(m.ptr, &k)
		key.Layers, key.KVFormat, key.StateFormat = uint32(k.layers), uint32(k.kv_format), uint32(k.state_format)
		key.ChunkRows, key.DraftWindow = uint32(k.chunk_rows), uint32(k.draft_window)
		m.prefix = prefixIdentity{ready: true, key: key, limit: int(k.context_tokens)}
	}
	key := m.prefix.key
	key.Scope = scope
	return key, m.prefix.limit, nil
}

// NewSessionFor returns a session for a request with this prompt. When the
// model's idle session last served a prompt under the same scope, the leading
// prefill chunks both prompts share stay in place, and the request's first
// Prefill, which must be given the whole prompt, computes only the rows after
// them. Scores are those of a new session. A prompt that shares less than a
// chunk, or any mismatch, prefills in full.
func (m *Model) NewSessionFor(capacity int, scope string, prompt []int32) (*Session, error) {
	if capacity <= 0 || uint64(capacity) > math.MaxUint32 {
		return nil, errors.New("invalid Inkling context capacity")
	}
	key, limit, err := m.prefixKey(scope)
	if err != nil {
		return m.NewSession(capacity)
	}
	id := key.String()
	if s := m.resume(capacity, id, prompt, key); s != nil {
		return s, nil
	}
	s, err := m.NewSession(prefixReserve(capacity, limit))
	if err != nil {
		return nil, err
	}
	s.key = id
	// The reserve is for later prompts. This request still stops at capacity.
	if err = nativeError(C.fak_ink_reset(s.ptr, C.uint32_t(capacity))); err == nil {
		err = nativeError(C.fak_ink_keep_boundaries(s.ptr, prefixBoundaries))
	}
	if err != nil {
		s.failure = err
		s.Close()
		return nil, err
	}
	return s, nil
}

func (m *Model) resume(capacity int, id string, prompt []int32, key PrefixKey) *Session {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ptr == nil || m.idle == nil || m.idleCapacity < capacity || m.idleKey != id || len(prompt) == 0 || len(prompt) > capacity {
		return nil
	}
	var shared, oldest, chunks C.uint32_t
	if nativeError(C.fak_ink_shared(m.idle, (*C.int32_t)(unsafe.Pointer(&prompt[0])), C.uint32_t(len(prompt)), &shared, &oldest, &chunks)) != nil {
		return nil
	}
	keep := reuseChunks(int(shared), len(prompt), int(oldest), int(chunks), int(key.ChunkRows), int(key.DraftWindow))
	if keep == 0 {
		return nil
	}
	ptr, reserved := m.idle, m.idleCapacity
	m.idle, m.idleCapacity, m.idleKey = nil, 0, ""
	if nativeError(C.fak_ink_rewind(ptr, C.uint32_t(capacity), C.uint32_t(keep))) != nil {
		C.fak_ink_free_session(ptr)
		return nil
	}
	held := slices.Clone(prompt[:keep*int(key.ChunkRows)])
	return &Session{model: m, ptr: ptr, capacity: reserved, key: id, held: held, whole: len(prompt), reused: len(held)}
}

// Reused is how many leading prompt tokens the session held when it opened.
func (s *Session) Reused() int { return s.reused }
