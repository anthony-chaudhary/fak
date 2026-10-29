package model

import (
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41ModelWKVEntry struct {
	packed     *V41DenseWKV
	backend    compute.V41FP8WKVBackend
	projection *V41DenseWKVProjection
}

// Model views share this owner through weightCloserState, never a global map.
// Active sessions hold the existing model-weight lease. mu orders first upload,
// projection and final retirement; native errors retain recovery handles.
type v41ModelWKVStore struct {
	mu      sync.Mutex
	entries map[string]*v41ModelWKVEntry
	closed  bool
}

func (m *Model) v41ModelWKVStore() *v41ModelWKVStore {
	if m == nil {
		return nil
	}
	weightCloserInitMu.Lock()
	owner := m.weightCloser
	weightCloserInitMu.Unlock()
	if owner == nil {
		return nil
	}
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.closed {
		return nil
	}
	return owner.v41DensePacked
}

func (m *Model) v41PackedWKVPresent(name string) bool {
	store := m.v41ModelWKVStore()
	if store == nil {
		return false
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	return !store.closed && store.entries[name] != nil
}

// InstallV41DenseWKV explicitly attaches one exact packed pair to a model before
// sessions start. It does not open the generic safetensors loader or admit any
// numerical profile. Both a named semantics request and qualified native WKV
// capability are required; the current native backend remains unqualified.
// Ordinary Q8/F32 stores are never replaced or silently reinterpreted.
func (m *Model) InstallV41DenseWKV(layer int, shard *V41PackedShard, bound int64, backend compute.V41FP8WKVBackend) error {
	if m == nil || !m.Cfg.IsDeepSeekV41() || m.Cfg.DeepSeekV41 == nil || m.Cfg.DeepSeekV41.NumericalProfile == nil || strings.TrimSpace(m.Cfg.DeepSeekV41.NumericalProfile.Semantics) == "" ||
		m.lora != nil || m.prism != nil || m.Cfg.HiddenSize != 5120 || layer < 0 || layer >= m.Cfg.NumLayers || backend == nil || !backend.Caps().DeviceMemory || !backend.V41FP8WKVSupported() || !backend.V41FP8WKVQualified() {
		return fmt.Errorf("%w: explicit qualified model WKV attachment required", ErrV41PackedUnsupported)
	}
	if !m.holdModelWeights() {
		return fmt.Errorf("%w: model weights are closing", ErrV41PackedUnsupported)
	}
	defer m.releaseWeightSession()
	name := layerName(layer, "attn.wkv.weight")
	if _, _, exists := m.residentShape(name); exists {
		return fmt.Errorf("%w: WKV already resident", ErrV41PackedUnsupported)
	}
	packed, err := shard.ReadV41DenseWKV(layer, bound)
	if err != nil {
		return err
	}
	owner := m.ensureWeightCloser()
	owner.mu.Lock()
	defer owner.mu.Unlock()
	if owner.closing || owner.closed || owner.sessions != 1 {
		return fmt.Errorf("%w: model changed during WKV intake", ErrV41PackedUnsupported)
	}
	if owner.v41DensePacked == nil {
		owner.v41DensePacked = &v41ModelWKVStore{entries: make(map[string]*v41ModelWKVEntry)}
	}
	store := owner.v41DensePacked
	store.mu.Lock()
	defer store.mu.Unlock()
	if store.closed || store.entries[name] != nil {
		return fmt.Errorf("%w: duplicate or closed WKV owner", ErrV41PackedUnsupported)
	}
	store.entries[name] = &v41ModelWKVEntry{packed: packed, backend: backend}
	return nil
}

func (owner *weightCloserState) releaseV41DensePackedStore() error {
	owner.mu.Lock()
	store := owner.v41DensePacked
	owner.mu.Unlock()
	if store == nil {
		return nil
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	store.closed = true
	var result error
	for name, entry := range store.entries {
		if entry.projection != nil {
			if err := entry.projection.Release(); err != nil {
				result = errors.Join(result, err)
				continue
			}
		}
		delete(store.entries, name)
	}
	// Keep failed handles attached to the retired owner for native recovery;
	// they are never republished to an ordinary session.
	return result
}

// v41PackedWKVProjection is reached from the canonical session projection core.
// Presence selects an exclusive route: backend mismatch, lost qualification,
// continuation requests and any native failure refuse, never host-fallback.
func (s *Session) v41PackedWKVProjection(layer int, leaf string, panel []float32, out, in, rows int, continuation *v41ProjectionContinuation) (result []float32, outcome v41DenseProjectionOutcome, cause error) {
	if leaf != "attn.wkv.weight" || s == nil || s.M == nil {
		return nil, v41ProjectionDeclined, nil
	}
	store := s.M.v41ModelWKVStore()
	if store == nil {
		return nil, v41ProjectionDeclined, nil
	}
	store.mu.Lock()
	defer store.mu.Unlock()
	entry := store.entries[layerName(layer, leaf)]
	if entry == nil {
		return nil, v41ProjectionDeclined, nil
	}
	fail := func(err error) ([]float32, v41DenseProjectionOutcome, error) { return nil, v41ProjectionError, err }
	if store.closed || continuation != nil || s.M.lora != nil || s.M.prism != nil || entry.backend != s.Backend || !entry.backend.V41FP8WKVSupported() || !entry.backend.V41FP8WKVQualified() {
		return fail(fmt.Errorf("%w: selected model WKV scope unavailable", ErrV41PackedUnsupported))
	}
	// The panel dispatcher submits at most128 real rows. Pad only the final
	// tile, including decode, with zero rows; GEMM and activation scale groups
	// are row-local, so padding cannot participate in any real row reduction.
	if rows <= 0 || rows > 128 || out != 512 || in != 5120 || len(panel) != rows*5120 {
		return fail(errV41ProjectionResult)
	}
	for _, v := range panel {
		if !finite32(v) {
			return fail(errV41ProjectionResult)
		}
	}
	s.ensureOpenBackendSession()
	opened := s.M.v41NowNanos()
	successful := 0
	// These counters cover activation upload and F32 readback only. First-bind
	// weight conversion/upload is backend-owned; cold total transfer accounting
	// requires returned backend counters, not an inferred scale-layout byte size.
	var uploaded, readback int64
	defer func() { s.M.v41NoteDenseProjection(1, 0, successful, 0, uploaded, readback, opened) }()
	if entry.projection == nil {
		p, err := entry.packed.Bind(entry.backend)
		if err != nil {
			return fail(err)
		}
		entry.projection = p
	}
	input := panel
	if rows != 128 {
		input = make([]float32, 128*5120)
		copy(input, panel)
	}
	be := entry.backend
	var owned []compute.Tensor
	defer v41ReleaseHeadTensors(be, &owned, &cause)
	x := be.Upload(compute.NewF32(be, []int{128, 5120}, input), compute.F32)
	owned = append(owned, x)
	uploaded = 128 * 5120 * 4
	y, err := entry.projection.Project(x, 128)
	if err != nil {
		return fail(err)
	}
	owned = append(owned, y)
	if y.Dtype != compute.BF16 || y.Layout != compute.RowMajor || !sameShape(y.Shape, []int{128, 512}) {
		return fail(errV41ProjectionResult)
	}
	values := be.Read(y)
	readback = 128 * 512 * 4
	if len(values) != 128*512 {
		return fail(errV41ProjectionResult)
	}
	for _, v := range values {
		if !finite32(v) {
			return fail(errV41ProjectionResult)
		}
	}
	result = append([]float32(nil), values[:rows*512]...)
	// Retire before publishing success; the shared cleanup preserves a primary
	// fault while attempting every secondary release, without any host retry.
	v41ReleaseHeadTensors(be, &owned, &cause)
	successful = rows
	return result, v41ProjectionHandled, nil
}
