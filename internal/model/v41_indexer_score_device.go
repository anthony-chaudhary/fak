package model

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This session-local seam accepts already projected, normalized, rotated and
// scaled operands. Nil preserves the scalar scorer. Once selected there is no
// decline or host replay, including when downstream selection rejects NaN.
type v41IndexerScoreFunc func(layer int, q, keys, weights []float32, heads, headDim, rows int) ([]float32, error)

// Health is bound only when the device scorer may be selected: the backend
// admits the operation, or admission failed at bind time. Healthy capability
// absence binds no health seam, so the host path pays no per-call device probe
// (#13791). A bind-time admission fault keeps the seam, so the first call still
// reports the shared-backend fault instead of silently scoring on the host.
type v41IndexerScoreHealthFunc func(layer int) error

// V41IndexerScoreOperationError excludes score-boundary health failures and
// selected score/selection failures from structural-error full-history replay.
type V41IndexerScoreOperationError struct {
	Layer int
	Cause error
}

func (e *V41IndexerScoreOperationError) Error() string {
	return fmt.Sprintf("model: V4.1 indexer score boundary failed at layer %d: %v", e.Layer, e.Cause)
}

func (e *V41IndexerScoreOperationError) Unwrap() error { return e.Cause }

var errV41IndexerScoreResult = errors.New("model: invalid selected V4.1 indexer score result")

// Only the synchronous Step composer supplies this carrier. Its source can be
// the producer's layer state or a reader's shared registry; arbitrary flat-key
// callbacks never acquire publication authority.
type v41IndexerKeyView struct {
	owner  *V41AttentionState
	source int
}

const v41IndexerKeyPrefix = "v41-indexer-keys/"

// A backend whose indexer score reads only the leading rows*width elements of
// its key buffer may score a [rows,width] view of a row log with spare row
// capacity. Exact-extent backends (current Vulkan) omit it and keep the
// exact-size log.
type v41IndexerScoreSpareKeyBackend interface {
	V41IndexerScoreSpareKeyCapacity() bool
}

func v41IndexerScoreSpareKeyCapacity(b compute.Backend) bool {
	spare, ok := b.(v41IndexerScoreSpareKeyBackend)
	return ok && spare.V41IndexerScoreSpareKeyCapacity()
}

type v41IndexerKeyEntry struct {
	view     v41IndexerKeyView
	forward  *v41ForwardState
	model    *Model
	epoch    uint64
	pos      int
	rows     int
	capacity int
	width    int
	anchor   []float32
	tensor   compute.Tensor
}

type v41IndexerKeyCache struct {
	s       *Session
	ops     compute.V41RowLogBackend
	entries map[int]v41IndexerKeyEntry
	// Include orphaned handles from a prior callback binding. Rebinding cannot
	// make already-live session allocations disappear from the budget ledger.
	handles           map[string]compute.Tensor
	validatedRows     int
	descriptorLookups int
}

type v41IndexerKeyCall struct {
	cache      *v41IndexerKeyCache
	layer      int
	entry      v41IndexerKeyEntry
	key        string
	rows       [][]float32
	old        compute.Tensor
	copyRows   int
	replace    bool
	budget     int64
	extraBytes int64
	claimed    bool
}

func (c *v41IndexerKeyCache) protected(t compute.Tensor) bool {
	for key, held := range c.handles {
		live, ok := c.s.halW[key]
		if ok && live.Buf() == held.Buf() && t.Buf() == held.Buf() {
			return true
		}
	}
	return false
}

func (c *v41IndexerKeyCache) charged() (int64, bool) {
	var total int64
	for key, held := range c.handles {
		live, ok := c.s.halW[key]
		if !ok || live.Buf() != held.Buf() {
			continue
		}
		if live.Backend() != c.s.Backend || live.Dtype != compute.F32 || live.Layout != compute.RowMajor || live.Quant != nil || len(live.Shape) != 2 {
			return 0, false
		}
		bytes, ok := checkedMul(int64(live.Shape[0]), int64(live.Shape[1]), 4)
		if !ok || bytes > math.MaxInt64-total {
			return 0, false
		}
		total += bytes
	}
	return total, true
}

// Publication rows are immutable owned allocations. publish only appends;
// rollback restores/deletes the prospective end; reset/reseed replace the map.
// Thus a still-present terminal anchor identifies the complete cached prefix.
// Keeping the slice alive prevents address reuse. Cursor equality alone is not
// used as identity, and neither caller-owned flats nor snapshots carry a cache.
func (o *v41DeviceGatherOps) residentIndexerScore(m *Model, layer, pos int, rows [][]float32, view *v41IndexerKeyView, score v41IndexerScoreFunc) v41IndexerScoreFunc {
	if o == nil || o.owner == nil || o.indexerKeyBudgetBytes <= 0 || o.indexerKeyCall != nil || score == nil || view == nil || view.owner == nil || len(rows) == 0 {
		return nil
	}
	s := o.owner
	st := s.v41Forward
	if s.M != m || s.Backend == nil || s.halClosed || s.halFailure != nil || st == nil || st.callbackOwner != s || st.gatherOps != o || len(st.history) != pos ||
		layer < 0 || layer >= m.Cfg.NumLayers || view.source < 0 || view.source >= m.Cfg.NumLayers || m.Cfg.MaxPositionEmbeddings <= 0 || pos < 0 || pos >= m.Cfg.MaxPositionEmbeddings {
		return nil
	}
	plan, err := m.v41AttentionPlan(layer)
	if err != nil {
		return nil
	}
	expected, source := st.attn, plan.KVSourceLayer
	if v41OwnsCompressedRows(plan) {
		expected, source = st.layerState(layer), layer
	}
	width, end, group := m.Cfg.IndexHeadDim, len(rows), max(1, plan.kvGroupSize(m.Cfg))
	if view.owner != expected || view.source != source || width <= 0 || view.owner.indexWidth() != width || view.owner.indexPublishedEnd[source] != end ||
		end > (pos+1)/group {
		return nil
	}
	ops, ok := s.Backend.(compute.V41RowLogBackend)
	if !ok || !ops.V41RowLogAvailable() {
		return nil
	}
	c := o.indexerKeys
	if c == nil {
		c = &v41IndexerKeyCache{s: s, ops: ops, entries: map[int]v41IndexerKeyEntry{}, handles: map[string]compute.Tensor{}}
		for key, t := range s.halW {
			if strings.HasPrefix(key, v41IndexerKeyPrefix) {
				c.handles[key] = t
			}
		}
	}
	charged, valid := c.charged()
	if !valid || charged > o.indexerKeyBudgetBytes {
		return nil
	}
	key := fmt.Sprintf("%s%d", v41IndexerKeyPrefix, layer)
	old := s.halW[key]
	entry, hit := c.entries[layer]
	hit = hit && entry.view == *view && entry.forward == st && entry.model == m && entry.epoch == s.cacheGeometryEpoch && entry.width == width &&
		entry.pos+1 == pos && entry.rows > 0 && entry.rows <= end && entry.capacity >= entry.rows &&
		old.Backend() == s.Backend && old.Buf() != nil && old.Buf() == entry.tensor.Buf() && slices.Equal(old.Shape, []int{entry.capacity, width})
	if hit {
		anchor, present := view.owner.indexPublication(v41AttentionPublicationKey{sourceLayer: source, start: entry.rows - 1, end: entry.rows})
		hit = present && len(anchor) == width && len(entry.anchor) == width && &anchor[0] == &entry.anchor[0]
	}
	copyRows, capacity := 0, 0
	if hit {
		copyRows, capacity = entry.rows, entry.capacity
	}
	// Verify descriptor provenance against the authoritative store without a
	// retained payload copy/hash. An anchor hit already proves rows before the
	// anchor are the cached, validated publications; walk only anchor onward.
	// On a hit, caller rows below the anchor are trusted unchecked: the only
	// caller passes borrowIndexKeys rows, and scoring reads the device copy.
	for i := max(0, copyRows-1); i < end; i++ {
		c.descriptorLookups++
		published, ok := view.owner.indexPublication(v41AttentionPublicationKey{sourceLayer: source, start: i, end: i + 1})
		if row := rows[i]; !ok || len(row) != width || len(published) != width || &row[0] != &published[0] {
			return nil
		}
	}
	spare := v41IndexerScoreSpareKeyCapacity(s.Backend)
	replace := !hit || end > capacity || (!spare && capacity != end)
	budget, peak := o.indexerKeyBudgetBytes, charged
	if stage := end - copyRows; stage > 0 {
		stageBytes, ok := checkedMul(int64(stage), int64(width), 4)
		if !ok || stageBytes > budget-peak {
			return nil
		}
		peak += stageBytes
	}
	if replace {
		sizes := []int{end}
		if grown := min(capacity, m.Cfg.MaxPositionEmbeddings/group/2) * 2; spare && grown > end {
			sizes = []int{grown, end}
		}
		capacity = 0
		for _, size := range sizes {
			if nextBytes, ok := checkedMul(int64(size), int64(width), 4); ok && nextBytes <= budget-peak {
				capacity, peak = size, peak+nextBytes
				break
			}
		}
		if capacity == 0 {
			return nil
		}
	}
	call := &v41IndexerKeyCall{cache: c, layer: layer, key: key, rows: rows, old: old, copyRows: copyRows, replace: replace, budget: budget, extraBytes: peak - charged,
		entry: v41IndexerKeyEntry{view: *view, forward: st, model: m, epoch: s.cacheGeometryEpoch, pos: pos, rows: end, capacity: capacity, width: width, anchor: rows[end-1]}}
	o.indexerKeys = c
	return func(layer int, q, flat, weights []float32, heads, width, count int) ([]float32, error) {
		if o.indexerKeyCall != nil || s.v41Forward != st || st.gatherOps != o || layer != call.layer {
			return nil, errV41IndexerScoreResult
		}
		// The original bound callback remains authoritative, including its
		// captured admission error. No extra early backend admission is run.
		// This carrier exists only while that owned synchronous call executes.
		o.indexerKeyCall = call
		defer func() { o.indexerKeyCall = nil }()
		values, err := score(layer, q, flat, weights, heads, width, count)
		if err == nil && !call.claimed {
			return nil, errV41IndexerScoreResult
		}
		return values, err
	}
}

func (p *v41IndexerKeyCall) validate(q, flat, weights []float32, heads, width, rows int) error {
	if p.claimed {
		return errV41IndexerScoreResult
	}
	p.claimed = true
	count, ok := checkedMulInt(rows, width)
	if !ok || flat != nil || width != p.entry.width || rows != len(p.rows) {
		return errV41IndexerScoreResult
	}
	if err := v41ValidateIndexerScoreGeometry(len(q), count, len(weights), heads, width, rows); err != nil {
		return err
	}
	check := func(name string, values []float32, offset int) error {
		for i, value := range values {
			if !finite32(value) {
				return fmt.Errorf("%w: indexer score %s element %d is non-finite", ErrV41ForwardStage, name, offset+i)
			}
		}
		return nil
	}
	if err := check("query", q, 0); err != nil {
		return err
	}
	// Device rows before copyRows were finite-checked when appended and are
	// immutable; the anchor hit that set copyRows proves they are unchanged.
	for i := p.copyRows; i < len(p.rows); i++ {
		row := p.rows[i]
		if len(row) != width {
			return errV41IndexerScoreResult
		}
		p.cache.validatedRows++
		if err := check("keys", row, i*width); err != nil {
			return err
		}
	}
	return check("weights", weights, 0)
}

func (p *v41IndexerKeyCall) acquire(query compute.Tensor) (compute.Tensor, error) {
	c, s := p.cache, p.cache.s
	charged, valid := c.charged()
	if !valid || charged > p.budget || p.extraBytes > p.budget-charged || s.v41Forward != p.entry.forward || s.M != p.entry.model || s.cacheGeometryEpoch != p.entry.epoch || len(s.v41Forward.history) != p.entry.pos {
		return compute.Tensor{}, errV41IndexerScoreResult
	}
	if s.halW == nil {
		s.halW = make(map[string]compute.Tensor)
	}
	n, dst := p.entry.rows-p.copyRows, p.old
	if !p.replace {
		if live, present := s.halW[p.key]; !present || live.Buf() != p.old.Buf() {
			return compute.Tensor{}, errV41IndexerScoreResult
		}
	}
	if p.stages() {
		if _, present := s.halW[p.key+"/next"]; present {
			return compute.Tensor{}, errV41IndexerScoreResult
		}
		if _, present := s.halW[p.key+"/staging"]; present {
			return compute.Tensor{}, errV41IndexerScoreResult
		}
	}
	// All cache-owned allocations, including staging, enter halW immediately.
	// Failure cleanup then flushes the owning session before retiring them.
	adopt := func(key string, t compute.Tensor, rows int) bool {
		if t.Buf() == nil || t.Backend() != s.Backend || t.Buf() == query.Buf() || c.protected(t) {
			return false
		}
		s.halW[key], c.handles[key] = t, t
		return t.Dtype == compute.F32 && t.Layout == compute.RowMajor && t.Quant == nil && slices.Equal(t.Shape, []int{rows, p.entry.width})
	}
	if p.replace {
		next, err := c.ops.V41RowLogAlloc(p.entry.capacity, p.entry.width)
		owned := adopt(p.key+"/next", next, p.entry.capacity)
		if err != nil {
			return compute.Tensor{}, err
		}
		if !owned {
			return compute.Tensor{}, errV41IndexerScoreResult
		}
		if p.copyRows > 0 {
			if err := c.ops.V41RowLogCopy(next, 0, p.old, 0, p.copyRows); err != nil {
				return compute.Tensor{}, err
			}
		}
		dst = next
	}
	// Appending past the committed rows leaves the validated prefix untouched,
	// so a failure before commit cannot invalidate the cached entry.
	if n > 0 {
		var values []float32
		if n == 1 {
			values = p.rows[p.copyRows]
		} else {
			values = make([]float32, 0, n*p.entry.width)
			for _, row := range p.rows[p.copyRows:] {
				values = append(values, row...)
			}
		}
		staged := s.uploadHostF32([]int{n, p.entry.width}, values, compute.MemoryActivation, "V4.1 indexer score key append")
		if !adopt(p.key+"/staging", staged, n) {
			return compute.Tensor{}, errV41IndexerScoreResult
		}
		if err := c.ops.V41RowLogCopy(dst, p.copyRows, staged, 0, n); err != nil {
			return compute.Tensor{}, err
		}
	}
	p.entry.tensor = dst
	view := dst
	view.Shape = []int{p.entry.rows, p.entry.width}
	return view, nil
}

func (p *v41IndexerKeyCall) stages() bool { return p.replace || p.entry.rows > p.copyRows }

func (p *v41IndexerKeyCall) commit() {
	c, s := p.cache, p.cache.s
	retire := func(key string) {
		if t, present := s.halW[key]; present {
			delete(s.halW, key)
			delete(c.handles, key)
			s.Backend.Free(t)
		}
	}
	if p.stages() {
		retire(p.key + "/staging")
	}
	if p.replace {
		retire(p.key)
		delete(s.halW, p.key+"/next")
		delete(c.handles, p.key+"/next")
		s.halW[p.key], c.handles[p.key] = p.entry.tensor, p.entry.tensor
	}
	c.entries[p.layer] = p.entry
}

func v41IndexerScoreOperationErr(layer int, cause error) error {
	return &V41IndexerScoreOperationError{Layer: layer, Cause: v41StageErr(v41StageIndexer, layer, cause)}
}

// A successful score dispatch can still produce NaN from finite inputs. The
// host selectors retain their rejection policy; close the selected session at
// its public boundary after the normal forward rollback, without replay.
func (s *Session) v41IndexerScoreGuard() {
	if primary := recover(); primary != nil {
		panic(s.v41IndexerScorePanic(primary))
	}
}

// Keep the selected-error policy shared with the legacy batch adapter. Unknown
// payloads retain their identity; only the selected score marker retires here.
func (s *Session) v41IndexerScorePanic(primary any) any {
	if err, ok := primary.(error); ok && !s.halClosed {
		var selected *V41IndexerScoreOperationError
		if errors.As(err, &selected) {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-indexer-score", Layer: selected.Layer, Stage: "selection", Cause: err,
			}
			s.halFailure = closed
			s.closeBackendOwned()
			return v41IndexerScoreOperationErr(selected.Layer, closed)
		}
	}
	return primary
}

// Conversion runs only after the owned forward and its defers finish. A nil
// panic remains present even in legacy mode; Goexit never returns from the
// inner call and therefore cannot become an ordinary result or conversion.
func (s *Session) v41IndexerScoreRecovering(forward func() []float32) (result []float32) {
	completed := false
	var primary any
	func() {
		defer func() { primary = recover() }()
		result = forward()
		completed = true
	}()
	if !completed {
		panic(s.v41IndexerScorePanic(primary))
	}
	return result
}

// Keep the existing publication/selection code on the nil path. The selected
// path changes only the score producer; the existing selectors retain ties,
// newest-block pinning, masks, -1 sentinels, infinity handling and NaN rejection.
func v41IndexRowsWithScore(layer int, q, keys, weights []float32, heads, headDim, rows, topKBlocks, blockSize, topK int, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc) ([]int32, error) {
	return v41IndexRowsSelected(layer, q, keys, weights, heads, headDim, rows, topKBlocks, blockSize, topK, score, health, nil)
}

// v41IndexRowsSelected additionally routes the candidate-block mask and final
// top-k through a selected device index-select seam. That seam owns the same
// tie order, pinning and -1 sentinels; its failure propagates without a host
// re-selection (Ref #13668).
func v41IndexRowsSelected(layer int, q, keys, weights []float32, heads, headDim, rows, topKBlocks, blockSize, topK int, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc, sel v41IndexSelectFunc) ([]int32, error) {
	return v41IndexRowsSelectedWithBorrow(layer, q, keys, weights, heads, headDim, rows, topKBlocks, blockSize, topK, score, health, sel, nil)
}

func v41IndexRowsSelectedWithBorrow(layer int, q, keys, weights []float32, heads, headDim, rows, topKBlocks, blockSize, topK int, score v41IndexerScoreFunc, health v41IndexerScoreHealthFunc, sel v41IndexSelectFunc, borrow *v41IndexerScoreBorrow) ([]int32, error) {
	if health != nil {
		if err := health(layer); err != nil {
			return nil, v41IndexerScoreOperationErr(layer, err)
		}
	}
	if score == nil && sel == nil {
		pub, err := NewV41IndexerPublication(layer, q, keys, weights, heads, headDim, rows, topKBlocks, blockSize, topK, 0)
		if err != nil {
			return nil, v41StageErr(v41StageIndexer, layer, err)
		}
		return pub.Rows(), nil
	}
	if layer < 0 || topK < 0 {
		return nil, v41IndexerScoreOperationErr(layer, fmt.Errorf("%w: invalid index publication layer or top-k", ErrV41ForwardStage))
	}
	var values []float32
	var err error
	if score == nil {
		if values, err = V41IndexerScore(q, keys, weights, heads, headDim, rows); err != nil {
			return nil, v41StageErr(v41StageIndexer, layer, err)
		}
	} else {
		var keyInput v41IndexerScoreKeys
		if borrow != nil {
			score, keys, keyInput = borrow.resolve(keys)
		}
		if keyInput.borrow != nil {
			values, err = borrow.binding.native.core(layer, q, keys, weights, heads, headDim, rows, keyInput)
		} else {
			values, err = score(layer, q, keys, weights, heads, headDim, rows)
		}
		if err == nil && len(values) != rows {
			err = errV41IndexerScoreResult
		}
		if err != nil {
			return nil, v41IndexerScoreOperationErr(layer, err)
		}
	}
	if sel != nil {
		selected, err := sel(layer, values, topK, topKBlocks, blockSize)
		if err == nil && len(selected) != min(topK, rows) {
			err = errV41IndexSelectResult
		}
		if err != nil {
			return nil, v41IndexerScoreOperationErr(layer, err)
		}
		return selected, nil
	}
	var mask []bool
	if topKBlocks > 0 {
		mask, err = V41SelectCandidateBlocks(values, rows, topKBlocks, blockSize)
		if err != nil {
			return nil, v41IndexerScoreOperationErr(layer, err)
		}
	}
	selected, err := V41SelectIndexRows(values, rows, topK, 0, mask)
	if err != nil {
		return nil, v41IndexerScoreOperationErr(layer, err)
	}
	return selected, nil
}

// Check the original host operands before any upload, even for an empty row
// set. The native ABI uses signed 32-bit element indices and host byte counts.
func v41ValidateIndexerScoreInputs(q, keys, weights []float32, heads, headDim, rows int) error {
	if err := v41ValidateIndexerScoreGeometry(len(q), len(keys), len(weights), heads, headDim, rows); err != nil {
		return err
	}
	for _, operand := range []struct {
		name   string
		values []float32
	}{{"query", q}, {"keys", keys}, {"weights", weights}} {
		for i, value := range operand.values {
			if !finite32(value) {
				return fmt.Errorf("%w: indexer score %s element %d is non-finite", ErrV41ForwardStage, operand.name, i)
			}
		}
	}
	return nil
}

func v41ValidateIndexerScoreGeometry(qLen, keyLen, weightLen, heads, headDim, rows int) error {
	qSize, qOK := checkedMulInt(heads, headDim)
	keySize, keyOK := checkedMulInt(rows, headDim)
	_, qBytesOK := checkedMulInt(qSize, 4)
	_, keyBytesOK := checkedMulInt(keySize, 4)
	_, weightBytesOK := checkedMulInt(heads, 4)
	_, outBytesOK := checkedMulInt(rows, 4)
	if heads <= 0 || headDim <= 0 || rows < 0 || heads > math.MaxInt32 || headDim > math.MaxInt32 || rows > math.MaxInt32 ||
		!qOK || !keyOK || qSize > math.MaxInt32 || keySize > math.MaxInt32 ||
		!qBytesOK || !keyBytesOK || !weightBytesOK || !outBytesOK ||
		qLen != qSize || keyLen != keySize || weightLen != heads {
		return fmt.Errorf("%w: invalid indexer score geometry heads=%d dim=%d rows=%d", ErrV41ForwardStage, heads, headDim, rows)
	}
	return nil
}

func (s *Session) v41IndexerScoreHealthFunc() v41IndexerScoreHealthFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	operation, ok := s.Backend.(compute.V41IndexerScoreBackend)
	if !ok {
		return nil
	}
	// Healthy absence at bind time selects the host scorer for the session, so
	// skip the per-call admission probe (#13791): on Vulkan it takes vulkanMu
	// on every indexer call for a device op that never runs. A bind-time
	// admission fault keeps the seam so the first call still reports it.
	if supported, err := operation.V41IndexerScoreAdmission(); err == nil && !supported {
		return nil
	}
	return func(layer int) error {
		s.ensureOpenBackendSession()
		if _, err := operation.V41IndexerScoreAdmission(); err != nil {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-indexer-score", Layer: layer, Stage: "admission", Cause: err,
			}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		return nil
	}
}

// Only the backend's native production capability selects this seam. Name,
// Class, generic device memory and F32 support alone cannot select it.
// Scores are returned as an owned host copy; this is not device-resident top-k
// or whole-model GPU execution. Snapshots never persist the callback.
func (s *Session) v41IndexerScoreFunc() v41IndexerScoreFunc {
	return s.v41BindIndexerScore().ordinary
}

func (s *Session) v41BindIndexerScore() v41BoundIndexerScore {
	if s == nil || s.M == nil || s.Backend == nil {
		return v41BoundIndexerScore{}
	}
	model, backend := s.M, s.Backend
	if !backend.Caps().DeviceMemory || !compute.BackendSupportsDeviceWeightDtype(backend, compute.F32) {
		return v41BoundIndexerScore{}
	}
	operation, ok := backend.(compute.V41IndexerScoreBackend)
	if !ok {
		return v41BoundIndexerScore{}
	}
	supported, admissionErr := operation.V41IndexerScoreAdmission()
	if admissionErr == nil && !supported {
		return v41BoundIndexerScore{}
	}
	core := func(layer int, q, keys, weights []float32, heads, headDim, rows int, keyInput v41IndexerScoreKeys) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		var resident *v41IndexerKeyCall
		if st := s.v41Forward; st != nil && st.gatherOps != nil && st.gatherOps.owner == s {
			resident = st.gatherOps.indexerKeyCall
		}
		stage := "payload"
		closeFailure := func(err error) error {
			closed := &BackendForwardOperationError{
				Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-indexer-score", Layer: layer, Stage: stage, Cause: err,
			}
			s.halFailure = closed
			s.closeBackendOwned()
			return closed
		}
		var primary any
		complete := false
		func() {
			defer func() { primary = recover() }()
			result, cause = func() (result []float32, cause error) {
				validate := func() error { return v41ValidateIndexerScoreInputs(q, keys, weights, heads, headDim, rows) }
				if resident != nil {
					validate = func() error { return resident.validate(q, keys, weights, heads, headDim, rows) }
				}
				if err := validate(); err != nil {
					return nil, closeFailure(err)
				}
				stage = "admission"
				if admissionErr != nil {
					return nil, closeFailure(admissionErr)
				}
				if _, err := operation.V41IndexerScoreAdmission(); err != nil {
					return nil, closeFailure(err)
				}
				if rows == 0 {
					return make([]float32, 0), nil
				}
				run := func() (result []float32, cause error) {
					var owned []compute.Tensor
					complete := false
					defer func() {
						failed := !complete || cause != nil
						pending := owned
						owned = nil
						var cleanup any
						present := false
						for i := len(pending) - 1; i >= 0; i-- {
							func() {
								returned := false
								defer func() {
									value := recover()
									if !returned && !present {
										cleanup, present = value, true
									}
								}()
								s.Backend.Free(pending[i])
								returned = true
							}()
						}
						if present && !failed {
							panic(cleanup)
						}
					}()
					result, cause = func() (result []float32, cause error) {
						fresh := func(x compute.Tensor) bool {
							if x.Buf() == nil || x.Backend() != s.Backend {
								return false
							}
							if resident != nil && resident.cache.protected(x) {
								return false
							}
							for _, previous := range owned {
								if previous.Buf() == x.Buf() {
									return false
								}
							}
							owned = append(owned, x)
							return true
						}
						shapeOK := func(x compute.Tensor, shape []int) bool {
							return x.Dtype == compute.F32 && x.Layout == compute.RowMajor && x.Quant == nil && slices.Equal(x.Shape, shape)
						}
						upload := func(name string, shape []int, values []float32) (compute.Tensor, error) {
							stage = name + " upload"
							x := s.uploadHostF32(shape, values, compute.MemoryActivation, "V4.1 indexer score "+name)
							if !fresh(x) || !shapeOK(x, shape) {
								return compute.Tensor{}, errV41IndexerScoreResult
							}
							return x, nil
						}
						qt, err := upload("query", []int{heads, headDim}, q)
						if err != nil {
							return nil, err
						}
						var kt compute.Tensor
						if resident == nil {
							// Revalidate after the existing admission and query upload, just
							// before dynamic backend use of the frozen host key operand.
							keys = keyInput.beforeUpload(keys)
							kt, err = upload("keys", []int{rows, headDim}, keys)
						} else {
							stage = "resident keys"
							kt, err = resident.acquire(qt)
						}
						if err != nil {
							return nil, err
						}
						wt, err := upload("weights", []int{heads}, weights)
						if err != nil {
							return nil, err
						}
						stage = "score"
						out, err := operation.V41IndexerScore(qt, kt, wt, rows, heads, headDim)
						ownedOutput := fresh(out) // release a fresh partial output even on error
						if err != nil {
							return nil, err
						}
						stage = "output validation"
						if !ownedOutput || !shapeOK(out, []int{rows}) {
							return nil, errV41IndexerScoreResult
						}
						stage = "readback"
						// The native score primitive marks this tensor for checked Read.
						values, validRead := compute.ReadV41IndexerScores(s.Backend, out, rows)
						if !validRead {
							return nil, errV41IndexerScoreResult
						}
						// Finite inputs may produce +/-Inf or NaN. Do not clamp, rescale,
						// round, or reject here: the unchanged selector owns that policy.
						result = values
						if resident != nil {
							// Checked Read is the completion boundary for both the score
							// and every prefix/staging copy feeding its exact-sized input.
							resident.commit()
						}
						return result, nil
					}()
					complete = true
					return result, cause
				}
				result, cause = run()
				if cause != nil {
					return nil, closeFailure(cause)
				}
				return result, nil
			}()
			complete = true
		}()
		if !complete {
			r := primary
			if err, ok := r.(error); ok {
				var closed *BackendForwardOperationError
				if errors.As(err, &closed) {
					panic(r)
				}
				var backend *compute.BackendError
				if errors.As(err, &backend) {
					result, cause = nil, closeFailure(err)
					return
				}
			}
			if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
				if original, ok := r.(error); ok {
					err = original
				}
				result, cause = nil, closeFailure(err)
				return
			}
			if !s.halClosed {
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
			}
			panic(r)
		}
		return result, cause
	}
	ordinary := func(layer int, q, keys, weights []float32, heads, headDim, rows int) ([]float32, error) {
		return core(layer, q, keys, weights, heads, headDim, rows, v41IndexerScoreKeys{})
	}
	return v41BoundIndexerScore{ordinary: ordinary, core: core, owner: s, model: model, backend: backend}
}
