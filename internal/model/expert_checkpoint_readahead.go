package model

import "fmt"

// ggufReadAheadRange is an optional mapped-reader capability. It starts an OS
// payload readahead hint without a host buffer or a Go worker. A nil error
// reports an accepted hint only; demand staging still owns the actual read.
type ggufReadAheadRange interface {
	ReadAheadRange(off, size int64) error
}

// ExpertCheckpointReadaheadStats separates optional OS advice from completed
// checkpoint reads. Bytes counts payload spans for accepted hints, excluding
// page-boundary rounding; it is not physical I/O or resident-byte accounting.
type ExpertCheckpointReadaheadStats struct {
	Hints       int    `json:"hints"`
	Bytes       int64  `json:"bytes"`
	Unavailable int    `json:"unavailable"`
	Failures    int    `json:"failures"`
	LastError   string `json:"last_error,omitempty"`
}

// readAhead advises the same indexed Q2/Q3 source the borrowed demand path
// will read. Advice failures are recorded, but cannot fail otherwise valid
// inference; the later read and upload retain their ordinary error behavior.
func (t *ExpertCheckpointTier) readAhead(name string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	entry, ok := t.index[name]
	if !ok || (entry.quant != ExpertCheckpointQ2K && entry.quant != ExpertCheckpointQ3K) ||
		t.pool.Budget() != 0 || t.overlays[entry.fused] != nil {
		t.mu.Unlock()
		return
	}
	src := t.shards[entry.shard]
	if src.PageCacheEnabled() {
		t.mu.Unlock()
		return
	}
	_, borrowed := src.r.(ggufReadOnlyRange)
	r, supported := src.r.(ggufReadAheadRange)
	if !borrowed || !supported {
		t.readahead.Unavailable++
		t.mu.Unlock()
		return
	}
	off, size, ok := src.expertReadSpan(entry.fused, entry.expert)
	t.mu.Unlock()
	var err error
	if !ok {
		err = fmt.Errorf("%w: readahead span for %s", ErrGGUFExpertMetadata, name)
	} else {
		err = r.ReadAheadRange(off, size)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if err != nil {
		t.readahead.Failures++
		t.readahead.LastError = fmt.Sprintf("%s: %v", name, err)
		return
	}
	t.readahead.Hints++
	t.readahead.Bytes += size
}
