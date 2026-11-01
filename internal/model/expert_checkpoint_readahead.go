package model

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

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

// readAhead advises the same indexed source the borrowed demand path
// will read. Advice failures are recorded, but cannot fail otherwise valid
// inference; the later read and upload retain their ordinary error behavior.
func (t *ExpertCheckpointTier) readAhead(name string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	entry, ok := t.index[name]
	if !ok || !expertReadAheadDtypeEnabled(entry.dtype()) ||
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

func expertReadAheadDtypeEnabled(dt compute.Dtype) bool {
	switch dt {
	case compute.Q2_K, compute.Q3_K:
		return true
	case compute.Q4_K, compute.Q5_K, compute.Q6_K, compute.IQ2_XXS, compute.IQ2_XS, compute.IQ1_M:
		return expertMappedReadaheadQualification
	default:
		return false
	}
}

type expertReadAheadSpan struct {
	shard    int
	off, end int64
	reader   ggufReadAheadRange
	name     string
}

// readAheadCoalesced adapts llama.cpp's sorted range union to indexed checkpoint
// payloads. Unlike page-rounded address merging, it never bridges a payload hole.
// The existing reader owns page alignment and EOF clamping. Advice is synchronous;
// no payload buffer, cache residency, worker, or model lifetime is created here.
func (t *ExpertCheckpointTier) readAheadCoalesced(names []string) {
	if t == nil || !expertMappedReadaheadQualification || len(names) == 0 {
		return
	}
	t.mu.Lock()
	if t.pool.Budget() != 0 {
		t.mu.Unlock()
		return
	}
	spans := make([]expertReadAheadSpan, 0, len(names))
	for _, name := range names {
		entry, ok := t.index[name]
		if !ok || !expertReadAheadDtypeEnabled(entry.dtype()) || t.overlays[entry.fused] != nil {
			continue
		}
		src := t.shards[entry.shard]
		if src.PageCacheEnabled() {
			continue
		}
		_, borrowed := src.r.(ggufReadOnlyRange)
		reader, supported := src.r.(ggufReadAheadRange)
		if !borrowed || !supported {
			t.readahead.Unavailable++
			continue
		}
		off, size, valid := src.expertReadSpan(entry.fused, entry.expert)
		// Subtraction validates the addition even at an int64-sized source.
		if !valid || off < 0 || size <= 0 || off > src.size || size > src.size-off {
			t.readahead.Failures++
			t.readahead.LastError = fmt.Sprintf("%v: readahead span for %s", ErrGGUFExpertMetadata, name)
			continue
		}
		spans = append(spans, expertReadAheadSpan{shard: entry.shard, off: off, end: off + size, reader: reader, name: name})
	}
	t.mu.Unlock()
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].shard != spans[j].shard {
			return spans[i].shard < spans[j].shard
		}
		return spans[i].off < spans[j].off
	})
	merged := spans[:0]
	for _, span := range spans {
		if n := len(merged); n > 0 && merged[n-1].shard == span.shard && span.off <= merged[n-1].end {
			merged[n-1].end = max(merged[n-1].end, span.end)
		} else {
			merged = append(merged, span)
		}
	}
	for _, span := range merged {
		size := span.end - span.off
		err := span.reader.ReadAheadRange(span.off, size)
		t.mu.Lock()
		if err != nil {
			t.readahead.Failures++
			t.readahead.LastError = fmt.Sprintf("%s: readahead offset %d size %d: %v", span.name, span.off, size, err)
		} else {
			// Hints counts accepted calls; Bytes is the selected payload union,
			// not page-rounded bytes, physical I/O, or proof of residency.
			t.readahead.Hints++
			t.readahead.Bytes += size
		}
		t.mu.Unlock()
	}
}
