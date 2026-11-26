package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// inkling2DFlash2Cache keeps opened draft checkpoints for the process so a
// request reopening one reuses its mapping, its device store and the weight
// views wired into it instead of mapping and wiring ~1 GB again.
var inkling2DFlash2Cache struct {
	sync.Mutex
	entries map[string]*inkling2DFlash2Shared
	clock   uint64
}

// inkling2DFlash2IdleCap is how many unreferenced checkpoints stay mapped.
const inkling2DFlash2IdleCap = 2

// inkling2DFlash2Uncached, set by a test, opens every checkpoint afresh: the
// A side of a residency comparison.
var inkling2DFlash2Uncached bool

// inkling2DFlash2Shared is one cached checkpoint. Handles share its model,
// whose fields are read-only once bound; each request still builds its own
// rings, scratch and chooser.
type inkling2DFlash2Shared struct {
	path  string
	cfg   []byte
	info  os.FileInfo
	model *Inkling2DFlash2Model
	refs  int
	used  uint64
	stale bool

	storeOnce sync.Once
	store     *inkling2DFlash2Store
}

func (e *inkling2DFlash2Shared) same(cfg []byte, info os.FileInfo) bool {
	return bytes.Equal(e.cfg, cfg) && os.SameFile(e.info, info) && e.info.Size() == info.Size() &&
		e.info.ModTime().Equal(info.ModTime())
}

// deviceStore is the checkpoint's device store, made on first use; nil when
// the device cannot alias the mapping.
func (e *inkling2DFlash2Shared) deviceStore() *inkling2DFlash2Store {
	e.storeOnce.Do(func() {
		if len(e.model.mapped) > 0 {
			e.store, _ = openInkling2DFlash2Store(e.model.mapped)
		}
	})
	return e.store
}

func (e *inkling2DFlash2Shared) free() {
	if e.store != nil {
		e.store.drainClose()
	}
	_ = e.model.Close()
}

// openInkling2DFlash2Cached returns a handle on the cached checkpoint of dir,
// opening it when absent or when config.json or the weights file changed.
func openInkling2DFlash2Cached(dir string, cfg []byte) (*Inkling2DFlash2Model, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	info, err := os.Stat(filepath.Join(abs, inkling2DFlash2Weights))
	if err != nil {
		return nil, err
	}
	c := &inkling2DFlash2Cache
	c.Lock()
	defer c.Unlock()
	if c.entries == nil {
		c.entries = map[string]*inkling2DFlash2Shared{}
	}
	e := c.entries[abs]
	if e != nil && !e.same(cfg, info) {
		delete(c.entries, abs)
		e.stale = true
		if e.refs == 0 {
			e.free()
		}
		e = nil
	}
	if e == nil {
		m, err := openInkling2DFlash2Mapped(dir, cfg)
		if err != nil {
			return nil, err
		}
		e = &inkling2DFlash2Shared{path: abs, cfg: bytes.Clone(cfg), info: info, model: m}
		c.entries[abs] = e
	}
	e.refs++
	c.clock++
	e.used = c.clock
	h := *e.model
	h.Dir, h.closer, h.shared = dir, &inkling2DFlash2Handle{e: e}, e
	return &h, nil
}

// inkling2DFlash2Handle releases one reference on Close; a second Close is a no-op.
type inkling2DFlash2Handle struct {
	e    *inkling2DFlash2Shared
	once sync.Once
}

func (h *inkling2DFlash2Handle) Close() error {
	h.once.Do(func() {
		c := &inkling2DFlash2Cache
		c.Lock()
		defer c.Unlock()
		h.e.refs--
		if h.e.refs > 0 {
			return
		}
		if h.e.stale {
			h.e.free()
			return
		}
		inkling2DFlash2TrimIdle()
	})
	return nil
}

// inkling2DFlash2TrimIdle unmaps the least recently used idle checkpoints past
// the cap; the cache lock is held.
func inkling2DFlash2TrimIdle() {
	c := &inkling2DFlash2Cache
	for {
		var idle int
		var lru *inkling2DFlash2Shared
		for _, e := range c.entries {
			if e.refs == 0 {
				idle++
				if lru == nil || e.used < lru.used {
					lru = e
				}
			}
		}
		if idle <= inkling2DFlash2IdleCap {
			return
		}
		delete(c.entries, lru.path)
		lru.free()
	}
}

// inkling2DFlash2Faults is a page-fault count of the process.
type inkling2DFlash2Faults struct{ major, minor int64 }

func (f inkling2DFlash2Faults) sub(g inkling2DFlash2Faults) inkling2DFlash2Faults {
	return inkling2DFlash2Faults{f.major - g.major, f.minor - g.minor}
}

// inkling2DFlash2FaultCounter, set by a test, reads the process page faults.
var inkling2DFlash2FaultCounter func() inkling2DFlash2Faults

func inkling2DFlash2FaultsNow() inkling2DFlash2Faults {
	if inkling2DFlash2FaultCounter == nil {
		return inkling2DFlash2Faults{}
	}
	return inkling2DFlash2FaultCounter()
}

// inkling2DFlash2OpenStats is the last open's wall time and faults, recorded
// only while a test reads faults.
type inkling2DFlash2OpenStats struct {
	took   time.Duration
	faults inkling2DFlash2Faults
}

var inkling2DFlash2LastOpen inkling2DFlash2OpenStats
