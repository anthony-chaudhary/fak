// Adapted from ai-dynamo/aiperf
// (Apache-2.0, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES):
// single-init shared tokenizer preload so concurrent workers reuse one loaded
// tokenizer instance rather than loading per worker.
package main

import (
	"sync"
	"sync/atomic"
)

// SharedTokenizerHandle represents a read-only preloaded tokenizer shared
// across worker goroutines.
type SharedTokenizerHandle struct {
	ModelID   string
	VocabSize int
}

// SharedTokenizerPreloader loads a tokenizer at most once per process/preloader
// and shares the resulting handle across all concurrent callers.
type SharedTokenizerPreloader struct {
	once   sync.Once
	handle *SharedTokenizerHandle
	err    error
	loads  atomic.Int64
	loader func() (*SharedTokenizerHandle, error)
}

// NewSharedTokenizerPreloader constructs a preloader backed by loader.
func NewSharedTokenizerPreloader(loader func() (*SharedTokenizerHandle, error)) *SharedTokenizerPreloader {
	return &SharedTokenizerPreloader{loader: loader}
}

// Get returns the shared tokenizer handle, invoking loader exactly once even
// under concurrent calls.
func (p *SharedTokenizerPreloader) Get() (*SharedTokenizerHandle, error) {
	p.once.Do(func() {
		p.loads.Add(1)
		p.handle, p.err = p.loader()
	})
	return p.handle, p.err
}

// LoadCount returns the number of times the underlying loader executed.
func (p *SharedTokenizerPreloader) LoadCount() int64 {
	return p.loads.Load()
}
