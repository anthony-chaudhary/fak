package main

import (
	"sync"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestTokenizerSharedOnce(t *testing.T) {
	p := NewSharedTokenizerPreloader(func() (*SharedTokenizerHandle, error) {
		return &SharedTokenizerHandle{ModelID: "qwen3.8", VocabSize: 151936}, nil
	})

	const workers = 16
	handles := make([]*SharedTokenizerHandle, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func(idx int) {
			defer wg.Done()
			h, err := p.Get()
			if err != nil {
				t.Errorf("worker %d Get: %v", idx, err)
				return
			}
			handles[idx] = h
		}(i)
	}
	wg.Wait()

	if p.LoadCount() != 1 {
		t.Fatalf("LoadCount = %d, want 1", p.LoadCount())
	}
	for i := 1; i < workers; i++ {
		if handles[i] != handles[0] {
			t.Fatalf("worker %d received distinct handle pointer", i)
		}
	}
}
