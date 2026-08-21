package main

import (
	"bytes"
	"sync"
	"testing"
)

func TestSessionNewDeterminism(t *testing.T) {
	const runs = 32
	results := make(chan string, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			deps := sessionNewTestDeps("windows", "")
			var stdout, stderr bytes.Buffer
			if code := runSessionNewWith(&stdout, &stderr, []string{"--dry-run", "same λ prompt"}, deps); code != 0 {
				t.Errorf("code=%d stderr=%s", code, stderr.String())
				return
			}
			results <- stdout.String()
		}()
	}
	wg.Wait()
	close(results)
	var first string
	for got := range results {
		if first == "" {
			first = got
			continue
		}
		if got != first {
			t.Fatalf("non-deterministic receipt:\nfirst=%s\ngot=%s", first, got)
		}
	}
}
