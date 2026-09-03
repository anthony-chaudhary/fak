package xprobe

import (
	"sync"
	"testing"
)

func TestPing(t *testing.T) {
	const want = "pong"
	if got := Ping(); got != want {
		t.Fatalf("Ping() = %q, want %q", got, want)
	}
}

func TestPingDeterministicRepeated(t *testing.T) {
	const want = "pong"
	for i := 0; i < 100; i++ {
		if got := Ping(); got != want {
			t.Fatalf("iteration %d: Ping() = %q, want %q", i, got, want)
		}
	}
}

func TestPingConcurrent(t *testing.T) {
	const (
		goroutines = 50
		iterations = 100
		want       = "pong"
	)

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func(workerID int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				if got := Ping(); got != want {
					t.Errorf("worker %d iteration %d: Ping() = %q, want %q", workerID, j, got, want)
				}
			}
		}(i)
	}

	wg.Wait()
}
