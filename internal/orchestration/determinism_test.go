package orchestration

import (
	"bytes"
	"sync"
	"testing"
)

func TestResolveDeterminismConcurrent(t *testing.T) {
	workers := 3
	task := TaskSpec{Schema: "fak-orchestration-task/1", ID: "determinism", WorkClass: WorkRigor, MaxWorkers: &workers}
	caps := HarnessCapabilities{SupportNative, SupportNative, SupportNative, SupportNative, SupportNative, SupportNative}
	const runs = 64
	results := make(chan []byte, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := Resolve(OrchestrationProfile{Name: ProfileAuto}, task, caps)
			if err != nil {
				t.Error(err)
				return
			}
			b, err := StableJSON(r)
			if err != nil {
				t.Error(err)
				return
			}
			results <- b
		}()
	}
	wg.Wait()
	close(results)
	var first []byte
	for got := range results {
		if first == nil {
			first = got
			continue
		}
		if !bytes.Equal(first, got) {
			t.Fatalf("non-deterministic JSON:\n%s\n%s", first, got)
		}
	}
}
