package zaitask

import (
	"reflect"
	"sync"
	"testing"
)

func TestZAIProductRoutingDeterminism(t *testing.T) {
	inputs := []struct{ prompt, class string }{
		{"summarize this diff", "bounded"},
		{"design architecture", "frontier"},
		{"", "bounded"},
		{"fix the test", ""},
	}
	baseline := make([]Suitability, len(inputs))
	for i, in := range inputs {
		baseline[i] = Classify(in.prompt, in.class)
	}
	const runs = 100
	errCh := make(chan string, runs)
	var wg sync.WaitGroup
	for i := 0; i < runs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got := make([]Suitability, len(inputs))
			for i, in := range inputs {
				got[i] = Classify(in.prompt, in.class)
			}
			if !reflect.DeepEqual(got, baseline) {
				errCh <- "routing changed across identical inputs"
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for failure := range errCh {
		t.Error(failure)
	}
}
