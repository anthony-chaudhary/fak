package model

import (
	"os"
	"os/exec"
	"strconv"
	"sync"
	"testing"
)

const lmHeadRouteRaceChildEnv = "FAK_TEST_LMHEAD_ROUTE_RACE_CHILD"

// TestLMHeadRouteConcurrentWithQ8StoreWrites: LMHeadRoute resolves the head through
// residentHeadName, which probes q8w while decode can write it under q8Mu. An unlocked
// probe aborts the process with a concurrent map read and map write, which no recover
// can catch, so the hammer runs in a child process and this test asserts it exits clean.
func TestLMHeadRouteConcurrentWithQ8StoreWrites(t *testing.T) {
	if os.Getenv(lmHeadRouteRaceChildEnv) == "1" {
		hammerLMHeadRouteWithQ8Writes(t)
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestLMHeadRouteConcurrentWithQ8StoreWrites$", "-test.count=1")
	cmd.Env = append(os.Environ(), lmHeadRouteRaceChildEnv+"=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("LMHeadRoute under concurrent q8w writes did not exit clean: %v\n%s", err, out)
	}
}

func hammerLMHeadRouteWithQ8Writes(t *testing.T) {
	m := NewSynthetic(qwen35HybridQ4KTestCfg())
	m.Quantize()
	want := m.LMHeadRoute()
	probe := &q8Tensor{}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		keys := make([]string, 64)
		for i := range keys {
			keys[i] = "lazy.fill." + strconv.Itoa(i)
		}
		for {
			select {
			case <-stop:
				return
			default:
			}
			q8Mu.Lock()
			for _, k := range keys {
				m.q8w[k] = probe
			}
			for _, k := range keys {
				delete(m.q8w, k)
			}
			q8Mu.Unlock()
		}
	}()
	defer func() {
		close(stop)
		wg.Wait()
	}()
	for i := 0; i < 20000; i++ {
		if got := m.LMHeadRoute(); got != want {
			t.Fatalf("LMHeadRoute = %q during concurrent q8w writes, want %q", got, want)
		}
	}
}
