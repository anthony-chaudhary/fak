package microagent

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"
)

type claimedWakeAgent struct {
	ID string `json:"id"`

	blankStarted chan struct{}
	releaseBlank chan struct{}
	blankOnce    *sync.Once
}

func (a *claimedWakeAgent) Step(context.Context, Gateway) (bool, error) {
	return false, nil
}

func (a *claimedWakeAgent) Freeze() ([]byte, error) {
	return json.Marshal(struct {
		ID string `json:"id"`
	}{ID: a.ID})
}

func (a *claimedWakeAgent) Thaw(b []byte) error {
	var state struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(b, &state); err != nil {
		return err
	}
	a.ID = state.ID
	return nil
}

func (a *claimedWakeAgent) Blank() Hibernable {
	if a.blankStarted != nil {
		a.blankOnce.Do(func() { close(a.blankStarted) })
		<-a.releaseBlank
	}
	return &claimedWakeAgent{
		blankStarted: a.blankStarted,
		releaseBlank: a.releaseBlank,
		blankOnce:    a.blankOnce,
	}
}

func TestWarmBandAcquireClaimsWakeBeforeRestoring(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	band, err := NewWarmBand(WarmBandConfig{
		High: 2,
		Dir:  t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()
	if err := band.Enroll("agent", &claimedWakeAgent{
		ID:           "agent",
		blankStarted: started,
		releaseBlank: release,
		blankOnce:    &sync.Once{},
	}); err != nil {
		t.Fatalf("Enroll: %v", err)
	}

	type result struct {
		h   Hibernable
		err error
	}
	first := make(chan result, 1)
	go func() {
		h, err := band.Acquire(context.Background(), "agent")
		first <- result{h: h, err: err}
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("Acquire never reached the restore boundary")
	}

	band.mu.Lock()
	claimed := band.warming["agent"]
	band.mu.Unlock()
	if !claimed {
		close(release)
		t.Fatal("Acquire reached Blank without claiming the id's wake")
	}

	second := make(chan result, 1)
	go func() {
		h, err := band.Acquire(context.Background(), "agent")
		second <- result{h: h, err: err}
	}()
	close(release)

	var got [2]result
	for i, ch := range []<-chan result{first, second} {
		select {
		case got[i] = <-ch:
		case <-time.After(time.Second):
			t.Fatal("concurrent Acquire did not finish")
		}
		if got[i].err != nil {
			t.Fatalf("Acquire %d: %v", i+1, got[i].err)
		}
	}
	if got[0].h != got[1].h {
		t.Fatal("concurrent Acquire restored two live values for one id")
	}
	band.Retire("agent")
}

func TestWarmBandRefillSkipsClaimedWake(t *testing.T) {
	band, err := NewWarmBand(WarmBandConfig{
		High:    2,
		MaxWarm: 1,
		Dir:     t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()
	if err := band.Enroll("agent", &claimedWakeAgent{
		ID:        "agent",
		blankOnce: &sync.Once{},
	}); err != nil {
		t.Fatalf("Enroll: %v", err)
	}

	band.mu.Lock()
	band.warming["agent"] = true
	band.mu.Unlock()
	if band.refillOne() {
		t.Fatal("refill consumed a snapshot already claimed by Acquire")
	}
	if !band.store.Parked("agent") || band.reserve.Len() != 0 {
		t.Fatalf("claimed snapshot moved: parked=%v warm=%d", band.store.Parked("agent"), band.reserve.Len())
	}
	band.mu.Lock()
	delete(band.warming, "agent")
	band.mu.Unlock()
}

func TestWarmBandRefillKeepsWakeClaimUntilWarmPublish(t *testing.T) {
	band, err := NewWarmBand(WarmBandConfig{
		High:    2,
		MaxWarm: 1,
		Dir:     t.TempDir(),
	})
	if err != nil {
		t.Fatalf("NewWarmBand: %v", err)
	}
	defer band.Close()
	if err := band.Enroll("agent", &claimedWakeAgent{
		ID:        "agent",
		blankOnce: &sync.Once{},
	}); err != nil {
		t.Fatalf("Enroll: %v", err)
	}

	band.reserve.mu.Lock()
	done := make(chan bool, 1)
	go func() { done <- band.refillOne() }()
	deadline := time.Now().Add(time.Second)
	for band.store.Parked("agent") && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if band.store.Parked("agent") {
		band.reserve.mu.Unlock()
		t.Fatal("refill never restored the claimed snapshot")
	}
	if band.mu.TryLock() {
		claimed := band.warming["agent"]
		band.mu.Unlock()
		if !claimed {
			band.reserve.mu.Unlock()
			t.Fatal("refill dropped the wake claim before publishing the warm value")
		}
	}
	band.reserve.mu.Unlock()

	select {
	case progressed := <-done:
		if !progressed {
			t.Fatal("refill did not publish the restored value")
		}
	case <-time.After(time.Second):
		t.Fatal("refill did not finish after warm publication unblocked")
	}
	if !band.reserve.Warm("agent") {
		t.Fatal("restored value was not published to the warm reserve")
	}
	band.mu.Lock()
	claimed := band.warming["agent"]
	band.mu.Unlock()
	if claimed {
		t.Fatal("wake claim remained after warm publication")
	}
}
