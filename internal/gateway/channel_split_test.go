package gateway

import (
	"reflect"
	"testing"
	"time"
)

func TestChannelSplit(t *testing.T) {
	t.Parallel()
	// The worker-facing views are directional by type: dispatch is receive-only,
	// returns is send-only, so the worker cannot write dispatch or read its own returns.
	cs := NewChannelSplit(1, 0)
	if reflect.TypeOf(cs.Dispatch()).ChanDir() != reflect.RecvDir {
		t.Fatal("Dispatch must be receive-only")
	}
	if reflect.TypeOf(cs.Returns()).ChanDir() != reflect.SendDir {
		t.Fatal("Returns must be send-only")
	}

	// A producer that never reads results must not stall dispatch: the worker's
	// return send blocks only the worker, and dispatch of already-queued work continues.
	cs = NewChannelSplit(4, 1)
	for i := 0; i < 3; i++ {
		cs.Submit(ChannelSplitJob{ID: i})
	}
	cs.Close()
	done := make(chan int)
	go func() {
		n := 0
		for j := range cs.Dispatch() {
			if n == 0 {
				cs.Returns() <- ChannelSplitResult{ID: j.ID}
			}
			n++
		}
		done <- n
	}()
	select {
	case n := <-done:
		if n != 3 {
			t.Fatalf("drained %d jobs, want 3", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("dispatch stalled behind unread results")
	}
	if r := <-cs.Results(); r.ID != 0 {
		t.Fatalf("result id = %d, want 0", r.ID)
	}
}
