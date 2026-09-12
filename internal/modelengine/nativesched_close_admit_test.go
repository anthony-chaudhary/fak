package modelengine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestNativeSchedulerCloseDuringPrefillCannotPublishWorkAfterCloseAndWait(t *testing.T) {
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	first, err := s.AdmitTokenIDs(context.Background(), "start-and-idle", []int{1})
	if err != nil {
		t.Fatalf("initial AdmitTokenIDs: %v", err)
	}
	for range first.Tokens() {
	}
	if _, err := first.Result(); err != nil {
		t.Fatalf("initial Result: %v", err)
	}

	prefillEntered := make(chan struct{})
	releasePrefill := make(chan struct{})
	nowCalls := 0
	s.now = func() time.Time {
		nowCalls++
		if nowCalls == 1 {
			close(prefillEntered)
			<-releasePrefill
		}
		return time.Now()
	}

	type admitResult struct {
		req abi.EngineRequest
		err error
	}
	admitted := make(chan admitResult, 1)
	go func() {
		req, err := s.AdmitTokenIDs(context.Background(), "audit-close-race", []int{1, 2, 3})
		admitted <- admitResult{req: req, err: err}
	}()

	select {
	case <-prefillEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("admission did not reach blocked prefill")
	}

	closeResult := make(chan error, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		closeResult <- s.CloseAndWait(ctx)
	}()

	deadline := time.Now().Add(2 * time.Second)
	for {
		s.mu.Lock()
		closed := s.closed
		s.mu.Unlock()
		if closed {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("CloseAndWait did not enter Close")
		}
		time.Sleep(time.Millisecond)
	}
	select {
	case err := <-closeResult:
		if err != nil {
			t.Fatalf("CloseAndWait: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("CloseAndWait did not return while the untracked admission was blocked in prefill")
	}
	close(releasePrefill)

	got := <-admitted
	if got.err != nil {
		if !errors.Is(got.err, errSchedClosed) {
			t.Fatalf("AdmitTokenIDs error = %v, want errSchedClosed", got.err)
		}
		return
	}
	if got.req == nil {
		t.Fatal("AdmitTokenIDs returned nil request and nil error")
	}
	defer func() {
		s.mu.Lock()
		got.req.(*schedLane).finish(nil, context.Canceled)
		s.mu.Unlock()
	}()
	t.Fatal("AdmitTokenIDs succeeded after CloseAndWait returned; its lane is orphaned in waiting with no scheduler loop")
}
