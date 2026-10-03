package kernel

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1s
func TestDiscardSubmissionDoesNotExecuteAndConsumesHandle(t *testing.T) {
	setup()
	engine := &countEngine{}
	abi.RegisterEngine("discard-test", engine)
	abi.RegisterAdjudicator(0, fakeAdj{abi.Verdict{Kind: abi.VerdictAllow}})
	k := New("discard-test")
	h, verdict := k.Submit(context.Background(), call("write_probe", `{}`))
	if verdict.Kind != abi.VerdictAllow {
		t.Fatalf("Submit=%+v", verdict)
	}
	if !k.DiscardSubmission(h) {
		t.Fatal("first discard did not consume pending submission")
	}
	if k.DiscardSubmission(h) {
		t.Fatal("second discard consumed the same handle")
	}
	if result, err := k.Reap(context.Background(), h); err == nil || result != nil {
		t.Fatalf("Reap discarded handle=(%+v,%v), want nil/error", result, err)
	}
	if got := atomic.LoadInt64(&engine.n); got != 0 {
		t.Fatalf("discard invoked engine %d times", got)
	}
}
