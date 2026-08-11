package lifecycleadapter

import (
	"context"
	"testing"
	"time"
)

func TestBuiltinNegotiationDoesNotPretendOSSuspensionIsCheckpoint(t *testing.T) {
	now := time.Now()
	for name, a := range map[string]Adapter{"codex": Codex(), "claude": Claude()} {
		doc := a.Capabilities()
		if doc.ApplicationCheckpoint {
			t.Errorf("%s claims application checkpoint", name)
		}
		req := Request{TransactionID: "tx", ForestID: "f", MemberID: name, Generation: 1, Operation: Checkpoint, Deadline: now.Add(time.Second)}
		n, r := Execute(context.Background(), req, a)
		if n.Supported || r.State != ResultUnsupported {
			t.Errorf("%s checkpoint=%+v/%+v", name, n, r)
		}
		req.Operation = Pause
		n, r = Execute(context.Background(), req, a)
		if !n.Supported || r.State != ResultCompleted {
			t.Errorf("%s pause=%+v/%+v", name, n, r)
		}
		req.Operation = Resume
		n, r = Execute(context.Background(), req, a)
		if !n.Supported || r.State != ResultCompleted {
			t.Errorf("%s resume=%+v/%+v", name, n, r)
		}
	}
}
func TestNativeFAKCheckpointAndReadiness(t *testing.T) {
	req := Request{TransactionID: "tx", ForestID: "f", MemberID: "h", Generation: 2, Operation: Checkpoint, Deadline: time.Now().Add(time.Second)}
	n, r := Execute(context.Background(), req, NativeFAK())
	if !n.Supported || !n.Document.ApplicationCheckpoint || r.State != ResultCompleted || r.ReadbackRef == "" {
		t.Fatalf("native=%+v/%+v", n, r)
	}
}
func TestUnknownAndVersionSkewFailClosed(t *testing.T) {
	req := Request{TransactionID: "tx", ForestID: "f", MemberID: "x", Generation: 1, Operation: Pause, Deadline: time.Now().Add(time.Second)}
	if n, r := Execute(context.Background(), req, Unknown("mystery")); n.Supported || r.State != ResultUnsupported {
		t.Fatalf("unknown=%+v/%+v", n, r)
	}
	skew := Custom(CapabilityDocument{Protocol: "fak-lifecycle-adapter/99", AdapterKind: "custom", Operations: []Operation{Pause}}, nil)
	if n, r := Execute(context.Background(), req, skew); n.Supported || r.State != ResultUnsupported {
		t.Fatalf("skew=%+v/%+v", n, r)
	}
}
func TestAdapterInvocationIsDeadlineBoundAndIndependentlyInjected(t *testing.T) {
	a := Custom(CapabilityDocument{Protocol: ProtocolVersion, AdapterKind: "custom", Operations: []Operation{Pause}}, func(ctx context.Context, _ Request) Result { <-ctx.Done(); return Result{} })
	req := Request{TransactionID: "tx", ForestID: "f", MemberID: "x", Generation: 1, Operation: Pause, Deadline: time.Now().Add(10 * time.Millisecond)}
	_, r := Execute(context.Background(), req, a)
	if r.State != ResultFailed || r.Reason != "adapter deadline exceeded" {
		t.Fatalf("result=%+v", r)
	}
}
