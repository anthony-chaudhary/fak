package fabricmap

import (
	"context"
	"errors"
	"testing"
	"time"
)

func authorizedForExecution(operation Operation) AuthorizedRoute {
	return AuthorizedRoute{Access: Access{Principal: "p", Tenant: "t", Operation: operation}, Route: Route{From: "L3", To: "L1", Links: []Link{{ID: "ssd-nic", From: "L3", To: "nic", Transport: "rdma"}, {ID: "nic-gpu", From: "nic", To: "L1", Transport: "gpudirect"}}}}
}
func completeReceipt(r AuthorizationRequest, bytes uint64) HopReceipt {
	now := time.Now().UTC()
	return HopReceipt{HopIndex: r.HopIndex, LinkID: r.Link.ID, From: r.Link.From, To: r.Link.To, Transport: r.Link.Transport, Bytes: bytes, StartedAt: now, CompletedAt: now.Add(time.Millisecond), Integrity: Integrity{Algorithm: "sha256", Digest: "abc"}, State: HopComplete}
}

func TestExecuteEmitsCompleteDirectedReceipts(t *testing.T) {
	authorized := authorizedForExecution(OperationCopy)
	result, err := Execute(context.Background(), authorized, 4096, ExecutorFunc(func(_ context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
		return completeReceipt(r, n), nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	if result.State != ExecutionComplete || result.Recovery != RecoveryNone || len(result.Hops) != 2 {
		t.Fatalf("result = %+v", result)
	}
	if result.Hops[0].From != "L3" || result.Hops[1].To != "L1" {
		t.Fatalf("direction lost: %+v", result.Hops)
	}
}

func TestPartialCopyCanReplanButMoveNeedsOperator(t *testing.T) {
	for _, test := range []struct {
		operation Operation
		want      RecoveryAction
	}{{OperationCopy, RecoveryAlternateRoute}, {OperationMove, RecoveryOperator}, {OperationWriteBack, RecoveryOperator}} {
		t.Run(string(test.operation), func(t *testing.T) {
			result, err := Execute(context.Background(), authorizedForExecution(test.operation), 8, ExecutorFunc(func(_ context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
				if r.HopIndex == 1 {
					return HopReceipt{}, errors.New("link down")
				}
				return completeReceipt(r, n), nil
			}))
			if err == nil {
				t.Fatal("expected failure")
			}
			if result.State != ExecutionPartial || result.Recovery != test.want {
				t.Fatalf("result = %+v", result)
			}
			if result.Hops[0].State != HopComplete || result.Hops[1].State != HopFailed {
				t.Fatalf("hop states = %+v", result.Hops)
			}
		})
	}
}

func TestFirstHopFailureIsRetryableAndNotPartial(t *testing.T) {
	result, err := Execute(context.Background(), authorizedForExecution(OperationMove), 8, ExecutorFunc(func(context.Context, AuthorizationRequest, uint64) (HopReceipt, error) {
		return HopReceipt{}, errors.New("down")
	}))
	if err == nil || result.State != ExecutionFailed || result.Recovery != RecoveryRetry {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}

func TestInvalidReceiptCannotClaimEndToEndSuccess(t *testing.T) {
	result, err := Execute(context.Background(), authorizedForExecution(OperationCopy), 8, ExecutorFunc(func(_ context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
		receipt := completeReceipt(r, n)
		receipt.To = "wrong-destination"
		return receipt, nil
	}))
	if err == nil {
		t.Fatal("expected witness failure")
	}
	if result.State == ExecutionComplete || result.Recovery != RecoveryOperator {
		t.Fatalf("result = %+v", result)
	}
}

func TestMissingIntegrityCannotClaimSuccess(t *testing.T) {
	result, err := Execute(context.Background(), authorizedForExecution(OperationRead), 8, ExecutorFunc(func(_ context.Context, r AuthorizationRequest, n uint64) (HopReceipt, error) {
		receipt := completeReceipt(r, n)
		receipt.Integrity = Integrity{}
		return receipt, nil
	}))
	if err == nil || result.State == ExecutionComplete {
		t.Fatalf("result=%+v err=%v", result, err)
	}
}
