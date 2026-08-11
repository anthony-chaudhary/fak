package fabricmap

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestTransportRegistryDispatchesOpaqueTechnology(t *testing.T) {
	registry := NewTransportRegistry()
	called := false
	err := registry.Register("future-storage-fabric-v9", AdapterFunc(func(_ context.Context, link Link, access Access, transfer Transfer) (HopReceipt, error) {
		called = true
		if transfer.Source != "ssd" || transfer.Destination != "gpu" || transfer.Operation != OperationCopy {
			t.Fatalf("transfer = %+v", transfer)
		}
		now := time.Now().UTC()
		return HopReceipt{LinkID: link.ID, From: transfer.Source, To: transfer.Destination, Transport: link.Transport, Bytes: transfer.Bytes, StartedAt: now, CompletedAt: now.Add(time.Millisecond), Integrity: Integrity{Algorithm: "sha256", Digest: "abc"}, State: HopComplete}, nil
	}))
	if err != nil {
		t.Fatal(err)
	}
	request := AuthorizationRequest{Access: Access{Principal: "p", Tenant: "t", Operation: OperationCopy}, Link: Link{ID: "direct", From: "ssd", To: "gpu", Transport: "future-storage-fabric-v9"}}
	receipt, err := registry.ExecuteHop(context.Background(), request, 4096)
	if err != nil {
		t.Fatal(err)
	}
	if !called || receipt.From != "ssd" || receipt.To != "gpu" {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestTransportRegistryRefusesUnknownAndReverseDirection(t *testing.T) {
	registry := NewTransportRegistry()
	request := AuthorizationRequest{Access: Access{Operation: OperationRead}, Link: Link{ID: "up", From: "L3", To: "L1", Transport: "gds"}}
	if _, err := registry.ExecuteHop(context.Background(), request, 1); !errors.Is(err, ErrAdapterMissing) {
		t.Fatalf("unknown error = %v", err)
	}
	if err := registry.Register("gds", AdapterFunc(func(_ context.Context, link Link, _ Access, transfer Transfer) (HopReceipt, error) {
		now := time.Now()
		return HopReceipt{LinkID: link.ID, From: transfer.Destination, To: transfer.Source, Transport: link.Transport, Bytes: transfer.Bytes, StartedAt: now, CompletedAt: now, Integrity: Integrity{Algorithm: "x", Digest: "y"}, State: HopComplete}, nil
	})); err != nil {
		t.Fatal(err)
	}
	if _, err := registry.ExecuteHop(context.Background(), request, 1); err == nil {
		t.Fatal("reverse receipt accepted")
	}
}

func TestTransportRegistryExecutesWholeAuthorizedRoute(t *testing.T) {
	registry := NewTransportRegistry()
	for _, transport := range []string{"rdma", "gpudirect-storage"} {
		name := transport
		if err := registry.Register(name, AdapterFunc(func(_ context.Context, link Link, _ Access, transfer Transfer) (HopReceipt, error) {
			now := time.Now().UTC()
			return HopReceipt{LinkID: link.ID, From: transfer.Source, To: transfer.Destination, Transport: link.Transport, Bytes: transfer.Bytes, StartedAt: now, CompletedAt: now.Add(time.Millisecond), Integrity: Integrity{Algorithm: "sha256", Digest: link.ID}, State: HopComplete}, nil
		})); err != nil {
			t.Fatal(err)
		}
	}
	authorized := AuthorizedRoute{Access: Access{Principal: "p", Tenant: "t", Operation: OperationCopy}, Route: Route{From: "ssd", To: "gpu", Links: []Link{{ID: "ssd-nic", From: "ssd", To: "nic", Transport: "rdma"}, {ID: "nic-gpu", From: "nic", To: "gpu", Transport: "gpudirect-storage"}}}}
	result, err := Execute(context.Background(), authorized, 8192, registry)
	if err != nil {
		t.Fatal(err)
	}
	if result.State != ExecutionComplete || len(result.Hops) != 2 {
		t.Fatalf("result = %+v", result)
	}
}

func TestTransportRegistrationIsDeterministic(t *testing.T) {
	registry := NewTransportRegistry()
	adapter := AdapterFunc(func(context.Context, Link, Access, Transfer) (HopReceipt, error) { return HopReceipt{}, nil })
	if err := registry.Register("z", adapter); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register("a", adapter); err != nil {
		t.Fatal(err)
	}
	if err := registry.Register("a", adapter); err == nil {
		t.Fatal("duplicate registration accepted")
	}
	got := registry.Transports()
	if len(got) != 2 || got[0] != "a" || got[1] != "z" {
		t.Fatalf("transports = %v", got)
	}
}
