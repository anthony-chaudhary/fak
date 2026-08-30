package vdso

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type recordingResultStore struct {
	refs []abi.Ref
	err  error
}

func (s *recordingResultStore) Store(_ context.Context, ref abi.Ref) error {
	s.refs = append(s.refs, ref)
	return s.err
}

func TestWriteThroughResultStoreReceivesIdenticalRef(t *testing.T) {
	ctx := context.Background()
	want := abi.Ref{
		Kind:   abi.RefBlob,
		Digest: "sha256:write-through-witness",
		Handle: 41,
		Len:    4096,
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeTenant,
	}
	var resident []abi.Ref
	durable := &recordingResultStore{}

	receipt, err := writeThroughResult(ctx, want, func(_ context.Context, ref abi.Ref) (bool, error) {
		resident = append(resident, ref)
		return true, nil
	}, durable)
	if err != nil {
		t.Fatalf("writeThroughResult: %v", err)
	}
	if len(resident) != 1 || len(durable.refs) != 1 {
		t.Fatalf("store calls resident=%d durable=%d, want 1/1", len(resident), len(durable.refs))
	}
	if !reflect.DeepEqual(resident[0], want) || !reflect.DeepEqual(durable.refs[0], want) {
		t.Fatalf("stores received different refs:\n resident=%+v\n durable=%+v\n want=%+v",
			resident[0], durable.refs[0], want)
	}
	if !receipt.Resident || !receipt.Durable || receipt.Replication != ResultFullyReplicated {
		t.Fatalf("receipt=%+v, want fully replicated", receipt)
	}
}

func TestWriteThroughDurableFailureIsTypedAndLookupStaysPartial(t *testing.T) {
	ctx := context.Background()
	delegateErr := errors.New("durable store unavailable")
	durable := &recordingResultStore{err: delegateErr}
	v := New(8)
	v.SetWriteThroughResultStore(durable)

	call := roCall("read_replication_fixture", `{"id":"9946"}`)
	ref := abi.Ref{
		Kind:   abi.RefInline,
		Digest: "sha256:partial-write-through-witness",
		Inline: []byte(`{"value":"resident"}`),
		Len:    int64(len(`{"value":"resident"}`)),
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeAgent,
	}
	result := &abi.Result{Call: call, Status: abi.StatusOK, Payload: ref}

	receipt, err := v.StoreResult(ctx, call, result)
	var partial *PartialReplicationError
	if !errors.As(err, &partial) {
		t.Fatalf("StoreResult error=%T %v, want *PartialReplicationError", err, err)
	}
	if !errors.Is(err, delegateErr) {
		t.Fatalf("StoreResult error does not unwrap delegate failure: %v", err)
	}
	if !receipt.Resident || receipt.Durable || receipt.Replication != ResultPartiallyReplicated {
		t.Fatalf("receipt=%+v, want resident partial replication", receipt)
	}
	if !reflect.DeepEqual(partial.Receipt, receipt) {
		t.Fatalf("typed partial receipt=%+v, want returned receipt=%+v", partial.Receipt, receipt)
	}
	if len(durable.refs) != 1 || !reflect.DeepEqual(durable.refs[0], ref) {
		t.Fatalf("durable refs=%+v, want the inserted ref %+v", durable.refs, ref)
	}

	hit, ok := v.Lookup(ctx, call)
	if !ok {
		t.Fatal("Lookup after partial replication missed the resident result")
	}
	if got := hit.Meta[MetaResultReplication]; got != string(ResultPartiallyReplicated) {
		t.Fatalf("Lookup replication=%q, want %q", got, ResultPartiallyReplicated)
	}
	if hit.Meta[MetaResultReplication] == string(ResultFullyReplicated) {
		t.Fatal("Lookup claimed full replication after the durable write failed")
	}
}

func TestWriteThroughDefaultRemainsResidentOnly(t *testing.T) {
	ctx := context.Background()
	v := New(8)
	call := roCall("read_resident_only_fixture", `{"id":"default"}`)
	ref := abi.Ref{
		Kind:   abi.RefInline,
		Digest: "sha256:resident-only-witness",
		Inline: []byte(`{"value":"local"}`),
		Len:    int64(len(`{"value":"local"}`)),
	}

	receipt, err := v.StoreResult(ctx, call, &abi.Result{Call: call, Status: abi.StatusOK, Payload: ref})
	if err != nil {
		t.Fatalf("StoreResult default: %v", err)
	}
	if !receipt.Resident || receipt.Durable || receipt.Replication != ResultResidentOnly {
		t.Fatalf("default receipt=%+v, want resident-only", receipt)
	}
	hit, ok := v.Lookup(ctx, call)
	if !ok {
		t.Fatal("Lookup after default resident insertion missed")
	}
	if got := hit.Meta[MetaResultReplication]; got != string(ResultResidentOnly) {
		t.Fatalf("Lookup replication=%q, want %q", got, ResultResidentOnly)
	}
}
