package harnesskit

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"testing"
)

func TestBuilderFreezesPortableSpec(t *testing.T) {
	profile := Profile{ID: "support", Extensions: []Extension{{ID: "kb", Plane: PlaneTools, Compatibility: ContractVersion, Provenance: Provenance{Source: "example.org/kb", Version: "v1.2.3"}}}}
	product, err := New("acme/support", "v1").WithProfile(profile).Build()
	if err != nil {
		t.Fatal(err)
	}
	profile.Extensions[0].ID = "mutated"
	got := product.Spec()
	if got.Profile.Extensions[0].ID != "kb" {
		t.Fatalf("product was not frozen: %#v", got)
	}
	got.Profile.Extensions[0].ID = "also-mutated"
	if product.Spec().Profile.Extensions[0].ID != "kb" {
		t.Fatal("Spec leaked mutable storage")
	}
}

func TestBuilderRejectsUnsupportedAndDuplicateExtensions(t *testing.T) {
	valid := Extension{ID: "x", Plane: PlaneTools, Compatibility: ContractVersion, Provenance: Provenance{Source: "example.org/x", Version: "v1"}}
	tests := []struct {
		name       string
		extensions []Extension
		code       Code
	}{
		{"unsupported", []Extension{{ID: "x", Plane: "kernel", Compatibility: ContractVersion, Provenance: valid.Provenance}}, CodeUnsupported},
		{"duplicate", []Extension{valid, valid}, CodeConflict},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := New("p", "v1").WithProfile(Profile{ID: "default", Extensions: tt.extensions}).Build()
			var contractErr *Error
			if !errors.As(err, &contractErr) || contractErr.Code != tt.code {
				t.Fatalf("got %v, want code %s", err, tt.code)
			}
		})
	}
}

func TestPublicContractIsCompleteAndMachineReadable(t *testing.T) {
	data, err := ContractJSON()
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip Contract
	if err := json.Unmarshal(data, &roundTrip); err != nil {
		t.Fatal(err)
	}
	got := PublicContract()
	if !reflect.DeepEqual(roundTrip, got) {
		t.Fatalf("round trip changed contract\n got %#v\nwant %#v", roundTrip, got)
	}
	if len(got.Planes) != 7 || len(got.Lifecycle) != 6 || len(got.Errors) < 7 || got.Security == "" || got.Compatibility == "" {
		t.Fatalf("incomplete contract: %#v", got)
	}
	if got.Ownership["factory_runtime"] != OwnershipHost {
		t.Fatalf("runtime ownership missing: %#v", got.Ownership)
	}
	if got.Hardware.PromotionEvidence == "" || got.Hardware.InvalidatingAssumption == "" {
		t.Fatalf("hardware contract missing: %#v", got.Hardware)
	}
	if got.Instructions.SchemaVersion != InstructionContractVersion || got.Instructions.Ownership == "" || got.Instructions.Security == "" || got.Instructions.Cancellation == "" {
		t.Fatalf("instruction composition contract missing: %#v", got.Instructions)
	}
	if got.Tools.Version != ToolContractVersion || got.Tools.Security == "" || got.Tools.Observability == "" {
		t.Fatalf("tool contract missing: %#v", got.Tools)
	}
}

type testStream struct{ ch chan int }

func (s testStream) Send(ctx context.Context, v int) error {
	select {
	case s.ch <- v:
		return nil
	case <-ctx.Done():
		return &Error{Code: CodeCanceled, Op: "send", Err: context.Cause(ctx)}
	}
}
func (s testStream) Recv(ctx context.Context) (int, error) {
	select {
	case v, ok := <-s.ch:
		if !ok {
			return 0, io.EOF
		}
		return v, nil
	case <-ctx.Done():
		return 0, &Error{Code: CodeCanceled, Op: "recv", Err: context.Cause(ctx)}
	}
}
func (s testStream) Close() error { close(s.ch); return nil }

func TestStreamContractMakesBackpressureCancelable(t *testing.T) {
	var stream Stream[int] = testStream{ch: make(chan int)}
	ctx, cancel := context.WithCancelCause(context.Background())
	cancel(errors.New("stop"))
	err := stream.Send(ctx, 1)
	var contractErr *Error
	if !errors.As(err, &contractErr) || contractErr.Code != CodeCanceled || !errors.Is(err, context.Cause(ctx)) {
		t.Fatalf("unexpected cancellation: %v", err)
	}
}
