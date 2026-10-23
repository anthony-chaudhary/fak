package compute

import (
	"context"
	"errors"
	"sync"
	"testing"
)

type ownedTransferTestDomain struct{ mu sync.Mutex }

func (d *ownedTransferTestDomain) ForwardExecutionOwner() *sync.Mutex { return &d.mu }

type ownedTransferTestCompletion struct {
	result  OwnedTransferResult
	calls   int
	inspect func()
}

func (c *ownedTransferTestCompletion) Query() OwnedTransferResult {
	c.calls++
	if c.inspect != nil {
		c.inspect()
	}
	return c.result
}

func (c *ownedTransferTestCompletion) Wait(ctx context.Context) OwnedTransferResult {
	c.calls++
	if c.inspect != nil {
		c.inspect()
	}
	if err := ctx.Err(); err != nil {
		return OwnedTransferResult{State: OwnedTransferPending, Err: err}
	}
	return c.result
}

// fak-test:runtime fast est=10ms lane=default
func TestOwnedTransferCompletionLifecycle(t *testing.T) {
	queryErr := errors.New("query failed")
	waitErr := errors.New("wait failed")
	for _, tc := range []struct {
		name      string
		class     MemoryClass
		result    OwnedTransferResult
		wait      bool
		cancel    bool
		wantState OwnedTransferState
		wantErr   error
	}{
		{"delayed-query", MemoryWeights, OwnedTransferResult{State: OwnedTransferComplete}, false, false, OwnedTransferComplete, nil},
		{"delayed-wait", MemoryOffload, OwnedTransferResult{State: OwnedTransferComplete}, true, false, OwnedTransferComplete, nil},
		{"query-error", MemoryKVCache, OwnedTransferResult{State: OwnedTransferPending, Err: queryErr}, false, false, OwnedTransferUnknown, queryErr},
		{"wait-error", MemoryActivation, OwnedTransferResult{State: OwnedTransferFailed, Err: waitErr}, true, false, OwnedTransferFailed, waitErr},
		{"unknown", MemoryUnknown, OwnedTransferResult{State: OwnedTransferUnknown}, false, false, OwnedTransferUnknown, ErrOwnedTransferUnknown},
		{"failed-without-error", MemoryScratchpad, OwnedTransferResult{State: OwnedTransferFailed}, false, false, OwnedTransferFailed, ErrOwnedTransferFailed},
		{"cancelled-wait", MemoryOffload, OwnedTransferResult{State: OwnedTransferComplete}, true, true, OwnedTransferUnknown, context.Canceled},
		{"complete-with-error", MemoryWeights, OwnedTransferResult{State: OwnedTransferComplete, Err: queryErr}, false, false, OwnedTransferFailed, queryErr},
		{"invalid-state", MemorySignalBuffer, OwnedTransferResult{State: OwnedTransferState(255)}, false, false, OwnedTransferUnknown, ErrOwnedTransferUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			domain := &ownedTransferTestDomain{}
			// The actual Go backing owner, not an address or allocation ID, is retained.
			source := &struct{ borrowed []float32 }{borrowed: []float32{1, 2, 3}}
			destination := &struct{ allocation int }{allocation: 1}
			descriptor := OwnedTransferDescriptor{
				Class: tc.class, Domain: domain, SourceOwner: source, DestinationOwner: destination,
			}
			completion := &ownedTransferTestCompletion{result: OwnedTransferResult{State: OwnedTransferPending}}
			transfer, err := NewOwnedTransfer(descriptor, completion)
			if err != nil {
				t.Fatal(err)
			}
			assertRetained := func() {
				t.Helper()
				got := transfer.Descriptor()
				if got.Class != tc.class || got.Domain != domain || transfer.Domain() != &domain.mu || got.SourceOwner != source || got.DestinationOwner != destination {
					t.Fatalf("lost class, domain, or retained owners: %+v", got)
				}
			}
			assertRetained()
			completion.inspect = func() {
				assertRetained()
				if transfer.Ready() || transfer.Query().State != OwnedTransferPending || transfer.Wait(context.Background()).State != OwnedTransferPending {
					t.Fatal("reentrant observation became Ready")
				}
				if _, retired, err := transfer.Retire(); retired || err != ErrOwnedTransferPending {
					t.Fatalf("retired inside completion callback: %t, %v", retired, err)
				}
			}
			for _, observe := range []func() OwnedTransferResult{transfer.Query, func() OwnedTransferResult { return transfer.Wait(context.Background()) }} {
				if got := observe(); got.State != OwnedTransferPending || got.Err != nil {
					t.Fatalf("delayed completion = %+v", got)
				}
				if transfer.Ready() {
					t.Fatal("pending transfer published as Ready")
				}
				if owners, retired, err := transfer.Retire(); retired || err != ErrOwnedTransferPending || owners.SourceOwner != nil || owners.DestinationOwner != nil {
					t.Fatalf("premature ownership return: %+v, %t, %v", owners, retired, err)
				}
				assertRetained()
			}
			completion.result = tc.result
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.cancel {
				cancel()
			}
			var got OwnedTransferResult
			if tc.wait {
				got = transfer.Wait(ctx)
			} else {
				got = transfer.Query()
			}
			if got.State != tc.wantState || got.Err != tc.wantErr {
				t.Fatalf("completion = %+v, want state %v and original error %v", got, tc.wantState, tc.wantErr)
			}
			assertRetained()
			wantReady := tc.wantState == OwnedTransferComplete
			if transfer.Ready() != wantReady {
				t.Fatalf("Ready = %t, want %t", transfer.Ready(), wantReady)
			}
			// A later optimistic result cannot erase a failure or uncertainty, and
			// a later provider error cannot rewrite a checked successful completion.
			calls := completion.calls
			completion.result = OwnedTransferResult{State: OwnedTransferComplete}
			if wantReady {
				completion.result = OwnedTransferResult{State: OwnedTransferFailed, Err: waitErr}
			}
			if again := transfer.Query(); again != got {
				t.Fatalf("terminal result changed: %+v -> %+v", got, again)
			}
			if again := transfer.Wait(context.Background()); again != got || completion.calls != calls {
				t.Fatalf("terminal completion was re-observed: %+v, calls %d -> %d", again, calls, completion.calls)
			}
			for attempt := 0; attempt < 2; attempt++ {
				owners, retired, err := transfer.Retire()
				if !wantReady {
					if retired || err != tc.wantErr || owners.SourceOwner != nil || owners.DestinationOwner != nil {
						t.Fatalf("uncertain/failed ownership returned: %+v, %t, %v", owners, retired, err)
					}
					assertRetained()
					continue
				}
				if err != nil || retired != (attempt == 0) {
					t.Fatalf("retirement %d = %t, %v", attempt, retired, err)
				}
				if attempt == 0 && (owners.Class != tc.class || owners.Domain != domain || owners.SourceOwner != source || owners.DestinationOwner != destination) {
					t.Fatalf("wrong ownership returned: %+v", owners)
				}
				if attempt != 0 && (owners.SourceOwner != nil || owners.DestinationOwner != nil) {
					t.Fatal("duplicate ownership return")
				}
				if retained := transfer.Descriptor(); retained.SourceOwner != nil || retained.DestinationOwner != nil || transfer.Ready() {
					t.Fatal("retired transfer retained ownership or still allowed publication")
				}
			}
		})
	}
	newTransfer := func(t *testing.T, completion OwnedTransferCompletion) (*OwnedTransfer, OwnedTransferDescriptor) {
		t.Helper()
		descriptor := OwnedTransferDescriptor{Class: MemoryWeights, Domain: &ownedTransferTestDomain{}, SourceOwner: new(int), DestinationOwner: new(int)}
		transfer, err := NewOwnedTransfer(descriptor, completion)
		if err != nil {
			t.Fatal(err)
		}
		return transfer, descriptor
	}
	t.Run("simultaneous-observations", func(t *testing.T) {
		entered, release := make(chan struct{}), make(chan struct{})
		released := false
		defer func() {
			if !released {
				close(release)
			}
		}()
		completion := &ownedTransferTestCompletion{
			result: OwnedTransferResult{State: OwnedTransferComplete},
			inspect: func() {
				close(entered)
				<-release
			},
		}
		transfer, descriptor := newTransfer(t, completion)
		result := make(chan OwnedTransferResult, 1)
		go func() { result <- transfer.Query() }()
		<-entered
		for _, got := range []OwnedTransferResult{transfer.Query(), transfer.Wait(context.Background())} {
			if got.State != OwnedTransferPending || got.Err != nil {
				t.Fatalf("overlapping observation = %+v", got)
			}
		}
		if owners, retired, err := transfer.Retire(); retired || err != ErrOwnedTransferPending || owners.SourceOwner != nil || owners.DestinationOwner != nil {
			t.Fatalf("retired during provider call: %+v, %t, %v", owners, retired, err)
		}
		if owners := transfer.Descriptor(); transfer.Ready() || owners.SourceOwner != descriptor.SourceOwner || owners.DestinationOwner != descriptor.DestinationOwner {
			t.Fatal("overlapping observation lost retention or published early")
		}
		close(release)
		released = true
		if got := <-result; got.State != OwnedTransferComplete || got.Err != nil || completion.calls != 1 || !transfer.Ready() {
			t.Fatalf("serialized provider result = %+v, calls = %d", got, completion.calls)
		}
	})
	t.Run("racing-retirement", func(t *testing.T) {
		transfer, descriptor := newTransfer(t, &ownedTransferTestCompletion{result: OwnedTransferResult{State: OwnedTransferComplete}})
		if got := transfer.Query(); got.State != OwnedTransferComplete || got.Err != nil {
			t.Fatalf("completion = %+v", got)
		}
		type retirement struct {
			owners  OwnedTransferDescriptor
			retired bool
			err     error
		}
		start := make(chan struct{})
		results := make(chan retirement, 2)
		for i := 0; i < 2; i++ {
			go func() {
				<-start
				owners, retired, err := transfer.Retire()
				results <- retirement{owners, retired, err}
			}()
		}
		close(start)
		handoffs := 0
		for i := 0; i < 2; i++ {
			got := <-results
			if got.err != nil {
				t.Fatal(got.err)
			}
			if got.retired {
				handoffs++
				if got.owners.SourceOwner != descriptor.SourceOwner || got.owners.DestinationOwner != descriptor.DestinationOwner || got.owners.Domain != descriptor.Domain || got.owners.Class != descriptor.Class {
					t.Fatalf("wrong retirement handoff: %+v", got.owners)
				}
			} else if got.owners.SourceOwner != nil || got.owners.DestinationOwner != nil {
				t.Fatal("duplicate retirement returned owners")
			}
		}
		if handoffs != 1 || transfer.Ready() {
			t.Fatalf("retirement handoffs = %d, Ready = %t", handoffs, transfer.Ready())
		}
	})
	t.Run("provider-panic", func(t *testing.T) {
		marker := errors.New("provider panic")
		completion := &ownedTransferTestCompletion{inspect: func() { panic(marker) }}
		transfer, descriptor := newTransfer(t, completion)
		var recovered any
		func() {
			defer func() { recovered = recover() }()
			transfer.Query()
		}()
		if recovered != marker {
			t.Fatalf("provider panic changed: %v", recovered)
		}
		if got := transfer.Query(); got.State != OwnedTransferUnknown || got.Err != ErrOwnedTransferUnknown || completion.calls != 1 || transfer.Ready() {
			t.Fatalf("panic completion = %+v, calls = %d", got, completion.calls)
		}
		if owners, retired, err := transfer.Retire(); retired || err != ErrOwnedTransferUnknown || owners.SourceOwner != nil || owners.DestinationOwner != nil {
			t.Fatalf("panic released ownership: %+v, %t, %v", owners, retired, err)
		}
		if owners := transfer.Descriptor(); owners.SourceOwner != descriptor.SourceOwner || owners.DestinationOwner != descriptor.DestinationOwner {
			t.Fatal("panic dropped retained owners")
		}
	})
	t.Run("invalid-and-zero-values-fail-closed", func(t *testing.T) {
		descriptor := OwnedTransferDescriptor{Class: MemoryWeights, Domain: &ownedTransferTestDomain{}, SourceOwner: new(int), DestinationOwner: new(int)}
		completion := &ownedTransferTestCompletion{}
		var nilDomain *ownedTransferTestDomain
		var nilSource []byte
		var nilCompletion *ownedTransferTestCompletion
		for _, invalidate := range []func(*OwnedTransferDescriptor){
			func(d *OwnedTransferDescriptor) { d.Class = "" },
			func(d *OwnedTransferDescriptor) { d.Domain = nilDomain },
			func(d *OwnedTransferDescriptor) { d.SourceOwner = nilSource },
			func(d *OwnedTransferDescriptor) { d.DestinationOwner = nil },
		} {
			invalid := descriptor
			invalidate(&invalid)
			if transfer, err := NewOwnedTransfer(invalid, completion); transfer != nil || err != ErrOwnedTransferInvalid {
				t.Fatalf("invalid descriptor accepted: %v, %v", transfer, err)
			}
		}
		if transfer, err := NewOwnedTransfer(descriptor, nilCompletion); transfer != nil || err != ErrOwnedTransferInvalid {
			t.Fatalf("typed-nil completion accepted: %v, %v", transfer, err)
		}
		for _, transfer := range []*OwnedTransfer{nil, {}} {
			if transfer.Ready() || transfer.Query().State != OwnedTransferUnknown {
				t.Fatal("uninitialized transfer appeared complete")
			}
			if _, retired, err := transfer.Retire(); retired || err != ErrOwnedTransferInvalid {
				t.Fatalf("uninitialized transfer retired: %t, %v", retired, err)
			}
		}
	})
}
