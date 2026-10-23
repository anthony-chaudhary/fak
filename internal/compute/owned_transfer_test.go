package compute

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestOwnedTransferCompletionLifecycle(t *testing.T) {
	queryFailure := errors.New("query failed")
	waitFailure := errors.New("wait failed")
	for _, tc := range []struct {
		name  string
		state TransferCompletionState
		cause error
		want  TransferCompletionState
	}{
		{"delayed-completion", TransferComplete, nil, TransferComplete},
		{"query-failure", TransferUnknown, queryFailure, TransferFailed},
		{"wait-failure", TransferUnknown, waitFailure, TransferFailed},
		{"cancelled-wait", TransferUnknown, context.Canceled, TransferFailed},
		{"error-overrides-success", TransferComplete, waitFailure, TransferFailed},
		{"unknown", TransferUnknown, nil, TransferUnknown},
		{"invalid-state", TransferCompletionState(255), nil, TransferUnknown},
		{"failed-without-cause", TransferFailed, nil, TransferFailed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			borrowed := []byte{7, 8, 9}
			destination := &struct{ live bool }{true}
			domain := new(sync.Mutex)
			releases := 0
			d := OwnedTransferDescriptor{
				Class: MemoryOffload, DomainID: "domain-1", SourceID: "borrowed-1", DestinationID: "device-1",
				Domain: domain, Source: borrowed, Destination: destination,
				Release: func() { releases++ },
			}
			transfer, err := NewOwnedTransfer(d)
			if err != nil {
				t.Fatal(err)
			}
			initial := transfer.Witness()
			// The descriptor and witness are values: caller edits cannot alter the
			// stored class/identity or the retained references.
			d.Class, d.Source, d.Destination = MemoryScratchpad, nil, nil
			borrowed = nil
			if err := transfer.Retire(); !errors.Is(err, ErrTransferNotComplete) {
				t.Fatalf("early retirement: %v", err)
			}
			pending := transfer.Observe(TransferPending, nil)
			if pending != initial || releases != 0 {
				t.Fatalf("pending changed ownership: %+v, releases %d", pending, releases)
			}
			assertRetained := func() {
				t.Helper()
				source, ok := transfer.descriptor.Source.([]byte)
				if !ok || len(source) != 3 || source[0] != 7 || transfer.descriptor.Destination != destination || transfer.descriptor.Domain != domain || releases != 0 {
					t.Fatal("lost borrowed backing, destination, or domain before safe retirement")
				}
			}
			assertRetained()
			observed := transfer.Observe(tc.state, tc.cause)
			if observed.State != tc.want || observed.Class != MemoryOffload || observed.DomainID != "domain-1" || observed.SourceID != "borrowed-1" || observed.DestinationID != "device-1" || observed.Retired {
				t.Fatalf("completion witness: %+v", observed)
			}
			if transfer.CompletionError() != tc.cause {
				t.Fatal("original completion error lost")
			}
			if tc.cause != nil && observed.Failure != tc.cause.Error() {
				t.Fatal("failure text lost")
			}
			// Neither duplicate nor contradictory completion rewrites first evidence.
			for _, state := range []TransferCompletionState{tc.state, TransferComplete, TransferPending, TransferFailed} {
				if got := transfer.Observe(state, queryFailure); got != observed {
					t.Fatalf("terminal evidence overwritten: %+v", got)
				}
			}
			if initial.State != TransferPending || initial.Retired {
				t.Fatal("prior witness was mutated")
			}
			changed := observed
			changed.State, changed.Class, changed.Failure = TransferComplete, MemoryWeights, "edited"
			if changed == observed {
				t.Fatal("fixture did not change returned witness")
			}
			if transfer.Witness() != observed {
				t.Fatal("returned witness aliases mutable state")
			}
			for i := 0; i < 2; i++ {
				err = transfer.Retire()
				if tc.want == TransferComplete {
					if err != nil || releases != 1 || !transfer.Witness().Retired || transfer.descriptor.Source != nil || transfer.descriptor.Destination != nil {
						t.Fatalf("successful retirement: err %v, releases %d", err, releases)
					}
				} else {
					wantErr := tc.cause
					if wantErr == nil {
						wantErr = ErrTransferNotComplete
					}
					if err != wantErr || transfer.Witness().Retired {
						t.Fatalf("unsafe retirement: %v", err)
					}
					assertRetained()
				}
			}
			if observed.Retired {
				t.Fatal("retirement mutated earlier completion witness")
			}
		})
	}
	if transfer, err := NewOwnedTransfer(OwnedTransferDescriptor{}); err == nil || transfer != nil {
		t.Fatal("incomplete descriptor admitted")
	}
	var zero OwnedTransfer
	if got := zero.Observe(TransferComplete, nil); got.State != TransferUnknown || zero.Retire() != ErrTransferNotComplete {
		t.Fatal("zero-value helper admitted success without retention")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestOwnedTransferConcurrentCompletionRetirement(t *testing.T) {
	const callers = 8
	var releases atomic.Int32
	var releaseFinished atomic.Bool
	entered := make(chan struct{}, callers)
	allowRelease := make(chan struct{})
	var unblock sync.Once
	defer unblock.Do(func() { close(allowRelease) })
	transfer, err := NewOwnedTransfer(OwnedTransferDescriptor{
		Class: MemoryOffload, DomainID: "concurrent-domain", SourceID: "source", DestinationID: "destination",
		Domain: new(sync.Mutex), Source: []byte{1}, Destination: new(byte),
		Release: func() {
			releases.Add(1)
			entered <- struct{}{}
			<-allowRelease
			releaseFinished.Store(true)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	before := transfer.Witness()
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	start := make(chan struct{})
	observations := make(chan OwnedTransferWitness, callers)
	for i := 0; i < callers; i++ {
		go func() {
			<-start
			observations <- transfer.Observe(TransferComplete, nil)
		}()
	}
	close(start)
	completed := before
	completed.State = TransferComplete
	for i := 0; i < callers; i++ {
		select {
		case got := <-observations:
			if got != completed {
				t.Fatalf("concurrent completion changed witness: %+v", got)
			}
		case <-deadline.C:
			t.Fatal("concurrent completion did not finish")
		}
	}
	// Hold the release callback open while the duplicate retirees are launched.
	// Every successful return must follow callback completion, not merely its start.
	type result struct {
		err      error
		finished bool
		witness  OwnedTransferWitness
	}
	results := make(chan result, callers)
	ready := make(chan struct{}, callers)
	startRetire := make(chan struct{})
	for i := 0; i < callers; i++ {
		go func() {
			ready <- struct{}{}
			<-startRetire
			err := transfer.Retire()
			results <- result{err, releaseFinished.Load(), transfer.Witness()}
		}()
	}
	for i := 0; i < callers; i++ {
		select {
		case <-ready:
		case <-deadline.C:
			t.Fatal("retire callers did not become ready")
		}
	}
	close(startRetire)
	select {
	case <-entered:
	case <-deadline.C:
		t.Fatal("release callback did not start")
	}
	if releases.Load() != 1 || releaseFinished.Load() {
		t.Fatal("release gate did not hold exactly one callback")
	}
	unblock.Do(func() { close(allowRelease) })
	retired := completed
	retired.Retired = true
	for i := 0; i < callers; i++ {
		select {
		case got := <-results:
			if got.err != nil || !got.finished || got.witness != retired {
				t.Fatalf("retirement returned before stable release: %+v", got)
			}
		case <-deadline.C:
			t.Fatal("concurrent retirement did not finish")
		}
	}
	if releases.Load() != 1 || before.State != TransferPending || before.Retired || completed.Retired {
		t.Fatal("duplicate release or mutation of prior value snapshots")
	}
}
