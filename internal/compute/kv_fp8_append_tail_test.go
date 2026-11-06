package compute

import (
	"errors"
	"runtime"
	"testing"
	"time"
)

// These host-transport controls assert the transaction's own fence/retention
// decisions, not delayed CUDA execution, actual attention, or producer parity.
// fak-test:runtime fast est=100ms lane=default
func TestKVFP8AppendTailAndBoundaryControl(t *testing.T) {
	savedOwners, savedAppend := kvFP8RetainedOwners, kvFP8AppendRetained
	defer func() { kvFP8RetainedOwners, kvFP8AppendRetained = savedOwners, savedAppend }()

	t.Run("both layers stop before the next dependent operation", func(t *testing.T) {
		// Prefix, then validate/ingest/read for each layer, then final tail.
		want := [8][3]int{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1},
			{2, 1, 1}, {2, 2, 1}, {2, 2, 2}, {2, 2, 2}}
		for at := 1; at <= 8; at++ {
			o, f := kvFP8AppendFixture(t)
			f.rejectCheck = at
			beforeList := kvFP8AppendRetained
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), f.run)
			if !errors.Is(err, ErrKVFP8NonFinite) {
				t.Fatalf("boundary %d: %v", at, err)
			}
			got := [3]int{f.validations, f.ingests, f.reads}
			if got != want[at-1] {
				t.Fatalf("boundary %d crossed dependent stage: got %v want %v", at, got, want[at-1])
			}
			checks := at
			if at > 1 && at < 8 {
				checks++ // run returned after an observed numerical rejection
			}
			if f.checks != checks || f.base.releases != 11 || o.stage != nil ||
				o.retained || o.backend.stopped.Load() || kvFP8AppendRetained != beforeList {
				t.Fatalf("boundary %d: checks=%d releases=%d retained=%v", at, f.checks, f.base.releases, o.retained)
			}
		}
	})

	t.Run("caught callback panic closes before caller can retry", func(t *testing.T) {
		for _, phase := range []string{"validate", "ingest", "read"} {
			t.Run(phase, func(t *testing.T) {
				o, f := kvFP8AppendFixture(t)
				old := o.image
				ops := f.ops()
				validate, ingest := ops.validate, ops.ingest
				panicked := false
				if phase == "validate" {
					ops.validate = func(in kvFP8AppendInput, status kvFP8Allocation) int {
						code := validate(in, status)
						if !panicked {
							panicked = true
							panic("callback-unwind")
						}
						return code
					}
				}
				if phase == "ingest" {
					ops.ingest = func(dst [kvFP8Planes]kvFP8Allocation, prefix int, in kvFP8AppendInput, status kvFP8Allocation) int {
						code := ingest(dst, prefix, in, status)
						if !panicked {
							panicked = true
							panic("callback-unwind")
						}
						return code
					}
				}
				beforeList := kvFP8AppendRetained
				var tx *kvFP8AppendTransaction
				var recovered any
				var retryAppend, retryConsume error
				closedBeforeRetry, retryRead, retryEffects := false, false, false
				err := kvFP8AppendEager(o, 4, 1, 1<<20, ops, func(current *kvFP8AppendTransaction) error {
					tx = current
					input := f.input(0, 1)
					func() {
						defer func() { recovered = recover() }()
						if err := current.append(0, input); err != nil {
							return
						}
						if phase == "read" {
							_ = current.consume(0, func(kvFP8AppendLayer) int {
								f.reads++
								panic("callback-unwind")
							})
						}
					}()
					closedBeforeRetry = o.uncertain && o.backend.stopped.Load() &&
						current.retained && kvFP8AppendRetained == current && f.stops > 0
					before := [4]int{f.validations, f.ingests, f.reads, f.checks}
					retryAppend = current.append(0, input)
					retryConsume = current.consume(0, func(kvFP8AppendLayer) int { retryRead = true; return 0 })
					retryEffects = before != [4]int{f.validations, f.ingests, f.reads, f.checks}
					return nil // caller also ignores both retry refusals
				})
				if err == nil || recovered != "callback-unwind" || !closedBeforeRetry ||
					retryAppend == nil || retryConsume == nil || retryRead || retryEffects ||
					tx == nil || tx.active || o.image != old || o.stage != tx.stage ||
					!o.retained || f.base.releases != 0 || kvFP8AppendRetained != tx ||
					tx.retainNext != beforeList || len(tx.inputs) != o.layers || tx.inputs[0].root == nil {
					t.Fatal("recovered callback panic left a reusable or unrooted transaction", err)
				}
			})
		}
	})

	for _, mode := range []string{"normal", "panic", "goexit"} {
		t.Run("ignored numerical error tail/"+mode, func(t *testing.T) {
			o, f := kvFP8AppendFixture(t)
			old := o.image
			view, err := o.borrow(0, kvFP8Key)
			if err != nil || !view.Ready() {
				t.Fatal("fixture needs a live borrowed view", err)
			}
			f.rejectCheck = 2 // input validation is complete but numerically invalid
			ops := f.ops()
			finish, release := ops.eager.finish, ops.eager.owner.release
			pendingTail, earlyRelease := false, false
			tailFences := 0
			ops.eager.finish = func(status *kvFP8Allocation) kvFP8Completion {
				if pendingTail {
					tailFences++
					pendingTail = false
				}
				return finish(status)
			}
			ops.eager.owner.release = func(a kvFP8Allocation) kvFP8Retirement {
				if pendingTail {
					earlyRelease = true
				}
				return release(a)
			}
			beforeList := kvFP8AppendRetained
			var tx *kvFP8AppendTransaction
			var stage *kvFP8Image
			var appendErr, result error
			var recovered any
			normal := false
			done := make(chan struct{})
			go func() {
				defer close(done)
				defer func() { recovered = recover() }()
				result = kvFP8AppendEager(o, 4, 1, 1<<20, ops, func(current *kvFP8AppendTransaction) error {
					tx, stage = current, current.stage
					appendErr = current.append(0, f.input(0, 1))
					if !errors.Is(appendErr, ErrKVFP8NonFinite) {
						return appendErr
					}
					// Model a caller that ignores rejection and queues tail work.
					// The sentinel survives until a later finish callback runs.
					pendingTail = true
					switch mode {
					case "panic":
						panic("append-tail-unwind")
					case "goexit":
						runtime.Goexit()
					}
					return nil
				})
				normal = true
			}()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("append tail control did not terminate")
			}
			if tx == nil || tx.active || o.image != old || view.Ready() ||
				!errors.Is(appendErr, ErrKVFP8NonFinite) || o.numericalFailure != 8 ||
				!errors.Is(o.primaryFailure, ErrKVFP8NonFinite) || earlyRelease {
				t.Fatal("lost rejection, stale-view invalidation, or control ownership")
			}
			if mode == "normal" {
				if !normal || recovered != nil || !errors.Is(result, ErrKVFP8NonFinite) ||
					pendingTail || tailFences != 1 || f.checks != 3 || f.stops != 0 ||
					o.stage != nil || o.retained || o.backend.stopped.Load() ||
					f.base.releases != 11 || tx.inputs != nil || kvFP8AppendRetained != beforeList {
					t.Fatal("ignored rejection tail was not fenced before known-complete discard", result)
				}
			} else {
				if normal || !pendingTail || tailFences != 0 || f.checks != 2 || f.stops != 1 ||
					!o.retained || !o.backend.stopped.Load() || o.stage != stage || f.base.releases != 0 ||
					kvFP8AppendRetained != tx || tx.owner != o || tx.stage != stage ||
					len(tx.inputs) != o.layers || tx.inputs[0].root == nil || tx.retainNext != beforeList {
					t.Fatal("abnormal tail did not retain this exact transaction and both images")
				}
				if mode == "panic" && recovered != "append-tail-unwind" {
					t.Fatal("panic was replaced", recovered)
				}
				if mode == "goexit" && recovered != nil {
					t.Fatal("Goexit became a panic", recovered)
				}
			}
			checks, stops, retained := f.checks, f.stops, kvFP8AppendRetained
			readCalled := false
			if tx.append(0, kvFP8AppendInput{}) == nil ||
				tx.consume(0, func(kvFP8AppendLayer) int { readCalled = true; return 0 }) == nil ||
				readCalled || f.checks != checks || f.stops != stops || kvFP8AppendRetained != retained {
				t.Fatal("inactive escaped transaction performed work")
			}
		})
	}
}
