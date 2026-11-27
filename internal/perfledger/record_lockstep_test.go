package perfledger

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func lockstepWaitBounded(t *testing.T, barrier *LockstepBarrier) error {
	t.Helper()
	errCh := make(chan error, 1)
	go func() {
		// Wait with context.Background() (no timeout), matching RecordsManager.
		errCh <- barrier.Wait(context.Background())
	}()
	select {
	case err := <-errCh:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("LockstepBarrier.Wait(context.Background()) hung")
		return nil
	}
}

func assertLockstepCode(t *testing.T, err error, want LockstepViolationCode) {
	t.Helper()
	if err == nil {
		t.Fatalf("err = nil, want violation %q", want)
	}
	if !errors.Is(err, ErrLockstepViolation) {
		t.Fatalf("errors.Is(err, ErrLockstepViolation) = false for %T (%v)", err, err)
	}
	var le *LockstepError
	if errors.As(err, &le) {
		if !le.HasCode(want) {
			t.Fatalf("LockstepError violations = %+v, want code %q", le.Violations, want)
		}
		return
	}
	var lv *LockstepViolation
	if errors.As(err, &lv) {
		if lv.Code != want {
			t.Fatalf("LockstepViolation.Code = %q, want %q", lv.Code, want)
		}
		return
	}
	t.Fatalf("unexpected error type %T, want *LockstepViolation or *LockstepError", err)
}

// fak-test:runtime fast est=2s lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestRecordLockstep(t *testing.T) {
	t.Parallel()

	t.Run("normal_completion", func(t *testing.T) {
		t.Parallel()
		ledgerPath := filepath.Join(t.TempDir(), "lockstep.jsonl")
		w := OpenWriter(ledgerPath, 0)
		defer func() { _ = w.Close() }()

		barrier := NewLockstepBarrier(NewWriterRecordSink(w))
		credits := []Credit{
			{ID: "c1", Phase: "profile", RequestID: "r1", Locality: LocalitySelfHosted, IssuedAt: time.UnixMilli(1000)},
			{ID: "c2", Phase: "profile", RequestID: "r2", Locality: LocalitySelfHosted, IssuedAt: time.UnixMilli(1010)},
			{ID: "c-noop", Phase: "profile", NoRequest: true},
		}
		for _, c := range credits {
			if err := barrier.RegisterCredit(c); err != nil {
				t.Fatalf("RegisterCredit(%q): %v", c.ID, err)
			}
		}

		var wg sync.WaitGroup
		outcomes := make([]CreditOutcome, len(credits))
		for i, c := range credits {
			i, c := i, c
			wg.Add(1)
			go func() {
				defer wg.Done()
				outcomes[i] = ExecuteCreditWithClock(
					context.Background(),
					c,
					barrier,
					barrier,
					func() time.Time { return time.UnixMilli(2000) },
					func(_ context.Context, cc *CreditContext) error {
						if cc.Credit().NoRequest {
							return nil
						}
						rec := NewRecord(time.UnixMilli(2000), "stop", LocalitySelfHosted, 256, 32, 0, 200*time.Millisecond, 50*time.Millisecond)
						return cc.EmitRecord(rec)
					},
				)
			}()
		}
		wg.Wait()

		for i, out := range outcomes {
			if out.Err != nil || out.Panicked || out.FallbackRecordSent || out.FallbackEmitErr != nil || out.ReturnErr != nil || out.Violation != nil {
				t.Fatalf("outcome[%d] unexpected error/fallback: %+v", i, out)
			}
			wantRecord := !credits[i].NoRequest
			if out.Return.RecordEmitted != wantRecord || out.Return.Cancelled {
				t.Fatalf("outcome[%d].Return = %+v, want RecordEmitted=%v Cancelled=false", i, out.Return, wantRecord)
			}
		}

		barrier.Finalize(2)
		if err := lockstepWaitBounded(t, barrier); err != nil {
			t.Fatalf("Wait: %v", err)
		}
		if err := barrier.Verify(); err != nil {
			t.Fatalf("Verify: %v", err)
		}
		stats := barrier.Stats()
		if stats.Completed != 2 || stats.NoRequest != 1 || stats.Records != 2 || stats.Violations != 0 {
			t.Fatalf("stats = %+v, want Completed=2 NoRequest=1 Records=2 Violations=0", stats)
		}
		if err := w.Close(); err != nil {
			t.Fatalf("writer Close: %v", err)
		}
		rows, _ := ReadTail(ledgerPath)
		if len(rows) != 2 {
			t.Fatalf("persisted rows = %d, want 2", len(rows))
		}
		sum := Summarize(rows)
		if sum.Count != 2 || sum.Errors != 0 {
			t.Fatalf("summary = %+v, want Count=2 Errors=0", sum)
		}
	})

	t.Run("out_of_order_return_before_record_still_syncs", func(t *testing.T) {
		t.Parallel()
		barrier := NewLockstepBarrier(nil)
		key := CreditKey{Phase: "profile", CreditID: "async-1"}
		if err := barrier.ReturnCredit(CreditReturn{Key: key, RecordEmitted: true}); err != nil {
			t.Fatalf("ReturnCredit: %v", err)
		}
		barrier.Finalize(1)

		waitDone := make(chan error, 1)
		go func() {
			waitDone <- barrier.Wait(context.Background())
		}()

		select {
		case err := <-waitDone:
			t.Fatalf("barrier unblocked before matching record arrived: %v", err)
		case <-time.After(20 * time.Millisecond):
		}

		if err := barrier.EmitCreditRecord(key, NewRecord(time.UnixMilli(3000), "stop", LocalitySelfHosted, 16, 4, 0, 100*time.Millisecond, 20*time.Millisecond)); err != nil {
			t.Fatalf("EmitCreditRecord: %v", err)
		}
		select {
		case err := <-waitDone:
			if err != nil {
				t.Fatalf("Wait returned error after record arrived: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Wait hung after record arrived")
		}
	})

	t.Run("worker_error_and_panic_emit_contained_failure_record", func(t *testing.T) {
		t.Parallel()
		barrier := NewLockstepBarrier(nil)
		fixedNow := func() time.Time { return time.UnixMilli(5000) }

		// 1) Worker fails with upstream error before emitting a record.
		errCredit := Credit{ID: "err-1", Phase: "profile", RequestID: "req-err", Locality: LocalitySelfHosted, IssuedAt: time.UnixMilli(4500)}
		outErr := ExecuteCreditWithClock(context.Background(), errCredit, barrier, barrier, fixedNow, func(_ context.Context, _ *CreditContext) error {
			return errors.New("upstream 502")
		})
		if outErr.Err == nil || outErr.Panicked || !outErr.FallbackRecordSent || outErr.FallbackEmitErr != nil || outErr.Violation != nil {
			t.Fatalf("outErr = %+v", outErr)
		}
		if outErr.Return.Cancelled || !outErr.Return.RecordEmitted {
			t.Fatalf("outErr.Return = %+v, want non-cancelled with RecordEmitted=true", outErr.Return)
		}

		// 2) Worker fails with DeadlineExceeded (non-cancelled timeout -> ErrorDeadline).
		dlCredit := Credit{ID: "dl-1", Phase: "profile", RequestID: "req-dl", Locality: LocalitySelfHosted, IssuedAt: time.UnixMilli(4000)}
		outDL := ExecuteCreditWithClock(context.Background(), dlCredit, barrier, barrier, fixedNow, func(_ context.Context, _ *CreditContext) error {
			return context.DeadlineExceeded
		})
		if !outDL.FallbackRecordSent || outDL.Return.Cancelled || !outDL.Return.RecordEmitted {
			t.Fatalf("outDL = %+v", outDL)
		}

		// 3) Worker panics mid-flight -> panic recovered, failure record emitted in defer.
		panicCredit := Credit{ID: "panic-1", Phase: "profile", RequestID: "req-panic", Locality: LocalitySelfHosted, IssuedAt: time.UnixMilli(4800)}
		outPanic := ExecuteCreditWithClock(context.Background(), panicCredit, barrier, barrier, fixedNow, func(_ context.Context, _ *CreditContext) error {
			panic("worker nil deref")
		})
		if !outPanic.Panicked || outPanic.Err == nil || !outPanic.FallbackRecordSent || outPanic.FallbackEmitErr != nil || outPanic.Violation != nil {
			t.Fatalf("outPanic = %+v", outPanic)
		}
		if outPanic.Return.Cancelled || !outPanic.Return.RecordEmitted {
			t.Fatalf("outPanic.Return = %+v, want non-cancelled with RecordEmitted=true", outPanic.Return)
		}

		// 4) Worker silently returns nil without emitting -> defer still guarantees a failure record.
		silentCredit := Credit{ID: "silent-1", Phase: "profile", RequestID: "req-silent", Locality: LocalitySelfHosted}
		outSilent := ExecuteCreditWithClock(context.Background(), silentCredit, barrier, barrier, fixedNow, func(_ context.Context, _ *CreditContext) error {
			return nil
		})
		if !outSilent.FallbackRecordSent || !outSilent.Return.RecordEmitted || outSilent.Violation != nil {
			t.Fatalf("outSilent = %+v", outSilent)
		}

		barrier.Finalize(4)
		if err := lockstepWaitBounded(t, barrier); err != nil {
			t.Fatalf("Wait hung or failed despite fallback records: %v", err)
		}
		if err := barrier.Verify(); err != nil {
			t.Fatalf("Verify: %v", err)
		}
		recs := barrier.Records()
		if len(recs) != 4 {
			t.Fatalf("records len = %d, want 4", len(recs))
		}
		if recs[0].Error != ErrorUpstream || recs[0].E2EMS != 500 || recs[0].RequestID != "req-err" {
			t.Fatalf("recs[0] = %+v", recs[0])
		}
		if recs[1].Error != ErrorDeadline || recs[1].E2EMS != 1000 || recs[1].RequestID != "req-dl" {
			t.Fatalf("recs[1] = %+v", recs[1])
		}
		if recs[2].Error != ErrorUpstream || recs[2].E2EMS != 200 || recs[2].RequestID != "req-panic" {
			t.Fatalf("recs[2] = %+v", recs[2])
		}
		if recs[3].Error != ErrorUpstream || recs[3].RequestID != "req-silent" {
			t.Fatalf("recs[3] = %+v", recs[3])
		}

		// 5) Containment: if emitting the fallback failure record itself errors OR
		// panics, it must NOT abort CreditReturn delivery; instead the return
		// proceeds and closed detection flags the lockstep break.
		for _, mode := range []string{"sink_error", "sink_panic"} {
			mode := mode
			t.Run(mode, func(t *testing.T) {
				t.Parallel()
				failingBarrier := NewLockstepBarrier(nil)
				var returnDelivered atomic.Bool
				var deliveredReturn CreditReturn
				retObserver := CreditReturnSinkFunc(func(ret CreditReturn) error {
					returnDelivered.Store(true)
					deliveredReturn = ret
					return failingBarrier.ReturnCredit(ret)
				})
				brokenSink := CreditRecordSinkFunc(func(CreditKey, Record) error {
					if mode == "sink_panic" {
						panic("sink exploded")
					}
					return errors.New("sink disk full")
				})
				out := ExecuteCredit(context.Background(), Credit{ID: "broken-1", Phase: "profile"}, brokenSink, retObserver, func(_ context.Context, _ *CreditContext) error {
					return errors.New("worker boom")
				})
				if out.FallbackRecordSent || out.FallbackEmitErr == nil {
					t.Fatalf("expected contained fallback emit failure, got %+v", out)
				}
				if !returnDelivered.Load() {
					t.Fatal("credit return was aborted; expected ReturnCredit to be invoked despite fallback emit failure")
				}
				if deliveredReturn.RecordEmitted || deliveredReturn.Cancelled {
					t.Fatalf("deliveredReturn = %+v, want RecordEmitted=false Cancelled=false", deliveredReturn)
				}
				// Barrier rejects the broken return with ViolationCompletedWithoutRecord.
				assertLockstepCode(t, out.ReturnErr, ViolationCompletedWithoutRecord)
				if out.Return.RecordEmitted || out.Return.Cancelled {
					t.Fatalf("out.Return = %+v, want RecordEmitted=false Cancelled=false", out.Return)
				}
				if out.Violation == nil || out.Violation.Code != ViolationCompletedWithoutRecord {
					t.Fatalf("out.Violation = %+v, want %q", out.Violation, ViolationCompletedWithoutRecord)
				}
				vs := out.Violations()
				var sawEmitFail, sawMissing bool
				for _, v := range vs {
					if v.Code == ViolationFailureEmitFailed {
						sawEmitFail = true
					}
					if v.Code == ViolationCompletedWithoutRecord {
						sawMissing = true
					}
				}
				if !sawEmitFail || !sawMissing {
					t.Fatalf("out.Violations() = %+v, want both %q and %q", vs, ViolationFailureEmitFailed, ViolationCompletedWithoutRecord)
				}

				failingBarrier.Finalize(1)
				err := lockstepWaitBounded(t, failingBarrier)
				assertLockstepCode(t, err, ViolationCompletedWithoutRecord)
			})
		}

		// 6) Containment on ReturnCredit panic.
		panicReturnSink := CreditReturnSinkFunc(func(CreditReturn) error {
			panic("return channel broken")
		})
		outRetPanic := ExecuteCredit(context.Background(), Credit{ID: "ret-panic", NoRequest: true}, nil, panicReturnSink, func(_ context.Context, _ *CreditContext) error {
			return nil
		})
		if outRetPanic.ReturnErr == nil {
			t.Fatalf("expected contained ReturnErr on panicking return sink")
		}
	})

	t.Run("cancellation_vs_non_cancelled", func(t *testing.T) {
		t.Parallel()
		barrier := NewLockstepBarrier(nil)

		// A) Pre-cancelled context does not invoke fn, marks Cancelled=true, emits no record.
		preCtx, preCancel := context.WithCancel(context.Background())
		preCancel()
		var ran atomic.Bool
		outPre := ExecuteCredit(preCtx, Credit{ID: "cancel-pre", Phase: "profile"}, barrier, barrier, func(_ context.Context, _ *CreditContext) error {
			ran.Store(true)
			return nil
		})
		if ran.Load() {
			t.Fatal("pre-cancelled credit should not have invoked worker callback")
		}
		if !outPre.Return.Cancelled || outPre.Return.RecordEmitted || outPre.FallbackRecordSent || outPre.Violation != nil {
			t.Fatalf("outPre = %+v", outPre)
		}

		// B) Mid-flight cancellation via context.Canceled without emitting emits no failure record.
		midCtx, midCancel := context.WithCancel(context.Background())
		outMid := ExecuteCredit(midCtx, Credit{ID: "cancel-mid", Phase: "profile"}, barrier, barrier, func(ctx context.Context, _ *CreditContext) error {
			midCancel()
			return ctx.Err()
		})
		if !outMid.Return.Cancelled || outMid.Return.RecordEmitted || outMid.FallbackRecordSent || outMid.Violation != nil {
			t.Fatalf("outMid = %+v", outMid)
		}

		// C) Explicit cc.Cancel() mid-flight without record.
		outCC := ExecuteCredit(context.Background(), Credit{ID: "cancel-explicit", Phase: "profile"}, barrier, barrier, func(_ context.Context, cc *CreditContext) error {
			cc.Cancel()
			return errors.New("aborted after cancel")
		})
		if !outCC.Return.Cancelled || outCC.Return.RecordEmitted || outCC.FallbackRecordSent || outCC.Violation != nil {
			t.Fatalf("outCC = %+v", outCC)
		}

		// D) Mid-flight cancellation AFTER emitting an ErrorClientCanceled record must NOT duplicate-emit.
		outCancelAfterEmit := ExecuteCredit(context.Background(), Credit{ID: "cancel-after-emit", Phase: "profile"}, barrier, barrier, func(_ context.Context, cc *CreditContext) error {
			if err := cc.EmitRecord(NewFailureRecord(time.UnixMilli(6000), LocalitySelfHosted, ErrorClientCanceled, 499, 150*time.Millisecond, 0)); err != nil {
				return err
			}
			cc.Cancel()
			return context.Canceled
		})
		if !outCancelAfterEmit.Return.Cancelled || !outCancelAfterEmit.Return.RecordEmitted || outCancelAfterEmit.FallbackRecordSent || outCancelAfterEmit.Violation != nil {
			t.Fatalf("outCancelAfterEmit = %+v", outCancelAfterEmit)
		}

		// E) Non-cancelled completed credit MUST emit a record, and a prior
		// cancelled-with-record credit cannot satisfy Finalize(1) on its behalf.
		compKey := CreditKey{Phase: "profile", CreditID: "completed-1"}
		barrier.Finalize(1)
		waitDone := make(chan error, 1)
		go func() {
			waitDone <- barrier.Wait(context.Background())
		}()
		select {
		case err := <-waitDone:
			t.Fatalf("cancelled credit's record falsely satisfied completed barrier: %v", err)
		case <-time.After(20 * time.Millisecond):
		}

		outComp := ExecuteCredit(context.Background(), Credit{ID: compKey.CreditID, Phase: compKey.Phase}, barrier, barrier, func(_ context.Context, _ *CreditContext) error {
			return context.DeadlineExceeded
		})
		if outComp.Return.Cancelled || !outComp.Return.RecordEmitted || !outComp.FallbackRecordSent {
			t.Fatalf("outComp = %+v", outComp)
		}
		select {
		case err := <-waitDone:
			if err != nil {
				t.Fatalf("Wait failed after completed credit finished: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("Wait hung after completed credit finished")
		}

		if err := barrier.Verify(); err != nil {
			t.Fatalf("Verify: %v", err)
		}
		stats := barrier.Stats()
		if stats.Completed != 1 || stats.Cancelled != 4 || stats.Records != 2 || stats.Violations != 0 {
			t.Fatalf("stats = %+v, want Completed=1 Cancelled=4 Records=2 Violations=0", stats)
		}
	})

	t.Run("closed_violation_detection", func(t *testing.T) {
		t.Parallel()

		// 1) Closed vocabulary integrity: non-empty, unique, matches tested codes.
		seenVocab := make(map[LockstepViolationCode]bool, len(LockstepViolations))
		for _, code := range LockstepViolations {
			if code == ViolationNone {
				t.Fatal("LockstepViolations must not contain empty ViolationNone")
			}
			if seenVocab[code] {
				t.Fatalf("duplicate code %q in LockstepViolations", code)
			}
			seenVocab[code] = true
		}

		detected := make(map[LockstepViolationCode]bool)
		markDetected := func(code LockstepViolationCode, err error) {
			t.Helper()
			assertLockstepCode(t, err, code)
			if !seenVocab[code] {
				t.Fatalf("code %q missing from closed LockstepViolations slice", code)
			}
			detected[code] = true
		}

		// ViolationCompletedWithoutRecord: stateless + barrier fail-closed.
		{
			b := NewLockstepBarrier(nil)
			ret := CreditReturn{Key: CreditKey{Phase: "p", CreditID: "c1"}, Cancelled: false, NoRequest: false, RecordEmitted: false}
			markDetected(ViolationCompletedWithoutRecord, ValidateCreditReturn(ret))
			markDetected(ViolationCompletedWithoutRecord, b.ReturnCredit(ret))
			b.Finalize(1)
			markDetected(ViolationCompletedWithoutRecord, lockstepWaitBounded(t, b))
		}

		// ViolationEmptyCreditID across stateless, barrier, context, and ExecuteCredit.
		{
			b := NewLockstepBarrier(nil)
			markDetected(ViolationEmptyCreditID, ValidateCreditReturn(CreditReturn{}))
			markDetected(ViolationEmptyCreditID, b.RegisterCredit(Credit{}))
			markDetected(ViolationEmptyCreditID, b.EmitCreditRecord(CreditKey{}, Record{}))
			markDetected(ViolationEmptyCreditID, b.ReturnCredit(CreditReturn{}))
			out := ExecuteCredit(context.Background(), Credit{}, b, b, nil)
			markDetected(ViolationEmptyCreditID, out.Violation)
		}

		// ViolationNoRequestEmittedRecord across stateless, CreditContext, and barrier.
		{
			b := NewLockstepBarrier(nil)
			markDetected(ViolationNoRequestEmittedRecord, ValidateCreditReturn(CreditReturn{
				Key:           CreditKey{CreditID: "nr1"},
				NoRequest:     true,
				RecordEmitted: true,
			}))
			cc := NewCreditContext(Credit{ID: "nr2", NoRequest: true}, b)
			markDetected(ViolationNoRequestEmittedRecord, cc.EmitRecord(Record{}))

			// Return as NoRequest first, then attempt to emit record.
			if err := b.ReturnCredit(CreditReturn{Key: CreditKey{CreditID: "nr3"}, NoRequest: true}); err != nil {
				t.Fatalf("ReturnCredit nr3: %v", err)
			}
			markDetected(ViolationNoRequestEmittedRecord, b.EmitCreditRecord(CreditKey{CreditID: "nr3"}, Record{}))

			// Emit record first, then return claiming NoRequest.
			if err := b.EmitCreditRecord(CreditKey{CreditID: "nr4"}, Record{}); err != nil {
				t.Fatalf("EmitCreditRecord nr4: %v", err)
			}
			markDetected(ViolationNoRequestEmittedRecord, b.ReturnCredit(CreditReturn{Key: CreditKey{CreditID: "nr4"}, NoRequest: true}))
		}

		// ViolationDuplicateCreditReturn.
		{
			b := NewLockstepBarrier(nil)
			ret := CreditReturn{Key: CreditKey{CreditID: "dup-ret"}, Cancelled: true}
			if err := b.ReturnCredit(ret); err != nil {
				t.Fatalf("first ReturnCredit: %v", err)
			}
			markDetected(ViolationDuplicateCreditReturn, b.ReturnCredit(ret))
		}

		// ViolationDuplicateRecordEmit via CreditContext and LockstepBarrier.
		{
			b := NewLockstepBarrier(nil)
			cc := NewCreditContext(Credit{ID: "dup-rec"}, b)
			if err := cc.EmitRecord(Record{}); err != nil {
				t.Fatalf("first EmitRecord: %v", err)
			}
			markDetected(ViolationDuplicateRecordEmit, cc.EmitRecord(Record{}))
			markDetected(ViolationDuplicateRecordEmit, b.EmitCreditRecord(CreditKey{CreditID: "dup-rec"}, Record{}))
		}

		// ViolationMissingRecord: credit return claims RecordEmitted=true but barrier never saw record.
		{
			b := NewLockstepBarrier(nil)
			if err := b.ReturnCredit(CreditReturn{Key: CreditKey{CreditID: "lost-rec"}, RecordEmitted: true}); err != nil {
				t.Fatalf("ReturnCredit: %v", err)
			}
			b.Finalize(1)
			markDetected(ViolationMissingRecord, b.Verify())

			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Millisecond)
			defer cancel()
			err := b.Wait(ctx)
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("Wait timeout err = %v, want context.DeadlineExceeded", err)
			}
			markDetected(ViolationMissingRecord, err)
		}

		// ViolationOrphanRecord: record arrived without a CreditReturn.
		{
			b := NewLockstepBarrier(nil)
			if err := b.EmitCreditRecord(CreditKey{CreditID: "orphan-1"}, Record{}); err != nil {
				t.Fatalf("EmitCreditRecord: %v", err)
			}
			b.Finalize(0)
			markDetected(ViolationOrphanRecord, b.Verify())
		}

		// ViolationIncompleteCredits: registered credit never returned, or completed < expected.
		{
			b := NewLockstepBarrier(nil)
			if err := b.RegisterCredit(Credit{ID: "in-flight"}); err != nil {
				t.Fatalf("RegisterCredit: %v", err)
			}
			b.FinalizeRegistered()
			markDetected(ViolationIncompleteCredits, b.Verify())

			b2 := NewLockstepBarrier(nil)
			b2.Finalize(2)
			markDetected(ViolationIncompleteCredits, b2.Verify())
		}

		// ViolationFailureEmitFailed via CreditOutcome.Violations().
		{
			out := CreditOutcome{
				Return:          CreditReturn{Key: CreditKey{CreditID: "fail-emit"}, RecordEmitted: false},
				FallbackEmitErr: errors.New("write refused"),
			}
			vs := out.Violations()
			if len(vs) == 0 {
				t.Fatal("expected violations for failed fallback emit")
			}
			var found bool
			for i := range vs {
				if vs[i].Code == ViolationFailureEmitFailed {
					found = true
					vCopy := vs[i]
					markDetected(ViolationFailureEmitFailed, &vCopy)
				}
			}
			if !found {
				t.Fatalf("violations = %+v, want %q", vs, ViolationFailureEmitFailed)
			}
		}

		// Ensure every closed code in LockstepViolations was exercised and detected.
		for _, code := range LockstepViolations {
			if !detected[code] {
				t.Fatalf("closed violation code %q was not exercised by TestRecordLockstep", code)
			}
		}
	})
}
