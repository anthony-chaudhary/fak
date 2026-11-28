// Adapted from ai-dynamo/aiperf, src/aiperf/workers/worker.py:1551
// (together with CreditContext in src/aiperf/credit/structs.py and the
// completion barrier in src/aiperf/records/records_manager.py), pinned at
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0. Changes were made: adapted from Python
// asyncio/ZMQ worker finally-block lockstep into idiomatic Go execution guards,
// a thread-safe no-timeout completion barrier, and closed violation detection
// wired to perfledger.Record.
package perfledger

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ErrLockstepViolation is the sentinel wrapped by LockstepViolation and
// LockstepError so callers can check errors.Is(err, ErrLockstepViolation).
var ErrLockstepViolation = errors.New("perfledger: record lockstep violation")

// ErrWriterRejected reports that NewWriterRecordSink could not enqueue a row
// because the Writer was nil, closed, or its bounded queue was full.
var ErrWriterRejected = errors.New("perfledger: writer rejected record")

// LockstepViolationCode is the closed vocabulary of credit<->record lockstep
// invariant failures.
type LockstepViolationCode string

const (
	ViolationNone                   LockstepViolationCode = ""
	ViolationEmptyCreditID          LockstepViolationCode = "empty_credit_id"
	ViolationCompletedWithoutRecord LockstepViolationCode = "completed_without_record"
	ViolationNoRequestEmittedRecord LockstepViolationCode = "no_request_emitted_record"
	ViolationDuplicateCreditReturn  LockstepViolationCode = "duplicate_credit_return"
	ViolationDuplicateRecordEmit    LockstepViolationCode = "duplicate_record_emit"
	ViolationMissingRecord          LockstepViolationCode = "missing_record"
	ViolationOrphanRecord           LockstepViolationCode = "orphan_record"
	ViolationIncompleteCredits      LockstepViolationCode = "incomplete_credits"
	ViolationFailureEmitFailed      LockstepViolationCode = "failure_emit_failed"
)

// LockstepViolations lists the closed lockstep violation vocabulary in
// canonical order.
var LockstepViolations = []LockstepViolationCode{
	ViolationEmptyCreditID,
	ViolationCompletedWithoutRecord,
	ViolationNoRequestEmittedRecord,
	ViolationDuplicateCreditReturn,
	ViolationDuplicateRecordEmit,
	ViolationMissingRecord,
	ViolationOrphanRecord,
	ViolationIncompleteCredits,
	ViolationFailureEmitFailed,
}

// LockstepViolation describes one concrete break of the credit<->record
// lockstep invariant with a closed machine-readable Code.
type LockstepViolation struct {
	Code   LockstepViolationCode
	Key    CreditKey
	Detail string
	Cause  error
}

func (v *LockstepViolation) Error() string {
	if v == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("perfledger: lockstep violation [")
	b.WriteString(string(v.Code))
	b.WriteByte(']')
	if v.Key.Phase != "" || v.Key.CreditID != "" {
		fmt.Fprintf(&b, " phase=%q credit_id=%q", v.Key.Phase, v.Key.CreditID)
	}
	if v.Detail != "" {
		b.WriteString(": ")
		b.WriteString(v.Detail)
	}
	if v.Cause != nil {
		b.WriteString(": ")
		b.WriteString(v.Cause.Error())
	}
	return b.String()
}

func (v *LockstepViolation) Unwrap() error {
	if v == nil {
		return nil
	}
	return v.Cause
}

func (v *LockstepViolation) Is(target error) bool {
	return target == ErrLockstepViolation
}

// LockstepError aggregates one or more closed LockstepViolations detected by
// LockstepBarrier.Wait or LockstepBarrier.Verify.
type LockstepError struct {
	Violations []LockstepViolation
}

func (e *LockstepError) Error() string {
	if e == nil || len(e.Violations) == 0 {
		return ""
	}
	if len(e.Violations) == 1 {
		v := e.Violations[0]
		return v.Error()
	}
	codes := make([]string, len(e.Violations))
	for i := range e.Violations {
		codes[i] = string(e.Violations[i].Code)
	}
	return fmt.Sprintf("perfledger: %d lockstep violations (%s): %s",
		len(e.Violations), strings.Join(codes, ","), e.Violations[0].Error())
}

func (e *LockstepError) Is(target error) bool {
	return target == ErrLockstepViolation
}

// HasCode reports whether any violation in e matches code.
func (e *LockstepError) HasCode(code LockstepViolationCode) bool {
	if e == nil {
		return false
	}
	for i := range e.Violations {
		if e.Violations[i].Code == code {
			return true
		}
	}
	return false
}

// CreditKey uniquely identifies one issued credit within a phase, mirroring
// aiperf's composite in-flight key (phase, credit_id).
type CreditKey struct {
	Phase    string
	CreditID string
}

// Credit is the scheduled unit of work tracked across execution, record
// emission, and return to the completion barrier.
type Credit struct {
	ID        string
	Phase     string
	RequestID string
	Locality  string
	IssuedAt  time.Time
	// NoRequest marks virtual credits that issue no inference request and are
	// exempt from the completed-credit record requirement.
	NoRequest bool
}

// Key returns the Credit's composite CreditKey.
func (c Credit) Key() CreditKey {
	return CreditKey{Phase: c.Phase, CreditID: c.ID}
}

// CreditReturn is the worker's terminal return signal sent in defer/finally
// regardless of success, error, panic, or cancellation.
type CreditReturn struct {
	Key           CreditKey
	Cancelled     bool
	NoRequest     bool
	RecordEmitted bool
}

// RequiresRecord reports whether this return is a completed (non-cancelled,
// request-bearing) credit that MUST be accompanied by an emitted record.
func (r CreditReturn) RequiresRecord() bool {
	return !r.Cancelled && !r.NoRequest
}

// ValidateCreditReturn enforces the closed stateless lockstep contract on a
// CreditReturn: non-empty CreditID, no emitted record on a NoRequest virtual
// credit, and every completed (non-cancelled, non-NoRequest) credit must carry
// RecordEmitted=true.
func ValidateCreditReturn(r CreditReturn) *LockstepViolation {
	if r.Key.CreditID == "" {
		return &LockstepViolation{
			Code:   ViolationEmptyCreditID,
			Key:    r.Key,
			Detail: "credit_id must be non-empty",
		}
	}
	if r.NoRequest && r.RecordEmitted {
		return &LockstepViolation{
			Code:   ViolationNoRequestEmittedRecord,
			Key:    r.Key,
			Detail: "no_request virtual credit must not emit a record",
		}
	}
	if r.RequiresRecord() && !r.RecordEmitted {
		return &LockstepViolation{
			Code:   ViolationCompletedWithoutRecord,
			Key:    r.Key,
			Detail: "completed non-cancelled credit returned without an emitted record",
		}
	}
	return nil
}

// CreditRecordSink receives records emitted for individual credits.
type CreditRecordSink interface {
	EmitCreditRecord(key CreditKey, rec Record) error
}

// CreditRecordSinkFunc adapts a function into a CreditRecordSink.
type CreditRecordSinkFunc func(key CreditKey, rec Record) error

func (f CreditRecordSinkFunc) EmitCreditRecord(key CreditKey, rec Record) error {
	return f(key, rec)
}

// CreditReturnSink receives terminal CreditReturn signals.
type CreditReturnSink interface {
	ReturnCredit(ret CreditReturn) error
}

// CreditReturnSinkFunc adapts a function into a CreditReturnSink.
type CreditReturnSinkFunc func(ret CreditReturn) error

func (f CreditReturnSinkFunc) ReturnCredit(ret CreditReturn) error {
	return f(ret)
}

type writerRecordSink struct {
	w *Writer
}

// NewWriterRecordSink adapts a perfledger.Writer into a CreditRecordSink.
// If w is nil or refuses the record (closed or queue full), EmitCreditRecord
// returns ErrWriterRejected so lockstep never treats a dropped write as emitted.
func NewWriterRecordSink(w *Writer) CreditRecordSink {
	return writerRecordSink{w: w}
}

func (s writerRecordSink) EmitCreditRecord(_ CreditKey, rec Record) error {
	if s.w == nil || !s.w.Offer(rec) {
		return ErrWriterRejected
	}
	return nil
}

// CreditContext holds worker-side per-credit lockstep state, including
// RecordEmitted, Cancelled, and Returned. Methods are concurrency-safe.
type CreditContext struct {
	emitMu sync.Mutex
	mu     sync.Mutex

	credit Credit
	sink   CreditRecordSink
	now    func() time.Time

	recordEmitted bool
	cancelled     bool
	returned      bool
}

// NewCreditContext builds per-credit execution state bound to sink.
func NewCreditContext(credit Credit, sink CreditRecordSink) *CreditContext {
	return &CreditContext{
		credit: credit,
		sink:   sink,
		now:    time.Now,
	}
}

// WithClock overrides the clock used for unset record timestamps and failure
// duration calculation (primarily for deterministic tests).
func (c *CreditContext) WithClock(now func() time.Time) *CreditContext {
	if c != nil && now != nil {
		c.mu.Lock()
		c.now = now
		c.mu.Unlock()
	}
	return c
}

// Credit returns a copy of the bound Credit.
func (c *CreditContext) Credit() Credit { return c.credit }

// Key returns the Credit's composite CreditKey.
func (c *CreditContext) Key() CreditKey { return c.credit.Key() }

// RecordEmitted reports whether a record has been successfully committed for
// this credit.
func (c *CreditContext) RecordEmitted() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.recordEmitted
}

// Cancelled reports whether this credit has been marked cancelled.
func (c *CreditContext) Cancelled() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.cancelled
}

// Returned reports whether this credit has already finalized its return.
func (c *CreditContext) Returned() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.returned
}

// Cancel marks the credit cancelled before/during execution.
func (c *CreditContext) Cancel() {
	c.mu.Lock()
	c.cancelled = true
	c.mu.Unlock()
}

// EmitRecord validates lockstep constraints, fills unset record identity
// defaults from the Credit, forwards rec to the configured CreditRecordSink,
// and marks RecordEmitted=true iff the sink succeeds without error or panic.
func (c *CreditContext) EmitRecord(rec Record) error {
	key := c.Key()
	if key.CreditID == "" {
		return &LockstepViolation{
			Code:   ViolationEmptyCreditID,
			Key:    key,
			Detail: "credit_id must be non-empty",
		}
	}
	if c.credit.NoRequest {
		return &LockstepViolation{
			Code:   ViolationNoRequestEmittedRecord,
			Key:    key,
			Detail: "no_request virtual credit must not emit a record",
		}
	}

	c.emitMu.Lock()
	defer c.emitMu.Unlock()

	c.mu.Lock()
	if c.returned {
		c.mu.Unlock()
		return &LockstepViolation{
			Code:   ViolationDuplicateCreditReturn,
			Key:    key,
			Detail: "cannot emit record after credit has returned",
		}
	}
	if c.recordEmitted {
		c.mu.Unlock()
		return &LockstepViolation{
			Code:   ViolationDuplicateRecordEmit,
			Key:    key,
			Detail: "record already emitted for credit",
		}
	}
	nowFn := c.now
	c.mu.Unlock()

	if c.sink == nil {
		return errors.New("perfledger: credit record sink is nil")
	}

	if rec.Schema == "" {
		rec.Schema = Schema
	}
	if rec.RequestID == "" {
		if c.credit.RequestID != "" {
			rec.RequestID = c.credit.RequestID
		} else {
			rec.RequestID = c.credit.ID
		}
	}
	if rec.Locality == "" {
		rec.Locality = c.credit.Locality
	}
	if rec.UnixMS == 0 {
		rec.UnixMS = nowFn().UnixMilli()
	}

	if err := c.sink.EmitCreditRecord(key, rec); err != nil {
		return err
	}

	c.mu.Lock()
	c.recordEmitted = true
	c.mu.Unlock()
	return nil
}

func (c *CreditContext) nowTime() time.Time {
	c.mu.Lock()
	nowFn := c.now
	c.mu.Unlock()
	if nowFn == nil {
		return time.Now()
	}
	return nowFn()
}

func (c *CreditContext) finalizeReturn(cancelled bool) CreditReturn {
	c.mu.Lock()
	defer c.mu.Unlock()
	if cancelled {
		c.cancelled = true
	}
	c.returned = true
	return CreditReturn{
		Key:           c.credit.Key(),
		Cancelled:     c.cancelled,
		NoRequest:     c.credit.NoRequest,
		RecordEmitted: c.recordEmitted,
	}
}

// CreditOutcome captures the complete terminal observation of one credit run
// executed under ExecuteCredit.
type CreditOutcome struct {
	Return             CreditReturn
	Err                error
	Panicked           bool
	FallbackRecordSent bool
	FallbackEmitErr    error
	ReturnErr          error
	Violation          *LockstepViolation
}

// Violations returns every closed lockstep violation observed for this
// execution (empty when valid).
func (o CreditOutcome) Violations() []LockstepViolation {
	var out []LockstepViolation
	if o.FallbackEmitErr != nil {
		out = append(out, LockstepViolation{
			Code:   ViolationFailureEmitFailed,
			Key:    o.Return.Key,
			Detail: "contained fallback failure record emit failed",
			Cause:  o.FallbackEmitErr,
		})
	}
	if v := ValidateCreditReturn(o.Return); v != nil {
		out = append(out, *v)
	} else if o.Violation != nil && o.Violation.Code != ViolationFailureEmitFailed {
		out = append(out, *o.Violation)
	}
	return out
}

// ExecuteCredit runs fn with a per-credit CreditContext and enforces aiperf's
// credit<->record lockstep invariant in defer/finally:
//   - If ctx is already cancelled (context.Canceled) before execution begins,
//     the credit returns immediately with Cancelled=true and no record.
//   - Regardless of whether fn returns normally, returns an error, returns
//     early without emitting a record, or panics, defer ensures any
//     non-cancelled request-bearing credit that has not yet emitted a record
//     emits a fallback failure Record.
//   - The fallback emit is contained (errors and panics are recovered into
//     FallbackEmitErr) so it can never abort CreditReturn delivery.
//   - NoRequest virtual credits and cancelled credits that have not emitted a
//     record do not trigger fallback failure records.
func ExecuteCredit(
	ctx context.Context,
	credit Credit,
	recSink CreditRecordSink,
	retSink CreditReturnSink,
	fn func(context.Context, *CreditContext) error,
) (outcome CreditOutcome) {
	return ExecuteCreditWithClock(ctx, credit, recSink, retSink, time.Now, fn)
}

// ExecuteCreditWithClock is ExecuteCredit with an explicit clock for
// deterministic timestamp and failure-duration measurement.
func ExecuteCreditWithClock(
	ctx context.Context,
	credit Credit,
	recSink CreditRecordSink,
	retSink CreditReturnSink,
	now func() time.Time,
	fn func(context.Context, *CreditContext) error,
) (outcome CreditOutcome) {
	if credit.ID == "" {
		v := &LockstepViolation{
			Code:   ViolationEmptyCreditID,
			Key:    credit.Key(),
			Detail: "credit_id must be non-empty",
		}
		return CreditOutcome{
			Return:    CreditReturn{Key: credit.Key(), NoRequest: credit.NoRequest},
			Err:       v,
			Violation: v,
		}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if now == nil {
		now = time.Now
	}

	cc := NewCreditContext(credit, recSink).WithClock(now)

	// Pre-flight cancellation mirror: a credit cancelled before execution begins
	// returns as Cancelled=true without executing fn or emitting a record.
	if errors.Is(ctx.Err(), context.Canceled) {
		ret := cc.finalizeReturn(true)
		var retErr error
		if retSink != nil {
			retErr = returnCreditContained(retSink, ret)
		}
		return CreditOutcome{
			Return:    ret,
			Err:       ctx.Err(),
			ReturnErr: retErr,
			Violation: ValidateCreditReturn(ret),
		}
	}

	var (
		workerErr error
		panicked  bool
	)

	defer func() {
		if r := recover(); r != nil {
			panicked = true
			workerErr = fmt.Errorf("perfledger: credit %s panicked: %v", credit.ID, r)
		}

		cancelled := cc.Cancelled() ||
			errors.Is(workerErr, context.Canceled) ||
			(!panicked && errors.Is(ctx.Err(), context.Canceled) && (workerErr != nil || !cc.RecordEmitted()))

		var (
			fallbackSent bool
			fallbackErr  error
		)
		if !cancelled && !credit.NoRequest && !cc.RecordEmitted() {
			errClass := failureClassForCreditError(workerErr, panicked)
			ts := cc.nowTime()
			var dur time.Duration
			if !credit.IssuedAt.IsZero() {
				if d := ts.Sub(credit.IssuedAt); d > 0 {
					dur = d
				}
			}
			failRec := NewFailureRecord(ts, credit.Locality, errClass, 0, dur, 0)
			if credit.RequestID != "" {
				failRec.RequestID = credit.RequestID
			} else {
				failRec.RequestID = credit.ID
			}
			if err := emitContained(cc, failRec); err != nil {
				fallbackErr = err
			} else {
				fallbackSent = true
			}
		}

		ret := cc.finalizeReturn(cancelled)
		var retErr error
		if retSink != nil {
			retErr = returnCreditContained(retSink, ret)
		}
		v := ValidateCreditReturn(ret)
		if v == nil && fallbackErr != nil {
			v = &LockstepViolation{
				Code:   ViolationFailureEmitFailed,
				Key:    ret.Key,
				Detail: "contained fallback failure record emit failed",
				Cause:  fallbackErr,
			}
		}
		outcome = CreditOutcome{
			Return:             ret,
			Err:                workerErr,
			Panicked:           panicked,
			FallbackRecordSent: fallbackSent,
			FallbackEmitErr:    fallbackErr,
			ReturnErr:          retErr,
			Violation:          v,
		}
	}()

	if fn == nil {
		if !credit.NoRequest {
			workerErr = errors.New("perfledger: nil credit runner")
		}
		return outcome
	}
	workerErr = fn(ctx, cc)
	return outcome
}

func failureClassForCreditError(err error, panicked bool) string {
	if !panicked && errors.Is(err, context.DeadlineExceeded) {
		return ErrorDeadline
	}
	return ErrorUpstream
}

func emitContained(cc *CreditContext, rec Record) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("perfledger: contained failure-record emit panic: %v", r)
		}
	}()
	return cc.EmitRecord(rec)
}

func returnCreditContained(sink CreditReturnSink, ret CreditReturn) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("perfledger: contained credit return panic: %v", r)
		}
	}()
	return sink.ReturnCredit(ret)
}

// LockstepStats summarizes credit and record counts observed by a LockstepBarrier.
type LockstepStats struct {
	Registered int
	Completed  int
	Cancelled  int
	NoRequest  int
	Records    int
	Violations int
}

type barrierCreditState struct {
	registered bool
	returned   bool
	ret        CreditReturn
	emitting   bool
	hasRecord  bool
	record     Record
}

// LockstepBarrier is the thread-safe completion barrier (analogous to aiperf's
// RecordsManager + CreditCallbackHandler barrier). It pairs emitted Records
// with returned Credits by CreditKey regardless of arrival order, allows Wait
// with no timeout once finalized without hanging when failures emit contained
// records, and fails closed with typed LockstepViolation errors whenever the
// invariant is broken.
type LockstepBarrier struct {
	mu          sync.Mutex
	downstream  CreditRecordSink
	credits     map[CreditKey]*barrierCreditState
	creditOrder []CreditKey
	recordOrder []CreditKey
	violations  []LockstepViolation

	expectedFinalized bool
	hasExplicitCount  bool
	expectedCompleted int

	completedCount int
	cancelledCount int
	noRequestCount int
	recordsCount   int

	notifyCh chan struct{}
}

// NewLockstepBarrier constructs a completion barrier that optionally forwards
// admitted records to downstream (which may be nil).
func NewLockstepBarrier(downstream CreditRecordSink) *LockstepBarrier {
	return &LockstepBarrier{
		downstream: downstream,
		credits:    make(map[CreditKey]*barrierCreditState),
		notifyCh:   make(chan struct{}),
	}
}

func (b *LockstepBarrier) signalLocked() {
	close(b.notifyCh)
	b.notifyCh = make(chan struct{})
}

func (b *LockstepBarrier) ensureCreditLocked(key CreditKey) *barrierCreditState {
	st, ok := b.credits[key]
	if !ok {
		st = &barrierCreditState{}
		b.credits[key] = st
		b.creditOrder = append(b.creditOrder, key)
	}
	return st
}

// RegisterCredit optionally pre-registers an issued credit so FinalizeRegistered
// and Verify can detect credits that never returned.
func (b *LockstepBarrier) RegisterCredit(credit Credit) error {
	key := credit.Key()
	b.mu.Lock()
	defer b.mu.Unlock()
	if key.CreditID == "" {
		v := LockstepViolation{
			Code:   ViolationEmptyCreditID,
			Key:    key,
			Detail: "cannot register credit with empty credit_id",
		}
		b.violations = append(b.violations, v)
		b.signalLocked()
		return &v
	}
	st := b.ensureCreditLocked(key)
	st.registered = true
	return nil
}

// EmitCreditRecord implements CreditRecordSink, tracking 1:1 credit->record
// emission and optionally forwarding to the downstream sink before committing.
func (b *LockstepBarrier) EmitCreditRecord(key CreditKey, rec Record) error {
	b.mu.Lock()
	if key.CreditID == "" {
		v := LockstepViolation{
			Code:   ViolationEmptyCreditID,
			Key:    key,
			Detail: "record emitted with empty credit_id",
		}
		b.violations = append(b.violations, v)
		b.signalLocked()
		b.mu.Unlock()
		return &v
	}
	st := b.ensureCreditLocked(key)
	if st.returned && st.ret.NoRequest {
		v := LockstepViolation{
			Code:   ViolationNoRequestEmittedRecord,
			Key:    key,
			Detail: "record emitted for no_request virtual credit",
		}
		b.violations = append(b.violations, v)
		b.signalLocked()
		b.mu.Unlock()
		return &v
	}
	if st.emitting || st.hasRecord {
		v := LockstepViolation{
			Code:   ViolationDuplicateRecordEmit,
			Key:    key,
			Detail: "duplicate record emitted for credit",
		}
		b.violations = append(b.violations, v)
		b.signalLocked()
		b.mu.Unlock()
		return &v
	}
	st.emitting = true
	downstream := b.downstream
	b.mu.Unlock()

	if downstream != nil {
		var completed bool
		defer func() {
			if !completed {
				b.mu.Lock()
				st.emitting = false
				b.signalLocked()
				b.mu.Unlock()
			}
		}()
		if err := downstream.EmitCreditRecord(key, rec); err != nil {
			b.mu.Lock()
			st.emitting = false
			b.signalLocked()
			b.mu.Unlock()
			completed = true
			return err
		}
		completed = true
	}

	b.mu.Lock()
	st.emitting = false
	if st.hasRecord {
		v := LockstepViolation{
			Code:   ViolationDuplicateRecordEmit,
			Key:    key,
			Detail: "duplicate record emitted for credit",
		}
		b.violations = append(b.violations, v)
		b.signalLocked()
		b.mu.Unlock()
		return &v
	}
	st.hasRecord = true
	st.record = rec
	b.recordsCount++
	b.recordOrder = append(b.recordOrder, key)
	b.signalLocked()
	b.mu.Unlock()
	return nil
}

// ReturnCredit implements CreditReturnSink, validating the return against the
// closed lockstep rules and advancing completion barrier state.
func (b *LockstepBarrier) ReturnCredit(ret CreditReturn) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	v := ValidateCreditReturn(ret)
	if ret.Key.CreditID == "" {
		if v == nil {
			v = &LockstepViolation{
				Code:   ViolationEmptyCreditID,
				Key:    ret.Key,
				Detail: "credit_id must be non-empty",
			}
		}
		b.violations = append(b.violations, *v)
		b.signalLocked()
		return v
	}

	st := b.ensureCreditLocked(ret.Key)
	if st.returned {
		dup := LockstepViolation{
			Code:   ViolationDuplicateCreditReturn,
			Key:    ret.Key,
			Detail: "credit returned more than once",
		}
		b.violations = append(b.violations, dup)
		b.signalLocked()
		return &dup
	}
	st.returned = true
	st.ret = ret

	if v != nil {
		b.violations = append(b.violations, *v)
		b.signalLocked()
		return v
	}
	if ret.NoRequest && st.hasRecord {
		nr := LockstepViolation{
			Code:   ViolationNoRequestEmittedRecord,
			Key:    ret.Key,
			Detail: "no_request virtual credit already had an emitted record",
		}
		b.violations = append(b.violations, nr)
		b.signalLocked()
		return &nr
	}

	switch {
	case ret.NoRequest:
		b.noRequestCount++
	case ret.Cancelled:
		b.cancelledCount++
	default:
		b.completedCount++
	}

	b.signalLocked()
	return nil
}

// Finalize declares the ground-truth count of completed (non-cancelled,
// non-NoRequest) credits expected before the completion barrier can unblock.
// Mirroring aiperf's RecordsManager, Wait blocks until at least expectedCompleted
// completed credits have returned AND every completed credit carries its record.
func (b *LockstepBarrier) Finalize(expectedCompleted int) {
	if expectedCompleted < 0 {
		expectedCompleted = 0
	}
	b.mu.Lock()
	b.expectedFinalized = true
	b.hasExplicitCount = true
	b.expectedCompleted = expectedCompleted
	b.signalLocked()
	b.mu.Unlock()
}

// FinalizeRegistered marks dispatch complete without an explicit numeric count,
// unblocking Wait once every registered credit has returned and every completed
// credit has received its emitted record.
func (b *LockstepBarrier) FinalizeRegistered() {
	b.mu.Lock()
	b.expectedFinalized = true
	b.hasExplicitCount = false
	b.signalLocked()
	b.mu.Unlock()
}

func (b *LockstepBarrier) isSatisfiedLocked() bool {
	if !b.expectedFinalized {
		return false
	}
	if b.hasExplicitCount && b.completedCount < b.expectedCompleted {
		return false
	}
	for _, key := range b.creditOrder {
		st := b.credits[key]
		if st.emitting {
			return false
		}
		if st.registered && !st.returned {
			return false
		}
		if st.returned && st.ret.RequiresRecord() && !st.hasRecord {
			return false
		}
		if st.hasRecord && !st.returned {
			return false
		}
	}
	return true
}

func (b *LockstepBarrier) auditViolationsLocked() []LockstepViolation {
	out := append([]LockstepViolation(nil), b.violations...)
	if b.hasExplicitCount && b.completedCount < b.expectedCompleted {
		out = append(out, LockstepViolation{
			Code:   ViolationIncompleteCredits,
			Detail: fmt.Sprintf("completed=%d expected=%d", b.completedCount, b.expectedCompleted),
		})
	}
	for _, key := range b.creditOrder {
		st := b.credits[key]
		if st.registered && !st.returned {
			out = append(out, LockstepViolation{
				Code:   ViolationIncompleteCredits,
				Key:    key,
				Detail: "registered credit did not return",
			})
		}
		if st.returned && st.ret.RequiresRecord() && !st.hasRecord && st.ret.RecordEmitted {
			out = append(out, LockstepViolation{
				Code:   ViolationMissingRecord,
				Key:    key,
				Detail: "credit return claimed record_emitted=true but barrier received no record",
			})
		}
		if st.hasRecord && !st.returned {
			out = append(out, LockstepViolation{
				Code:   ViolationOrphanRecord,
				Key:    key,
				Detail: "record emitted without corresponding credit return",
			})
		}
	}
	return out
}

// Wait blocks until the barrier is finalized and every completed credit is
// paired with an emitted record, failing closed immediately with *LockstepError
// if any lockstep violation is observed. Passing context.Background() waits with
// no timeout, mirroring aiperf's RecordsManager completion barrier.
func (b *LockstepBarrier) Wait(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		b.mu.Lock()
		if len(b.violations) > 0 {
			err := &LockstepError{Violations: append([]LockstepViolation(nil), b.violations...)}
			b.mu.Unlock()
			return err
		}
		if b.isSatisfiedLocked() {
			b.mu.Unlock()
			return nil
		}
		ch := b.notifyCh
		b.mu.Unlock()

		select {
		case <-ch:
		case <-ctx.Done():
			b.mu.Lock()
			if len(b.violations) > 0 {
				err := &LockstepError{Violations: append([]LockstepViolation(nil), b.violations...)}
				b.mu.Unlock()
				return err
			}
			if b.isSatisfiedLocked() {
				b.mu.Unlock()
				return nil
			}
			audit := b.auditViolationsLocked()
			b.mu.Unlock()
			if len(audit) > 0 {
				return fmt.Errorf("%w: %w", ctx.Err(), &LockstepError{Violations: audit})
			}
			return ctx.Err()
		}
	}
}

// Verify runs a non-blocking closed lockstep audit over the barrier's current
// state and returns *LockstepError if any invariant was violated or any credit
// remains incomplete/unpaired.
func (b *LockstepBarrier) Verify() error {
	b.mu.Lock()
	audit := b.auditViolationsLocked()
	b.mu.Unlock()
	if len(audit) == 0 {
		return nil
	}
	return &LockstepError{Violations: audit}
}

// Records returns all emitted records observed by the barrier in commit order.
func (b *LockstepBarrier) Records() []Record {
	b.mu.Lock()
	defer b.mu.Unlock()
	out := make([]Record, 0, len(b.recordOrder))
	for _, key := range b.recordOrder {
		if st := b.credits[key]; st != nil && st.hasRecord {
			out = append(out, st.record)
		}
	}
	return out
}

// Violations returns a snapshot of inline lockstep violations recorded so far.
func (b *LockstepBarrier) Violations() []LockstepViolation {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]LockstepViolation(nil), b.violations...)
}

// Stats returns a snapshot of credit and record counts tracked by the barrier.
func (b *LockstepBarrier) Stats() LockstepStats {
	b.mu.Lock()
	defer b.mu.Unlock()
	registered := 0
	for _, st := range b.credits {
		if st.registered {
			registered++
		}
	}
	return LockstepStats{
		Registered: registered,
		Completed:  b.completedCount,
		Cancelled:  b.cancelledCount,
		NoRequest:  b.noRequestCount,
		Records:    b.recordsCount,
		Violations: len(b.violations),
	}
}
