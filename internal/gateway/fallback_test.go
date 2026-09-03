package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type customHeaderFuncError struct {
	msg string
	h   http.Header
}

func (e *customHeaderFuncError) Error() string {
	return e.msg
}

func (e *customHeaderFuncError) Header() http.Header {
	return e.h
}

type customHeadersFuncError struct {
	msg string
	h   http.Header
}

func (e *customHeadersFuncError) Error() string {
	return e.msg
}

func (e *customHeadersFuncError) Headers() http.Header {
	return e.h
}

type customRespHeaderFuncError struct {
	msg string
	h   http.Header
}

func (e *customRespHeaderFuncError) Error() string {
	return e.msg
}

func (e *customRespHeaderFuncError) ResponseHeader() http.Header {
	return e.h
}

type customStructHeaderError struct {
	Header http.Header
	Msg    string
}

func (e *customStructHeaderError) Error() string {
	return e.Msg
}

type customHaltExceptionBoolError struct {
	msg       string
	exception bool
}

func (e *customHaltExceptionBoolError) Error() string {
	return e.msg
}

func (e *customHaltExceptionBoolError) IsHaltException() bool {
	return e.exception
}

func TestHaltExceptionHeader(t *testing.T) {
	cases := []struct {
		name string
		h    http.Header
		want bool
	}{
		{
			name: "nil header",
			h:    nil,
			want: false,
		},
		{
			name: "empty header",
			h:    http.Header{},
			want: false,
		},
		{
			name: "fak exception lowercase key",
			h:    http.Header{"x-fak-gateway-exception": []string{"true"}},
			want: true,
		},
		{
			name: "fak exception canonical key",
			h: func() http.Header {
				h := make(http.Header)
				h.Set("x-fak-gateway-exception", "true")
				return h
			}(),
			want: true,
		},
		{
			name: "fak exception uppercase value",
			h:    http.Header{"x-fak-gateway-exception": []string{"TRUE"}},
			want: true,
		},
		{
			name: "fak exception value with spaces",
			h:    http.Header{"x-fak-gateway-exception": []string{"  true  "}},
			want: true,
		},
		{
			name: "portkey exception lowercase key",
			h:    http.Header{"x-portkey-gateway-exception": []string{"true"}},
			want: true,
		},
		{
			name: "portkey exception canonical key",
			h: func() http.Header {
				h := make(http.Header)
				h.Set("x-portkey-gateway-exception", "true")
				return h
			}(),
			want: true,
		},
		{
			name: "portkey exception mixed case value",
			h:    http.Header{"X-Portkey-Gateway-Exception": []string{"True"}},
			want: true,
		},
		{
			name: "fak exception value false",
			h:    http.Header{"x-fak-gateway-exception": []string{"false"}},
			want: false,
		},
		{
			name: "fak exception value 0",
			h:    http.Header{"x-fak-gateway-exception": []string{"0"}},
			want: false,
		},
		{
			name: "unrelated headers",
			h: http.Header{
				"Content-Type": []string{"application/json"},
				"Retry-After":  []string{"60"},
			},
			want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := IsHaltExceptionHeader(tc.h)
			if got != tc.want {
				t.Errorf("IsHaltExceptionHeader() = %v, want %v", got, tc.want)
			}
		})
	}

	t.Run("MarkHaltException sets header", func(t *testing.T) {
		h := make(http.Header)
		MarkHaltException(h)
		if !IsHaltExceptionHeader(h) {
			t.Fatal("MarkHaltException did not make IsHaltExceptionHeader true")
		}
		if h.Get(HeaderHaltException) != "true" {
			t.Fatalf("h.Get(%q) = %q, want %q", HeaderHaltException, h.Get(HeaderHaltException), "true")
		}

		// nil safety
		MarkHaltException(nil)
	})
}

func TestHaltExceptionErrorClassification(t *testing.T) {
	fakHeader := make(http.Header)
	MarkHaltException(fakHeader)

	portkeyHeader := http.Header{"x-portkey-gateway-exception": []string{"true"}}
	plainHeader := http.Header{"Content-Type": []string{"text/plain"}}

	cases := []struct {
		name string
		err  error
		want bool
	}{
		{
			name: "nil error",
			err:  nil,
			want: false,
		},
		{
			name: "standard error",
			err:  errors.New("connection reset by peer"),
			want: false,
		},
		{
			name: "sentinel ErrHaltException",
			err:  ErrHaltException,
			want: true,
		},
		{
			name: "wrapped ErrHaltException",
			err:  fmt.Errorf("failed upstream call: %w", ErrHaltException),
			want: true,
		},
		{
			name: "deeply wrapped ErrHaltException",
			err:  fmt.Errorf("layer 1: %w", fmt.Errorf("layer 2: %w", ErrHaltException)),
			want: true,
		},
		{
			name: "UpstreamError with fak header",
			err: &UpstreamError{
				Status: http.StatusUnauthorized,
				Header: fakHeader,
			},
			want: true,
		},
		{
			name: "UpstreamError with portkey header",
			err: &UpstreamError{
				Status: http.StatusForbidden,
				Header: portkeyHeader,
			},
			want: true,
		},
		{
			name: "UpstreamError with plain header",
			err: &UpstreamError{
				Status: http.StatusInternalServerError,
				Header: plainHeader,
			},
			want: false,
		},
		{
			name: "UpstreamError wrapping agent.UpstreamStatusError with fak header",
			err: &UpstreamError{
				Err:    &agent.UpstreamStatusError{Status: http.StatusUnauthorized, Body: "invalid api key"},
				Header: fakHeader,
			},
			want: true,
		},
		{
			name: "UpstreamError wrapping agent.UpstreamStatusError without header",
			err: &UpstreamError{
				Err:    &agent.UpstreamStatusError{Status: http.StatusInternalServerError, Body: "server error"},
				Header: plainHeader,
			},
			want: false,
		},
		{
			name: "agent.UpstreamStatusError without header wrapper",
			err:  &agent.UpstreamStatusError{Status: http.StatusInternalServerError, Body: "server error"},
			want: false,
		},
		{
			name: "custom error with Header() method",
			err: &customHeaderFuncError{
				msg: "content policy violation",
				h:   fakHeader,
			},
			want: true,
		},
		{
			name: "custom error with Headers() method",
			err: &customHeadersFuncError{
				msg: "quota exhausted",
				h:   portkeyHeader,
			},
			want: true,
		},
		{
			name: "custom error with ResponseHeader() method",
			err: &customRespHeaderFuncError{
				msg: "unrecoverable auth error",
				h:   fakHeader,
			},
			want: true,
		},
		{
			name: "custom error struct with Header field",
			err: &customStructHeaderError{
				Header: fakHeader,
				Msg:    "invalid model permissions",
			},
			want: true,
		},
		{
			name: "custom error struct with plain Header field",
			err: &customStructHeaderError{
				Header: plainHeader,
				Msg:    "transient 503",
			},
			want: false,
		},
		{
			name: "custom error with IsHaltException() bool true",
			err: &customHaltExceptionBoolError{
				msg:       "explicit unrecoverable exception",
				exception: true,
			},
			want: true,
		},
		{
			name: "custom error with IsHaltException() bool false",
			err: &customHaltExceptionBoolError{
				msg:       "recoverable error",
				exception: false,
			},
			want: false,
		},
		{
			name: "NewHaltExceptionError helper with nil args",
			err:  NewHaltExceptionError(nil, nil),
			want: true,
		},
		{
			name: "NewHaltExceptionError helper wrapping existing error",
			err:  NewHaltExceptionError(errors.New("upstream auth failure"), nil),
			want: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := IsHaltException(tc.err)
			if got != tc.want {
				t.Errorf("IsHaltException() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestHaltExceptionFallbackChain(t *testing.T) {
	ctx := context.Background()

	t.Run("normal 5xx continues fallback to secondary", func(t *testing.T) {
		var primaryCalls atomic.Int32
		var fallback1Calls atomic.Int32
		var fallback2Calls atomic.Int32

		primary := func(_ context.Context) (string, error) {
			primaryCalls.Add(1)
			return "", &UpstreamError{
				Status: http.StatusInternalServerError,
				Header: http.Header{"Content-Type": []string{"application/json"}},
			}
		}
		fallback1 := func(_ context.Context) (string, error) {
			fallback1Calls.Add(1)
			return "", &UpstreamError{
				Status: http.StatusBadGateway,
				Header: http.Header{"Content-Type": []string{"application/json"}},
			}
		}
		fallback2 := func(_ context.Context) (string, error) {
			fallback2Calls.Add(1)
			return "success-replica-2", nil
		}

		executor := NewFallbackExecutor(primary, fallback1, fallback2)
		res, err := executor.Execute(ctx)
		if err != nil {
			t.Fatalf("expected successful fallback, got err: %v", err)
		}
		if res != "success-replica-2" {
			t.Fatalf("got res %q, want %q", res, "success-replica-2")
		}
		if primaryCalls.Load() != 1 || fallback1Calls.Load() != 1 || fallback2Calls.Load() != 1 {
			t.Fatalf("call counts: primary=%d fb1=%d fb2=%d, want 1/1/1",
				primaryCalls.Load(), fallback1Calls.Load(), fallback2Calls.Load())
		}
	})

	t.Run("auth failure with fak gateway exception header stops fallback immediately", func(t *testing.T) {
		var primaryCalls atomic.Int32
		var fallback1Calls atomic.Int32
		var fallback2Calls atomic.Int32

		h := make(http.Header)
		MarkHaltException(h)

		primary := func(_ context.Context) (string, error) {
			primaryCalls.Add(1)
			return "", &UpstreamError{
				Err:    errors.New("unauthorized: bad api key"),
				Status: http.StatusUnauthorized,
				Header: h,
			}
		}
		fallback1 := func(_ context.Context) (string, error) {
			fallback1Calls.Add(1)
			return "fb1-never-reached", nil
		}
		fallback2 := func(_ context.Context) (string, error) {
			fallback2Calls.Add(1)
			return "fb2-never-reached", nil
		}

		executor := NewFallbackExecutor(primary, fallback1, fallback2)
		_, err := executor.Execute(ctx)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !IsHaltException(err) {
			t.Fatalf("expected IsHaltException(err) == true, got %v", err)
		}
		if primaryCalls.Load() != 1 {
			t.Fatalf("primary calls = %d, want 1", primaryCalls.Load())
		}
		if fallback1Calls.Load() != 0 || fallback2Calls.Load() != 0 {
			t.Fatalf("fallbacks must not be called on gateway exception: fb1=%d fb2=%d",
				fallback1Calls.Load(), fallback2Calls.Load())
		}
	})

	t.Run("auth failure with portkey gateway exception header stops fallback immediately", func(t *testing.T) {
		var primaryCalls atomic.Int32
		var fallbackCalls atomic.Int32

		primary := func(_ context.Context) (string, error) {
			primaryCalls.Add(1)
			return "", &UpstreamError{
				Err:    errors.New("forbidden: account suspended"),
				Status: http.StatusForbidden,
				Header: http.Header{"x-portkey-gateway-exception": []string{"true"}},
			}
		}
		fallback := func(_ context.Context) (string, error) {
			fallbackCalls.Add(1)
			return "fb-never-reached", nil
		}

		executor := NewFallbackExecutor(primary, fallback)
		_, err := executor.Execute(ctx)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !IsHaltException(err) {
			t.Fatalf("expected gateway exception, got %v", err)
		}
		if primaryCalls.Load() != 1 || fallbackCalls.Load() != 0 {
			t.Fatalf("calls: primary=%d fallback=%d, want 1/0", primaryCalls.Load(), fallbackCalls.Load())
		}
	})

	t.Run("mid-chain gateway exception stops subsequent fallbacks", func(t *testing.T) {
		var primaryCalls atomic.Int32
		var fb1Calls atomic.Int32
		var fb2Calls atomic.Int32

		primary := func(_ context.Context) (string, error) {
			primaryCalls.Add(1)
			return "", errors.New("transient upstream 503")
		}
		fb1 := func(_ context.Context) (string, error) {
			fb1Calls.Add(1)
			return "", NewHaltExceptionError(errors.New("quota exceeded"), nil)
		}
		fb2 := func(_ context.Context) (string, error) {
			fb2Calls.Add(1)
			return "fb2-never-reached", nil
		}

		executor := NewFallbackExecutor(primary, fb1, fb2)
		_, err := executor.Execute(ctx)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !IsHaltException(err) {
			t.Fatalf("expected gateway exception from fb1, got %v", err)
		}
		if primaryCalls.Load() != 1 || fb1Calls.Load() != 1 || fb2Calls.Load() != 0 {
			t.Fatalf("calls: primary=%d fb1=%d fb2=%d, want 1/1/0",
				primaryCalls.Load(), fb1Calls.Load(), fb2Calls.Load())
		}
	})

	t.Run("response object carrying gateway exception header stops fallback", func(t *testing.T) {
		var primaryCalls atomic.Int32
		var fbCalls atomic.Int32

		primary := func(_ context.Context) (*http.Response, error) {
			primaryCalls.Add(1)
			resp := &http.Response{
				StatusCode: http.StatusBadRequest,
				Header:     http.Header{"x-fak-gateway-exception": []string{"true"}},
			}
			return resp, nil
		}
		fallback := func(_ context.Context) (*http.Response, error) {
			fbCalls.Add(1)
			return &http.Response{StatusCode: http.StatusOK}, nil
		}

		executor := NewFallbackExecutor(primary, fallback)
		_, err := executor.Execute(ctx)
		if err == nil {
			t.Fatal("expected ErrHaltException from response header, got nil")
		}
		if !errors.Is(err, ErrHaltException) {
			t.Fatalf("expected ErrHaltException, got %v", err)
		}
		if primaryCalls.Load() != 1 || fbCalls.Load() != 0 {
			t.Fatalf("calls: primary=%d fb=%d, want 1/0", primaryCalls.Load(), fbCalls.Load())
		}
	})

	t.Run("all runners fail with recoverable errors returns last error", func(t *testing.T) {
		primary := func(_ context.Context) (string, error) {
			return "", errors.New("err-primary")
		}
		fb := func(_ context.Context) (string, error) {
			return "", errors.New("err-fallback")
		}

		executor := NewFallbackExecutor(primary, fb)
		_, err := executor.Execute(ctx)
		if err == nil || err.Error() != "err-fallback" {
			t.Fatalf("expected err-fallback, got %v", err)
		}
	})

	t.Run("missing primary returns error", func(t *testing.T) {
		executor := &FallbackExecutor[string]{}
		_, err := executor.Execute(ctx)
		if err == nil {
			t.Fatal("expected error on missing primary")
		}
	})

	t.Run("context cancellation stops chain immediately", func(t *testing.T) {
		canceledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		var fbCalls atomic.Int32
		primary := func(c context.Context) (string, error) {
			return "", errors.New("normal error")
		}
		fb := func(c context.Context) (string, error) {
			fbCalls.Add(1)
			return "ok", nil
		}

		executor := NewFallbackExecutor(primary, fb)
		_, err := executor.Execute(canceledCtx)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
		if fbCalls.Load() != 0 {
			t.Fatalf("fallback was called after cancel: %d", fbCalls.Load())
		}
	})
}

func TestHaltExceptionReplicaFallbackAllowed(t *testing.T) {
	ctx := context.Background()

	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	deadlineCtx, cancelDeadline := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancelDeadline()
	time.Sleep(2 * time.Millisecond)

	h := make(http.Header)
	MarkHaltException(h)

	cases := []struct {
		name string
		ctx  context.Context
		err  error
		want bool
	}{
		{
			name: "nil error",
			ctx:  ctx,
			err:  nil,
			want: false,
		},
		{
			name: "canceled context",
			ctx:  canceledCtx,
			err:  errors.New("random failure"),
			want: false,
		},
		{
			name: "deadline exceeded context",
			ctx:  deadlineCtx,
			err:  errors.New("random failure"),
			want: false,
		},
		{
			name: "context.Canceled error",
			ctx:  ctx,
			err:  context.Canceled,
			want: false,
		},
		{
			name: "context.DeadlineExceeded error",
			ctx:  ctx,
			err:  context.DeadlineExceeded,
			want: false,
		},
		{
			name: "sentinel ErrHaltException returns false",
			ctx:  ctx,
			err:  ErrHaltException,
			want: false,
		},
		{
			name: "wrapped ErrHaltException returns false",
			ctx:  ctx,
			err:  fmt.Errorf("upstream rejected: %w", ErrHaltException),
			want: false,
		},
		{
			name: "UpstreamError with exception header returns false",
			ctx:  ctx,
			err:  &UpstreamError{Status: 401, Header: h},
			want: false,
		},
		{
			name: "NewHaltExceptionError returns false",
			ctx:  ctx,
			err:  NewHaltExceptionError(errors.New("auth denied"), nil),
			want: false,
		},
		{
			name: "transient network error returns true",
			ctx:  ctx,
			err:  errors.New("transient connection reset"),
			want: true,
		},
		{
			name: "agent.UpstreamStatusError 503 without exception header returns true",
			ctx:  ctx,
			err:  &agent.UpstreamStatusError{Status: http.StatusServiceUnavailable, Body: "overloaded"},
			want: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := replicaFallbackAllowed(tc.ctx, tc.err)
			if got != tc.want {
				t.Errorf("replicaFallbackAllowed() = %v, want %v", got, tc.want)
			}
		})
	}
}

type failoverTestPlanner struct {
	name  string
	err   error
	calls atomic.Int32
	comp  *agent.Completion
}

func (p *failoverTestPlanner) Model() string { return p.name }

func (p *failoverTestPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	if p.err != nil {
		return nil, p.err
	}
	return p.comp, nil
}

func TestHaltExceptionReplicaRouterCompleteStopsFallback(t *testing.T) {
	ctx := context.Background()

	t.Run("gateway exception stops replica router fallback", func(t *testing.T) {
		m := nativeReservationFleet(t, "w1", "w2")

		excErr := NewHaltExceptionError(errors.New("401 unauthorized"), nil)
		first := &reservationTestPlanner{worker: "w1", membership: m, err: excErr, wantBooked: 2}
		second := &reservationTestPlanner{worker: "w2", membership: m, wantBooked: 2}

		r := nativeReservationRouter(t, m, namedPickPolicy("w1"), first, second)

		_, err := r.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "test"}}, nil)
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !IsHaltException(err) {
			t.Fatalf("expected gateway exception, got %v", err)
		}
		if first.calls.Load() != 1 {
			t.Fatalf("first replica calls = %d, want 1", first.calls.Load())
		}
		if second.calls.Load() != 0 {
			t.Fatalf("second replica was called (%d times), but fallback should have stopped", second.calls.Load())
		}
	})

	t.Run("transient error allows replica router fallback", func(t *testing.T) {
		m := nativeReservationFleet(t, "w1", "w2")

		transientErr := errors.New("temporary 500 failure")
		first := &reservationTestPlanner{worker: "w1", membership: m, err: transientErr, wantBooked: 2}
		second := &reservationTestPlanner{worker: "w2", membership: m, wantBooked: 2}

		r := nativeReservationRouter(t, m, namedPickPolicy("w1"), first, second)

		comp, err := r.Complete(ctx, []agent.Message{{Role: agent.RoleUser, Content: "test"}}, nil)
		if err != nil {
			t.Fatalf("expected fallback to succeed, got %v", err)
		}
		if comp.Message.Content != "w2" {
			t.Fatalf("expected completion from w2, got %q", comp.Message.Content)
		}
		if first.calls.Load() != 1 {
			t.Fatalf("first replica calls = %d, want 1", first.calls.Load())
		}
		if second.calls.Load() != 1 {
			t.Fatalf("second replica calls = %d, want 1", second.calls.Load())
		}
	})
}
