package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
)

const (
	// HeaderHaltException is the header signaling an unrecoverable error that
	// should immediately abort fallback chains.
	HeaderHaltException = "x-fak-gateway-exception"

	// HeaderPortkeyHaltException is the Portkey-compatible gateway exception header.
	HeaderPortkeyHaltException = "x-portkey-gateway-exception"
)

// ErrHaltException is the sentinel error returned or wrapped when an unrecoverable
// gateway exception occurs.
var ErrHaltException = errors.New("gateway exception: unrecoverable error")

// UpstreamError wraps an underlying error with HTTP response headers,
// enabling gateway exception inspection and header propagation.
type UpstreamError struct {
	Err    error
	Status int
	Header http.Header
}

func (e *UpstreamError) Error() string {
	if e.Err != nil {
		return e.Err.Error()
	}
	if e.Status != 0 {
		return fmt.Sprintf("upstream error: HTTP %d", e.Status)
	}
	return "upstream error"
}

func (e *UpstreamError) Unwrap() error {
	return e.Err
}

func (e *UpstreamError) ResponseHeader() http.Header {
	if e == nil {
		return nil
	}
	return e.Header
}

// NewHaltExceptionError creates an UpstreamError marked with the gateway exception header.
func NewHaltExceptionError(err error, h http.Header) error {
	if h == nil {
		h = make(http.Header)
	}
	MarkHaltException(h)
	if err == nil {
		err = ErrHaltException
	}
	return &UpstreamError{
		Err:    err,
		Header: h,
	}
}

// IsHaltExceptionHeader reports whether h contains an unrecoverable gateway
// exception signal (x-fak-gateway-exception: true or x-portkey-gateway-exception: true).
func IsHaltExceptionHeader(h http.Header) bool {
	if h == nil {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(h.Get(HeaderHaltException)), "true") {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(h.Get(HeaderPortkeyHaltException)), "true") {
		return true
	}
	for k, vs := range h {
		if strings.EqualFold(k, HeaderHaltException) || strings.EqualFold(k, HeaderPortkeyHaltException) {
			for _, v := range vs {
				if strings.EqualFold(strings.TrimSpace(v), "true") {
					return true
				}
			}
		}
	}
	return false
}

// MarkHaltException sets the gateway exception header on h to "true".
func MarkHaltException(h http.Header) {
	if h != nil {
		h.Set(HeaderHaltException, "true")
	}
}

// extractHeader attempts to extract an http.Header from any value or error,
// checking common getter interfaces and struct fields.
func extractHeader(v any) (http.Header, bool) {
	if v == nil {
		return nil, false
	}
	type headerGetter interface {
		Header() http.Header
	}
	if hg, ok := v.(headerGetter); ok && hg != nil {
		return hg.Header(), true
	}
	type headersGetter interface {
		Headers() http.Header
	}
	if hg, ok := v.(headersGetter); ok && hg != nil {
		return hg.Headers(), true
	}
	type respHeaderGetter interface {
		ResponseHeader() http.Header
	}
	if hg, ok := v.(respHeaderGetter); ok && hg != nil {
		return hg.ResponseHeader(), true
	}
	type respHeadersGetter interface {
		ResponseHeaders() http.Header
	}
	if hg, ok := v.(respHeadersGetter); ok && hg != nil {
		return hg.ResponseHeaders(), true
	}

	val := reflect.ValueOf(v)
	if val.Kind() == reflect.Pointer {
		if val.IsNil() {
			return nil, false
		}
		val = val.Elem()
	}
	if val.Kind() == reflect.Struct {
		for _, name := range []string{"Header", "Headers", "ResponseHeader", "ResponseHeaders"} {
			f := val.FieldByName(name)
			if f.IsValid() && f.CanInterface() {
				if h, ok := f.Interface().(http.Header); ok {
					return h, true
				}
			}
		}
	}
	return nil, false
}

// walkError traverses an error and its unwrapped causes, invoking fn on each.
func walkError(err error, fn func(error) bool) bool {
	if err == nil {
		return false
	}
	if fn(err) {
		return true
	}
	switch u := err.(type) {
	case interface{ Unwrap() error }:
		return walkError(u.Unwrap(), fn)
	case interface{ Unwrap() []error }:
		for _, child := range u.Unwrap() {
			if walkError(child, fn) {
				return true
			}
		}
	}
	return false
}

// IsHaltException reports whether err represents an unrecoverable gateway exception,
// checking errors.Is(err, ErrHaltException), custom error markers, or if err unwraps
// to an error carrying the gateway exception header.
func IsHaltException(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, ErrHaltException) {
		return true
	}
	return walkError(err, func(e error) bool {
		if errors.Is(e, ErrHaltException) {
			return true
		}
		if gec, ok := e.(interface{ IsHaltException() bool }); ok && gec != nil {
			if gec.IsHaltException() {
				return true
			}
		}
		if gec, ok := e.(interface{ HaltException() bool }); ok && gec != nil {
			if gec.HaltException() {
				return true
			}
		}
		if h, ok := extractHeader(e); ok && h != nil {
			if IsHaltExceptionHeader(h) {
				return true
			}
		}
		return false
	})
}

// FallbackRunner executes an operation that produces a value of type T or an error.
type FallbackRunner[T any] = func(ctx context.Context) (T, error)

// FallbackExecutor executes a primary runner with fallback runners. If any runner
// fails with an unrecoverable gateway exception or returns a response carrying
// the gateway exception header, fallback is halted immediately.
type FallbackExecutor[T any] struct {
	Primary   FallbackRunner[T]
	Fallbacks []FallbackRunner[T]
}

// NewFallbackExecutor creates a FallbackExecutor with the given primary and fallback runners.
func NewFallbackExecutor[T any](primary FallbackRunner[T], fallbacks ...FallbackRunner[T]) *FallbackExecutor[T] {
	return &FallbackExecutor[T]{
		Primary:   primary,
		Fallbacks: fallbacks,
	}
}

// AddFallback appends a fallback runner.
func (fe *FallbackExecutor[T]) AddFallback(fallback FallbackRunner[T]) *FallbackExecutor[T] {
	if fallback != nil {
		fe.Fallbacks = append(fe.Fallbacks, fallback)
	}
	return fe
}

// Execute runs the primary runner, falling back to sequential fallback runners on
// recoverable errors. If any runner encounters an unrecoverable gateway exception
// (via IsHaltException or gateway exception response header), fallback is aborted
// immediately and the error is returned.
func (fe *FallbackExecutor[T]) Execute(ctx context.Context) (T, error) {
	var zero T
	if fe == nil || fe.Primary == nil {
		return zero, errors.New("fallback executor: missing primary runner")
	}
	runners := make([]FallbackRunner[T], 0, 1+len(fe.Fallbacks))
	runners = append(runners, fe.Primary)
	runners = append(runners, fe.Fallbacks...)
	return ExecuteFallbackChain(ctx, runners...)
}

// ExecuteWithRunners executes the given primary and fallback runners through the fallback chain.
func (fe *FallbackExecutor[T]) ExecuteWithRunners(ctx context.Context, primary FallbackRunner[T], fallbacks ...FallbackRunner[T]) (T, error) {
	runners := make([]FallbackRunner[T], 0, 1+len(fallbacks))
	if primary != nil {
		runners = append(runners, primary)
	}
	runners = append(runners, fallbacks...)
	return ExecuteFallbackChain(ctx, runners...)
}

// ExecuteFallbackChain executes runners in sequence until one succeeds.
// If any runner returns an error or response with a gateway exception, the chain
// stops immediately and returns the error without trying subsequent runners.
func ExecuteFallbackChain[T any](ctx context.Context, runners ...FallbackRunner[T]) (T, error) {
	var zero T
	if len(runners) == 0 {
		return zero, errors.New("fallback executor: no runners provided")
	}

	var lastErr error
	for _, runner := range runners {
		if runner == nil {
			continue
		}
		if ctx != nil && ctx.Err() != nil {
			return zero, ctx.Err()
		}

		res, err := runner(ctx)
		if err != nil {
			if IsHaltException(err) {
				return zero, err
			}
			if h, ok := extractHeader(res); ok && IsHaltExceptionHeader(h) {
				return zero, fmt.Errorf("%w: %w", ErrHaltException, err)
			}
			lastErr = err
			continue
		}

		// Check if response value itself signals a gateway exception header.
		if h, ok := extractHeader(res); ok && IsHaltExceptionHeader(h) {
			return res, ErrHaltException
		}

		return res, nil
	}

	if lastErr != nil {
		return zero, lastErr
	}
	return zero, errors.New("fallback executor: all runners failed")
}
