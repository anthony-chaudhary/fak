package gateway

import (
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
)

type upstreamObserveTransport struct {
	base           http.RoundTripper
	observe        func(status int, header http.Header)
	observeError   func(error)
	observeFailure func(UpstreamFailureReceipt)
	mu             sync.Mutex
	attempts       map[any]int
}

func (t *upstreamObserveTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	attempt := t.nextAttempt(req)
	resp, err := t.base.RoundTrip(req)
	if resp != nil && t.observe != nil {
		t.observe(resp.StatusCode, resp.Header)
	}
	if err != nil && t.observeError != nil && transientUpstreamTransportError(err) {
		t.observeError(err)
	}
	if t.observeFailure != nil {
		if err != nil {
			t.observeFailure(upstreamFailureReceipt(req, 0, nil, attempt, err))
		} else if resp != nil && resp.StatusCode >= 400 {
			t.observeFailure(upstreamFailureReceipt(req, resp.StatusCode, resp.Header, attempt, nil))
		} else if resp != nil && resp.Body != nil {
			resp.Body = &receiptBody{ReadCloser: resp.Body, req: req, status: resp.StatusCode, header: resp.Header.Clone(), attempt: attempt, emit: t.observeFailure}
		}
	}
	return resp, err
}

func (t *upstreamObserveTransport) nextAttempt(req *http.Request) int {
	if req == nil {
		return 1
	}
	key := any(req.Context())
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.attempts == nil {
		t.attempts = make(map[any]int)
	}
	t.attempts[key]++
	return t.attempts[key]
}

func wrapUpstreamObserver(client *http.Client, observe func(int, http.Header), observeError func(error), observeFailure func(UpstreamFailureReceipt)) {
	if client == nil || (observe == nil && observeError == nil && observeFailure == nil) {
		return
	}
	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}
	client.Transport = &upstreamObserveTransport{base: base, observe: observe, observeError: observeError, observeFailure: observeFailure}
}

func transientUpstreamTransportError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	if transientTransportError(err) {
		return true
	}
	low := strings.ToLower(err.Error())
	for _, needle := range []string{"connection reset", "connection refused", "broken pipe", "server closed idle connection", "tls handshake timeout"} {
		if strings.Contains(low, needle) {
			return true
		}
	}
	return false
}
