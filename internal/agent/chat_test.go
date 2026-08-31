package agent

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

type issue10321RoundTripFunc func(*http.Request) (*http.Response, error)

func (f issue10321RoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type issue10321ReadCloser struct {
	reader io.Reader
	err    error
}

func (r *issue10321ReadCloser) Read(p []byte) (int, error) {
	n, err := r.reader.Read(p)
	if err == io.EOF {
		return 0, r.err
	}
	return n, err
}

func (r *issue10321ReadCloser) Close() error { return nil }

func issue10321Response(contentType string, body io.ReadCloser) *http.Response {
	h := make(http.Header)
	h.Set("Content-Type", contentType)
	return &http.Response{StatusCode: http.StatusOK, Header: h, Body: body}
}

func TestCompleteRetriesHTTP200ReadFailureBeforeParsing(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")

	readCause := io.ErrUnexpectedEOF
	const good = `{"model":"m","choices":[{"message":{"role":"assistant","content":"recovered"},"finish_reason":"stop"}]}`
	attempts := 0
	p := NewHTTPPlanner("http://upstream.test", "m", "")
	p.Client = &http.Client{Transport: issue10321RoundTripFunc(func(*http.Request) (*http.Response, error) {
		attempts++
		if attempts == 1 {
			return issue10321Response("application/json", &issue10321ReadCloser{
				reader: strings.NewReader(`{"model":"m"`),
				err:    readCause,
			}), nil
		}
		return issue10321Response("application/json", io.NopCloser(strings.NewReader(good))), nil
	})}

	comp, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
	if comp.Message.Content != "recovered" {
		t.Fatalf("content = %q, want recovered", comp.Message.Content)
	}
}

func TestCompleteExhaustedHTTP200ReadsPreserveFinalCause(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "2")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")

	firstCause := errors.New("first buffered read failure")
	finalCause := io.ErrUnexpectedEOF
	attempts := 0
	p := NewHTTPPlanner("http://upstream.test", "m", "")
	p.Client = &http.Client{Transport: issue10321RoundTripFunc(func(*http.Request) (*http.Response, error) {
		attempts++
		cause := firstCause
		if attempts == 2 {
			cause = finalCause
		}
		return issue10321Response("application/json", &issue10321ReadCloser{
			reader: strings.NewReader(`{"choices":[`),
			err:    cause,
		}), nil
	})}

	_, err := p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "hi"}}, nil)
	if !errors.Is(err, finalCause) {
		t.Fatalf("err = %v, want wrapped final read cause", err)
	}
	if errors.Is(err, firstCause) {
		t.Fatalf("err = %v, unexpectedly retained first read cause", err)
	}
	if attempts != 2 {
		t.Fatalf("attempts = %d, want 2", attempts)
	}
}
