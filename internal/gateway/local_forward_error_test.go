package gateway

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=1ms lane=default

// upstreamErrorStatus must classify a local inference-engine forward failure on
// its own ladder — retryable as 503 "local_forward_retryable", non-retryable as
// 500 "local_forward_failed" — instead of masking it as the opaque 502
// "upstream model error" fallthrough, which hides the real cause from client and
// operator alike. No GPU is needed: the error is an ordinary agent value.

func TestUpstreamErrorStatus_LocalForwardErrorIsNotMaskedAs502(t *testing.T) {
	retryable := agent.NewLocalForwardError("prefill", errors.New("device OOM during prefill"), true)
	status, code, msg := upstreamErrorStatus(retryable)
	if status != http.StatusServiceUnavailable {
		t.Fatalf("retryable local forward failure should be 503, got %d", status)
	}
	if code != "local_forward_retryable" {
		t.Fatalf("retryable local forward failure should carry the distinct code, got %q", code)
	}
	if !strings.Contains(msg, "prefill") {
		t.Fatalf("message does not name the failed operation: %q", msg)
	}

	fatal := agent.NewLocalForwardError("decode", errors.New("unsupported tensor layout"), false)
	status, code, msg = upstreamErrorStatus(fatal)
	if status != http.StatusInternalServerError {
		t.Fatalf("non-retryable local forward failure should be 500, got %d", status)
	}
	if code != "local_forward_failed" {
		t.Fatalf("non-retryable local forward failure should carry the distinct code, got %q", code)
	}
	if !strings.Contains(msg, "decode") {
		t.Fatalf("message does not name the failed operation: %q", msg)
	}
}
