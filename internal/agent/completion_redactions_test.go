package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/wirescreen"
)

// completion_redactions_test.go is the #881 keystone witness (pillar A): the
// outbound span-redaction count computed in prepareUpstream (call.redacted, from
// RedactOutboundMessages on the non-passthrough re-marshal path) is transferred to
// Completion.PreSendRedactions at the assembly point, mirroring PreSendQuarantines.
// Before #881 the count was computed at stream.go and silently dropped; a caller
// could see what was quarantined but never that anything was redacted.

// stubChatServer answers any /chat/completions POST with a fixed "ok" turn. The
// request body is irrelevant to the assertion — the test reads the count off the
// returned Completion, not the wire.
func stubChatServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
}

// TestPreSendRedactionsSurfaced drives a full Complete round-trip with a
// credit-card-bearing outbound message. With the pii redactor active the span is
// redacted on the re-marshal path and Completion.PreSendRedactions is > 0; with the
// redactor inert (the default) the same call surfaces 0 and the path is unchanged.
func TestPreSendRedactionsSurfaced(t *testing.T) {
	// A Luhn-valid card number is flagged by the pii redactor (see transcript_redact_test).
	secretMsgs := []Message{
		{Role: "user", Content: "please charge card 4111 1111 1111 1111 now"},
	}

	t.Run("active redactor surfaces a positive count", func(t *testing.T) {
		restore := wirescreen.SetActiveRedactorForTest("pii")
		defer restore()

		ts := stubChatServer()
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner(string(ProviderOpenAI), ts.URL, "gpt-test", "sekret")
		if err != nil {
			t.Fatal(err)
		}
		comp, err := planner.Complete(context.Background(), secretMsgs, nil)
		if err != nil {
			t.Fatalf("complete: %v", err)
		}
		if comp.PreSendRedactions <= 0 {
			t.Fatalf("active redactor: PreSendRedactions = %d, want > 0 (the card span should have been redacted)", comp.PreSendRedactions)
		}
	})

	t.Run("inert redactor surfaces zero", func(t *testing.T) {
		restore := wirescreen.SetActiveRedactorForTest("") // "" -> inert/nil, the default
		defer restore()

		ts := stubChatServer()
		defer ts.Close()

		planner, err := NewProviderHTTPPlanner(string(ProviderOpenAI), ts.URL, "gpt-test", "sekret")
		if err != nil {
			t.Fatal(err)
		}
		comp, err := planner.Complete(context.Background(), secretMsgs, nil)
		if err != nil {
			t.Fatalf("complete: %v", err)
		}
		if comp.PreSendRedactions != 0 {
			t.Fatalf("inert redactor: PreSendRedactions = %d, want 0", comp.PreSendRedactions)
		}
	})
}
