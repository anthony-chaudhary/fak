package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// saturatedDeadlineServer primes the node estimator with the strix1 incident:
// six requests in flight, each a 1.6k-token prefill in 87 s plus a 315-token
// decode in 163 s. The estimator clock is injected and frozen.
func saturatedDeadlineServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	frozen := time.Unix(1_800_000_000, 0)
	est := deadlineadmit.NewEstimator(deadlineadmit.Config{}, func() time.Time { return frozen })
	srv.metrics.deadlineEst = est
	for i := 0; i < 6; i++ {
		est.Begin() // never ended: the six saturating requests stay in flight
	}
	est.Observe(deadlineadmit.Observation{PromptTokens: 1600, Prefill: 87 * time.Second, CompletionTokens: 315, Decode: 163 * time.Second})
	return srv
}

func postDeadlineChat(t *testing.T, ts *httptest.Server, headers map[string]string) *http.Response {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":    "test-model",
		"messages": []map[string]string{{"role": "user", "content": strings.Repeat("A", 6400)}}, // ~1.6k tokens
	})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { resp.Body.Close() })
	return resp
}

func TestChatCompletionsShedsRequestThatCannotMeetClientDeadline(t *testing.T) {
	ts := httptest.NewServer(saturatedDeadlineServer(t).Handler())
	defer ts.Close()

	resp := postDeadlineChat(t, ts, map[string]string{deadlineadmit.HeaderStainlessTimeout: "300"})
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	if got := resp.Header.Get(deadlineadmit.HeaderReject); got != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("%s = %q, want %q", deadlineadmit.HeaderReject, got, deadlineadmit.ReasonDeadlineInfeasible)
	}
	if resp.Header.Get("Retry-After") == "" {
		t.Fatal("Retry-After missing on deadline refusal")
	}
	var env struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&env); err != nil {
		t.Fatal(err)
	}
	if env.Error.Code != deadlineadmit.CodeDeadlineInfeasible {
		t.Fatalf("error.code = %q, want %q", env.Error.Code, deadlineadmit.CodeDeadlineInfeasible)
	}
}

func TestChatCompletionsWithoutDeadlineIsNotShed(t *testing.T) {
	ts := httptest.NewServer(saturatedDeadlineServer(t).Handler())
	defer ts.Close()

	resp := postDeadlineChat(t, ts, nil)
	if got := resp.Header.Get(deadlineadmit.HeaderReject); got != "" {
		t.Fatalf("undeclared-deadline request refused with %q (status %d)", got, resp.StatusCode)
	}
}

func TestAdmitClientDeadlineBindsDeadlineToRequestContext(t *testing.T) {
	srv := newTestServer(t)
	arrived := time.Now()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set(deadlineadmit.HeaderFakDeadlineMs, "5000")
	got, release, ok := srv.admitClientDeadline(httptest.NewRecorder(), r, arrived, 10, 10)
	if !ok {
		t.Fatal("unmeasured node refused a request")
	}
	defer release()
	dl, has := got.Context().Deadline()
	if !has || !dl.Equal(arrived.Add(5*time.Second)) {
		t.Fatalf("context deadline = (%v, %v), want %v", dl, has, arrived.Add(5*time.Second))
	}
	if n := srv.metrics.deadlineEstimator().InFlight(); n != 1 {
		t.Fatalf("in-flight = %d, want 1 while admitted", n)
	}
	release()
	if n := srv.metrics.deadlineEstimator().InFlight(); n != 0 {
		t.Fatalf("in-flight = %d, want 0 after release", n)
	}
}
