package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// haloPiDeadlineServer primes the node estimator with synthetic rates of
// 86 t/s prefill and 21 t/s decode, nothing else in flight. This fixture is
// an admission-policy control, not a hardware performance witness.
func haloPiDeadlineServer(t *testing.T) *Server {
	t.Helper()
	srv := newTestServer(t)
	frozen := time.Unix(1_800_000_000, 0)
	est := deadlineadmit.NewEstimator(deadlineadmit.Config{}, func() time.Time { return frozen })
	srv.metrics.deadlineEst = est
	est.Observe(deadlineadmit.Observation{PromptTokens: 8600, Prefill: 100 * time.Second, CompletionTokens: 210, Decode: 10 * time.Second})
	return srv
}

// haloPiTurns is a pi multi-turn read task: turn 1 is a 48k-token history,
// turn 2 resends it plus the reply and a 2k-token tool result.
func haloPiTurns() (turn1, turn2 []agent.Message) {
	turn1 = []agent.Message{
		{Role: "system", Content: strings.Repeat("s", 4000)},
		{Role: "user", Content: strings.Repeat("u", 188_000)},
	}
	turn2 = append(append([]agent.Message(nil), turn1...),
		agent.Message{Role: "assistant", Content: "reading the next file"},
		agent.Message{Role: "tool", Content: strings.Repeat("t", 8000)},
	)
	return turn1, turn2
}

func admitHaloTurn(t *testing.T, srv *Server, ctx context.Context, msgs []agent.Message, budget string) (*httptest.ResponseRecorder, func(), bool) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
	if budget != "" {
		r.Header.Set(deadlineadmit.HeaderStainlessTimeout, budget)
	}
	rec := httptest.NewRecorder()
	_, release, ok := srv.admitClientDeadlineMessages(rec, r, time.Now(), msgs, 0)
	return rec, release, ok
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineAdmissionDoesNotCreditReleasedHistory(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	turn1, turn2 := haloPiTurns()

	_, release, ok := admitHaloTurn(t, srv, context.Background(), turn1, "")
	if !ok {
		t.Fatal("turn 1 without a declared budget was refused")
	}
	release()
	if n := srv.metrics.deadlineEstimator().InFlight(); n != 0 {
		t.Fatalf("in-flight after release = %d, want 0", n)
	}

	// Admission and release establish no cache residency, even when the request
	// context is still healthy. The extending turn must retain its cold estimate.
	rec, release, ok := admitHaloTurn(t, srv, context.Background(), turn2, "600")
	if release != nil {
		defer release()
	}
	if ok || rec.Code != http.StatusServiceUnavailable ||
		rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("turn 2 after release: ok=%v status=%d reject=%q; want cold refusal %q",
			ok, rec.Code, rec.Header().Get(deadlineadmit.HeaderReject), deadlineadmit.ReasonDeadlineInfeasible)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineAdmissionRefusesColdHugePrompt(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	_, turn2 := haloPiTurns()

	rec, _, ok := admitHaloTurn(t, srv, context.Background(), turn2, "600")
	if ok || rec.Code != http.StatusServiceUnavailable ||
		rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("cold 50k turn: ok=%v status=%d reject=%q; want 503 %q",
			ok, rec.Code, rec.Header().Get(deadlineadmit.HeaderReject), deadlineadmit.ReasonDeadlineInfeasible)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineAdmissionDoesNotCreditCanceledHistory(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	turn1, turn2 := haloPiTurns()

	ctx, cancel := context.WithCancel(context.Background())
	_, release, ok := admitHaloTurn(t, srv, ctx, turn1, "")
	if !ok {
		t.Fatal("turn 1 without a declared budget was refused")
	}
	cancel()
	release()

	rec, _, ok := admitHaloTurn(t, srv, context.Background(), turn2, "600")
	if ok || rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("turn 2 after a canceled turn 1: ok=%v reject=%q; want cold refusal %q",
			ok, rec.Header().Get(deadlineadmit.HeaderReject), deadlineadmit.ReasonDeadlineInfeasible)
	}
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineAdmissionDoesNotCreditFailedHTTPRequest(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	planner := &chatDecodeTraceCountingPlanner{native: false}
	srv.planner = planner
	turn1, turn2 := haloPiTurns()

	post := func(messages []agent.Message, trace bool, budget string) *httptest.ResponseRecorder {
		t.Helper()
		body, err := json.Marshal(ChatRequest{Messages: messages, FakDecodeTrace: trace})
		if err != nil {
			t.Fatal(err)
		}
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
		r.Header.Set("Content-Type", "application/json")
		if budget != "" {
			r.Header.Set(deadlineadmit.HeaderStainlessTimeout, budget)
		}
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, r)
		if err := r.Context().Err(); err != nil {
			t.Fatalf("request context = %v, want healthy context", err)
		}
		return rec
	}

	// This request passes deadline admission without a budget, then fails before
	// inference because its planner cannot supply a native decode trace.
	rec := post(turn1, true, "")
	if rec.Code != http.StatusBadRequest || planner.calls != 0 ||
		!strings.Contains(rec.Body.String(), "fak_decode_trace requires a fak-native model route") {
		t.Fatalf("failed request: status=%d planner calls=%d body=%s; want non-native trace refusal before inference",
			rec.Code, planner.calls, rec.Body.String())
	}
	if n := srv.metrics.deadlineEstimator().InFlight(); n != 0 {
		t.Fatalf("in-flight after failed request = %d, want 0", n)
	}

	rec = post(turn2, false, "600")
	if rec.Code != http.StatusServiceUnavailable ||
		rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) ||
		rec.Header().Get("Retry-After") == "" || planner.calls != 0 {
		t.Fatalf("turn 2 after failed request: status=%d reject=%q retry=%q planner calls=%d; want cold 503 before inference",
			rec.Code, rec.Header().Get(deadlineadmit.HeaderReject), rec.Header().Get("Retry-After"), planner.calls)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineAdmissionPreservesExplicitCachedTokenCredit(t *testing.T) {
	for _, tc := range []struct {
		name      string
		cachedTok int
		wantAdmit bool
	}{
		{name: "cold", cachedTok: 0, wantAdmit: false},
		{name: "trusted-cache-input", cachedTok: 48_000, wantAdmit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := haloPiDeadlineServer(t)
			_, turn2 := haloPiTurns()
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			r.Header.Set(deadlineadmit.HeaderStainlessTimeout, "600")
			rec := httptest.NewRecorder()
			// The caller explicitly supplies trusted credit for this API control;
			// no physical cache or gateway history is claimed as residency proof.
			_, release, ok := srv.admitClientDeadlineCached(rec, r, time.Now(), estimateMessageContentTokens(turn2), tc.cachedTok, 0)
			if release != nil {
				defer release()
			}
			if ok != tc.wantAdmit {
				t.Fatalf("admit with cached=%d: ok=%v status=%d reject=%q; want %v",
					tc.cachedTok, ok, rec.Code, rec.Header().Get(deadlineadmit.HeaderReject), tc.wantAdmit)
			}
			if !ok && (rec.Code != http.StatusServiceUnavailable ||
				rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible)) {
				t.Fatalf("cold direct admission: status=%d reject=%q; want typed 503",
					rec.Code, rec.Header().Get(deadlineadmit.HeaderReject))
			}
		})
	}
}
