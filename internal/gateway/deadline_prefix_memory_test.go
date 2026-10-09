package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// haloPiDeadlineServer primes the node estimator with the 2026-10-09 strix3
// shape: 86 t/s prefill and 21 t/s decode, nothing else in flight.
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
func TestDeadlineAdmissionAdmitsWarmMultiTurnPromptThisNodeServed(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	turn1, turn2 := haloPiTurns()

	_, release, ok := admitHaloTurn(t, srv, context.Background(), turn1, "")
	if !ok {
		t.Fatal("turn 1 without a declared budget was refused")
	}
	release()

	rec, release, ok := admitHaloTurn(t, srv, context.Background(), turn2, "600")
	if !ok {
		t.Fatalf("warm turn 2 refused: status=%d %s=%q; want admission (48k of 50k tokens resident)",
			rec.Code, deadlineadmit.HeaderReject, rec.Header().Get(deadlineadmit.HeaderReject))
	}
	release()
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
func TestDeadlineAdmissionDoesNotRememberCanceledTurn(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	turn1, turn2 := haloPiTurns()

	ctx, cancel := context.WithCancel(context.Background())
	_, release, ok := admitHaloTurn(t, srv, ctx, turn1, "")
	if !ok {
		t.Fatal("turn 1 without a declared budget was refused")
	}
	cancel() // the client left before the turn finished: its prefix is not known resident
	release()

	rec, _, ok := admitHaloTurn(t, srv, context.Background(), turn2, "600")
	if ok || rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("turn 2 after a canceled turn 1: ok=%v reject=%q; want cold refusal %q",
			ok, rec.Header().Get(deadlineadmit.HeaderReject), deadlineadmit.ReasonDeadlineInfeasible)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=10ms lane=default
func TestDeadlinePrefixMemoryExpiresAndMatchesOnlyExactPrefixes(t *testing.T) {
	turn1, turn2 := haloPiTurns()
	mem := &deadlinePrefixMemory{entries: map[[32]byte]deadlinePrefixEntry{}}
	now := time.Unix(1_800_000_000, 0)
	chain1 := deadlinePrefixChain(turn1)
	mem.remember(chain1[len(chain1)-1], 48_000, now)

	if got := mem.residentTokens(deadlinePrefixChain(turn2), now.Add(time.Minute)); got != 48_000 {
		t.Fatalf("resident tokens for an extending turn = %d, want 48000", got)
	}
	edited := append([]agent.Message(nil), turn2...)
	edited[1] = agent.Message{Role: "user", Content: "a different history"}
	if got := mem.residentTokens(deadlinePrefixChain(edited), now.Add(time.Minute)); got != 0 {
		t.Fatalf("resident tokens for a diverged history = %d, want 0", got)
	}
	if got := mem.residentTokens(deadlinePrefixChain(turn2), now.Add(deadlinePrefixTTL+time.Second)); got != 0 {
		t.Fatalf("resident tokens past the TTL = %d, want 0", got)
	}
}
