package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// cacheReportingPlanner answers every turn and reports cachedTok of its prompt
// as a provider prompt-cache hit. One completion token keeps the estimator's
// synthetic rates untouched.
type cacheReportingPlanner struct {
	cachedTok int
	calls     int
}

func (p *cacheReportingPlanner) Complete(_ context.Context, msgs []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens:        estimateMessageContentTokens(msgs),
			CompletionTokens:    1,
			PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: p.cachedTok},
		},
	}, nil
}

func (*cacheReportingPlanner) Model() string { return "cache-reporting" }

// piToolTurns is a pi read task: turn 1 ends in a 47k-token tool result, turn 2
// resends it with the reply and a 2k-token tool result (about 50k tokens).
func piToolTurns(toolChars int) (turn1, turn2 []agent.Message) {
	turn1 = []agent.Message{
		{Role: "system", Content: strings.Repeat("s", 4000)},
		{Role: "user", Content: "read the repository"},
		{Role: "assistant", Content: "reading"},
		{Role: "tool", Content: strings.Repeat("t", toolChars)},
	}
	turn2 = append(append([]agent.Message(nil), turn1...),
		agent.Message{Role: "assistant", Content: "reading the next file"},
		agent.Message{Role: "tool", Content: strings.Repeat("n", 8000)},
	)
	return turn1, turn2
}

func postDeadlineCreditTurn(t *testing.T, srv *Server, messages []agent.Message, budget string) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(ChatRequest{Messages: messages})
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
	return rec
}

func wantDeadlineRefusal(t *testing.T, rec *httptest.ResponseRecorder, what string) {
	t.Helper()
	if rec.Code != http.StatusServiceUnavailable ||
		rec.Header().Get(deadlineadmit.HeaderReject) != string(deadlineadmit.ReasonDeadlineInfeasible) {
		t.Fatalf("%s: status=%d reject=%q; want 503 %q", what, rec.Code,
			rec.Header().Get(deadlineadmit.HeaderReject), deadlineadmit.ReasonDeadlineInfeasible)
	}
}

func deadlineCreditedTokens(t *testing.T, srv *Server) float64 {
	t.Helper()
	b, err := json.Marshal(srv.metrics.adjudicationSummary())
	if err != nil {
		t.Fatal(err)
	}
	var sum struct {
		Credit struct {
			Tokens float64 `json:"credited_tokens"`
		} `json:"deadline_cache_credit"`
	}
	if err := json.Unmarshal(b, &sum); err != nil {
		t.Fatal(err)
	}
	return sum.Credit.Tokens
}

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=100ms lane=default
func TestDeadlineCacheCreditAdmitsWarmMultiTurnPrefix(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	planner := &cacheReportingPlanner{cachedTok: 40_000}
	srv.planner = planner
	turn1, turn2 := piToolTurns(188_000)

	if rec := postDeadlineCreditTurn(t, srv, turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d body=%s", rec.Code, rec.Body.String())
	}
	rec := postDeadlineCreditTurn(t, srv, turn2, "600")
	if rec.Code != http.StatusOK || planner.calls != 2 {
		t.Fatalf("warm turn 2: status=%d reject=%q calls=%d; want admitted on the served cached prefix",
			rec.Code, rec.Header().Get(deadlineadmit.HeaderReject), planner.calls)
	}
	if got := deadlineCreditedTokens(t, srv); got < 40_000 {
		t.Fatalf("deadline_cache_credit.credited_tokens = %v, want the ~48k verified prefix", got)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineCacheCreditRefusesColdSameSizePrompt(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	planner := &cacheReportingPlanner{cachedTok: 40_000}
	srv.planner = planner
	turn1, _ := piToolTurns(188_000)
	if rec := postDeadlineCreditTurn(t, srv, turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d", rec.Code)
	}
	_, cold := piToolTurns(188_000)
	cold[1].Content = "a different task"
	wantDeadlineRefusal(t, postDeadlineCreditTurn(t, srv, cold, "600"), "cold conversation of the same size")
	if planner.calls != 1 {
		t.Fatalf("planner calls = %d, want 1", planner.calls)
	}
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineCacheCreditNoCreditOnPrefixMismatch(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	srv.planner = &cacheReportingPlanner{cachedTok: 40_000}
	turn1, turn2 := piToolTurns(188_000)
	if rec := postDeadlineCreditTurn(t, srv, turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d", rec.Code)
	}
	// The history was rewritten (as a compaction would): only the short head
	// still matches, so the large tool result is priced cold.
	turn2[3] = agent.Message{Role: "tool", Content: strings.Repeat("x", 188_000)}
	wantDeadlineRefusal(t, postDeadlineCreditTurn(t, srv, turn2, "600"), "rewritten history")
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineCacheCreditNeedsUpstreamCacheEvidence(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	srv.planner = &cacheReportingPlanner{cachedTok: 0}
	turn1, turn2 := piToolTurns(188_000)
	if rec := postDeadlineCreditTurn(t, srv, turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d", rec.Code)
	}
	wantDeadlineRefusal(t, postDeadlineCreditTurn(t, srv, turn2, "600"), "turn after an uncached served turn")
}

// fak-test:justify why=invariant when=changed:internal/gateway/**
// fak-test:runtime fast est=50ms lane=default
func TestDeadlineCacheCreditBoundsColdOverrun(t *testing.T) {
	srv := haloPiDeadlineServer(t)
	srv.planner = &cacheReportingPlanner{cachedTok: 80_000}
	// ~100k tokens prices ~1170 s cold, beyond 1.5x the 540 s ceiling: a wrong
	// credit could overrun the deadline by more than the bound, so none is given.
	turn1, turn2 := piToolTurns(400_000)
	if rec := postDeadlineCreditTurn(t, srv, turn1, ""); rec.Code != http.StatusOK {
		t.Fatalf("turn 1: status=%d", rec.Code)
	}
	wantDeadlineRefusal(t, postDeadlineCreditTurn(t, srv, turn2, "600"), "warm turn past the cold-overrun bound")
}
