package gateway

// Witness for issue #14451: two tenants on the OpenAI and Anthropic routes,
// including one cancelled stream — ledger rows exist for every request, the
// principal is never client-forgeable, per-tenant sums match the response
// usage fields, the hash chain verifies, and rows survive a restart.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// newUsageLedgerTestServer wires a minimal Server with metrics and a ledger
// file under t.TempDir().
func newUsageLedgerTestServer(t *testing.T) (*Server, *UsageLedger, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "usage-ledger.jsonl")
	l, err := OpenUsageLedger(path)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	t.Cleanup(func() { l.Close() })
	s := &Server{metrics: newGatewayMetrics(time.Now()), model: "test-model"}
	s.SetUsageLedger(l)
	return s, l, path
}

func TestUsageLedgerTwoTenantReplay(t *testing.T) {
	s, _, path := newUsageLedgerTestServer(t)

	type turn struct {
		principal string
		wire      string
		usage     agent.Usage
		finish    string
	}
	turns := []turn{
		{"tenant-a", "openai_chat_completions", agent.Usage{PromptTokens: 100, CompletionTokens: 40, TotalTokens: 140, CacheReadInputTokens: 60}, "stop"},
		{"tenant-b", "anthropic_messages", agent.Usage{PromptTokens: 250, CompletionTokens: 80, TotalTokens: 330, CacheReadInputTokens: 120}, "end_turn"},
		{"tenant-a", "openai_responses", agent.Usage{PromptTokens: 50, CompletionTokens: 10, TotalTokens: 60}, "length"},
		{"tenant-b", "openai_chat_completions", agent.Usage{PromptTokens: 30, CompletionTokens: 5, TotalTokens: 35}, "cancelled"}, // aborted stream keeps its partial row
	}
	for i, tr := range turns {
		trace := "trace-" + tr.principal + "-" + itoa(uint64(i+1))
		s.logInferenceTurnAs(trace, tr.wire, false, tr.usage, tr.finish, time.Millisecond, false, false, tr.principal)
	}

	rows, err := ReadUsageLedger(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if len(rows) != len(turns) {
		t.Fatalf("ledger rows = %d, want one per request (%d)", len(rows), len(turns))
	}
	if err := VerifyUsageLedger(rows); err != nil {
		t.Fatalf("chain verify: %v", err)
	}
	var wantA, wantB UsageLedgerSum
	for i, tr := range turns {
		row := rows[i]
		wantTrace := "trace-" + tr.principal + "-" + itoa(uint64(i+1))
		if row.Principal != tr.principal {
			t.Fatalf("row %d principal = %q, want %q (must come from the authenticated key)", i, row.Principal, tr.principal)
		}
		if row.RequestID != wantTrace {
			t.Fatalf("row %d request_id = %q, want %q", i, row.RequestID, wantTrace)
		}
		if row.Wire != tr.wire || row.Status != usageLedgerStatus(tr.finish) {
			t.Fatalf("row %d wire/status = %q/%q, want %q/%q", i, row.Wire, row.Status, tr.wire, usageLedgerStatus(tr.finish))
		}
		if row.PromptTokens != int64(tr.usage.PromptTokens) || row.CompletionTokens != int64(tr.usage.CompletionTokens) ||
			row.CachedPromptTokens != int64(tr.usage.CachedPromptTokens()) || row.TotalTokens != int64(tr.usage.TotalTokens) {
			t.Fatalf("row %d token axes = %+v, want %+v", i, row, tr.usage)
		}
		target := &wantA
		if tr.principal == "tenant-b" {
			target = &wantB
		}
		target.Requests++
		target.PromptTokens += int64(tr.usage.PromptTokens)
		target.CompletionTokens += int64(tr.usage.CompletionTokens)
		target.CachedPromptTokens += int64(tr.usage.CachedPromptTokens())
		target.TotalTokens += int64(tr.usage.TotalTokens)
	}
	// Export surface: whole-file sums_by_principal equals the replay totals.
	rep := s.usageLedgerReport(usageLedgerDefaultN)
	if !rep.Enabled || !rep.ChainOK {
		t.Fatalf("report enabled/chain_ok = %v/%v, want true/true", rep.Enabled, rep.ChainOK)
	}
	if got := rep.SumsByPrincipal["tenant-a"]; got != wantA {
		t.Fatalf("export tenant-a sums = %+v, want %+v", got, wantA)
	}
	if got := rep.SumsByPrincipal["tenant-b"]; got != wantB {
		t.Fatalf("export tenant-b sums = %+v, want %+v", got, wantB)
	}

	// Restart continuation: reopen the file, the chain still verifies, and the
	// next row resumes seq + prev_hash from the recovered head.
	l2, err := OpenUsageLedger(path)
	if err != nil {
		t.Fatalf("reopen after restart: %v", err)
	}
	defer l2.Close()
	if err := l2.Append(UsageLedgerRow{
		RequestID: "trace-tenant-a-5", Principal: "tenant-a", Wire: "openai_chat_completions",
		Status: UsageLedgerStatusCompleted, PromptTokens: 7, CompletionTokens: 3, TotalTokens: 10,
	}); err != nil {
		t.Fatalf("append after restart: %v", err)
	}
	rows2, err := ReadUsageLedger(path)
	if err != nil {
		t.Fatalf("re-read after restart: %v", err)
	}
	if len(rows2) != len(turns)+1 {
		t.Fatalf("post-restart rows = %d, want %d", len(rows2), len(turns)+1)
	}
	last := rows2[len(rows2)-1]
	if last.Seq != uint64(len(rows2)) || last.PrevHash != rows2[len(rows2)-2].Hash {
		t.Fatalf("restart row did not continue the chain: %+v", last)
	}
	if err := VerifyUsageLedger(rows2); err != nil {
		t.Fatalf("post-restart chain: %v", err)
	}
}

// TestUsageLedgerPrincipalNotClientForgeable drives the REAL auth seam: two
// keyset keys, and tenant-b presents tenant-a's pre-claimed trace id plus a
// spoofed X-Fak-Principal header. The ledger row must attribute to the
// presenting key, never to the claimed identity.
func TestUsageLedgerPrincipalNotClientForgeable(t *testing.T) {
	s, _, path := newUsageLedgerTestServer(t)
	s.keyset = newKeyset(map[string]string{"key-a": "tenant-a", "key-b": "tenant-b"})
	// Tenant-a claims the trace id first (first-writer-wins ownership table).
	s.bindTraceOwner("trace-spoof", "tenant-a")

	var sawPrincipal string
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawPrincipal = principalFromContext(r.Context())
		// The route boundary captures the principal exactly as the migrated
		// handlers do, then funnels the turn.
		s.logInferenceTurnAs("trace-spoof", "openai_chat_completions", false,
			agent.Usage{PromptTokens: 11, CompletionTokens: 2, TotalTokens: 13}, "stop",
			time.Millisecond, false, false, principalFromContext(r.Context()))
		w.WriteHeader(http.StatusNoContent)
	})
	h := s.withAuth(next)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer key-b")
	req.Header.Set("X-Fak-Principal", "tenant-a") // spoof: must lose to the presenting key
	req.Header.Set("X-Trace-Id", "trace-spoof")   // pre-claimed by tenant-a
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("authed request status = %d, want 204", rec.Code)
	}
	if sawPrincipal != "tenant-b" {
		t.Fatalf("context principal = %q, want tenant-b (the presenting key)", sawPrincipal)
	}
	rows, err := ReadUsageLedger(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if len(rows) != 1 || rows[0].Principal != "tenant-b" {
		t.Fatalf("ledger attribution = %+v, want the presenting key tenant-b", rows)
	}
}

// TestUsageLedgerExportEndpoint exercises GET /v1/fak/usage/ledger: last-N
// rows, whole-file sums_by_principal, chain_ok.
func TestUsageLedgerExportEndpoint(t *testing.T) {
	s, _, _ := newUsageLedgerTestServer(t)
	s.logInferenceTurnAs("t-a", "openai_chat_completions", false,
		agent.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11}, "stop",
		time.Millisecond, false, false, "tenant-a")
	s.logInferenceTurnAs("t-b", "anthropic_messages", false,
		agent.Usage{PromptTokens: 20, CompletionTokens: 2, TotalTokens: 22}, "end_turn",
		time.Millisecond, false, false, "tenant-b")

	rec := httptest.NewRecorder()
	s.handleFakUsageLedger(rec, httptest.NewRequest(http.MethodGet, "/v1/fak/usage/ledger", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var rep usageLedgerReport
	if err := json.Unmarshal(rec.Body.Bytes(), &rep); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if !rep.Enabled || !rep.ChainOK {
		t.Fatalf("report enabled/chain_ok = %v/%v, want true/true", rep.Enabled, rep.ChainOK)
	}
	if len(rep.Rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rep.Rows))
	}
	a := rep.SumsByPrincipal["tenant-a"]
	b := rep.SumsByPrincipal["tenant-b"]
	if a.Requests != 1 || a.PromptTokens != 10 || a.CompletionTokens != 1 || a.TotalTokens != 11 {
		t.Fatalf("tenant-a sums = %+v", a)
	}
	if b.Requests != 1 || b.PromptTokens != 20 || b.CompletionTokens != 2 || b.TotalTokens != 22 {
		t.Fatalf("tenant-b sums = %+v", b)
	}
}

// TestUsageLedgerTamperDetected flips one stored token axis and requires the
// chain to break at that row.
func TestUsageLedgerTamperDetected(t *testing.T) {
	s, _, path := newUsageLedgerTestServer(t)
	s.logInferenceTurnAs("t1", "openai_chat_completions", false,
		agent.Usage{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11}, "stop",
		time.Millisecond, false, false, "tenant-a")
	s.logInferenceTurnAs("t2", "openai_chat_completions", false,
		agent.Usage{PromptTokens: 20, CompletionTokens: 1, TotalTokens: 21}, "stop",
		time.Millisecond, false, false, "tenant-b")

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read raw ledger: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	if len(lines) != 2 {
		t.Fatalf("ledger lines = %d, want 2", len(lines))
	}
	tampered := strings.Replace(lines[1], `"completion_tokens":1`, `"completion_tokens":9`, 1)
	if tampered == lines[1] {
		t.Fatalf("tamper pattern not found in %q", lines[1])
	}
	if err := os.WriteFile(path, []byte(lines[0]+"\n"+tampered+"\n"), 0o600); err != nil {
		t.Fatalf("write tampered ledger: %v", err)
	}
	if err := VerifyUsageLedgerFile(path); err == nil {
		t.Fatal("tampered chain verified clean; want a break")
	}
	// A corrupted chain must refuse to extend (fail-closed at Open).
	if _, err := OpenUsageLedger(path); err == nil {
		t.Fatal("corrupt ledger opened; want refusal")
	}
}

// TestUsageLedgerOffLeavesRequestPathUnchanged pins Law A2: with no ledger
// enabled the funnel runs clean and writes nothing.
func TestUsageLedgerOffLeavesRequestPathUnchanged(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage-ledger.jsonl")
	s := &Server{metrics: newGatewayMetrics(time.Now()), model: "test-model"}
	s.logInferenceTurnAs("t-off", "openai_chat_completions", false,
		agent.Usage{PromptTokens: 3, CompletionTokens: 1, TotalTokens: 4}, "stop",
		time.Millisecond, false, false, "tenant-a")
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("ledger file created while the ledger was off: stat err=%v", err)
	}
	rep := s.usageLedgerReport(usageLedgerDefaultN)
	if rep.Enabled || len(rep.Rows) != 0 {
		t.Fatalf("off report = %+v, want disabled/empty", rep)
	}
}
