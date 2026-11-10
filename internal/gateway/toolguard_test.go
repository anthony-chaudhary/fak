package gateway

import (
	"context"
	"fmt"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// toolguard_test.go — the #14327 raw pass-through witnesses: the ToolGuard
// posture vocabulary, the library default staying armed, and both interception
// seams (proposed-call adjudication + inbound result admission) going inert
// under an explicit OFF while the armed floor keeps screening.

// TestParseToolGuardMode locks the closed ToolGuard posture vocabulary: empty
// resolves ON (the library/harness default), on-spellings arm, off-spellings
// select raw pass-through (surrounding whitespace trims to the token), and any
// other token fails loud.
func TestParseToolGuardMode(t *testing.T) {
	cases := []struct {
		raw     string
		want    string
		wantErr bool
	}{
		{"", ToolGuardOn, false},
		{"on", ToolGuardOn, false},
		{"ON", ToolGuardOn, false},
		{"1", ToolGuardOn, false},
		{"true", ToolGuardOn, false},
		{"yes", ToolGuardOn, false},
		{"enabled", ToolGuardOn, false},
		{"off", ToolGuardOff, false},
		{"OFF", ToolGuardOff, false},
		{"0", ToolGuardOff, false},
		{"false", ToolGuardOff, false},
		{"no", ToolGuardOff, false},
		{"disabled", ToolGuardOff, false},
		{"raw", ToolGuardOff, false},
		{" raw ", ToolGuardOff, false}, // surrounding spaces trim to the "raw" token
		{"bogus", "", true},
		{"maybe", "", true},
	}
	for _, tc := range cases {
		got, err := ParseToolGuardMode(tc.raw)
		if tc.wantErr {
			if err == nil {
				t.Errorf("ParseToolGuardMode(%q) = %q, want error", tc.raw, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseToolGuardMode(%q) unexpected error: %v", tc.raw, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ParseToolGuardMode(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

// TestToolGuardDefaultArmed is the regression lock for the flip: a Server built
// with the default config (no ToolGuard set) stays ARMed — the library's
// historical fail-closed floor is unchanged for every existing caller.
func TestToolGuardDefaultArmed(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	srv, err := New(Config{EngineID: "mock", Model: "m"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	if srv.toolGuardOff {
		t.Fatal("default Config (no ToolGuard) must stay armed: toolGuardOff = true")
	}
}

// poisonResult carries PII (email, phone, SSN) AND a credential in one tool
// result — the content the armed floor must screen out and raw mode must pass
// through byte-identical.
const poisonResult = `{"page":"contact jane.doe@example.com phone +1-555-0132 ssn 123-45-6789; api_key=` +
	`sk-abcdef0123456789abcdef0123 was found in env"}`

func poisonMessages() []agent.Message {
	return []agent.Message{
		{Role: agent.RoleUser, Content: "run lookup"},
		{
			Role:      agent.RoleAssistant,
			Content:   "calling fetch_url",
			ToolCalls: []agent.ToolCall{{ID: "call_1", Function: agent.Func{Name: "fetch_url", Arguments: `{}`}}},
		},
		{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: poisonResult},
	}
}

// TestToolGuardRawPassthroughResultAdmission proves the result-side seam goes
// inert under an explicit OFF: zero ResultAdmissions come back and the inbound
// tool result reaches the model byte-identical — the PII/secret bytes are still
// present verbatim because no screen, masking, or taint stamp ran.
func TestToolGuardRawPassthroughResultAdmission(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	srv, err := New(Config{EngineID: "mock", Model: "m", ToolGuard: ToolGuardOff})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	msgs := poisonMessages()
	adms, err := srv.admitInboundResults(context.Background(), msgs, nil, "raw-trace-14327")
	if err != nil {
		t.Fatalf("admitInboundResults (raw): %v", err)
	}
	if adms != nil {
		t.Fatalf("raw passthrough returned %d ResultAdmissions, want none", len(adms))
	}
	if msgs[2].Content != poisonResult {
		t.Fatalf("raw passthrough mutated the inbound tool result: %q", msgs[2].Content)
	}
	for _, want := range []string{"jane.doe@example.com", "+1-555-0132", "123-45-6789", "api_key=sk-abcdef0123456789abcdef0123"} {
		if !strings.Contains(msgs[2].Content, want) {
			t.Fatalf("raw passthrough lost verbatim bytes %q: %q", want, msgs[2].Content)
		}
	}
}

// TestToolGuardArmedStillScreens proves the flip only happens under the
// explicit OFF: the same poisoned result on a default (armed) Server comes back
// with non-empty admissions and the raw credential bytes gone from the
// forwarded content (masked or quarantined by the armed floor).
func TestToolGuardArmedStillScreens(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	led := ifc.NewLedger()
	abi.RegisterResultAdmitter(10, ctxmmu.New())
	abi.RegisterResultAdmitter(20, ifc.NewStampGate(led, ifc.Policy{}))
	srv, err := New(Config{EngineID: "mock", Model: "m"}) // no ToolGuard => armed
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	msgs := poisonMessages()
	adms, err := srv.admitInboundResults(context.Background(), msgs, nil, "armed-trace-14327")
	if err != nil {
		t.Fatalf("admitInboundResults (armed): %v", err)
	}
	if len(adms) == 0 {
		t.Fatalf("armed server returned no ResultAdmissions for the inbound tool result")
	}
	if strings.Contains(msgs[2].Content, "sk-abcdef0123456789abcdef0123") {
		t.Fatalf("armed floor failed to mask/quarantine the credential: %q", msgs[2].Content)
	}
}

// TestToolGuardRawPassthroughProposedCalls proves the proposed-call seam goes
// inert under an explicit OFF: every proposed call is kept verbatim (the
// caller's own slice, unmodified), with zero adjudications, drops, or served
// hits — no deny rungs, no vDSO/reuse probes, no kernel Decide fold.
func TestToolGuardRawPassthroughProposedCalls(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	srv, err := New(Config{EngineID: "mock", Model: "m", ToolGuard: ToolGuardOff})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	calls := []agent.ToolCall{{ID: "call_p1", Function: agent.Func{Name: "web_search", Arguments: `{"query":"x"}`}}}
	kept, adjs, dropped, servedText, servedHits := srv.adjudicateProposedServed(context.Background(), calls, "raw-proposed-14327")
	if len(kept) != len(calls) || !reflect.DeepEqual(kept, calls) {
		t.Fatalf("raw passthrough kept = %v, want the input calls verbatim %v", kept, calls)
	}
	if &kept[0] != &calls[0] {
		t.Fatal("raw passthrough returned a copy instead of the caller's slice")
	}
	if len(adjs) != 0 {
		t.Fatalf("raw passthrough produced %d adjudications, want zero", len(adjs))
	}
	if dropped != 0 || servedText != "" || servedHits != 0 {
		t.Fatalf("raw passthrough: dropped=%d servedText=%q servedHits=%d, want 0/\"\"/0", dropped, servedText, servedHits)
	}
}

// TestToolGuardFailLoud proves an explicit but unparsable ToolGuard value fails
// New at boot rather than silently raw-passing or silently arming.
func TestToolGuardFailLoud(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	_, err := New(Config{EngineID: "mock", Model: "m", ToolGuard: "maybe"})
	if err == nil {
		t.Fatal("New with unparsable ToolGuard must fail loud")
	}
	if !strings.Contains(err.Error(), "tool guard") {
		t.Fatalf("error must name the tool guard boundary: %v", err)
	}
}

// TestToolGuardRawResponsesRestoreContinuation closes the third interception
// seam (#14327 cross-validation finding 1): the buffered /v1/responses restore
// continuation is fak-owned serving, but its kernel folds (the Decide fold on
// the restore call and the result-admission fold on the restored bytes) are
// guard processing. With hostile rungs registered — an everything-Defer
// adjudicator (DEFAULT_DENY via the fold) and an everything-quarantine result
// admitter — raw mode must STILL deliver the restored bytes: both folds
// provably never ran.
func TestToolGuardRawResponsesRestoreContinuation(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model", ToolGuard: ToolGuardOff})
	abi.RegisterAdjudicator(1, deferReadAdj{})
	abi.RegisterResultAdmitter(0, quarantineAdmitter{})
	const trace = "toolguard-raw-restore"
	const body = "restorable evidence under raw pass-through"
	id := ctxplan.Digest([]byte(body))
	srv.stashRestore(trace, id, "evidence", []byte(body))
	planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{
		toolCallTurn("restore", "fak_context_restore", fmt.Sprintf(`{"id":%q}`, id)),
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}},
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the original task."})
	if planner.turn != 2 {
		t.Fatalf("raw restore did not continue to the next sample: samples=%d resp=%+v", planner.turn, resp)
	}
	sawRestored := false
	for _, msg := range planner.capturedMsgs[1] {
		if msg.Role == agent.RoleTool && msg.ToolCallID == "restore" && strings.Contains(msg.Content, "restorable evidence") {
			sawRestored = true
		}
	}
	if !sawRestored {
		t.Fatalf("raw mode did not deliver the restored bytes to the model (a kernel fold ran): %+v", planner.capturedMsgs[1])
	}
}

// TestToolGuardArmedResponsesRestoreStillScreens is the armed mirror of the raw
// witness: the SAME hostile rungs with the guard armed (the default) refuse the
// restore — the Decide fold DEFAULT_DENYs the call and the restored bytes never
// reach the model. Locks that the raw-mode skip above is a posture change, not
// a hole punched in the armed floor.
func TestToolGuardArmedResponsesRestoreStillScreens(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{EngineID: "test", Model: "test-model"})
	abi.RegisterAdjudicator(1, deferReadAdj{})
	abi.RegisterResultAdmitter(0, quarantineAdmitter{})
	const trace = "toolguard-armed-restore"
	const body = "restorable evidence the armed floor must withhold"
	id := ctxplan.Digest([]byte(body))
	srv.stashRestore(trace, id, "evidence", []byte(body))
	planner := &scriptedMultiTurnPlanner{turns: []*agent.Completion{
		toolCallTurn("restore", "fak_context_restore", fmt.Sprintf(`{"id":%q}`, id)),
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}},
	}}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"input": "Finish the original task."})
	for _, msg := range planner.capturedMsgs[1] {
		if msg.Role == agent.RoleTool && msg.ToolCallID == "restore" {
			if strings.Contains(msg.Content, body) {
				t.Fatalf("armed floor delivered the restored bytes to the model: %q", msg.Content)
			}
			if !strings.Contains(msg.Content, `"error"`) {
				t.Fatalf("armed floor neither withheld nor explained the restore: %q", msg.Content)
			}
		}
	}
	_ = resp
}
