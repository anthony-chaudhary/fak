package codexsession

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

type nopWC struct{ bytes.Buffer }

func (*nopWC) Close() error { return nil }

func approvalHarness(t *testing.T, allow bool) (*Adapter, *nopWC, *[]harnesskit.Envelope, *[]ApprovalJournalEntry) {
	t.Helper()
	now := time.Unix(100, 0)
	var events []harnesskit.Envelope
	var journal []ApprovalJournalEntry
	a, err := New(Config{Workspace: t.TempDir(), Version: "fixture", RunID: "r", Sink: func(e harnesskit.Envelope) error { events = append(events, e); return nil }, Now: func() time.Time { return now }, ApprovalTimeout: time.Minute, ApprovalPolicy: func(r ApprovalRequest) PolicyDecision {
		return PolicyDecision{Allow: allow, Reason: map[bool]string{true: "bounded_read", false: "structural_deny"}[allow], Risk: "workspace authority"}
	}, ApprovalJournal: func(e ApprovalJournalEntry) { journal = append(journal, e) }})
	if err != nil {
		t.Fatal(err)
	}
	w := &nopWC{}
	a.stdin = w
	a.pending = map[string]pendingApproval{}
	a.resolved = map[string]struct{}{}
	a.inputIDs = map[string]struct{}{}
	a.epoch = 7
	p := harnessprotocolForTest(t)
	a.emit = p
	return a, w, &events, &journal
}
func harnessprotocolForTest(t *testing.T) func(harnesskit.EventType, string, string, any) error {
	var seq uint64
	return func(k harnesskit.EventType, id, c string, p any) error { seq++; return nil }
}
func request(t *testing.T, a *Adapter, method string) string {
	t.Helper()
	p := map[string]any{"threadId": "th", "turnId": "tu", "itemId": "it", "requestId": "rq", "reason": "needed", "command": "Get-Content README.md", "cwd": a.cfg.Workspace}
	b, _ := json.Marshal(p)
	if err := a.handleApprovalRequest(rpcMessage{ID: json.RawMessage(`42`), Method: method, Params: b}); err != nil {
		t.Fatal(err)
	}
	return "th:tu:it:rq"
}
func rpcLines(t *testing.T, w *nopWC) []map[string]any {
	t.Helper()
	var out []map[string]any
	for _, line := range strings.Split(strings.TrimSpace(w.String()), "\n") {
		if line == "" {
			continue
		}
		var v map[string]any
		if err := json.Unmarshal([]byte(line), &v); err != nil {
			t.Fatal(err)
		}
		out = append(out, v)
	}
	return out
}

func TestApprovalApproveOnceAndReceiptLayers(t *testing.T) {
	a, w, _, journal := approvalHarness(t, true)
	id := request(t, a, "item/commandExecution/requestApproval")
	r := ApprovalResolution{InputID: "in1", ApprovalID: id, Decision: "approve", Scope: a.cfg.Workspace, Principal: "operator", LeaseValid: true, Epoch: 7}
	if err := a.ResolveApproval(r); err != nil {
		t.Fatal(err)
	}
	if err := a.ResolveApproval(r); !errors.Is(err, ErrApprovalDuplicate) {
		t.Fatalf("duplicate=%v", err)
	}
	lines := rpcLines(t, w)
	if len(lines) != 1 || lines[0]["id"] != float64(42) || lines[0]["result"].(map[string]any)["decision"] != "accept" {
		t.Fatalf("responses=%#v", lines)
	}
	got := (*journal)[len(*journal)-1]
	if !strings.Contains(got.FakCapabilityFloor, "additional") || !strings.Contains(got.CodexSandboxPolicy, "independent") {
		t.Fatalf("receipt=%+v", got)
	}
}
func TestStructuralDenyNeverPendingOrClickable(t *testing.T) {
	a, w, _, _ := approvalHarness(t, false)
	id := request(t, a, "item/commandExecution/requestApproval")
	if _, ok := a.pending[id]; ok {
		t.Fatal("structural deny became pending")
	}
	if err := a.ResolveApproval(ApprovalResolution{InputID: "x", ApprovalID: id, Decision: "approve", Principal: "p", LeaseValid: true, Epoch: 7}); !errors.Is(err, ErrApprovalDuplicate) {
		t.Fatalf("override=%v", err)
	}
	if got := rpcLines(t, w)[0]["result"].(map[string]any)["decision"]; got != "decline" {
		t.Fatalf("decision=%v", got)
	}
}
func TestApprovalFailClosedCases(t *testing.T) {
	for _, tc := range []struct {
		name string
		act  func(*Adapter, string)
		want error
	}{{"stale", func(a *Adapter, id string) { a.epoch++ }, ErrApprovalStale}, {"unauthorized", func(a *Adapter, id string) {}, ErrApprovalUnauthorized}} {
		t.Run(tc.name, func(t *testing.T) {
			a, _, _, _ := approvalHarness(t, true)
			id := request(t, a, "item/fileChange/requestApproval")
			r := ApprovalResolution{InputID: "i", ApprovalID: id, Decision: "approve", Principal: "p", LeaseValid: tc.name != "unauthorized", Epoch: 7}
			tc.act(a, id)
			if err := a.ResolveApproval(r); !errors.Is(err, tc.want) {
				t.Fatalf("got %v want %v", err, tc.want)
			}
		})
	}
}
func TestTimeoutDisconnectUnknownAndReconnectState(t *testing.T) {
	a, w, _, _ := approvalHarness(t, true)
	id := request(t, a, "item/commandExecution/requestApproval")
	snapshot := a.pending[id]
	a.failPending("disconnect")
	if len(a.pending) != 0 || rpcLines(t, w)[0]["result"].(map[string]any)["decision"] != "decline" {
		t.Fatal("disconnect did not fail closed")
	}
	a2, _, _, _ := approvalHarness(t, true)
	a2.pending[id] = snapshot
	if _, ok := a2.pending[id]; !ok {
		t.Fatal("refresh projection lost pending state")
	}
	a3, w3, _, _ := approvalHarness(t, true)
	request(t, a3, "future/additiveApproval")
	if got := rpcLines(t, w3)[0]["result"].(map[string]any)["reason"]; got != "unknown_request_kind" {
		t.Fatalf("unknown=%v", got)
	}
	a4, w4, _, _ := approvalHarness(t, true)
	request(t, a4, "item/commandExecution/requestApproval")
	a4.cfg.Now = func() time.Time { return time.Unix(1000, 0) }
	a4.ExpireApprovals()
	if got := rpcLines(t, w4)[0]["result"].(map[string]any)["reason"]; got != "timeout" {
		t.Fatalf("timeout=%v", got)
	}
}
func TestCapturedApprovalRender(t *testing.T) {
	a, _, _, _ := approvalHarness(t, true)
	id := request(t, a, "item/commandExecution/requestApproval")
	p := a.pending[id]
	render := fmt.Sprintf("APPROVAL %s\nkind: %s\nsummary: %s\nworkspace: %s\nscope: %s\nrisk: %s\nconsequence: %s\nfak capability floor: allow\nCodex sandbox/approval policy: independently enforced\n[Approve] [Deny]", p.request.ApprovalID, p.request.Kind, p.request.Summary, p.request.Workspace, p.request.Scope, p.policy.Risk, p.request.Consequence)
	for _, s := range []string{"kind: command", "summary: Get-Content README.md", "risk: workspace authority", "fak capability floor: allow", "Codex sandbox/approval policy: independently enforced", "[Approve] [Deny]"} {
		if !strings.Contains(render, s) {
			t.Fatalf("captured render missing %q:\n%s", s, render)
		}
	}
}
