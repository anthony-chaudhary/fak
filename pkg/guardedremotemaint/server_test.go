package guardedremotemaint

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

type facadeAuthority struct {
	mu                     sync.Mutex
	grant                  remotemaint.Grant
	receipt                remotemaint.Receipt
	resolve, validate, run int
	requests               []remotemaint.Request
}

func (a *facadeAuthority) Resolve(_ context.Context, req remotemaint.Request) (remotemaint.Grant, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.resolve++
	a.requests = append(a.requests, req)
	return a.grant, nil
}

func (a *facadeAuthority) Validate(context.Context, remotemaint.Grant) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.validate++
	return nil
}

func (a *facadeAuthority) Execute(context.Context, remotemaint.Grant) (remotemaint.Receipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.run++
	return a.receipt, nil
}

func (a *facadeAuthority) counts() (int, int, int) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.resolve, a.validate, a.run
}

type facadeResultDeny struct{ active *atomic.Bool }

func (g facadeResultDeny) Tools() []string        { return []string{canonicalTool} }
func (g facadeResultDeny) Caps() []abi.Capability { return nil }
func (g facadeResultDeny) Admit(_ context.Context, _ *abi.ToolCall, result *abi.Result) abi.Verdict {
	if !g.active.Load() {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "guardedremotemaint-test-inactive"}
	}
	result.Payload = abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"untrusted":"must-not-leak"}`)}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "guardedremotemaint-test-result-deny"}
}

type facadeCallRecorder struct {
	active     *atomic.Bool
	mu         sync.Mutex
	principals []string
	results    int
	resultMeta map[string]string
}

func (r *facadeCallRecorder) Tools() []string        { return []string{canonicalTool} }
func (r *facadeCallRecorder) Caps() []abi.Capability { return nil }
func (r *facadeCallRecorder) Adjudicate(_ context.Context, call *abi.ToolCall) abi.Verdict {
	if r.active.Load() && call != nil {
		r.mu.Lock()
		r.principals = append(r.principals, call.Meta[vdso.MetaPrincipal])
		r.mu.Unlock()
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "guardedremotemaint-test-recorder"}
}

func (r *facadeCallRecorder) Admit(_ context.Context, _ *abi.ToolCall, result *abi.Result) abi.Verdict {
	if r.active.Load() && result != nil {
		r.mu.Lock()
		r.results++
		r.resultMeta = make(map[string]string, len(result.Meta))
		for key, value := range result.Meta {
			r.resultMeta[key] = value
		}
		r.mu.Unlock()
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "guardedremotemaint-test-recorder"}
}

func (r *facadeCallRecorder) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.principals...)
}

func (r *facadeCallRecorder) resultSnapshot() (int, map[string]string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.results, r.resultMeta
}

func facadeGrant(now time.Time) remotemaint.Grant {
	return remotemaint.Grant{
		ProfileID: "facade-profile", OperationDigest: strings.Repeat("1", 64), ContextID: strings.Repeat("2", 64),
		Target: "maintenance.example.invalid", HostKey: "SHA256:" + base64.RawStdEncoding.EncodeToString(make([]byte, 32)),
		Operation: remotemaint.OperationSystemdProfileInstallReload, PayloadSHA256: strings.Repeat("3", 64),
		Holder: "facade-agent", BootID: "boot-facade", PriorStateSHA256: strings.Repeat("4", 64),
		Paths:        []string{"/etc/systemd/system/fak.service"},
		PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "maintenance-stage."}},
		Units:        []string{"fak.service"},
		LeasePins:    []remotemaint.LeasePin{{NodeID: "node-1", ID: "lease-1", Epoch: 1, ExpiresAt: now.Add(10 * time.Minute).UTC()}},
		ExpiresAt:    now.Add(5 * time.Minute).UTC(),
	}
}

func facadeReceipt(g remotemaint.Grant, now time.Time) remotemaint.Receipt {
	return remotemaint.Receipt{
		Schema: remotemaint.ReceiptSchema, GrantDigest: remotemaint.GrantDigest(g), ProfileID: g.ProfileID,
		OperationDigest: g.OperationDigest, ContextID: g.ContextID, Target: g.Target, HostKey: g.HostKey,
		Operation: g.Operation, BootID: g.BootID, StartedAt: now.Add(-time.Millisecond).UTC(),
		EndedAt: now.UTC(), Outcome: remotemaint.OutcomeApplied, Verified: true,
		BackupDisposition: remotemaint.BackupNotRequired,
	}
}

func newFacadeAuthority() *facadeAuthority {
	now := time.Now().UTC()
	g := facadeGrant(now)
	return &facadeAuthority{grant: g, receipt: facadeReceipt(g, now)}
}

func facadeRPC(t *testing.T, s *AgentSession, apiBearer, mcpBearer, method string, params any) (int, []byte) {
	t.Helper()
	body, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 41, "method": method, "params": params})
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "http://fak.invalid/mcp", bytes.NewReader(body))
	if apiBearer != "" {
		req.Header.Set("Authorization", "Bearer "+apiBearer)
	}
	if mcpBearer != "" {
		req.Header.Set("X-Fak-MCP-Session", "Bearer "+mcpBearer)
	}
	req.Header.Set("X-Fak-Principal-Class", "self-model")
	rec := httptest.NewRecorder()
	s.handler.ServeHTTP(rec, req)
	return rec.Code, rec.Body.Bytes()
}

func facadeToolParams(name string, args any) map[string]any {
	return map[string]any{"name": name, "arguments": args}
}

func assertFacadeCounts(t *testing.T, a *facadeAuthority, resolve, validate, execute int) {
	t.Helper()
	r, v, e := a.counts()
	if r != resolve || v != validate || e != execute {
		t.Fatalf("authority calls resolve=%d validate=%d execute=%d, want %d/%d/%d", r, v, e, resolve, validate, execute)
	}
}

// fak-test:runtime fast est=1s lane=short
func TestGuardedRemoteMaintenanceFacadeKernelBoundary(t *testing.T) {
	const agentID = "source-agent-14032"

	t.Run("authenticated HTTP and MCP doors precede authority", func(t *testing.T) {
		a := newFacadeAuthority()
		s, err := NewAgentSession(HostConfig{Authority: a, AgentID: agentID})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		args := map[string]any{"profile_id": a.grant.ProfileID, "operation_digest": a.grant.OperationDigest}
		for _, tc := range []struct{ api, mcp string }{
			{"", s.mcpBearer}, {"wrong-api", s.mcpBearer}, {s.apiBearer, ""}, {s.apiBearer, "wrong-mcp"},
		} {
			facadeRPC(t, s, tc.api, tc.mcp, "tools/call", facadeToolParams(canonicalTool, args))
			assertFacadeCounts(t, a, 0, 0, 0)
		}
	})

	t.Run("tool surface hides shell and nested syscall", func(t *testing.T) {
		a := newFacadeAuthority()
		s, err := NewAgentSession(HostConfig{Authority: a, AgentID: agentID})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		status, body := facadeRPC(t, s, s.apiBearer, s.mcpBearer, "tools/list", map[string]any{})
		if status != http.StatusOK || !bytes.Contains(body, []byte(canonicalTool)) || bytes.Contains(body, []byte(`"Bash"`)) || bytes.Contains(body, []byte(`"fak_syscall"`)) {
			t.Fatalf("tools/list status=%d body=%s", status, body)
		}
		for _, hidden := range []string{"Bash", "fak_syscall"} {
			facadeRPC(t, s, s.apiBearer, s.mcpBearer, "tools/call", facadeToolParams(hidden, map[string]any{"command": "echo unsafe", "tool": canonicalTool}))
			assertFacadeCounts(t, a, 0, 0, 0)
		}
	})

	t.Run("invalid shapes and aliases never reach authority", func(t *testing.T) {
		a := newFacadeAuthority()
		s, err := NewAgentSession(HostConfig{Authority: a, AgentID: agentID})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		bad := []map[string]any{
			facadeToolParams(canonicalTool, map[string]any{"profile_id": a.grant.ProfileID, "operation_digest": a.grant.OperationDigest, "target": "attacker.invalid"}),
			facadeToolParams(canonicalTool+"_alias", map[string]any{"profile_id": a.grant.ProfileID, "operation_digest": a.grant.OperationDigest}),
			facadeToolParams(canonicalTool, map[string]any{"profile_id": a.grant.ProfileID, "operation_digest": "short"}),
		}
		for _, params := range bad {
			facadeRPC(t, s, s.apiBearer, s.mcpBearer, "tools/call", params)
			assertFacadeCounts(t, a, 0, 0, 0)
		}
	})

	for _, tc := range []struct {
		name   string
		mutate func(*remotemaint.Grant)
	}{
		{"credential scope", func(g *remotemaint.Grant) {
			g.Paths = []string{"/etc/systemd/system/id_ed25519"}
			g.Units = []string{"id_ed25519"}
		}},
		{"self configuration scope", func(g *remotemaint.Grant) { g.Paths = []string{"/srv/agent/.fak/guard/remote-maintenance.json"} }},
		{"malformed grant", func(g *remotemaint.Grant) { g.ProfileID = "different-profile" }},
	} {
		t.Run("resolved "+tc.name+" is denied before effect", func(t *testing.T) {
			a := newFacadeAuthority()
			tc.mutate(&a.grant)
			s, err := NewAgentSession(HostConfig{Authority: a, AgentID: agentID})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			_, _ = s.Execute(context.Background(), remotemaint.Request{ProfileID: "facade-profile", OperationDigest: strings.Repeat("1", 64)})
			_, _, executed := a.counts()
			if executed != 0 {
				t.Fatalf("unsafe resolved grant executed %d times", executed)
			}
		})
	}

	t.Run("canonical request crosses registered kernel as nonhuman agent", func(t *testing.T) {
		var active atomic.Bool
		active.Store(true)
		recorder := &facadeCallRecorder{active: &active}
		abi.RegisterAdjudicator(1<<30, recorder)
		abi.RegisterResultAdmitter(1<<30, recorder)
		t.Cleanup(func() { active.Store(false) })
		a := newFacadeAuthority()
		s, err := NewAgentSession(HostConfig{Authority: a, AgentID: agentID})
		if err != nil {
			t.Fatal(err)
		}
		out, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: a.grant.ProfileID, OperationDigest: a.grant.OperationDigest})
		if err != nil {
			results, meta := recorder.resultSnapshot()
			t.Fatalf("canonical default-floor refusal: outcome=%+v receipt=%+v err=%v admitted=%d meta=%v", out, out.Receipt, err, results, meta)
		}
		if !out.Authorized || out.Effect != "completed" || out.Receipt == nil || out.Receipt.GrantDigest != remotemaint.GrantDigest(a.grant) || out.TraceID == "" {
			t.Fatalf("outcome=%+v", out)
		}
		assertFacadeCounts(t, a, 1, 2, 1)
		if len(a.requests) != 1 || a.requests[0].ProfileID != a.grant.ProfileID {
			t.Fatalf("authority requests=%+v", a.requests)
		}
		active.Store(false)
		principals := recorder.snapshot()
		if len(principals) < 2 {
			t.Fatalf("recorded kernel calls=%v, want initial and resource decisions", principals)
		}
		for _, principal := range principals {
			if principal != agentID || principal == "human" {
				t.Fatalf("kernel call attributed to %q, want nonhuman agent %q", principal, agentID)
			}
		}
		call := &abi.ToolCall{Tool: canonicalTool, Meta: map[string]string{vdso.MetaPrincipal: agentID}}
		if got := (authenticatedAgentGate{agentID: agentID}).Adjudicate(context.Background(), call); got.Kind != abi.VerdictDefer {
			t.Fatalf("source agent attribution=%+v", got)
		}
		call.Meta[vdso.MetaPrincipal] = "human"
		if got := (authenticatedAgentGate{agentID: agentID}).Adjudicate(context.Background(), call); got.Kind != abi.VerdictDeny {
			t.Fatalf("human promotion accepted: %+v", got)
		}
		s.Close()
		if !s.closed || s.apiBearer != "" || s.mcpBearer != "" || s.handler != nil {
			t.Fatal("Close retained private credentials or handler")
		}
	})

	t.Run("post-effect result denial returns only bound unknown receipt", func(t *testing.T) {
		var active atomic.Bool
		active.Store(true)
		gate := facadeResultDeny{active: &active}
		abi.RegisterResultAdmitter(1<<30, gate)
		t.Cleanup(func() { active.Store(false) })
		a := newFacadeAuthority()
		s, err := NewAgentSession(HostConfig{Authority: a, AgentID: agentID})
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		out, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: a.grant.ProfileID, OperationDigest: a.grant.OperationDigest})
		active.Store(false)
		if !errors.Is(err, remotemaint.ErrUnknownEffect) || !out.Authorized || out.Effect != "attempted_unknown" || out.Receipt == nil || out.Receipt.Outcome != remotemaint.OutcomeUnknownEffect || out.Receipt.Verified {
			t.Fatalf("outcome=%+v err=%v", out, err)
		}
		if raw, _ := json.Marshal(out); bytes.Contains(raw, []byte("must-not-leak")) {
			t.Fatalf("denied result leaked: %s", raw)
		}
		assertFacadeCounts(t, a, 1, 2, 1)
		if out.Receipt.ProfileID != a.grant.ProfileID || out.Receipt.ContextID != a.grant.ContextID {
			t.Fatalf("unknown receipt lost bound identity: %+v", out.Receipt)
		}
	})
}
