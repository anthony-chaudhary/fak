package guardedremotemaint

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
	"github.com/anthony-chaudhary/fak/internal/vdso"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

const canonicalTool = "fak_remote_maintenance"

var errMCPToolCall = errors.New("guarded remote maintenance: MCP tool call failed")

// HostConfig binds one finite host authority to one non-human agent session.
// AgentID is authenticated attribution; it never represents user approval.
type HostConfig struct {
	Authority remotemaint.Authority
	AgentID   string
}

// Outcome is the gateway's bounded account of authorization and any attempted
// effect. Effect is one of not_attempted, completed, rolled_back, or
// attempted_unknown. Receipt is present only when the gateway returned a
// validated, result-admitted maintenance receipt (including its safe unknown
// substitution after a post-effect result denial).
type Outcome struct {
	Authorized  bool
	Verdict     string
	Reason      string
	DecidedBy   string
	Disposition string
	TraceID     string
	Effect      string
	Receipt     *remotemaint.Receipt
}

// AgentSession owns an in-process gateway and its unexported credentials.
type AgentSession struct {
	mu        sync.RWMutex
	server    *gateway.Server
	handler   http.Handler
	apiBearer string
	mcpBearer string
	closed    bool
}

// NewAgentSession constructs and authenticates a finite remote-maintenance MCP
// session. It performs no maintenance operation.
func NewAgentSession(cfg HostConfig) (*AgentSession, error) {
	agentID := strings.TrimSpace(cfg.AgentID)
	if cfg.Authority == nil {
		return nil, errors.New("guarded remote maintenance: authority is required")
	}
	if agentID == "" || agentID != cfg.AgentID || len(agentID) > 256 {
		return nil, errors.New("guarded remote maintenance: invalid agent id")
	}
	apiBearer, err := mintBearer()
	if err != nil {
		return nil, fmt.Errorf("guarded remote maintenance: mint API bearer: %w", err)
	}
	mcpBearer, err := mintBearer()
	if err != nil {
		return nil, fmt.Errorf("guarded remote maintenance: mint MCP bearer: %w", err)
	}
	traceID, err := mintTraceID()
	if err != nil {
		return nil, fmt.Errorf("guarded remote maintenance: mint trace identity: %w", err)
	}
	chain, err := guardedChain(agentID)
	if err != nil {
		return nil, err
	}
	srv, err := gateway.New(gateway.Config{
		EngineID:         "inkernel",
		Model:            "guarded-remote-maintenance",
		RequireKey:       apiBearer,
		AllowLAN:         false,
		MCPSessionBearer: mcpBearer,
		MCPAgentID:       agentID,
		DefaultTraceID:   traceID,
		ExposeTools:      []string{canonicalTool},
		Adjudicators:     chain,
		VDSO:             false,
	})
	if err != nil {
		return nil, fmt.Errorf("guarded remote maintenance: gateway: %w", err)
	}
	srv.SetRemoteMaintenanceAuthority(finiteAuthority{Authority: cfg.Authority})
	s := &AgentSession{server: srv, handler: srv.Handler(), apiBearer: apiBearer, mcpBearer: mcpBearer}
	if err := s.initialize(context.Background()); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

// Execute sends one canonical request through the authenticated gateway HTTP
// and MCP layers. It never calls Authority directly.
func (s *AgentSession) Execute(ctx context.Context, req remotemaint.Request) (Outcome, error) {
	if s == nil {
		return Outcome{}, errors.New("guarded remote maintenance: nil session")
	}
	if ctx == nil {
		return Outcome{}, errors.New("guarded remote maintenance: nil context")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return Outcome{}, errors.New("guarded remote maintenance: invalid request")
	}
	if _, err := remotemaint.ParseRequest(raw); err != nil {
		return Outcome{}, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return Outcome{}, errors.New("guarded remote maintenance: session closed")
	}
	var wire gateway.SyscallResponse
	if err := s.callRPC(ctx, 2, "tools/call", map[string]any{
		"name": canonicalTool, "arguments": json.RawMessage(raw),
	}, &wire); err != nil {
		if errors.Is(err, errMCPToolCall) {
			return Outcome{Verdict: "UNKNOWN", Effect: "attempted_unknown"}, remotemaint.ErrUnknownEffect
		}
		return Outcome{}, err
	}
	out := Outcome{
		Authorized: wire.Verdict.Kind == "ALLOW", Verdict: wire.Verdict.Kind,
		Reason: wire.Verdict.Reason, DecidedBy: wire.Verdict.By,
		Disposition: wire.Verdict.Disposition, TraceID: wire.TraceID,
		Effect: "not_attempted",
	}
	if !out.Authorized {
		return out, remotemaint.ErrAuthority
	}
	if wire.Result == nil || strings.TrimSpace(wire.Result.Content) == "" {
		out.Effect = "attempted_unknown"
		return out, remotemaint.ErrUnknownEffect
	}
	var receipt remotemaint.Receipt
	if err := json.Unmarshal([]byte(wire.Result.Content), &receipt); err != nil {
		out.Effect = "attempted_unknown"
		return out, remotemaint.ErrUnknownEffect
	}
	out.Receipt = &receipt
	switch receipt.Outcome {
	case remotemaint.OutcomeApplied, remotemaint.OutcomeUnchanged:
		if !successfulReceipt(req, wire.Result, receipt) {
			out.Effect = "attempted_unknown"
			return out, remotemaint.ErrUnknownEffect
		}
		out.Effect = "completed"
		return out, nil
	case remotemaint.OutcomeRolledBack:
		out.Effect = "rolled_back"
		return out, fmt.Errorf("%w: effect rolled back", remotemaint.ErrAuthority)
	case remotemaint.OutcomeUnknownEffect:
		out.Effect = "attempted_unknown"
		return out, remotemaint.ErrUnknownEffect
	case remotemaint.OutcomeRollbackFailed:
		out.Effect = "attempted_unknown"
		return out, remotemaint.ErrRollbackFailed
	default:
		out.Effect = "attempted_unknown"
		return out, remotemaint.ErrUnknownEffect
	}
}

func successfulReceipt(req remotemaint.Request, result *gateway.ResultEnvelope, receipt remotemaint.Receipt) bool {
	if result == nil || result.Status != "OK" || result.Meta["remote_maintenance_authorization"] != "ALLOW" {
		return false
	}
	admission := result.Meta["remote_maintenance_result_admission"]
	if admission != "ALLOW" && admission != "DEFER" {
		return false
	}
	if receipt.Schema != remotemaint.ReceiptSchema || !receipt.Verified || receipt.ProfileID != req.ProfileID ||
		receipt.OperationDigest != req.OperationDigest {
		return false
	}
	digest, err := hex.DecodeString(receipt.GrantDigest)
	return err == nil && len(digest) == 32
}

// Close releases the embedded gateway and erases the facade's bearer copies.
func (s *AgentSession) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	s.apiBearer, s.mcpBearer = "", ""
	s.handler = nil
	if s.server != nil {
		s.server.Close()
		s.server = nil
	}
}

func (s *AgentSession) initialize(ctx context.Context) error {
	var result struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if err := s.callRPC(ctx, 1, "initialize", map[string]any{
		"protocolVersion": "2024-11-05",
		"clientInfo":      map[string]string{"name": "fak-guarded-remote-maintenance", "version": "1"},
	}, &result); err != nil {
		return err
	}
	if result.ProtocolVersion == "" {
		return errors.New("guarded remote maintenance: MCP initialization failed")
	}
	return nil
}

func (s *AgentSession) callRPC(ctx context.Context, id int, method string, params any, into any) error {
	body, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": id, "method": method, "params": params})
	if err != nil {
		return errors.New("guarded remote maintenance: encode MCP request")
	}
	req := httptest.NewRequest(http.MethodPost, "http://fak.invalid/mcp", bytes.NewReader(body)).WithContext(ctx)
	req.Header.Set("Authorization", "Bearer "+s.apiBearer)
	req.Header.Set("X-Fak-MCP-Session", "Bearer "+s.mcpBearer)
	req.Header.Set("X-Fak-Principal-Class", "self-model")
	rec := httptest.NewRecorder()
	s.handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		return fmt.Errorf("guarded remote maintenance: authenticated MCP refused (status %d)", rec.Code)
	}
	var envelope struct {
		Result json.RawMessage `json:"result"`
		Error  *struct {
			Code int `json:"code"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil || envelope.Error != nil || len(envelope.Result) == 0 {
		if method == "tools/call" {
			return errMCPToolCall
		}
		return errors.New("guarded remote maintenance: MCP request refused")
	}
	if method != "tools/call" {
		if err := json.Unmarshal(envelope.Result, into); err != nil {
			return errors.New("guarded remote maintenance: malformed MCP response")
		}
		return nil
	}
	var toolResult struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(envelope.Result, &toolResult); err != nil || len(toolResult.Content) != 1 || toolResult.Content[0].Type != "text" {
		return errMCPToolCall
	}
	if err := json.Unmarshal([]byte(toolResult.Content[0].Text), into); err != nil {
		return errMCPToolCall
	}
	return nil
}

func mintBearer() (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(raw), nil
}

// mintTraceID creates a non-secret policy-ledger namespace independently of
// both credentials. One session keeps one trace so its IFC state accumulates;
// separate sessions cannot collide on the gateway-local gw-N fallback.
func mintTraceID() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	return "guarded-rm-" + hex.EncodeToString(raw), nil
}

func guardedChain(agentID string) ([]abi.Adjudicator, error) {
	registered := abi.Adjudicators()
	chain := make([]abi.Adjudicator, 0, len(registered)+1)
	replaced := 0
	for _, rung := range registered {
		if policy, ok := rung.(*adjudicator.Adjudicator); ok && policy == adjudicator.Default {
			chain = append(chain, adjudicator.New(strictPolicy()))
			replaced++
			continue
		}
		chain = append(chain, rung)
	}
	if replaced != 1 {
		return nil, fmt.Errorf("guarded remote maintenance: authoritative policy registrations=%d, want 1", replaced)
	}
	return append(chain, authenticatedAgentGate{agentID: agentID}), nil
}

func strictPolicy() adjudicator.Policy {
	p := adjudicator.DefaultPolicy()
	p.Posture = adjudicator.PostureFailClosed
	p.Allow = map[string]bool{canonicalTool: true}
	p.AllowPrefix = nil
	p.Complain = nil
	p.AdvisoryReasons = nil
	p.OperatorApprovals = nil
	p.Profile = nil
	p.SelfModifyGlobs = append(p.SelfModifyGlobs, ".git/", ".fak/guard/", ".hermes/")
	p.BlockedPathGlobs = append(p.BlockedPathGlobs, ".env", ".aws/", ".npmrc", ".netrc", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa", "/etc/", ".credentials.json", ".gitconfig", ".bashrc", ".bash_profile", ".zshrc", ".profile", "gh/hosts.yml", ".pypirc", ".docker/config.json", ".kube/config", ".terraformrc")
	return p
}

type authenticatedAgentGate struct{ agentID string }

func (g authenticatedAgentGate) Caps() []abi.Capability { return nil }
func (g authenticatedAgentGate) Tools() []string        { return []string{canonicalTool} }
func (g authenticatedAgentGate) Adjudicate(_ context.Context, call *abi.ToolCall) abi.Verdict {
	if call != nil && call.Tool == canonicalTool && call.Meta[vdso.MetaPrincipal] == g.agentID {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "guardedremotemaint.authenticated-agent"}
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonTrustViolation, By: "guardedremotemaint.authenticated-agent"}
}

type finiteAuthority struct{ remotemaint.Authority }
