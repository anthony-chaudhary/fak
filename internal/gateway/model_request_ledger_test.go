package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/promptaudit"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

func TestNativeModelRequestReconstructsExactPlannerBoundaryAfterReopen(t *testing.T) {
	dir := t.TempDir()
	ledger, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("Open ledger: %v", err)
	}
	oldOpen := openNativeModelRequestLedger
	openNativeModelRequestLedger = func() (*sessionledger.Ledger, error) { return ledger, nil }
	t.Cleanup(func() { openNativeModelRequestLedger = oldOpen })

	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterCapability(a2achan.CapA2ASend)
	abi.RegisterCapability(a2achan.CapA2ARecv)
	srv, err := New(Config{
		EngineID:       "localtools",
		Model:          "record-model",
		VDSO:           true,
		Native:         true,
		NativeMaxTurns: 4,
		DecideSession: func(_ context.Context, _ string) SessionVerdict {
			return SessionVerdict{Proceed: true, MaxTokens: 64}
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	recorder := &nativeWireRecorder{}
	srv.planner = recorder
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	const trace = "native-wire-trace"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	if v := a2achan.Default.Send(context.Background(), "operator", key, a2achan.Shared([]byte("switch to the receipt plan")), a2achan.CapA2ASend); v.Kind != abi.VerdictAllow {
		t.Fatalf("enqueue steer: %v", v.Kind)
	}
	t.Cleanup(func() {
		for {
			if _, _, ok := a2achan.Default.TryRecv(context.Background(), key, a2achan.CapA2ARecv); !ok {
				return
			}
		}
	})

	status, body := postNativeMessages(t, ts, map[string]any{
		"model":      "record-model",
		"max_tokens": 64,
		"messages": []map[string]any{
			{"role": "user", "content": "look up order A-1"},
		},
		"tools": wireTools(),
	})
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", status, body)
	}
	boundaryMessages, boundaryTools := recorder.firstTurn(t)

	reopened, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	rebuilt, receipt, err := reopened.ReconstructModelRequest(trace, "")
	if err != nil {
		t.Fatalf("reconstruct native request: %v", err)
	}
	wire := modelRequestFromPlannerBoundary(t, boundaryMessages, boundaryTools)
	if err := sessionledger.VerifyModelRequest(wire, rebuilt); err != nil {
		t.Fatalf("ledger reconstruction differs from planner boundary: %v", err)
	}
	wireDigest, err := sessionledger.CanonicalModelRequestDigest(wire)
	if err != nil {
		t.Fatalf("digest planner boundary: %v", err)
	}
	if receipt.RequestSHA256 != wireDigest {
		t.Fatalf("receipt digest = %q, planner-boundary digest = %q", receipt.RequestSHA256, wireDigest)
	}
	if receipt.RequestID == "" || rebuilt.Identity.RequestID != receipt.RequestID || receipt.LedgerEntry == "" {
		t.Fatalf("durable request identity is incomplete: receipt=%+v request=%+v", receipt, rebuilt.Identity)
	}

	var sawSystem, sawUser, sawSteer bool
	for _, segment := range rebuilt.Segments {
		var message agent.Message
		if err := json.Unmarshal(segment.Content, &message); err != nil {
			t.Fatalf("decode reconstructed message: %v", err)
		}
		sawSystem = sawSystem || message.Role == agent.RoleSystem
		sawUser = sawUser || message.Content == "look up order A-1"
		sawSteer = sawSteer || (segment.Kind == "injected_directive" && strings.Contains(message.Content, "switch to the receipt plan"))
	}
	if !sawSystem || !sawUser || !sawSteer {
		t.Fatalf("reconstructed segments missing system=%v user=%v injected-steer=%v: %+v", sawSystem, sawUser, sawSteer, rebuilt.Segments)
	}
	wantTools, _ := json.Marshal(boundaryTools)
	if !bytes.Equal(rebuilt.Tools, wantTools) {
		t.Fatalf("reconstructed tools = %s, planner-boundary tools = %s", rebuilt.Tools, wantTools)
	}
	t.Logf("native model request reconstructed after reopen: request=%s prompt=%s", receipt.RequestSHA256, receipt.PromptSHA256)
}

func TestNativeModelRequestReceiptFailureStopsBeforePlanner(t *testing.T) {
	oldOpen := openNativeModelRequestLedger
	openNativeModelRequestLedger = func() (*sessionledger.Ledger, error) {
		return nil, errors.New("receipt store unavailable")
	}
	t.Cleanup(func() { openNativeModelRequestLedger = oldOpen })

	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	srv, err := New(Config{
		EngineID: "localtools", Model: "record-model", VDSO: true, Native: true, NativeMaxTurns: 1,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	recorder := &nativeWireRecorder{}
	srv.planner = recorder
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	status, _ := postNativeMessages(t, ts, map[string]any{
		"model": "record-model", "max_tokens": 64,
		"messages": []map[string]any{{"role": "user", "content": "must not reach the model"}},
	})
	if status != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 when receipt persistence fails", status)
	}
	if calls := recorder.calls(); calls != 0 {
		t.Fatalf("planner calls = %d, want zero after receipt failure", calls)
	}
}

func modelRequestFromPlannerBoundary(t *testing.T, messages []agent.Message, tools []agent.ToolDef) sessionledger.ModelRequest {
	t.Helper()
	segments := make([]sessionledger.ModelRequestSegment, 0, len(messages))
	for _, message := range messages {
		raw, err := json.Marshal(message)
		if err != nil {
			t.Fatalf("marshal planner message: %v", err)
		}
		segments = append(segments, sessionledger.ModelRequestSegment{
			Kind: "message", Source: promptaudit.SourceUnknown, Content: raw,
		})
	}
	rawTools, err := json.Marshal(tools)
	if err != nil {
		t.Fatalf("marshal planner tools: %v", err)
	}
	return sessionledger.ModelRequest{
		Identity: sessionledger.ModelRequestIdentity{
			Model:     "record-model",
			Turn:      1,
			MaxTokens: 64,
		},
		Segments: segments,
		Tools:    rawTools,
	}
}
