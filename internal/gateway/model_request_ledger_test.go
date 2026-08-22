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
	"time"

	"github.com/anthony-chaudhary/fak/internal/a2achan"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/promptaudit"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

type blockedInputClaimPlanner struct {
	started chan struct{}
	release chan struct{}
	seen    [][]agent.Message
}

func (p *blockedInputClaimPlanner) Model() string { return "blocked-input-claim" }

func (p *blockedInputClaimPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.seen = append(p.seen, append([]agent.Message(nil), messages...))
	select {
	case p.started <- struct{}{}:
	default:
	}
	<-p.release
	return &agent.Completion{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "done"},
		Usage:   agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}, nil
}

func TestNativeInputClaimsPrecedeRequestsAcrossBlockedProvider(t *testing.T) {
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
		EngineID: "localtools", Model: "record-model", VDSO: true, Native: true, NativeMaxTurns: 1,
		DecideSession: func(context.Context, string) SessionVerdict {
			return SessionVerdict{Proceed: true, MaxTokens: 64}
		},
		SteerSession: func(ctx context.Context, traceID, principal, text string) error {
			key := a2achan.ChannelKey{Locale: a2achan.Session, ID: traceID}
			if v := a2achan.Default.Send(ctx, principal, key, a2achan.Shared([]byte(text)), a2achan.CapA2ASend); v.Kind != abi.VerdictAllow {
				return errors.New("steer refused by input bus")
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	planner := &blockedInputClaimPlanner{started: make(chan struct{}, 1), release: make(chan struct{})}
	t.Cleanup(func() {
		select {
		case <-planner.release:
		default:
			close(planner.release)
		}
	})
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	const trace = "native-wire-trace"
	key := a2achan.ChannelKey{Locale: a2achan.Session, ID: trace}
	t.Cleanup(func() {
		for {
			if _, _, ok := a2achan.Default.TryRecv(context.Background(), key, a2achan.CapA2ARecv); !ok {
				return
			}
		}
	})
	enqueue := func(payload string) {
		t.Helper()
		body, err := json.Marshal(SteerRequest{Text: payload})
		if err != nil {
			t.Fatalf("marshal steer %q: %v", payload, err)
		}
		response, err := http.Post(ts.URL+"/v1/fak/session/"+trace+"/steer", "application/json", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("post steer %q: %v", payload, err)
		}
		response.Body.Close()
		if response.StatusCode != http.StatusAccepted {
			t.Fatalf("steer %q status = %d, want 202", payload, response.StatusCode)
		}
	}
	enqueue("A")

	type response struct {
		status int
		body   []byte
	}
	first := make(chan response, 1)
	go func() {
		status, body := postNativeMessages(t, ts, map[string]any{
			"model": "record-model", "max_tokens": 64,
			"messages": []map[string]any{{"role": "user", "content": "request one"}},
		})
		first <- response{status: status, body: body}
	}()
	select {
	case <-planner.started:
	case <-time.After(5 * time.Second):
		t.Fatal("provider fixture did not block")
	}
	preDispatch, err := ledger.Chain(trace)
	if err != nil {
		t.Fatalf("read pre-dispatch ledger: %v", err)
	}
	var preKinds []string
	for _, entry := range preDispatch {
		preKinds = append(preKinds, entry.Kind)
	}
	steerRecords := sessionctl.ReadSteerNextRecords(trace)
	if len(steerRecords) != 1 || steerRecords[0].Move.Payload != "A" {
		t.Fatalf("pre-dispatch steer records = %+v; ledger kinds=%v", steerRecords, preKinds)
	}
	if len(preKinds) < 2 || preKinds[len(preKinds)-2] != sessionledger.KindInputClaimReceipt || preKinds[len(preKinds)-1] != sessionledger.KindModelRequestReceipt {
		t.Fatalf("pre-dispatch ledger kinds = %v, want input claim immediately before model request", preKinds)
	}
	enqueue("B")
	close(planner.release)
	result := <-first
	if result.status != http.StatusOK {
		t.Fatalf("request 1 status = %d; body=%s", result.status, result.body)
	}
	status, body := postNativeMessages(t, ts, map[string]any{
		"model": "record-model", "max_tokens": 64,
		"messages": []map[string]any{{"role": "user", "content": "request two"}},
	})
	if status != http.StatusOK {
		t.Fatalf("request 2 status = %d; body=%s", status, body)
	}
	secondSteerRecords := sessionctl.ReadSteerNextRecords(trace)
	if len(secondSteerRecords) != 1 || secondSteerRecords[0].Move.Payload != "B" {
		t.Fatalf("request 2 steer records = %+v, want B", secondSteerRecords)
	}

	reopened, err := sessionledger.Open(dir)
	if err != nil {
		t.Fatalf("reopen ledger: %v", err)
	}
	chain, err := reopened.Chain(trace)
	if err != nil {
		t.Fatalf("read reopened chain: %v", err)
	}
	var receiptKinds []string
	var claims []sessionledger.InputClaimReceipt
	var requests []sessionledger.ModelRequestReceipt
	for _, entry := range chain {
		if entry.Kind == "input_claim_receipt" || entry.Kind == sessionledger.KindModelRequestReceipt {
			receiptKinds = append(receiptKinds, entry.Kind)
		}
		switch entry.Kind {
		case sessionledger.KindInputClaimReceipt:
			var receipt sessionledger.InputClaimReceipt
			if err := json.Unmarshal(entry.Content, &receipt); err != nil {
				t.Fatalf("decode input claim receipt: %v", err)
			}
			claims = append(claims, receipt)
		case sessionledger.KindModelRequestReceipt:
			var receipt sessionledger.ModelRequestReceipt
			if err := json.Unmarshal(entry.Content, &receipt); err != nil {
				t.Fatalf("decode model request receipt: %v", err)
			}
			requests = append(requests, receipt)
		}
	}
	want := []string{"input_claim_receipt", sessionledger.KindModelRequestReceipt, "input_claim_receipt", sessionledger.KindModelRequestReceipt}
	if strings.Join(receiptKinds, ",") != strings.Join(want, ",") {
		t.Fatalf("claim/request receipt order after reopen = %v, want %v; planner=%+v", receiptKinds, want, planner.seen)
	}
	if len(claims) != 2 || len(requests) != 2 {
		t.Fatalf("reopened claims=%d requests=%d, want 2/2", len(claims), len(requests))
	}
	for i, wantInput := range []string{"A", "B"} {
		claim, claimReceipt, err := reopened.ReconstructInputClaim(trace, claims[i].ClaimID)
		if err != nil {
			t.Fatalf("reconstruct claim %d: %v", i+1, err)
		}
		if claimReceipt.State != sessionledger.InputClaimClaimed || len(claim.Inputs) != 1 {
			t.Fatalf("claim %d after reopen = %+v receipt=%+v", i+1, claim, claimReceipt)
		}
		var input agent.Message
		if err := json.Unmarshal(claim.Inputs[0], &input); err != nil {
			t.Fatalf("decode claim %d input: %v", i+1, err)
		}
		if input.Content != wantInput {
			t.Fatalf("claim %d input = %q, want exactly %q", i+1, input.Content, wantInput)
		}
		rebuiltRequest, requestReceipt, err := reopened.ReconstructModelRequest(trace, requests[i].RequestID)
		if err != nil {
			t.Fatalf("reconstruct request %d: %v", i+1, err)
		}
		if requestReceipt.InputClaim == nil || requestReceipt.InputClaim.ClaimID != claimReceipt.ClaimID ||
			requestReceipt.InputClaim.InputSHA256 != claimReceipt.InputSHA256 || requestReceipt.InputClaim.InputCount != claimReceipt.InputCount {
			t.Fatalf("request %d claim binding = %+v, claim receipt = %+v", i+1, requestReceipt.InputClaim, claimReceipt)
		}
		steerCounts := map[string]int{"A": 0, "B": 0}
		for _, segment := range rebuiltRequest.Segments {
			var message agent.Message
			if err := json.Unmarshal(segment.Content, &message); err != nil {
				t.Fatalf("decode request %d segment: %v", i+1, err)
			}
			if _, ok := steerCounts[message.Content]; ok {
				steerCounts[message.Content]++
			}
		}
		otherInput := []string{"B", "A"}[i]
		if steerCounts[wantInput] != 1 || steerCounts[otherInput] != 0 {
			t.Fatalf("request %d steer inputs = %v, want exactly %q", i+1, steerCounts, wantInput)
		}
	}
}

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
