package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestFeatureProofAppliedProducersAndReadSurfaces(t *testing.T) {
	engineID := "feature-proof/" + t.Name()
	abi.RegisterEngine(engineID, engine.MockEngine)
	s, err := New(Config{EngineID: engineID, Model: "claude-test", Provider: "anthropic", BaseURL: "http://127.0.0.1:1", CompactHistoryBudget: 120, ElideResultBytes: 2048, RequireKey: "snapshot-admin-secret"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	s.readBearer = "snapshot-read-secret"
	for _, tc := range []struct {
		feature ServeFeature
		raw     []byte
	}{
		{FeatureCompactHistory, compactWireBody(t, 20)},
		{FeatureElideResults, elideWireBody(t)},
	} {
		req, err := agent.DecodeAnthropicMessagesRequest(tc.raw)
		if err != nil {
			t.Fatal(err)
		}
		before := append([]byte(nil), req.Raw...)
		if tc.feature == FeatureCompactHistory {
			fired, reason := s.compactAnthropicRawWithReason(req, 0, "")
			if !fired {
				t.Fatalf("compaction fixture did not fire: %s", reason)
			}
		} else {
			s.maybeElideAnthropicRaw(req)
		}
		if len(req.Raw) >= len(before) {
			t.Fatalf("%s producer did not shorten fixture", tc.feature)
		}
		snapshot := s.featureProofCollector().Snapshot()
		if len(snapshot.Receipts) == 0 {
			t.Fatalf("%s producer emitted no receipt", tc.feature)
		}
		last := snapshot.Receipts[len(snapshot.Receipts)-1]
		if last.Feature != tc.feature || last.Outcome != FeatureProofVerified || last.Rewrite == nil {
			t.Fatalf("applied producer missing verified receipt: %+v", last)
		}
		proof := last.Rewrite
		if proof.ShedBytes != len(before)-len(req.Raw) || proof.PreSHA256 != featureProofHash(before) || proof.PostSHA256 != featureProofHash(req.Raw) {
			t.Fatalf("receipt does not bind actual producer bytes: %+v", proof)
		}
	}
	want := s.featureProofCollector().Snapshot()
	handler := s.Handler()
	get := func(path, token string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		req.RemoteAddr = "203.0.113.8:40000"
		if token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		return rec
	}
	const endpoint = "/v1/fak/features/proof"
	if rec := get(endpoint, ""); rec.Code != http.StatusUnauthorized {
		t.Fatalf("proof endpoint unprotected: %d", rec.Code)
	}
	rec := get(endpoint, "snapshot-read-secret")
	if rec.Code != 200 || rec.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("proof read status/cache=%d/%q body=%s", rec.Code, rec.Header().Get("Cache-Control"), rec.Body.String())
	}
	var got FeatureProofSummary
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("proof endpoint differs from actual producer receipts: got=%+v want=%+v", got, want)
	}
	assertObservationPayloadFree(t, rec.Body.Bytes())
	obs, _ := getObservationSnapshot(t, s)
	var observed FeatureProofSummary
	decodeObservationData(t, obs.Sources.FeaturesProof, &observed)
	if !reflect.DeepEqual(observed, want) {
		t.Fatal("observation proof source disagrees with direct proof endpoint")
	}
	metrics := s.renderMetrics()
	for _, counter := range want.Counters {
		line := fmt.Sprintf("fak_feature_activations_total{feature=%q,outcome=%q} %d\n", counter.Feature, counter.Outcome, counter.Count)
		if !strings.Contains(metrics, line) {
			t.Fatalf("missing actual proof counter %s", line)
		}
		if counter.Outcome == FeatureProofVerified {
			line = fmt.Sprintf("fak_feature_saved_bytes_total{feature=%q} %d\n", counter.Feature, counter.SavedBytes)
			if !strings.Contains(metrics, line) {
				t.Fatalf("missing verified saved-byte counter %s", line)
			}
		}
	}
	post := httptest.NewRequest(http.MethodPost, endpoint, nil)
	post.Header.Set("Authorization", "Bearer snapshot-admin-secret")
	postRec := httptest.NewRecorder()
	handler.ServeHTTP(postRec, post)
	if postRec.Code != http.StatusMethodNotAllowed || postRec.Header().Get("Allow") != "GET" {
		t.Fatalf("proof accepted mutation: %d", postRec.Code)
	}
	if !reflect.DeepEqual(s.featureProofCollector().Snapshot(), want) {
		t.Fatal("diagnostic reads changed proof evidence")
	}
}

func TestNativeCompactionObserverComposesActivationAndProof(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	m := model.NewSynthetic(kvmmuSynthCfg())
	m.Quantize()
	p := agent.NewInKernelPlanner(m, newByteLevelTokenizer(t), "native-compaction-proof", false, nil, false)
	p.SetPromptShrinkLevers(80, false, false)
	collector, err := NewFeatureProofCollector(4)
	if err != nil {
		t.Fatal(err)
	}

	activationCalls := 0
	ctx := agent.WithNativeCompactionObserver(context.Background(), func(value agent.NativeCompactionObservation) {
		if value.DroppedMessages <= 0 {
			t.Fatalf("activation received non-applied observation: %+v", value)
		}
		activationCalls++
	})
	ctx = agent.WithNativeCompactionObserver(ctx, collector.recordNativeCompaction)
	if _, err := p.Complete(ctx, nativeFeatureProofMessages(), nil, agent.WithMaxTokens(1)); err != nil {
		t.Fatal(err)
	}
	if activationCalls != 1 {
		t.Fatalf("activation observer calls = %d, want 1", activationCalls)
	}
	snapshot := collector.Snapshot()
	if len(snapshot.Receipts) != 1 {
		t.Fatalf("proof receipts = %d, want 1", len(snapshot.Receipts))
	}
	receipt := snapshot.Receipts[0]
	if receipt.Feature != FeatureCompactHistory || receipt.Outcome != FeatureProofVerified || receipt.Rewrite == nil {
		t.Fatalf("native compaction proof receipt = %+v", receipt)
	}
	proof := receipt.Rewrite
	if proof.SerializationMethod != "typed_messages_json" || proof.ProtectedRegionsKind != "shared_whole_messages" ||
		proof.PreBytes <= proof.PostBytes || proof.ShedBytes != proof.PreBytes-proof.PostBytes ||
		proof.PreTokens == nil || proof.PostTokens == nil || *proof.PreTokens <= *proof.PostTokens {
		t.Fatalf("native compaction proof lacks quantitative evidence: %+v", proof)
	}
	if proof.PrePrefixSHA256 != proof.PostPrefixSHA256 || proof.PreSuffixSHA256 != proof.PostSuffixSHA256 {
		t.Fatalf("native compaction proof lacks structural identity: %+v", proof)
	}
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "native-proof-secret-marker") {
		t.Fatalf("proof retained compacted payload: %s", raw)
	}
	assertObservationPayloadFree(t, raw)

	idleCtx := agent.WithNativeCompactionObserver(context.Background(), func(agent.NativeCompactionObservation) {
		activationCalls++
	})
	idleCtx = agent.WithNativeCompactionObserver(idleCtx, collector.recordNativeCompaction)
	if _, err := p.Complete(idleCtx, []agent.Message{{Role: agent.RoleUser, Content: "short"}}, nil, agent.WithMaxTokens(1)); err != nil {
		t.Fatal(err)
	}
	if activationCalls != 1 || len(collector.Snapshot().Receipts) != 1 {
		t.Fatalf("idle native request published evidence: activation=%d receipts=%d", activationCalls, len(collector.Snapshot().Receipts))
	}
}

func nativeFeatureProofMessages() []agent.Message {
	return []agent.Message{
		{Role: agent.RoleSystem, Content: "System prompt invariant instructions."},
		{Role: agent.RoleUser, Content: "Read the file first."},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r1", Type: "function", Function: agent.Func{Name: "Read", Arguments: `{"path":"large.txt"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r1", Content: strings.Repeat("native-proof-secret-marker line\n", 30)},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "e1", Type: "function", Function: agent.Func{Name: "Edit", Arguments: `{"path":"large.txt"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "e1", Content: "ok"},
		{Role: agent.RoleAssistant, Content: "File edited."},
		{Role: agent.RoleUser, Content: "Middle turn 1"},
		{Role: agent.RoleAssistant, Content: "Middle reply 1"},
		{Role: agent.RoleUser, Content: "Middle turn 2"},
		{Role: agent.RoleAssistant, Content: "Middle reply 2"},
		{Role: agent.RoleUser, Content: "Latest query: what is final state?"},
	}
}
