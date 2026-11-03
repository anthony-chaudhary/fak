package gateway

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func causalGatewayFixture(t *testing.T) (Config, *causalEmbeddingProvider) {
	t.Helper()
	bos := 3
	m := model.NewSynthetic(model.Config{ModelType: "llama", HiddenSize: 8, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
		IntermediateSize: 12, VocabSize: 4, RMSNormEps: 1e-5, RopeTheta: 10000, MaxPositionEmbeddings: 32, BOSTokenID: &bos})
	tok, err := tokenizer.FromGGML([]string{"a", "b", "ab", "<s>"}, []string{"a b"}, []int32{1, 1, 1, 3}, "gpt2")
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Model: "tiny-causal", InKernelModel: m, Tokenizer: tok, EmbeddingPooling: "mean"}
	p, err := newCausalEmbeddingProvider(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, p
}

// fak-test:runtime fast est=200ms lane=default
func TestCausalEmbeddingOrdinaryHandlerInputsAndBindings(t *testing.T) {
	cfg, p := causalGatewayFixture(t)
	s := &Server{model: cfg.Model, causalEmbeddings: p, metrics: newGatewayMetrics(time.Now())}
	run := func(body string) (int, EmbeddingsResponse, http.Header) {
		t.Helper()
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body))
		s.handleEmbeddings(w, r)
		var out EmbeddingsResponse
		if w.Code == http.StatusOK {
			if err := json.Unmarshal(w.Body.Bytes(), &out); err != nil {
				t.Fatal(err)
			}
		}
		return w.Code, out, w.Header()
	}
	code, text, header := run(`{"model":"tiny-causal","input":"ab"}`)
	if code != 200 || text.Usage.PromptTokens != 2 || len(text.Data) != 1 || header.Get("X-Fak-Tokenizer-SHA256") != cfg.Tokenizer.Identity() || header.Get("X-Fak-Artifact-Identity") != "unknown" {
		t.Fatalf("text %d %+v %v", code, text, header)
	}
	code, ids, _ := run(`{"input":[3,2]}`)
	if code != 200 || ids.Usage.TotalTokens != 2 || !reflect.DeepEqual(text.Data, ids.Data) {
		t.Fatal("actual IDs differ from identical encoded text")
	}
	code, batch, _ := run(`{"input":[[3,2],[1]],"encoding_format":"base64","dimensions":8}`)
	if code != 200 || batch.Usage.TotalTokens != 3 || len(batch.Data) != 2 {
		t.Fatalf("batch %d %+v", code, batch)
	}
	for _, item := range batch.Data {
		v, ok := item.Embedding.(string)
		if !ok {
			t.Fatal("not base64")
		}
		b, err := base64.StdEncoding.DecodeString(v)
		if err != nil || len(b) != 8*4 {
			t.Fatal("bad base64 vector")
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingRequestRefusals(t *testing.T) {
	_, p := causalGatewayFixture(t)
	s := &Server{model: p.modelName, causalEmbeddings: p, metrics: newGatewayMetrics(time.Now())}
	for _, body := range []string{
		`{"input":[]}`, `{"input":[null]}`, `{"input":[1.5]}`, `{"input":[-1]}`, `{"input":[4]}`,
		`{"input":[[1],"a"]}`, `{"input":"ab","model":"wrong"}`, `{"input":"ab","dimensions":3}`,
		`{"input":"ab","encoding_format":"wrong"}`,
	} {
		w := httptest.NewRecorder()
		s.handleEmbeddings(w, httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(body)))
		if w.Code == 200 {
			t.Fatalf("accepted %s", body)
		}
	}
	p.gate <- struct{}{}
	w := httptest.NewRecorder()
	s.handleEmbeddings(w, httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"input":"ab"}`)))
	<-p.gate
	if w.Code != http.StatusTooManyRequests {
		t.Fatalf("busy %d", w.Code)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w = httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"input":"ab"}`)).WithContext(ctx)
	s.handleEmbeddings(w, r)
	if w.Code != http.StatusRequestTimeout {
		t.Fatalf("cancel %d", w.Code)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingEPRefusal(t *testing.T) {
	cfg, _ := causalGatewayFixture(t)
	cfg.ExpertParallelRanks = 2
	if _, err := newCausalEmbeddingProvider(cfg); err == nil {
		t.Fatal("collective-unaware route selected")
	}
	cfg.ExpertParallelRanks = 0
	cfg.InKernelQ4K = true
	if _, err := newCausalEmbeddingProvider(cfg); err == nil {
		t.Fatal("quantized route selected")
	}
	cfg.InKernelQ4K = false
	cfg.BaseURL = "http://unused.invalid"
	if _, err := newCausalEmbeddingProvider(cfg); err == nil {
		t.Fatal("proxy route selected")
	}
	cfg.EmbeddingPooling = ""
	if p, err := newCausalEmbeddingProvider(cfg); err != nil || p != nil {
		t.Fatal("unset selection changed legacy")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingHashDefaultPreserved(t *testing.T) {
	s := &Server{model: "hash"}
	w := httptest.NewRecorder()
	s.handleEmbeddings(w, httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"input":"a b","dimensions":3}`)))
	if w.Code != 200 || w.Header().Get("X-Fak-Embedding-Mode") != "" {
		t.Fatal("hash default changed")
	}
	var response EmbeddingsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	want, n := embedText("a b", 3)
	got, _ := json.Marshal(response.Data[0].Embedding)
	expected, _ := json.Marshal(want)
	if string(got) != string(expected) || response.Usage.TotalTokens != n {
		t.Fatal("hash output changed")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingDeviceOnlyPolicyRefused(t *testing.T) {
	cfg, _ := causalGatewayFixture(t)
	cfg.InKernelPlanner.RequireDeviceExecution = true
	if _, err := newCausalEmbeddingProvider(cfg); err == nil {
		t.Fatal("CPU embedding bypassed device-only policy")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestCausalEmbeddingSessionAndExactAdmission(t *testing.T) {
	_, p := causalGatewayFixture(t)
	s := &Server{model: p.modelName, causalEmbeddings: p, metrics: newGatewayMetrics(time.Now())}
	call := func() int {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/v1/embeddings", strings.NewReader(`{"input":[3,2]}`))
		r.Header.Set("X-Trace-Id", "embedding-governance")
		s.handleEmbeddings(w, r)
		return w.Code
	}
	s.observeSession = func(ctx context.Context, id string) SessionState { return SessionState{TraceID: id, Run: "paused"} }
	if code := call(); code != http.StatusConflict {
		t.Fatalf("paused model work accepted: %d", code)
	}
	s.observeSession = nil
	policy := DefaultAdmissionPolicy()
	policy.TokenBudget = 1
	ctl := NewAdmissionController(policy)
	s.SetAdmissionController(ctl)
	if code := call(); code == http.StatusOK {
		t.Fatal("exact two-token input escaped one-token budget")
	}
	if st := ctl.Stats(); st.Running != 0 || st.Waiting != 0 {
		t.Fatalf("refusal leaked %+v", st)
	}
	policy.TokenBudget = 2
	ctl = NewAdmissionController(policy)
	s.SetAdmissionController(ctl)
	var debited int
	s.debitSession = func(ctx context.Context, id string, u SessionUsage) SessionState {
		debited += u.ContextTokens
		return SessionState{TraceID: id, Run: "running"}
	}
	if code := call(); code != http.StatusOK {
		t.Fatalf("exact admitted input refused: %d", code)
	}
	if debited != 2 {
		t.Fatalf("usage debit %d, want actual IDs 2", debited)
	}
	if st := ctl.Stats(); st.Running != 0 || st.Waiting != 0 {
		t.Fatalf("success leaked %+v", st)
	}
	// Direct typed gate cancellation must also leave no slot/token reservation.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	lease, err := s.beginCausalEmbeddingAdmission(ctx, servedSessionTurn{traceID: "canceled"}, 2)
	if err == nil {
		lease.Release()
		t.Fatal("canceled admission accepted")
	}
}
