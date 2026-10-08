package agent_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func v41ProjectionHTTPModel(t *testing.T) *model.Model {
	t.Helper()
	// The flat deepseek41 component config is the supported public constructor
	// envelope used by V41EngramProjectionSourceNamesAndCollision and GGUF fixtures.
	cfg := model.Config{
		ModelType: "deepseek41", NumLayers: 1, HiddenSize: 64,
		NumHeads: 2, NumKVHeads: 1, HeadDim: 32,
		QKNopeHeadDim: 16, QKRopeHeadDim: 16, QLoraRank: 32,
		OLoraRank: 16, OGroups: 2, MoEIntermediateSize: 32, VocabSize: 8,
		NumExperts: model.V41RouterExperts, NumExpertsPerTok: model.V41RouterTopK,
		NSharedExperts: 1, RoutedScalingFactor: 1.5, RopeTheta: 10000,
		RMSNormEps: 1e-6, EOSTokenID: -1,
		HCMult: 4, HCEps: 1e-6, HCSinkhornIters: 20, CompressRatios: []int{0},
	}
	if !cfg.IsDeepSeekV41() {
		t.Fatal("flat component fixture lost V41 identity")
	}
	var tensors []model.NamedTensorF32
	seed := uint64(0x9E3779B97F4A7C15)
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			seed = seed*6364136223846793005 + 1442695040888963407
			value := (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.1
			switch {
			case strings.HasSuffix(name, "norm.weight"), strings.HasSuffix(name, "mhc.scale"):
				value = 1
			case strings.HasSuffix(name, "mhc.base"):
				value = 0
			}
			data[i] = value
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	const H, I = 64, 32
	add("model.embed_tokens.weight", 8, H)
	add("lm_head.weight", 8, H)
	add("model.norm.weight", H)
	const prefix = "model.layers.0."
	add(prefix+"attn_norm.weight", H)
	add(prefix+"ffn_norm.weight", H)
	add(prefix+"mhc.mixes.weight", 24, H)
	add(prefix+"mhc.base", 24)
	add(prefix+"mhc.scale", 3)
	add(prefix+"attn.wq_a.weight", 32, H)
	add(prefix+"attn.wq_b.weight", 64, 32)
	add(prefix+"attn.wkv.weight", 32, H)
	add(prefix+"attn.wo_a.weight", 16, 64)
	add(prefix+"attn.wo_b.weight", H, 32)
	add(prefix+"attn.sink", 2)
	add(prefix+"ffn.gate.weight", model.V41RouterExperts, H)
	add(prefix+"ffn.gate.e_score_correction_bias", model.V41RouterExperts)
	for _, stem := range []string{"ffn.shared_experts"} {
		add(prefix+stem+".w1.weight", I, H)
		add(prefix+stem+".w3.weight", I, H)
		add(prefix+stem+".w2.weight", H, I)
	}
	for e := 0; e < model.V41RouterExperts; e++ {
		stem := fmt.Sprintf("%sffn.experts.%d", prefix, e)
		add(stem+".w1.weight", I, H)
		add(stem+".w3.weight", I, H)
		add(stem+".w2.weight", H, I)
	}
	m, err := model.NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// [SW-VERIFIED] Exercises the supported CPU model through ordinary serving.
// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime fast est=1s lane=default
func TestV41PhaseAttributionProjectionOrdinaryHTTP(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")
	t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
	m := v41ProjectionHTTPModel(t)
	const prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
	tok, err := tokenizer.FromGGML([]string{prompt, "a", "b", "c", "d", "e", "f", "g"}, []string{"a b"}, []int32{3, 1, 1, 1, 1, 1, 1, 1}, "gpt2")
	if err != nil {
		t.Fatal(err)
	}
	planner := agent.NewInKernelPlanner(m, tok, "native", false, nil, false)
	srv, err := gateway.New(gateway.Config{EngineID: "mock", Model: "native", Logf: func(string, ...any) {}})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	srv.SetPlanner(planner)
	keys := []string{
		"head_projection_device_calls", "head_projection_host_calls", "head_projection_device_rows", "head_projection_host_rows",
		"head_projection_activation_upload_bytes", "head_projection_readback_bytes", "head_projection_nanos",
		"engram_projection_device_calls", "engram_projection_host_calls", "engram_projection_device_rows", "engram_projection_host_rows",
		"engram_projection_matmul_calls", "engram_projection_activation_upload_bytes", "engram_projection_readback_bytes",
		"engram_projection_host_weight_f32_bytes", "engram_projection_nanos",
	}
	var previous map[string]map[string]int64
	for request := 0; request < 2; request++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
		req.RemoteAddr = "127.0.0.1:50101"
		req.Header.Set("Content-Type", "application/json")
		completion := httptest.NewRecorder()
		srv.Handler().ServeHTTP(completion, req)
		if completion.Code != http.StatusOK {
			t.Fatalf("completion status=%d body=%s", completion.Code, completion.Body.String())
		}
		var response struct {
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(completion.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		if response.Usage.CompletionTokens != 2 {
			t.Fatalf("completion tokens=%d want 2", response.Usage.CompletionTokens)
		}
		reader := httptest.NewRecorder()
		get := httptest.NewRequest(http.MethodGet, "/debug/vars", nil)
		get.RemoteAddr = "127.0.0.1:50102"
		srv.Handler().ServeHTTP(reader, get)
		if reader.Code != http.StatusOK {
			t.Fatalf("debug reader status=%d", reader.Code)
		}
		var doc struct {
			MoE struct {
				Requests   int64           `json:"requests"`
				Checkpoint json.RawMessage `json:"checkpoint"`
				V41        struct {
					Scope       string                     `json:"scope"`
					Prefill     map[string]json.RawMessage `json:"prefill"`
					Decode      map[string]json.RawMessage `json:"decode"`
					Projections struct {
						Prefill map[string]json.RawMessage `json:"prefill"`
						Decode  map[string]json.RawMessage `json:"decode"`
					} `json:"projections"`
				} `json:"v41_phases"`
			} `json:"moe_residency"`
		}
		if err := json.Unmarshal(reader.Body.Bytes(), &doc); err != nil {
			t.Fatal(err)
		}
		if doc.MoE.V41.Scope != "model_lifetime" || doc.MoE.Requests != 0 || len(doc.MoE.Checkpoint) != 0 {
			t.Fatal("ordinary CPU reader must expose phases independently of ring/checkpoint")
		}
		sourceJSON, err := json.Marshal(m.V41ExpertFaultAttribution())
		if err != nil {
			t.Fatal(err)
		}
		var source map[string]map[string]json.RawMessage
		if err := json.Unmarshal(sourceJSON, &source); err != nil {
			t.Fatal(err)
		}
		for phase, fields := range map[string]map[string]json.RawMessage{"prefill": doc.MoE.V41.Prefill, "decode": doc.MoE.V41.Decode} {
			if len(fields) != 31 {
				t.Fatalf("legacy served %s fields=%d want 31", phase, len(fields))
			}
			for _, key := range keys {
				if _, exists := fields[key]; exists {
					t.Fatalf("legacy served %s gained projection field %s", phase, key)
				}
			}
		}
		current := map[string]map[string]int64{}
		for phase, fields := range map[string]map[string]json.RawMessage{"prefill": doc.MoE.V41.Projections.Prefill, "decode": doc.MoE.V41.Projections.Decode} {
			if len(fields) != 16 {
				t.Fatalf("served projection %s fields=%d want 16", phase, len(fields))
			}
			current[phase] = map[string]int64{}
			for _, key := range keys {
				var got, want *int64
				if err := json.Unmarshal(fields[key], &got); err != nil {
					t.Fatalf("served %s.%s missing numeric contract: %v", phase, key, err)
				}
				if err := json.Unmarshal(source[phase][key], &want); err != nil {
					t.Fatal(err)
				}
				if got == nil || want == nil {
					t.Fatalf("served/source %s.%s must not be null", phase, key)
				}
				if *got != *want {
					t.Fatalf("served %s.%s=%d source=%d", phase, key, *got, *want)
				}
				current[phase][key] = *got
			}
			if current[phase]["head_projection_host_calls"] <= 0 || current[phase]["head_projection_host_rows"] <= 0 || current[phase]["head_projection_nanos"] <= 0 {
				t.Fatalf("ordinary request did not execute actual %s head projection", phase)
			}
			if previous != nil && current[phase]["head_projection_host_calls"] <= previous[phase]["head_projection_host_calls"] {
				t.Fatalf("second request did not advance %s source", phase)
			}
		}
		previous = current
	}
}
