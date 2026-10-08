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

func v41CompressorHTTPModel(t *testing.T) *model.Model {
	t.Helper()
	cfg := model.Config{
		ModelType: "deepseek41", NumLayers: 2, HiddenSize: 64, NumHeads: 1, NumKVHeads: 1, HeadDim: 512,
		QKNopeHeadDim: 496, QKRopeHeadDim: 16, QLoraRank: 32, OLoraRank: 16, OGroups: 1,
		MoEIntermediateSize: 32, VocabSize: 8, NumExperts: model.V41RouterExperts, NumExpertsPerTok: model.V41RouterTopK,
		NSharedExperts: 1, RoutedScalingFactor: 1.5, RopeTheta: 10000, RMSNormEps: 1e-6, EOSTokenID: -1,
		Window: []int{-1}, IndexNHeads: 1, IndexHeadDim: 512, IndexTopK: 2,
		DeepSeekV41: &model.DeepSeekV41Config{HCMult: 4, HCEps: 1e-6, HCSinkhornIters: 20,
			CompressRatios: []int{2, 2}, KVSourceLayerIDs: []int{0}, IndexSourceLayerIDs: []int{0}, CandidateSourceLayerID: 20},
	}
	if !cfg.IsDeepSeekV41() {
		t.Fatal("public compressed fixture lost V41 identity")
	}
	var tensors []model.NamedTensorF32
	seed := uint64(0x6136815)
	add := func(name string, shape ...int) {
		n := 1
		for _, d := range shape {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			seed = seed*6364136223846793005 + 1442695040888963407
			v := (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.1
			if strings.HasSuffix(name, "norm.weight") || strings.HasSuffix(name, "mhc.scale") {
				v = 1
			}
			if strings.HasSuffix(name, "mhc.base") {
				v = 0
			}
			data[i] = v
		}
		tensors = append(tensors, model.NamedTensorF32{Name: name, Shape: shape, Data: data})
	}
	add("model.embed_tokens.weight", 8, 64)
	add("lm_head.weight", 8, 64)
	add("model.norm.weight", 64)
	for layer := 0; layer < 2; layer++ {
		p := fmt.Sprintf("model.layers.%d.", layer)
		add(p+"attn_norm.weight", 64)
		add(p+"ffn_norm.weight", 64)
		add(p+"mhc.mixes.weight", 256, 24)
		add(p+"mhc.base", 24)
		add(p+"mhc.scale", 3)
		add(p+"attn.wq_a.weight", 32, 64)
		add(p+"attn.wq_a_norm.weight", 32)
		add(p+"attn.wq_b.weight", 512, 32)
		add(p+"attn.wkv.weight", 512, 64)
		add(p+"attn.kv_norm.weight", 512)
		add(p+"attn.wo_a.weight", 16, 512)
		add(p+"attn.wo_b.weight", 64, 16)
		add(p+"attn.sink", 1)
		add(p+"attn.compressor.wkv.weight", 512, 64)
		add(p+"attn.compressor.wgate.weight", 512, 64)
		add(p+"attn.compressor.norm.weight", 512)
		add(p+"indexer.wq_b.weight", 512, 32)
		add(p+"indexer.wk.weight", 512, 512)
		add(p+"indexer.k_norm.weight", 512)
		add(p+"indexer.weights_proj.weight", 1, 64)
		add(p+"ffn.gate.weight", model.V41RouterExperts, 64)
		add(p+"ffn.gate.e_score_correction_bias", model.V41RouterExperts)
		add(p+"ffn.shared_experts.w1.weight", 32, 64)
		add(p+"ffn.shared_experts.w3.weight", 32, 64)
		add(p+"ffn.shared_experts.w2.weight", 64, 32)
		for expert := 0; expert < model.V41RouterExperts; expert++ {
			e := fmt.Sprintf("%sffn.experts.%d", p, expert)
			add(e+".w1.weight", 32, 64)
			add(e+".w3.weight", 32, 64)
			add(e+".w2.weight", 64, 32)
		}
	}
	m, err := model.NewFromF32Tensors(cfg, tensors)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime medium est=3s lane=default
func TestV41CompressorProjectionOrdinaryHTTPSourceAndReader(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "2")
	t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
	m := v41CompressorHTTPModel(t)
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
	keys := []string{"compressor_projection_device_calls", "compressor_projection_host_calls", "compressor_projection_device_rows", "compressor_projection_host_rows", "compressor_projection_activation_upload_bytes", "compressor_projection_readback_bytes", "compressor_projection_nanos"}
	var previous map[string]map[string]int64
	for request := 0; request < 2; request++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"native","messages":[{"role":"user","content":"hi"}],"max_tokens":2,"temperature":0}`))
		req.RemoteAddr = "127.0.0.1:50101"
		req.Header.Set("Content-Type", "application/json")
		completion := httptest.NewRecorder()
		srv.Handler().ServeHTTP(completion, req)
		if completion.Code != http.StatusOK {
			t.Fatalf("ordinary compressed completion status=%d body=%s", completion.Code, completion.Body.String())
		}
		var response struct {
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err = json.Unmarshal(completion.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		if response.Usage.CompletionTokens != 2 {
			t.Fatal("ordinary request did not decode two tokens")
		}
		get := httptest.NewRequest(http.MethodGet, "/debug/vars", nil)
		get.RemoteAddr = "127.0.0.1:50102"
		reader := httptest.NewRecorder()
		srv.Handler().ServeHTTP(reader, get)
		if reader.Code != http.StatusOK {
			t.Fatal("default compressor reader unavailable")
		}
		var doc struct {
			MoE struct {
				Requests   int64           `json:"requests"`
				Checkpoint json.RawMessage `json:"checkpoint"`
				V41        struct {
					Scope           string `json:"scope"`
					Prefill, Decode map[string]json.RawMessage
					Projections     map[string]map[string]json.RawMessage `json:"projections"`
					Compressor      map[string]map[string]json.RawMessage `json:"compressor"`
				} `json:"v41_phases"`
			} `json:"moe_residency"`
		}
		if err = json.Unmarshal(reader.Body.Bytes(), &doc); err != nil {
			t.Fatal(err)
		}
		if doc.MoE.V41.Scope != "model_lifetime" || doc.MoE.Requests != 0 || len(doc.MoE.Checkpoint) != 0 {
			t.Fatal("compressor source must remain default-readable independently of checkpoint/ring")
		}
		if len(doc.MoE.V41.Compressor) != 2 {
			t.Fatal("compressor reader must contain exactly prefill/decode")
		}
		sourceRaw, err := json.Marshal(m.V41ExpertFaultAttribution())
		if err != nil {
			t.Fatal(err)
		}
		var source map[string]map[string]json.RawMessage
		if err = json.Unmarshal(sourceRaw, &source); err != nil {
			t.Fatal(err)
		}
		current := map[string]map[string]int64{}
		for _, phase := range []string{"prefill", "decode"} {
			legacy := doc.MoE.V41.Prefill
			if phase == "decode" {
				legacy = doc.MoE.V41.Decode
			}
			if len(legacy) != 31 || len(doc.MoE.V41.Projections[phase]) != 16 {
				t.Fatal("compressor extension changed existing default reader shapes")
			}
			fields := doc.MoE.V41.Compressor[phase]
			if len(fields) != 7 {
				t.Fatalf("compressor %s must expose exactly seven numeric fields", phase)
			}
			current[phase] = map[string]int64{}
			for _, key := range keys {
				var got, want *int64
				if err = json.Unmarshal(fields[key], &got); err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(source[phase][key], &want); err != nil {
					t.Fatal(err)
				}
				if got == nil || want == nil || *got != *want {
					t.Fatalf("ordinary default reader differs from real model source %s.%s", phase, key)
				}
				current[phase][key] = *got
				if _, ok := legacy[key]; ok {
					t.Fatal("compressor field contaminated legacy phase")
				}
			}
			if current[phase]["compressor_projection_host_calls"] <= 0 || current[phase]["compressor_projection_host_rows"] <= 0 || current[phase]["compressor_projection_nanos"] <= 0 || current[phase]["compressor_projection_device_calls"] != 0 {
				t.Fatal("ordinary CPU request did not invoke actual compressor source")
			}
			if previous != nil && current[phase]["compressor_projection_host_calls"] <= previous[phase]["compressor_projection_host_calls"] {
				t.Fatal("second ordinary request did not advance compressor source")
			}
		}
		previous = current
	}
}
