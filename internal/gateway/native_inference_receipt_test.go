package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

func nativeReceiptServer(t *testing.T) *Server {
	t.Helper()
	// Gateway tests intentionally reset the process-global ABI registry. Restore
	// the production in-kernel engine so this wire witness is order-independent.
	abi.RegisterEngine(modelengine.EngineID, modelengine.Default)
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_SEED", "9070")
	cfg := kvmmuSynthCfg()
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	m.Quantize()
	tok := newByteLevelTokenizer(t)
	srv, err := New(Config{InKernelModel: m, Tokenizer: tok, Model: "synthetic-live"})
	if err != nil {
		t.Fatal(err)
	}
	return srv
}

func postNativeReceipt(t *testing.T, srv *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rr, req)
	return rr
}

func TestNativeInferenceReceiptProductionPath(t *testing.T) {
	srv := nativeReceiptServer(t)
	messages := []agent.Message{{Role: agent.RoleUser, Content: "measure"}}
	zero := 0.0
	const requestedTokens = 64
	body, _ := json.Marshal(ChatRequest{Model: "synthetic-live", Messages: messages, MaxTokens: requestedTokens, Temperature: &zero, Fak: &FakRequestExt{NativeInferenceReceipt: true}})
	rr := postNativeReceipt(t, srv, string(body))
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var resp ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Fak == nil || resp.Fak.NativeInferenceReceipt == nil {
		t.Fatalf("opt-in response missing fak.native_inference_receipt: %s", rr.Body.String())
	}
	receipt := resp.Fak.NativeInferenceReceipt
	if len(receipt.TokenIDs) != requestedTokens || len(receipt.TokenLogprobs) != len(receipt.TokenIDs) || resp.Usage.CompletionTokens != len(receipt.TokenIDs) {
		t.Fatalf("token ids/logprobs/usage = %d/%d/%d, want request-bound equal entries", len(receipt.TokenIDs), len(receipt.TokenLogprobs), resp.Usage.CompletionTokens)
	}
	for i, lp := range receipt.TokenLogprobs {
		if math.IsNaN(lp) || math.IsInf(lp, 0) || lp > 0 {
			t.Fatalf("logprob[%d]=%v, want finite normalized log probability", i, lp)
		}
	}
	for name, seconds := range map[string]float64{"prefill": receipt.PrefillSeconds, "ttft": receipt.TTFTSeconds, "decode": receipt.DecodeSeconds} {
		if seconds < 0 || math.IsNaN(seconds) || math.IsInf(seconds, 0) {
			t.Fatalf("%s_seconds=%v, want finite non-negative", name, seconds)
		}
	}
	if receipt.Model != "synthetic-live" || receipt.Engine != "inkernel" || receipt.Planner != "inkernel" || receipt.Owner != "fak" || receipt.Backend != "cpu-ref" || receipt.ForwardPath != "cpu/reference" || receipt.Q4K || receipt.FallbackActive {
		t.Fatalf("execution identity = %+v, want exact synthetic inkernel cpu/reference without Q4K or fallback", receipt)
	}
	wantSelection := model.NativeSelectionIdentity{
		Schema:              model.NativeSelectionIdentitySchemaV1,
		ModelRef:            "synthetic-live",
		Backend:             "cpu-ref",
		ForwardPath:         "cpu/reference",
		Quantization:        model.NativeSelectionQuantizationQ8_0,
		PrefillChunkTokens:  0,
		CPUOffloadExperts:   0,
		Q4KGateUpOutputSlab: false,
	}
	if receipt.NativeSelection != wantSelection {
		t.Fatalf("kernel selection = %+v, want %+v", receipt.NativeSelection, wantSelection)
	}
	wantDigest, err := receipt.NativeSelection.Digest()
	if err != nil {
		t.Fatalf("wire kernel selection is invalid: %v", err)
	}
	if receipt.NativeSelectionDigest != wantDigest {
		t.Fatalf("kernel selection digest = %q, recomputed %q", receipt.NativeSelectionDigest, wantDigest)
	}
	for _, want := range []string{`"kernel_selection":`, `"kernel_selection_digest":"sha256:`} {
		if !bytes.Contains(rr.Body.Bytes(), []byte(want)) {
			t.Fatalf("wire receipt missing %s: %s", want, rr.Body.String())
		}
	}
	if bytes.Contains(bytes.ToLower(rr.Body.Bytes()), []byte("llama.cpp")) {
		t.Fatalf("native receipt substituted an external engine: %s", rr.Body.String())
	}
	metrics := srv.renderMetrics()
	for _, want := range []string{
		`fak_native_runtime_info{engine="inkernel",backend="other",forward_path="other",model="synthetic",planner="inkernel",owner="fak"} 1`,
		`fak_native_receipt_requests_total{engine="inkernel",backend="other",forward_path="other"} 1`,
		`fak_native_receipt_signal_supported{signal="prefill"} 1`,
		`fak_native_receipt_signal_supported{signal="decode"} 1`,
		`fak_native_receipt_latest_stale 0`,
	} {
		if !strings.Contains(metrics, want) {
			t.Fatalf("production request did not reach /metrics: missing %q", want)
		}
	}
	if receipt.Qwen35MetalForwardSequence != nil || bytes.Contains(rr.Body.Bytes(), []byte(`"qwen35_metal_forward_sequence"`)) {
		t.Fatalf("CPU receipt acquired Metal sequence evidence: %+v", receipt.Qwen35MetalForwardSequence)
	}
	if receipt.Qwen35MetalStateIdentity != nil || bytes.Contains(rr.Body.Bytes(), []byte(`"qwen35_metal_state_identity"`)) {
		t.Fatalf("CPU receipt acquired Metal state identity: %+v", receipt.Qwen35MetalStateIdentity)
	}
	if receipt.CUDAImmutableWeightUploads != nil {
		t.Fatalf("CPU receipt acquired CUDA upload evidence: %+v", receipt.CUDAImmutableWeightUploads)
	}
}

func gatewayQwen35MetalStateIdentityFixture(authority string) *model.Qwen35MetalStateIdentityReceipt {
	receipt := &model.Qwen35MetalStateIdentityReceipt{
		Schema:              model.Qwen35MetalStateIdentitySchema,
		Available:           true,
		Authority:           authority,
		OwnerGeneration:     strings.Repeat("a", 64),
		Tokens:              32,
		TokenLineageSHA256:  strings.Repeat("b", 64),
		FullAttentionLayers: 1,
		GDNLayers:           1,
		StateCount:          5,
		States: []model.Qwen35MetalStateDigest{
			{Layer: 0, Role: model.Qwen35MetalStateRoleKRaw, Elements: 16, SHA256: strings.Repeat("c", 64)},
			{Layer: 0, Role: model.Qwen35MetalStateRoleKPost, Elements: 16, SHA256: strings.Repeat("d", 64)},
			{Layer: 0, Role: model.Qwen35MetalStateRoleV, Elements: 16, SHA256: strings.Repeat("e", 64)},
			{Layer: 1, Role: model.Qwen35MetalStateRoleGDNConv, Elements: 12, SHA256: strings.Repeat("f", 64)},
			{Layer: 1, Role: model.Qwen35MetalStateRoleGDNRecurrent, Elements: 16, SHA256: strings.Repeat("0", 64)},
		},
		DigestOperations:  7,
		DigestInputBytes:  8192,
		DigestNanoseconds: 4096,
		BindingSHA256:     strings.Repeat("1", 64),
	}
	if authority == model.Qwen35MetalStateAuthoritySequence {
		receipt.GDNSnapshotOps = 1
		receipt.GDNSeedOps = 1
		receipt.GDNStateD2HBytes = 112
		receipt.GDNStateH2DBytes = 112
	}
	return receipt
}

func TestNativeInferenceReceiptStateIdentityJSONPrivacyAccountingAndOmission(t *testing.T) {
	for _, authority := range []string{model.Qwen35MetalStateAuthorityControl, model.Qwen35MetalStateAuthoritySequence} {
		t.Run(authority, func(t *testing.T) {
			want := gatewayQwen35MetalStateIdentityFixture(authority)
			raw, err := json.Marshal(FakExt{NativeInferenceReceipt: &model.NativeInferenceReceipt{Qwen35MetalStateIdentity: want}})
			if err != nil {
				t.Fatal(err)
			}
			var decoded FakExt
			if err := json.Unmarshal(raw, &decoded); err != nil {
				t.Fatal(err)
			}
			if decoded.NativeInferenceReceipt == nil || !reflect.DeepEqual(decoded.NativeInferenceReceipt.Qwen35MetalStateIdentity, want) {
				t.Fatalf("state identity JSON round trip = %+v, want %+v", decoded.NativeInferenceReceipt, want)
			}

			var envelope map[string]json.RawMessage
			if err := json.Unmarshal(raw, &envelope); err != nil {
				t.Fatal(err)
			}
			var native map[string]json.RawMessage
			if err := json.Unmarshal(envelope["native_inference_receipt"], &native); err != nil {
				t.Fatal(err)
			}
			identityRaw, present := native["qwen35_metal_state_identity"]
			if !present {
				t.Fatalf("public receipt omitted opted-in state identity: %s", raw)
			}
			for _, field := range []string{
				`"schema":"fak.qwen35-metal-state-identity/1"`, `"available":true`, `"authority":"` + authority + `"`,
				`"owner_generation":"` + strings.Repeat("a", 64) + `"`, `"tokens":32`, `"token_lineage_sha256":"` + strings.Repeat("b", 64) + `"`,
				`"full_attention_layers":1`, `"gdn_layers":1`, `"state_count":5`, `"layer":0`, `"role":"full_attention_k_raw"`,
				`"elements":16`, `"sha256":"` + strings.Repeat("c", 64) + `"`, `"gdn_snapshot_ops":` + fmt.Sprint(want.GDNSnapshotOps),
				`"gdn_seed_ops":` + fmt.Sprint(want.GDNSeedOps), `"gdn_state_d2h_bytes":` + fmt.Sprint(want.GDNStateD2HBytes),
				`"gdn_state_h2d_bytes":` + fmt.Sprint(want.GDNStateH2DBytes), `"digest_operations":7`, `"digest_input_bytes":8192`,
				`"digest_nanoseconds":4096`, `"binding_sha256":"` + strings.Repeat("1", 64) + `"`,
			} {
				if !bytes.Contains(identityRaw, []byte(field)) {
					t.Fatalf("state identity JSON %s missing exact field %s", identityRaw, field)
				}
			}
			public := strings.ToLower(string(identityRaw))
			for _, forbidden := range []string{`"handle"`, `"pointer"`, `"ptr"`, `"path"`, `"tensor"`, `"values"`, `"chunks"`, `/users/private/weights`, `0xfeedface`, `3.1415926535`} {
				if strings.Contains(public, forbidden) {
					t.Fatalf("public state identity exposed forbidden native/content marker %q: %s", forbidden, identityRaw)
				}
			}
		})
	}

	raw, err := json.Marshal(FakExt{NativeInferenceReceipt: &model.NativeInferenceReceipt{}})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte(`"qwen35_metal_state_identity"`)) {
		t.Fatalf("unavailable identity emitted null/default object: %s", raw)
	}
}

func TestNativeInferenceReceiptDefaultWireOmission(t *testing.T) {
	srv := nativeReceiptServer(t)
	messages := []agent.Message{{Role: agent.RoleUser, Content: "plain"}}
	zero := 0.0
	body, _ := json.Marshal(ChatRequest{Model: "synthetic-live", Messages: messages, MaxTokens: 2, Temperature: &zero})
	rr := postNativeReceipt(t, srv, string(body))
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatal(err)
	}
	if _, present := raw["fak"]; present || bytes.Contains(rr.Body.Bytes(), []byte("native_inference_receipt")) {
		t.Fatalf("default response changed wire shape: %s", rr.Body.String())
	}
}

func TestNativeInferenceReceiptUnsupportedRequestsFailClosed(t *testing.T) {
	native := nativeReceiptServer(t)
	cases := map[string]string{
		"streaming":   `{"messages":[{"role":"user","content":"x"}],"stream":true,"fak":{"native_inference_receipt":true}}`,
		"temperature": `{"messages":[{"role":"user","content":"x"}],"temperature":0.1,"fak":{"native_inference_receipt":true}}`,
		"top-p":       `{"messages":[{"role":"user","content":"x"}],"top_p":0.5,"fak":{"native_inference_receipt":true}}`,
		"logit-bias":  `{"messages":[{"role":"user","content":"x"}],"logit_bias":{"1":1},"fak":{"native_inference_receipt":true}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			rr := postNativeReceipt(t, native, body)
			if rr.Code != http.StatusBadRequest || bytes.Contains(rr.Body.Bytes(), []byte(`"native_inference_receipt":{`)) {
				t.Fatalf("status=%d body=%s, want fail-closed 400 without receipt", rr.Code, rr.Body.String())
			}
		})
	}

	proxy, err := New(Config{Model: "mock"})
	if err != nil {
		t.Fatal(err)
	}
	rr := postNativeReceipt(t, proxy, `{"messages":[{"role":"user","content":"x"}],"temperature":0,"fak":{"native_inference_receipt":true}}`)
	if rr.Code != http.StatusBadGateway || bytes.Contains(rr.Body.Bytes(), []byte(`"native_inference_receipt":{`)) {
		t.Fatalf("non-native planner status=%d body=%s, want fail-closed 502 without stamped receipt", rr.Code, rr.Body.String())
	}
}

func TestNativeInferenceReceiptJSONShapeUsesChosenTokenArrays(t *testing.T) {
	raw, err := json.Marshal(FakExt{NativeInferenceReceipt: &model.NativeInferenceReceipt{
		TokenIDs:      []int{7},
		TokenLogprobs: []float64{-1},
		Qwen35MetalForwardSequence: &model.Qwen35MetalForwardSequenceReceipt{
			Path:                  model.Qwen35MetalGDNSequenceForwardPath,
			Available:             true,
			SelectorState:         model.Qwen35MetalSequenceSelectorOn,
			EvidenceState:         model.Qwen35MetalSequenceEvidenceExecuted,
			Tokens:                32,
			CommandBuffers:        1,
			Encoders:              7,
			IntermediateWaits:     0,
			IntermediateReadbacks: 0,
			TerminalWaits:         1,
			TerminalReadbacks:     1,
			HostUploadBytes:       65536,
			HostReadbackBytes:     16384,
			Committed:             true,
			CompletedWait:         true,
			TimingAvailable:       true,
			GPUMilliseconds:       2.5,
			WaitMilliseconds:      3.5,
		},
		CUDAImmutableWeightUploads: &model.NativeCUDAImmutableWeightUploadDelta{
			Before: model.NativeCUDAImmutableWeightUploadCounters{Calls: 4, TransferBytes: 1024, ResidentBytes: 512},
			After:  model.NativeCUDAImmutableWeightUploadCounters{Calls: 5, TransferBytes: 5120, ResidentBytes: 2560},
			Delta:  model.NativeCUDAImmutableWeightUploadCounters{Calls: 1, TransferBytes: 4096, ResidentBytes: 2048},
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"native_inference_receipt"`, `"token_ids":[7]`, `"token_logprobs":[-1]`, `"fallback_active":false`, `"qwen35_metal_forward_sequence"`, `"path":"metal/qwen35-gdn-preprojected-sequence-v1"`, `"selector_state":"on"`, `"evidence_state":"executed"`, `"tokens":32`, `"command_buffers":1`, `"encoders":7`, `"intermediate_waits":0`, `"intermediate_readbacks":0`, `"terminal_waits":1`, `"terminal_readbacks":1`, `"host_upload_bytes":65536`, `"host_readback_bytes":16384`, `"committed":true`, `"completed_wait":true`, `"timing_available":true`, `"gpu_milliseconds":2.5`, `"wait_milliseconds":3.5`, `"cuda_immutable_weight_uploads"`, `"before":{"calls":4,"transfer_bytes":1024,"resident_bytes":512}`, `"after":{"calls":5,"transfer_bytes":5120,"resident_bytes":2560}`, `"delta":{"calls":1,"transfer_bytes":4096,"resident_bytes":2048}`} {
		if !bytes.Contains(raw, []byte(field)) {
			t.Fatalf("receipt JSON %s missing %s", raw, field)
		}
	}
}

func gatewayQwen35MetalReceiptConfig() model.Config {
	return model.Config{
		HiddenSize: 256, NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 64,
		IntermediateSize: 256, VocabSize: 512, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
		LayerTypes:          []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim: 3, LinearKeyHeadDim: 64, LinearNumKeyHeads: 2,
		LinearValueHeadDim: 64, LinearNumValueHeads: 4, AttnOutputGate: true,
		FullAttentionInterval: 4, NormGain1p: true,
	}
}

func assertUnavailableMetalSequenceRefusal(t *testing.T, rr *httptest.ResponseRecorder) bool {
	t.Helper()
	if model.Qwen35MetalGDNPreprojectedSequenceAvailable() {
		return false
	}
	if rr.Code != http.StatusBadGateway {
		t.Fatalf("platform-unavailable Metal sequence status=%d body=%s, want fail-closed 502", rr.Code, rr.Body.String())
	}
	if bytes.Contains(rr.Body.Bytes(), []byte(`"native_inference_receipt"`)) || bytes.Contains(rr.Body.Bytes(), []byte(`"fallback_active"`)) {
		t.Fatalf("platform-unavailable Metal sequence forged execution/fallback evidence: %s", rr.Body.String())
	}
	return true
}

func TestNativeInferenceReceiptBackendNilQ4KMetalSequenceRequest(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "1")
	cfg := gatewayQwen35MetalReceiptConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	srv, err := New(Config{
		InKernelModel: m, Tokenizer: newByteLevelTokenizer(t), Model: "qwen38-metal-receipt",
		InKernelQ4K: true, Metal: true,
		InKernelPlanner: agent.InKernelPlannerConfig{Qwen35MetalGDNSequence: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	// With the generic hybrid renderer and thinking enabled, the thirteen-byte
	// user payload makes the real model prompt exactly P32.
	rr := postNativeReceipt(t, srv, `{"messages":[{"role":"user","content":"receipt-proof"}],"max_tokens":1,"temperature":0,"fak":{"native_inference_receipt":true}}`)
	if assertUnavailableMetalSequenceRefusal(t, rr) {
		return
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Fak == nil || response.Fak.NativeInferenceReceipt == nil {
		t.Fatalf("gateway omitted native receipt: %s", rr.Body.String())
	}
	native := response.Fak.NativeInferenceReceipt
	sequence := native.Qwen35MetalForwardSequence
	if native.Backend != "metal" || native.ForwardPath != model.Qwen35MetalGDNSequenceForwardPath || !native.Q4K || native.FallbackActive {
		t.Fatalf("execution identity=%+v, want backend-nil Q4_K Metal sequence without fallback", native)
	}
	if sequence == nil || !sequence.Available || sequence.SelectorState != model.Qwen35MetalSequenceSelectorOn || sequence.EvidenceState != model.Qwen35MetalSequenceEvidenceExecuted {
		t.Fatalf("sequence selector/evidence=%+v", sequence)
	}
	if sequence.Tokens != 32 || sequence.CommandBuffers != 1 || sequence.Encoders <= 1 || sequence.IntermediateWaits != 0 || sequence.IntermediateReadbacks != 0 || sequence.TerminalWaits != 1 || sequence.TerminalReadbacks != 1 {
		t.Fatalf("sequence accounting=%+v", sequence)
	}
	if sequence.HostUploadBytes == 0 || sequence.HostReadbackBytes == 0 || !sequence.Committed || !sequence.CompletedWait {
		t.Fatalf("sequence transfer/lifecycle=%+v", sequence)
	}
	for _, field := range []string{`"selector_state":"on"`, `"evidence_state":"executed"`, `"intermediate_waits":0`, `"intermediate_readbacks":0`} {
		if !bytes.Contains(rr.Body.Bytes(), []byte(field)) {
			t.Fatalf("gateway JSON missing %s: %s", field, rr.Body.String())
		}
	}
}

func TestNativeInferenceReceiptMetalSequenceRouteAccountingWire(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "off")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "1")
	m := model.NewSynthetic(gatewayQwen35MetalReceiptConfig())
	m.Quantize()
	srv, err := New(Config{
		InKernelModel: m, Tokenizer: newByteLevelTokenizer(t), Model: "qwen38-metal-wire-receipt",
		InKernelQ4K: true, Metal: true,
		InKernelPlanner: agent.InKernelPlannerConfig{Qwen35MetalGDNSequence: true},
	})
	if err != nil {
		t.Fatal(err)
	}
	rr := postNativeReceipt(t, srv, `{"messages":[{"role":"user","content":"receipt-proof"}],"max_tokens":1,"temperature":0,"fak":{"native_inference_receipt":true}}`)
	if assertUnavailableMetalSequenceRefusal(t, rr) {
		return
	}
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var envelope struct {
		Fak struct {
			Native json.RawMessage `json:"native_inference_receipt"`
		} `json:"fak"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(envelope.Fak.Native, &wire); err != nil {
		t.Fatalf("native receipt is not a public JSON object: %v", err)
	}
	assertWireValue := func(field, want string) {
		t.Helper()
		if got := string(wire[field]); got != want {
			t.Fatalf("native receipt %s=%s, want %s; receipt=%s", field, got, want, envelope.Fak.Native)
		}
	}
	assertWireValue("backend", `"metal"`)
	assertWireValue("forward_path", `"metal/qwen35-gdn-preprojected-sequence-v1"`)
	assertWireValue("q4k", "true")
	assertWireValue("fallback_active", "false")

	var sequence map[string]json.RawMessage
	if err := json.Unmarshal(wire["qwen35_metal_forward_sequence"], &sequence); err != nil {
		t.Fatalf("sequence accounting is not a public JSON object: %v", err)
	}
	for field, want := range map[string]string{
		"path":                   `"metal/qwen35-gdn-preprojected-sequence-v1"`,
		"available":              "true",
		"selector_state":         `"on"`,
		"evidence_state":         `"executed"`,
		"tokens":                 "32",
		"command_buffers":        "1",
		"intermediate_waits":     "0",
		"intermediate_readbacks": "0",
		"terminal_waits":         "1",
		"terminal_readbacks":     "1",
		"committed":              "true",
		"completed_wait":         "true",
	} {
		if got := string(sequence[field]); got != want {
			t.Fatalf("sequence receipt %s=%s, want %s; receipt=%s", field, got, want, wire["qwen35_metal_forward_sequence"])
		}
	}
	for _, field := range []string{"encoders", "host_upload_bytes", "host_readback_bytes"} {
		var got uint64
		if err := json.Unmarshal(sequence[field], &got); err != nil || got == 0 {
			t.Fatalf("sequence receipt %s=%s, want positive accounting", field, sequence[field])
		}
	}
}

func TestNativeInferenceReceiptBackendNilQ4KMetalControlAndTypedAbsence(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "off")
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "on")
	t.Setenv("FAK_INKERNEL_MAX_TOKENS", "1")
	cfg := gatewayQwen35MetalReceiptConfig()
	m := model.NewSynthetic(cfg)
	m.Quantize()
	srv, err := New(Config{
		InKernelModel: m, Tokenizer: newByteLevelTokenizer(t), Model: "qwen38-metal-control-receipt",
		InKernelQ4K: true, Metal: true,
		InKernelPlanner: agent.InKernelPlannerConfig{Qwen35MetalGDNSequence: false},
	})
	if err != nil {
		t.Fatal(err)
	}
	rr := postNativeReceipt(t, srv, `{"messages":[{"role":"user","content":"receipt-proof"}],"max_tokens":1,"temperature":0,"fak":{"native_inference_receipt":true}}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rr.Code, rr.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	native := response.Fak.NativeInferenceReceipt
	sequence := native.Qwen35MetalForwardSequence
	if native.Backend != "metal" || native.ForwardPath != "metal/qwen35-hybrid-session-v1" || !native.Q4K || native.FallbackActive {
		t.Fatalf("control execution identity=%+v", native)
	}
	wantEvidence := model.Qwen35MetalSequenceEvidenceNotSelected
	if !model.Qwen35MetalGDNPreprojectedSequenceAvailable() {
		wantEvidence = model.Qwen35MetalSequenceEvidenceUnavailable
	}
	if sequence == nil || sequence.Available || sequence.SelectorState != model.Qwen35MetalSequenceSelectorOff || sequence.EvidenceState != wantEvidence {
		t.Fatalf("control selector/evidence=%+v", sequence)
	}
	if sequence.CommandBuffers != 0 || sequence.Encoders != 0 || sequence.IntermediateWaits != 0 || sequence.IntermediateReadbacks != 0 || sequence.TerminalWaits != 0 || sequence.TerminalReadbacks != 0 {
		t.Fatalf("control invented sequence work=%+v", sequence)
	}

	unsupportedSession := model.NewSynthetic(cfg).NewSession()
	unsupported := unsupportedSession.Qwen35MetalForwardSequenceStatus()
	unsupportedSession.Close()
	if unsupported.SelectorState != model.Qwen35MetalSequenceSelectorOff || unsupported.EvidenceState != model.Qwen35MetalSequenceEvidenceUnsupported {
		t.Fatalf("unsupported session state=%+v", unsupported)
	}
	unavailable := model.Qwen35MetalForwardSequenceReceipt{
		Path: model.Qwen35MetalGDNSequenceForwardPath, SelectorState: model.Qwen35MetalSequenceSelectorOn,
		EvidenceState: model.Qwen35MetalSequenceEvidenceUnavailable,
	}
	raw, err := json.Marshal(FakExt{NativeInferenceReceipt: &model.NativeInferenceReceipt{Qwen35MetalForwardSequence: &unavailable}})
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"selector_state":"on"`, `"evidence_state":"unavailable"`} {
		if !bytes.Contains(raw, []byte(field)) {
			t.Fatalf("typed unavailable state missing %s: %s", field, raw)
		}
	}
}
