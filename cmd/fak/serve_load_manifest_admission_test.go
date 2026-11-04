package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// serve_load_manifest_admission_test.go — the layout-guard (lg-preflight-wire) serve-side
// witnesses: the manifest preflight refusal is typed (errors.Is on the sentinel, reason-token
// Has()), it aborts the load funnel before any arm selection or dispatch, and the shared
// admission gate carries it too. Assertions compare closed-vocabulary reason tokens, never
// error prose.

// ds41BrokenHeaderFile builds an in-memory deepseek41 header whose config parses but whose
// tensor directory carries none of the forward-requested operands: the census must refuse it
// TENSOR_MISSING off the header alone.
func ds41BrokenHeaderFile() *ggufload.File {
	u64 := func(v uint64) ggufload.Value { return ggufload.Value{Type: ggufload.TypeUint64, Value: v} }
	tokens := make([]ggufload.Value, 4)
	for i := range tokens {
		tokens[i] = ggufload.Value{Type: ggufload.TypeString, Value: "t"}
	}
	return &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                        {Type: ggufload.TypeString, Value: "deepseek41"},
			"deepseek41.block_count":                      u64(1),
			"deepseek41.embedding_length":                 u64(64),
			"deepseek41.attention.head_count":             u64(1),
			"deepseek41.expert_count":                     u64(4),
			"deepseek41.expert_feed_forward_length":       u64(8),
			"deepseek41.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-6)},
			"tokenizer.ggml.tokens":                       {Type: ggufload.TypeArray, Value: tokens},
		},
		Tensors: []ggufload.TensorInfo{},
	}
}

// TestServeAdmitManifestPreflightTypedRefusal pins the error-type contract: a refused header
// returns *ServeManifestPreflightRefusal wrapping ErrServeManifestPreflight, exposing the
// full report through Refusals()/Has() in the closed reason-token vocabulary.
//
// fak-test:runtime fast est=10ms
func TestServeAdmitManifestPreflightTypedRefusal(t *testing.T) {
	err := serveAdmitManifestPreflight("synthetic.gguf", ds41BrokenHeaderFile())
	var rr *ServeManifestPreflightRefusal
	if !errors.As(err, &rr) {
		t.Fatalf("error=%T %v, want *ServeManifestPreflightRefusal", err, err)
	}
	if !errors.Is(err, ErrServeManifestPreflight) {
		t.Fatal("refusal does not wrap ErrServeManifestPreflight")
	}
	if !rr.Has(newmodel.RefusalTensorMissing) {
		t.Fatalf("report has no TENSOR_MISSING refusal: %+v", rr.Report.Refusals)
	}
	named := false
	for _, r := range rr.Refusals() {
		if r.Reason == newmodel.RefusalTensorMissing && r.Axis == "token_embd.weight" {
			named = true
		}
	}
	if !named {
		t.Fatalf("TENSOR_MISSING refusals do not name token_embd.weight: %+v", rr.Refusals())
	}
}

// cleanLlamaHeaderFile is a minimal llama header every one of whose tensors the loader
// consumes: the admission helper must pass it with a nil error (the green path).
func cleanLlamaHeaderFile() *ggufload.File {
	u32 := func(v uint32) ggufload.Value { return ggufload.Value{Type: ggufload.TypeUint32, Value: v} }
	tokens := make([]ggufload.Value, 4)
	for i := range tokens {
		tokens[i] = ggufload.Value{Type: ggufload.TypeString, Value: "t"}
	}
	dims := []uint64{8, 8}
	return &ggufload.File{
		Metadata: map[string]ggufload.Value{
			"general.architecture":                   {Type: ggufload.TypeString, Value: "llama"},
			"llama.block_count":                      u32(1),
			"llama.embedding_length":                 u32(8),
			"llama.feed_forward_length":              u32(16),
			"llama.attention.head_count":             u32(2),
			"llama.attention.layer_norm_rms_epsilon": {Type: ggufload.TypeFloat32, Value: float32(1e-6)},
			"tokenizer.ggml.tokens":                  {Type: ggufload.TypeArray, Value: tokens},
		},
		Tensors: []ggufload.TensorInfo{
			{Name: "token_embd.weight", Dims: []uint64{8, 4}, Type: ggufload.TensorF32},
			{Name: "output_norm.weight", Dims: []uint64{8}, Type: ggufload.TensorF32},
			{Name: "output.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.attn_q.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.attn_k.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.attn_v.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.attn_output.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_gate.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_up.weight", Dims: dims, Type: ggufload.TensorF32},
			{Name: "blk.0.ffn_down.weight", Dims: dims, Type: ggufload.TensorF32},
		},
	}
}

// TestServeAdmitManifestPreflightPassesCleanHeader pins the no-false-positive side of the
// admission helper: a fully consumed header admits with a nil error.
//
// fak-test:runtime fast est=10ms
func TestServeAdmitManifestPreflightPassesCleanHeader(t *testing.T) {
	if err := serveAdmitManifestPreflight("clean.gguf", cleanLlamaHeaderFile()); err != nil {
		t.Fatalf("clean header refused: %v", err)
	}
}

// TestLoadServeInKernelModelPlacedErrRefusesBrokenDeepSeek41BeforeDispatch is the load-funnel
// witness: a deepseek41 checkpoint whose directory carries no forward operand is refused with
// the typed sentinel BEFORE arm selection or any payload read, so no broken V4.1 artifact can
// reach dispatch on any backend arm.
//
// fak-test:runtime fast est=50ms
func TestLoadServeInKernelModelPlacedErrRefusesBrokenDeepSeek41BeforeDispatch(t *testing.T) {
	const dim = 64
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 0, 8)
	writeKVStringForTest(&b, "general.architecture", "deepseek41")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "deepseek41.embedding_length", dim)
	writeKVUint32ForTest(&b, "deepseek41.block_count", 1)
	writeKVUint32ForTest(&b, "deepseek41.attention.head_count", 1)
	writeKVUint32ForTest(&b, "deepseek41.expert_count", 4)
	writeKVUint32ForTest(&b, "deepseek41.expert_feed_forward_length", 8)
	writeKVFloat32ForTest(&b, "deepseek41.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVStringArrayForTest(&b, "tokenizer.ggml.tokens", []string{"t", "t", "t", "t"})
	padToAlignmentForTest(&b, 32)

	path := filepath.Join(t.TempDir(), "broken_ds41.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	_, _, _, _, _, err := loadServeInKernelModelPlacedErr(path, nil, false, false, 0, nil, 1, nil)
	if err == nil {
		t.Fatal("load admitted a deepseek41 checkpoint with no forward-requested tensors")
	}
	var rr *ServeManifestPreflightRefusal
	if !errors.As(err, &rr) {
		t.Fatalf("error=%T %v, want *ServeManifestPreflightRefusal", err, err)
	}
	if !errors.Is(err, ErrServeManifestPreflight) {
		t.Fatal("load-funnel refusal does not wrap ErrServeManifestPreflight")
	}
	if !rr.Has(newmodel.RefusalTensorMissing) {
		t.Fatalf("load-funnel refusal carries no TENSOR_MISSING: %+v", rr.Refusals())
	}
}

// TestPreflightServeBackendForwardKeepsCapabilityScope pins the guard's boundary: the
// header-only capability gate checks ONLY backend/forward compatibility against the declared
// config (the sparse qwen35 fixture still admits exactly as pinned by the support-matrix
// tests), while artifact tensor coverage is refused fail-closed at the load funnel, where
// every arm funnels before dispatch.
//
// fak-test:runtime fast est=10ms
func TestPreflightServeBackendForwardKeepsCapabilityScope(t *testing.T) {
	be := &servePreflightGDNBackend{
		servePreflightBackend: newServePreflightBackend("cuda"),
		path:                  fakmodel.Qwen35GDNCUDAPath,
	}
	if _, err := preflightServeBackendForwardWith("synthetic.gguf", be, servePreflightOpen(t, "qwen35", nil)); err != nil {
		t.Fatalf("capability-gate scope changed: %v", err)
	}
}

// TestServeAdmitSafetensorsPreflightTypedRefusal pins the safetensors leg: a plan-level
// failure folds into the same typed refusal as the GGUF leg.
//
// fak-test:runtime fast est=10ms
func TestServeAdmitSafetensorsPreflightTypedRefusal(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	err := serveAdmitSafetensorsPreflight(dir)
	var rr *ServeManifestPreflightRefusal
	if !errors.As(err, &rr) {
		t.Fatalf("error=%T %v, want *ServeManifestPreflightRefusal", err, err)
	}
	if !errors.Is(err, ErrServeManifestPreflight) {
		t.Fatal("safetensors refusal does not wrap ErrServeManifestPreflight")
	}
	if !rr.Has(newmodel.RefusalHeaderConfigInvalid) {
		t.Fatalf("safetensors refusal carries no HEADER_CONFIG_INVALID: %+v", rr.Refusals())
	}
}

// writeKVStringArrayForTest appends one GGUF string-array key/value pair (the tokenizer
// spelling the Config vocab derivation reads).
func writeKVStringArrayForTest(b *bytes.Buffer, key string, values []string) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(ggufload.TypeString))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		writeStringForTest(b, v)
	}
}
