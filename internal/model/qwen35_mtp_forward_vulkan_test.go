//go:build vulkan && (windows || linux) && cgo

package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen35MTPTestPriorHiddenRows is a test-only producer for the real-artifact
// A/B witness in the external model_test package. It executes the checkpoint's
// normal quantized CPU target path and captures the final-layer residual before
// finalNorm. Row zero is MTP's bootstrap zero; row k uses target hidden k-1.
func Qwen35MTPTestPriorHiddenRows(m *Model, tokens []int, outputDir string) (rows [][]float32, err error) {
	if m == nil || m.Cfg.HiddenSize <= 0 || m.Cfg.NumLayers <= 0 || len(tokens) == 0 || outputDir == "" {
		return nil, fmt.Errorf("model: invalid MTP target-hidden producer input")
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			rows = nil
			err = fmt.Errorf("model: native target-hidden producer failed: %v", recovered)
		}
	}()
	s := m.NewSession()
	s.Quant, s.Q4K = true, true
	defer s.Close()
	rows = make([][]float32, len(tokens))
	rows[0] = make([]float32, m.Cfg.HiddenSize)
	for pos := 0; pos+1 < len(tokens); pos++ {
		if tokens[pos] < 0 || tokens[pos] >= m.Cfg.VocabSize {
			return nil, fmt.Errorf("model: target-hidden token %d at position %d outside [0,%d)", tokens[pos], pos, m.Cfg.VocabSize)
		}
		dir := filepath.Join(outputDir, fmt.Sprintf("position_%06d", pos))
		tap := &hiddenTap{dir: dir, pos: pos, ops: false}
		s.tap = tap
		_ = s.Step(tokens[pos])
		if tap.err != nil {
			return nil, fmt.Errorf("model: capture target hidden at position %d: %w", pos, tap.err)
		}
		payload, readErr := os.ReadFile(filepath.Join(dir, fmt.Sprintf("layer_%02d.f32", m.Cfg.NumLayers-1)))
		if readErr != nil {
			return nil, fmt.Errorf("model: read target hidden at position %d: %w", pos, readErr)
		}
		if len(payload) != m.Cfg.HiddenSize*4 {
			return nil, fmt.Errorf("model: target hidden at position %d has %d bytes, want %d", pos, len(payload), m.Cfg.HiddenSize*4)
		}
		row := make([]float32, m.Cfg.HiddenSize)
		for i := range row {
			row[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
			if math.IsNaN(float64(row[i])) || math.IsInf(float64(row[i]), 0) {
				return nil, fmt.Errorf("model: target hidden at position %d index %d is non-finite", pos, i)
			}
		}
		rows[pos+1] = row
	}
	return rows, nil
}

func TestQwen35MTPForwardUsesResidentVulkanDraftAndHead(t *testing.T) {
	backend, ok := compute.Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan backend is not registered")
		}
		t.Skip("Vulkan backend unavailable")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(backend.Tier()), strings.ToLower(expected)) {
		t.Fatalf("Vulkan device %q does not match required device %q", backend.Tier(), expected)
	}

	t.Run("forward_parity_receipt_and_lifetime", func(t *testing.T) {
		oracleModel := qwen35MTPVulkanOracleModel(t)
		oracle, err := oracleModel.NewQwen35MTPForward()
		if err != nil {
			t.Fatal(err)
		}
		defer oracle.Close()

		model := qwen35MTPVulkanOracleModel(t)
		resident, err := model.NewQwen35MTPForwardWithBackend(backend)
		if err != nil {
			t.Fatalf("construct resident MTP forward: %v", err)
		}

		steps := []struct {
			prior, embedding []float32
		}{
			{prior: []float32{0.4, -1.1, 2.3, 0.7}, embedding: []float32{-0.8, 1.6, 0.2, -2.1}},
			{prior: []float32{1.9, 0.3, -0.6, 2.4}, embedding: []float32{0.5, -1.4, 1.8, -0.2}},
		}
		for pos, step := range steps {
			want, err := oracle.Forward(pos, step.prior, step.embedding)
			if err != nil {
				t.Fatalf("CPU oracle position %d: %v", pos, err)
			}
			got, err := resident.Forward(pos, step.prior, step.embedding)
			if err != nil {
				t.Fatalf("Vulkan position %d: %v", pos, err)
			}
			assertQwen35MTPVulkanApprox(t, pos, got, want)
			if got := resident.draft.Cache.Len(); got != pos+1 {
				t.Fatalf("position %d host draft cache=%d, want %d", pos, got, pos+1)
			}
			if got := resident.draft.halKV.Len(); got != pos+1 {
				t.Fatalf("position %d resident draft KV=%d, want %d", pos, got, pos+1)
			}
		}

		beforeRefusal := resident.Receipt()
		if _, err := resident.Forward(1, steps[1].prior, steps[1].embedding); err == nil {
			t.Fatal("repeated resident position succeeded")
		} else {
			var stateErr *Qwen35MTPForwardError
			if !errors.As(err, &stateErr) || stateErr.Stage != "position" {
				t.Fatalf("repeated position error=%v, want typed position refusal", err)
			}
		}
		if got := resident.draft.halKV.Len(); got != len(steps) {
			t.Fatalf("repeated position mutated resident draft KV to %d", got)
		}
		if after := resident.Receipt(); after != beforeRefusal {
			t.Fatalf("repeated position mutated receipt: before=%+v after=%+v", beforeRefusal, after)
		}

		receipt := resident.Receipt()
		if receipt.Path != compute.Qwen35MTPDraftPath || receipt.Backend != backend.Name() {
			t.Fatalf("resident path receipt=%+v", receipt)
		}
		if receipt.Positions != uint64(len(steps)) || receipt.DecoderOperations != uint64(len(steps)) || receipt.HeadOperations != uint64(len(steps)) {
			t.Fatalf("resident operation receipt=%+v", receipt)
		}
		if receipt.SetupH2DBytes == 0 || receipt.SetupD2HBytes != 0 {
			t.Fatalf("setup transfer receipt=%+v", receipt)
		}
		wantBoundaryH2D := uint64(len(steps) * 2 * model.Cfg.HiddenSize * compute.F32.Bytes())
		wantBoundaryD2H := uint64(len(steps) * (model.Cfg.HiddenSize + model.Cfg.VocabSize) * compute.F32.Bytes())
		if receipt.BoundaryH2DBytes != wantBoundaryH2D || receipt.BoundaryD2HBytes != wantBoundaryD2H {
			t.Fatalf("boundary transfers H2D/D2H=%d/%d, want %d/%d", receipt.BoundaryH2DBytes, receipt.BoundaryD2HBytes, wantBoundaryH2D, wantBoundaryD2H)
		}
		if receipt.IntermediateH2DBytes != 0 || receipt.IntermediateD2HBytes != 0 {
			t.Fatalf("resident intermediates crossed host boundary: %+v", receipt)
		}

		if err := model.CloseWeights(); err == nil {
			t.Fatal("model weights closed while resident draft held the checkpoint")
		}
		resident.Close()
		if err := model.CloseWeights(); err != nil {
			t.Fatalf("close weights after resident draft: %v", err)
		}
	})

	t.Run("proposal_restore_keeps_target_and_backend_separate", func(t *testing.T) {
		model := qwen35MTPEnabledSyntheticModel(t)
		target := model.NewSession()
		target.captureTargetHidden = true
		defer target.Close()
		prompt := []int{2, 1}
		target.Prefill(prompt)

		draft, err := NewQwen35MTPDraftSessionWithBackend(target, 3, backend)
		if err != nil {
			t.Fatalf("construct resident draft session: %v", err)
		}
		defer draft.Close()
		proposal := draft.Propose(prompt)
		if err := draft.Err(); err != nil {
			t.Fatalf("resident proposal: %v", err)
		}
		if len(proposal) != 3 || draft.pending == nil {
			t.Fatalf("resident proposal=%v pending=%v, want depth 3 with rollback checkpoint", proposal, draft.pending != nil)
		}
		if target.Cache.Len() != len(prompt) {
			t.Fatalf("resident draft mutated target KV length to %d, want %d", target.Cache.Len(), len(prompt))
		}
		if draft.forward.draft.halKV.Len() != len(prompt)+2 {
			t.Fatalf("resident draft KV length=%d, want committed %d + speculative 2", draft.forward.draft.halKV.Len(), len(prompt))
		}

		if err := draft.restoreProposal(); err != nil {
			t.Fatalf("restore speculative proposal: %v", err)
		}
		if draft.forward.draft.halKV.Len() != len(prompt) || draft.forward.draft.Cache.Len() != len(prompt) {
			t.Fatalf("restored draft KV lengths resident/host=%d/%d, want %d", draft.forward.draft.halKV.Len(), draft.forward.draft.Cache.Len(), len(prompt))
		}
		if target.Cache.Len() != len(prompt) {
			t.Fatalf("proposal restore mutated target KV length to %d", target.Cache.Len())
		}

		oldForward := draft.forward
		if err := draft.recreateForward(); err != nil {
			t.Fatalf("recreate resident forward: %v", err)
		}
		if !oldForward.closed {
			t.Fatal("recreate left the prior resident forward open")
		}
		if draft.forward.resident == nil || draft.forward.Receipt().Path != compute.Qwen35MTPDraftPath || draft.forward.Receipt().Backend != backend.Name() {
			t.Fatalf("recreate silently lost resident backend: %+v", draft.forward.Receipt())
		}
		if target.Cache.Len() != len(prompt) {
			t.Fatalf("forward recreation mutated target KV length to %d", target.Cache.Len())
		}
	})
}

func assertQwen35MTPVulkanApprox(t *testing.T, pos int, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("position %d logits length=%d, CPU oracle=%d", pos, len(got), len(want))
	}
	var dot, gotNorm, wantNorm, maxDelta, maxReference float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("position %d logits[%d]=%v", pos, i, got[i])
		}
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("position %d CPU oracle logits[%d]=%v", pos, i, want[i])
		}
		dot += g * w
		gotNorm += g * g
		wantNorm += w * w
		maxDelta = math.Max(maxDelta, math.Abs(g-w))
		maxReference = math.Max(maxReference, math.Abs(w))
	}
	cosine := dot / math.Sqrt(gotNorm*wantNorm)
	if math.IsNaN(cosine) || math.IsInf(cosine, 0) || cosine < 0.99999 || maxDelta > 2e-5*math.Max(1, maxReference) {
		t.Fatalf("position %d Vulkan/CPU logits cosine=%.9f maxDelta=%g referenceMax=%g", pos, cosine, maxDelta, maxReference)
	}
	if gotArgmax, wantArgmax := argmaxF32(got), argmaxF32(want); gotArgmax != wantArgmax {
		t.Fatalf("position %d Vulkan argmax=%d, CPU oracle=%d", pos, gotArgmax, wantArgmax)
	}
}

// qwen35MTPVulkanOracleModel exercises the resident attention path instead of
// letting an all-zero Q/K/V fixture reduce this witness to an MLP-only check.
// HeadDim=4 with a 2-wide rotary prefix covers partial RoPE at position 1;
// non-zero Q/K norms, packed query gates, V/O projections, and positive eps
// make a broken attention, gate, or normalization operation observable.
func qwen35MTPVulkanOracleModel(t *testing.T) *Model {
	t.Helper()
	dense := func(rows, cols, seed int) []float32 {
		out := make([]float32, rows*cols)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				n := ((row+1)*17 + (col+1)*11 + seed*7) % 29
				out[row*cols+col] = float32(n-14) * 0.025
			}
		}
		return out
	}
	weights := map[string]testMTPWeight{
		"mtp.pre_fc_norm_hidden.weight":                {shape: []int{4}, data: []float32{0.02, -0.03, 0.04, -0.01}},
		"mtp.pre_fc_norm_embedding.weight":             {shape: []int{4}, data: []float32{-0.04, 0.01, 0.03, -0.02}},
		"mtp.fc.weight":                                {shape: []int{4, 8}, data: dense(4, 8, 1)},
		"mtp.norm.weight":                              {shape: []int{4}, data: []float32{0.03, -0.02, 0.01, 0.05}},
		"mtp.layers.0.input_layernorm.weight":          {shape: []int{4}, data: []float32{-0.02, 0.04, 0.01, -0.03}},
		"mtp.layers.0.post_attention_layernorm.weight": {shape: []int{4}, data: []float32{0.01, 0.03, -0.04, 0.02}},
		"mtp.layers.0.self_attn.q_norm.weight":         {shape: []int{4}, data: []float32{0.05, -0.02, 0.03, -0.01}},
		"mtp.layers.0.self_attn.k_norm.weight":         {shape: []int{4}, data: []float32{-0.01, 0.04, -0.03, 0.02}},
		"mtp.layers.0.self_attn.q_proj.weight":         {shape: []int{8, 4}, data: dense(8, 4, 2)},
		"mtp.layers.0.self_attn.k_proj.weight":         {shape: []int{4, 4}, data: dense(4, 4, 3)},
		"mtp.layers.0.self_attn.v_proj.weight":         {shape: []int{4, 4}, data: dense(4, 4, 4)},
		"mtp.layers.0.self_attn.o_proj.weight":         {shape: []int{4, 4}, data: dense(4, 4, 5)},
		"mtp.layers.0.mlp.gate_proj.weight":            {shape: []int{6, 4}, data: dense(6, 4, 6)},
		"mtp.layers.0.mlp.up_proj.weight":              {shape: []int{6, 4}, data: dense(6, 4, 7)},
		"mtp.layers.0.mlp.down_proj.weight":            {shape: []int{4, 6}, data: dense(4, 6, 8)},
		"lm_head.weight": {shape: []int{5, 4}, data: []float32{
			0.7, -0.2, 0.1, 0.3,
			-0.4, 0.8, -0.1, 0.2,
			0.2, 0.1, 0.9, -0.5,
			-0.3, -0.4, 0.2, 0.7,
			0.5, 0.6, -0.7, 0.1,
		}},
	}

	manifest := make(map[string]tensorMeta, len(weights))
	raw := make([]byte, 0)
	for name, weight := range weights {
		offset := len(raw)
		for _, value := range weight.data {
			var encoded [4]byte
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			raw = append(raw, encoded[:]...)
		}
		manifest[name] = tensorMeta{Dtype: "F32", Shape: weight.shape, Offset: offset, Nbytes: len(weight.data) * 4}
	}
	cfg := qwen35MTPTestConfig()
	cfg.HiddenSize = 4
	cfg.NumLayers = 1
	cfg.NumHeads = 1
	cfg.NumKVHeads = 1
	cfg.HeadDim = 4
	cfg.PartialRotaryFactor = 0.5
	cfg.IntermediateSize = 6
	cfg.VocabSize = 5
	cfg.RMSNormEps = 1e-5
	cfg.QKNormEps = 1e-5
	cfg.RopeTheta = 10000
	cfg.AttnOutputGate = true
	cfg.QKNorm = true
	cfg.NormGain1p = true
	return &Model{Cfg: cfg, manifest: manifest, raw: raw}
}
