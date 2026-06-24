package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"unsafe"
)

const cacheDir = ".cache/smollm2-135m"
const oracleDirsEnv = "FAK_ORACLE_DIRS"
const oracleRequiredFamiliesEnv = "FAK_ORACLE_REQUIRED_FAMILIES"
const glmOracleDir = ".cache/oracle-glm"
const glmOracleModel = "yujiepan/glm-5-tiny-random"
const glmOraclePromptIDsJSON = `[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]`
const glmOracleExportHint = "from fak/: python internal/model/export_oracle.py --online --trust-remote-code --model " +
	glmOracleModel + " --out " + glmOracleDir + " --prompt-ids-json '" + glmOraclePromptIDsJSON + "'"

// MiniMax-M3 MSA oracle. No public tiny `minimax_m3` checkpoint exists, so the fixture
// is built locally (à la yujiepan/glm-5-tiny-random) by make_minimax_m3_tiny.py — a small
// text-only MiniMaxM3VL decoder with random weights, instantiable on a plain CPU box with
// transformers>=5.12 (no GPU/artifact node needed). The exporter then translates the real
// config (flat index_* or nested sparse_attention_config) into fak's flat MSA axes.
const minimaxOracleDir = ".cache/oracle-minimax"
const minimaxOracleModel = ".cache/minimax-m3-tiny"
const minimaxOraclePromptIDsJSON = `[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]`
const minimaxOracleExportHint = "from fak/: python internal/model/make_minimax_m3_tiny.py " + minimaxOracleModel +
	" && python internal/model/export_oracle.py --online --model " + minimaxOracleModel +
	" --out internal/model/" + minimaxOracleDir + " --prompt-ids-json '" + minimaxOraclePromptIDsJSON + "'"

// qwen3_5 (Qwen3.6 / Qwen3-Next) Gated-DeltaNet oracle (#447). No public tiny `qwen3_5`
// checkpoint exists (only the 27B is published), so the fixture is built locally by
// make_qwen35_tiny.py — a small text-only Qwen3_5ForCausalLM with 3 linear_attention
// (Gated DeltaNet) layers + 1 gated full_attention layer, instantiable on a plain CPU
// box with transformers>=5.10 (no GPU / 27B artifact node). This is the witness for the
// "CPU reference first (bit-exact vs HF)" subtask: HF transformers (which we did NOT
// author) is the reference the pure-Go qwen35 forward must reproduce to f32 tolerance.
const qwen35OracleDir = ".cache/oracle-qwen35"
const qwen35OracleModel = ".cache/qwen35-tiny"
const qwen35OraclePromptIDsJSON = `[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]`
const qwen35OracleExportHint = "from fak/: python internal/model/make_qwen35_tiny.py " + qwen35OracleModel +
	" && python internal/model/export_oracle.py --online --model " + qwen35OracleModel +
	" --out internal/model/" + qwen35OracleDir + " --prompt-ids-json '" + qwen35OraclePromptIDsJSON + "'"

type oraclePrompt struct {
	Index     int              `json:"index"`
	Text      string           `json:"text"`
	Ids       []int            `json:"ids"`
	DSATraces []oracleDSATrace `json:"dsa_traces,omitempty"`
	MSATraces []oracleMSATrace `json:"msa_traces,omitempty"`

	HiddenShape []int `json:"hidden_shape"`
	LogitsShape []int `json:"logits_shape"`
	ArgmaxPos   []int `json:"argmax_per_pos"`
	GreedyIds   []int `json:"greedy_ids"`
}

type oracleDSATrace struct {
	Layer           int     `json:"layer"`
	Module          string  `json:"module"`
	Source          string  `json:"source"`
	TopKShape       []int   `json:"topk_shape"`
	TopKIndices     [][]int `json:"topk_indices"`
	AttnOutputShape []int   `json:"attn_output_shape"`
	AttnOutputFile  string  `json:"attn_output_file"`
}

// oracleMSATrace is the MiniMax-M3 lightning-indexer trace: BlockTopK is the per-query
// selected key-BLOCK indices (right-padded with -1) — one set per query, since the HF
// MiniMaxM3VLIndexer max-pools its block scores over all index heads (amax(dim=1)) before
// the top-k, so the selection is shared by every attention head. AttnOutput* is the
// per-layer attention output.
type oracleMSATrace struct {
	Layer           int     `json:"layer"`
	Module          string  `json:"module"`
	Source          string  `json:"source"`
	BlockTopKShape  []int   `json:"block_topk_shape"`
	BlockTopK       [][]int `json:"block_topk"`
	AttnOutputShape []int   `json:"attn_output_shape"`
	AttnOutputFile  string  `json:"attn_output_file"`
}

type evictionFixture struct {
	PrefixIds      []int `json:"prefix_ids"`
	PoisonIds      []int `json:"poison_ids"`
	QueryIds       []int `json:"query_ids"`
	NeverGreedy    []int `json:"never_greedy"`
	PoisonedGreedy []int `json:"poisoned_greedy"`
}

type oracleDoc struct {
	Model    string          `json:"model"`
	Prompts  []oraclePrompt  `json:"prompts"`
	Eviction evictionFixture `json:"eviction"`
}

type oracleRawDoc struct {
	Model   string          `json:"model"`
	Config  json.RawMessage `json:"config"`
	Prompts []oraclePrompt  `json:"prompts"`
}

// loadOracle / loadModel skip the test cleanly when the (gitignored, 538MB) export
// is absent, so CI without weights stays green; locally it is the real witness.
func loadFixture(t *testing.T) (*Model, oracleDoc) {
	t.Helper()
	m, doc := loadFixtureDir(t, cacheDir, true)
	return m, doc
}

func loadFixtureDir(t *testing.T, dir string, missingIsSkip bool) (*Model, oracleDoc) {
	t.Helper()
	// -short skips the weight-backed witnesses: loading the gitignored ~538MB f32
	// export is the slow, OOM-prone part of the WSL suite (see fak/test.sh + the
	// model-test OOM note). `-short` keeps the synthetic + architest invariants —
	// the ~95% of logic regressions — so it is a reliable 2s pre-commit/pre-push
	// gate. The full suite (no -short) still runs the real oracle locally.
	if testing.Short() {
		t.Skip("weight-backed oracle witness skipped under -short (load is the OOM/slow path)")
	}
	resolved, ok := resolveOracleDir(dir)
	if !ok {
		msg := "no exported weights in " + dir + "; run: python internal/model/export_oracle.py --out " + dir
		if missingIsSkip {
			t.Skip(msg)
		}
		t.Fatal(msg)
	}
	m, err := Load(resolved)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var doc oracleDoc
	if err := readJSON(filepath.Join(resolved, "oracle.json"), &doc); err != nil {
		t.Fatalf("oracle: %v", err)
	}
	return m, doc
}

func resolveOracleDir(dir string) (string, bool) {
	if _, err := os.Stat(filepath.Join(dir, "weights.f32")); err == nil {
		return dir, true
	}
	if filepath.IsAbs(dir) {
		return dir, false
	}
	repoRootRelative := filepath.Join("..", "..", dir)
	if _, err := os.Stat(filepath.Join(repoRootRelative, "weights.f32")); err == nil {
		return repoRootRelative, true
	}
	return dir, false
}

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func maxAbsDiff(a, b []float32) (float64, int) {
	mx, at := 0.0, -1
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > mx {
			mx, at = d, i
		}
	}
	return mx, at
}

// cosine similarity — the right witness for hidden states, whose absolute scale is
// dominated by a few outlier "massive activation" dims (a known late-layer
// residual-stream phenomenon) that make an absolute tolerance meaningless.
func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na)*math.Sqrt(nb) + 1e-30)
}

func argmax(v []float32) int {
	bi, bv := 0, v[0]
	for i, x := range v {
		if x > bv {
			bv, bi = x, i
		}
	}
	return bi
}

func oracleMatrixDirsFromEnv(raw string) ([]string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return []string{cacheDir}, false
	}
	seen := map[string]bool{}
	var dirs []string
	for _, dir := range splitOraclePathList(raw) {
		dir = strings.TrimSpace(dir)
		if dir == "" || seen[dir] {
			continue
		}
		seen[dir] = true
		dirs = append(dirs, dir)
	}
	return dirs, true
}

func splitOraclePathList(raw string) []string {
	if strings.Contains(raw, ";") {
		return strings.Split(raw, ";")
	}
	return filepath.SplitList(raw)
}

func oracleMatrixDirs() ([]string, bool) {
	return oracleMatrixDirsFromEnv(os.Getenv(oracleDirsEnv))
}

func oracleTestName(dir string) string {
	name := filepath.Base(filepath.Clean(dir))
	if name == "." || name == string(filepath.Separator) {
		name = dir
	}
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return r
		case r == '-' || r == '_':
			return r
		default:
			return '_'
		}
	}, name)
}

func normalizeOracleFamily(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	r := strings.NewReplacer("_", "", "-", "", " ", "")
	return r.Replace(s)
}

func requiredOracleFamilies(raw string) []string {
	seen := map[string]bool{}
	var out []string
	for _, item := range strings.Split(raw, ",") {
		item = normalizeOracleFamily(item)
		if item == "" || seen[item] {
			continue
		}
		seen[item] = true
		out = append(out, item)
	}
	return out
}

func missingOracleFamilies(required []string, have map[string]string) []string {
	var missing []string
	for _, want := range required {
		found := false
		for family := range have {
			if strings.Contains(family, want) {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, want)
		}
	}
	return missing
}

// TestForwardMatchesHFOracle is the rung-1 witness: the pure-Go forward pass must
// reproduce HF's per-layer hidden states and final logits to f32 tolerance, and
// pick the SAME next token at every position. HF authored the oracle, not us.
func TestForwardMatchesHFOracle(t *testing.T) {
	dirs, explicit := oracleMatrixDirs()
	if len(dirs) == 0 {
		t.Fatalf("%s is set but contains no oracle directories", oracleDirsEnv)
	}
	for _, dir := range dirs {
		dir := dir
		t.Run(oracleTestName(dir), func(t *testing.T) {
			m, doc := loadFixtureDir(t, dir, !explicit)
			resolved, _ := resolveOracleDir(dir)
			assertForwardMatchesHFOracle(t, resolved, m, doc)
		})
	}
}

func TestOptionalMistralSWAOracleNonVacuous(t *testing.T) {
	const dir = ".cache/oracle-mistral-swa"
	m, doc := loadFixtureDir(t, dir, true)
	if !strings.Contains(m.Cfg.archFamilyKey(), "mistral") {
		t.Fatalf("%s family = %q, want mistral", dir, m.Cfg.archFamilyKey())
	}
	nonVacuous := false
	for l := 0; l < m.Cfg.NumLayers; l++ {
		W := m.Cfg.windowForLayer(l)
		if W < 0 {
			continue
		}
		for _, p := range doc.Prompts {
			if len(p.Ids) > W {
				nonVacuous = true
				break
			}
		}
	}
	if !nonVacuous {
		t.Fatalf("%s has no prompt longer than its configured sliding window %v", dir, m.Cfg.Window)
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalLlama3OracleCoversScalingAndEOSList(t *testing.T) {
	const dir = ".cache/oracle-llama3"
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "llama") {
		t.Fatalf("%s family = %q, want llama", dir, cfg.archFamilyKey())
	}
	if cfg.RopeScaling != "llama3" || cfg.RopeFactor == 0 || cfg.RopeOrigContext == 0 {
		t.Fatalf("%s rope scaling = %q factor=%v orig=%d, want populated llama3 scaling",
			dir, cfg.RopeScaling, cfg.RopeFactor, cfg.RopeOrigContext)
	}
	if len(cfg.EOSTokenIDs) < 2 {
		t.Fatalf("%s EOS ids = %v, want a Llama-3-style EOS list", dir, cfg.EOSTokenIDs)
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalQwen3OracleCoversQKNorm(t *testing.T) {
	const dir = ".cache/oracle-qwen3"
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "qwen3") {
		t.Fatalf("%s family = %q, want qwen3", dir, cfg.archFamilyKey())
	}
	if !cfg.QKNorm {
		t.Fatalf("%s did not derive QKNorm=true from qwen3 metadata", dir)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for _, suffix := range []string{"q_norm", "k_norm"} {
			name := layerPrefix(l) + "self_attn." + suffix + ".weight"
			meta, ok := m.manifest[name]
			if !ok {
				t.Fatalf("%s missing qk-norm tensor %s", dir, name)
			}
			if len(meta.Shape) != 1 || meta.Shape[0] != cfg.HeadDim {
				t.Fatalf("%s tensor %s shape = %v, want [%d]", dir, name, meta.Shape, cfg.HeadDim)
			}
		}
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

// TestOptionalOLMo2OracleCoversPostNormFullProjQKNorm is the real-oracle witness
// for the OLMo 2 row of the #474 matrix. OLMo 2 is the POST-norm family (the norm
// runs AFTER each sub-layer on the raw residual, not before it) and it carries
// qk-norms over the WHOLE q/k projection — not per-head, the way Qwen3/Gemma3 do.
// Both axes are family-derived by the loader (BlockTopology=PostNorm in weights.go,
// QKNorm=true + the full-projection branch of applyQKNormCfg in arch.go); this proves
// the pure-Go forward reproduces HF for a real OLMo 2 decoder, not just the config.
//
// Reproduce the gitignored (~26MB) fixture on a plain CPU box (no GPU/artifact node):
//
//	from fak/: python internal/model/make_olmo2_tiny.py .cache/olmo2-tiny
//	  && python internal/model/export_oracle.py --online --model .cache/olmo2-tiny \
//	     --out internal/model/.cache/oracle-olmo2 \
//	     --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'
func TestOptionalOLMo2OracleCoversPostNormFullProjQKNorm(t *testing.T) {
	const dir = ".cache/oracle-olmo2"
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "olmo2") {
		t.Fatalf("%s family = %q, want olmo2", dir, cfg.archFamilyKey())
	}
	if cfg.BlockTopology != PostNorm {
		t.Fatalf("%s topology = %v, want PostNorm", dir, cfg.BlockTopology)
	}
	if !cfg.QKNorm {
		t.Fatalf("%s did not derive QKNorm=true from olmo2 metadata", dir)
	}
	// OLMo 2 is the POST-norm placement: no input_layernorm; the only per-layer
	// norms are post_attention_layernorm + post_feedforward_layernorm.
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		if _, ok := m.manifest[p+"input_layernorm.weight"]; ok {
			t.Fatalf("%s layer %d unexpectedly has input_layernorm (OLMo2 is PostNorm)", dir, l)
		}
		for _, suffix := range []string{"post_attention_layernorm.weight", "post_feedforward_layernorm.weight"} {
			if _, ok := m.manifest[p+suffix]; !ok {
				t.Fatalf("%s layer %d missing %s", dir, l, suffix)
			}
		}
		// qk-norm is over the FULL projection, not head_dim: q_norm is
		// NumHeads*HeadDim, k_norm is NumKVHeads*HeadDim (distinguishing OLMo2
		// from the per-head Qwen3/Gemma3 qk-norm).
		qn, ok := m.manifest[p+"self_attn.q_norm.weight"]
		if !ok {
			t.Fatalf("%s layer %d missing self_attn.q_norm.weight", dir, l)
		}
		if len(qn.Shape) != 1 || qn.Shape[0] != cfg.NumHeads*cfg.HeadDim {
			t.Fatalf("%s q_norm shape = %v, want full projection [%d]", dir, qn.Shape, cfg.NumHeads*cfg.HeadDim)
		}
		kn, ok := m.manifest[p+"self_attn.k_norm.weight"]
		if !ok {
			t.Fatalf("%s layer %d missing self_attn.k_norm.weight", dir, l)
		}
		if len(kn.Shape) != 1 || kn.Shape[0] != cfg.NumKVHeads*cfg.HeadDim {
			t.Fatalf("%s k_norm shape = %v, want full kv projection [%d]", dir, kn.Shape, cfg.NumKVHeads*cfg.HeadDim)
		}
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalQwen3MoEOracleCoversHybridDenseSparseLayers(t *testing.T) {
	const dir = ".cache/oracle-qwen3moe"
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "qwen3moe") {
		t.Fatalf("%s family = %q, want qwen3moe", dir, cfg.archFamilyKey())
	}
	if !cfg.IsMoE() {
		t.Fatalf("%s did not load MoE config: experts=%d topk=%d", dir, cfg.NumExperts, cfg.NumExpertsPerTok)
	}
	denseLayers, sparseLayers := 0, 0
	for l := 0; l < cfg.NumLayers; l++ {
		_, hasDense := m.manifest[layerName(l, "mlp.gate_proj.weight")]
		_, hasRouter := m.manifest[routerName(l)]
		if hasDense {
			denseLayers++
			if _, ok := m.ffnForLayer(l).(denseSwiGLU); !ok {
				t.Fatalf("%s layer %d selected %T for dense tensors, want denseSwiGLU", dir, l, m.ffnForLayer(l))
			}
			for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				if _, ok := m.manifest[layerName(l, "mlp."+suffix)]; !ok {
					t.Fatalf("%s dense layer %d missing mlp.%s", dir, l, suffix)
				}
			}
		}
		if hasRouter {
			sparseLayers++
			if _, ok := m.ffnForLayer(l).(moeFFN); !ok {
				t.Fatalf("%s layer %d selected %T for router, want moeFFN", dir, l, m.ffnForLayer(l))
			}
			for _, suffix := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
				if _, ok := m.manifest[expertName(l, 0, suffix)]; !ok {
					t.Fatalf("%s sparse layer %d missing expert 0 %s", dir, l, suffix)
				}
			}
		}
	}
	if denseLayers == 0 || sparseLayers == 0 {
		t.Fatalf("%s dense/sparse layer counts = %d/%d, want both > 0", dir, denseLayers, sparseLayers)
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalGemma3OracleCoversLocalGlobalAttention(t *testing.T) {
	const dir = ".cache/oracle-gemma3"
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "gemma3") {
		t.Fatalf("%s family = %q, want gemma3", dir, cfg.archFamilyKey())
	}
	if cfg.BlockTopology != SandwichNorm || !cfg.NormGain1p || !cfg.ActGeluTanh || !cfg.QKNorm {
		t.Fatalf("%s gemma3 axes topology=%v gain1p=%v gelu_tanh=%v qknorm=%v",
			dir, cfg.BlockTopology, cfg.NormGain1p, cfg.ActGeluTanh, cfg.QKNorm)
	}
	if cfg.QueryPreAttnScalar == 0 {
		t.Fatalf("%s query_pre_attn_scalar was not preserved", dir)
	}
	slidingLayer, fullLayer := -1, -1
	for l, typ := range cfg.LayerTypes {
		switch typ {
		case "sliding_attention":
			slidingLayer = l
		case "full_attention":
			fullLayer = l
		}
	}
	if slidingLayer < 0 || fullLayer < 0 {
		t.Fatalf("%s layer_types = %v, want both sliding_attention and full_attention", dir, cfg.LayerTypes)
	}
	if len(cfg.Window) <= slidingLayer || cfg.Window[slidingLayer] <= 0 {
		t.Fatalf("%s sliding layer %d window = %v, want positive window", dir, slidingLayer, cfg.Window)
	}
	if len(cfg.Window) <= fullLayer || cfg.Window[fullLayer] != -1 {
		t.Fatalf("%s full layer %d window = %v, want -1", dir, fullLayer, cfg.Window)
	}
	if len(cfg.RopeThetaPerLayer) <= fullLayer || cfg.RopeThetaPerLayer[slidingLayer] == cfg.RopeThetaPerLayer[fullLayer] {
		t.Fatalf("%s rope theta per layer = %v, want distinct local/global theta", dir, cfg.RopeThetaPerLayer)
	}
	for l := 0; l < cfg.NumLayers; l++ {
		for _, suffix := range []string{"q_norm", "k_norm"} {
			name := layerPrefix(l) + "self_attn." + suffix + ".weight"
			meta, ok := m.manifest[name]
			if !ok {
				t.Fatalf("%s missing qk-norm tensor %s", dir, name)
			}
			if len(meta.Shape) != 1 || meta.Shape[0] != cfg.HeadDim {
				t.Fatalf("%s tensor %s shape = %v, want [%d]", dir, name, meta.Shape, cfg.HeadDim)
			}
		}
	}
	nonVacuous := false
	for _, p := range doc.Prompts {
		if len(p.Ids) > cfg.Window[slidingLayer] {
			nonVacuous = true
			break
		}
	}
	if !nonVacuous {
		t.Fatalf("%s has no prompt longer than sliding window %d", dir, cfg.Window[slidingLayer])
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalDeepSeekV2OracleDocumentsMLABoundary(t *testing.T) {
	const dir = ".cache/oracle-deepseek-v2"
	resolved, ok := resolveOracleDir(dir)
	if !ok {
		t.Skip("no exported DeepSeek V2 oracle; run export_oracle.py with --trust-remote-code")
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	if !strings.Contains(cfg.archFamilyKey(), "deepseek") {
		t.Fatalf("%s family = %q, want deepseek", dir, cfg.archFamilyKey())
	}
	if cfg.QLoraRank <= 0 || cfg.KVLoraRank <= 0 || cfg.QKNopeHeadDim <= 0 || cfg.QKRopeHeadDim <= 0 || cfg.VHeadDim <= 0 {
		t.Fatalf("%s did not preserve MLA metadata: q_lora=%d kv_lora=%d nope=%d rope=%d v=%d",
			dir, cfg.QLoraRank, cfg.KVLoraRank, cfg.QKNopeHeadDim, cfg.QKRopeHeadDim, cfg.VHeadDim)
	}
	if cfg.HeadDim != cfg.QKNopeHeadDim+cfg.QKRopeHeadDim {
		t.Fatalf("%s head_dim=%d, want qk_nope+qk_rope=%d",
			dir, cfg.HeadDim, cfg.QKNopeHeadDim+cfg.QKRopeHeadDim)
	}
	if cfg.VHeadDim == cfg.HeadDim {
		t.Fatalf("%s fixture is not exercising DeepSeek's split qk/v head widths: head=%d v=%d",
			dir, cfg.HeadDim, cfg.VHeadDim)
	}
	if cfg.MoEIntermediateSize <= 0 || cfg.MoEIntermediateSize == cfg.IntermediateSize {
		t.Fatalf("%s moe_intermediate_size=%d intermediate_size=%d, want distinct DeepSeek dense/MoE widths",
			dir, cfg.MoEIntermediateSize, cfg.IntermediateSize)
	}

	var doc oracleDoc
	if err := readJSON(filepath.Join(resolved, "oracle.json"), &doc); err != nil {
		t.Fatalf("oracle: %v", err)
	}
	if len(doc.Prompts) == 0 {
		t.Fatalf("%s oracle contains no prompts", dir)
	}
	var man map[string]tensorMeta
	if err := readJSON(filepath.Join(resolved, "manifest.json"), &man); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	expectShape := func(name string, want []int) {
		t.Helper()
		meta, ok := man[name]
		if !ok {
			t.Fatalf("%s missing MLA tensor %s", dir, name)
		}
		if !sameShape(meta.Shape, want) {
			t.Fatalf("%s tensor %s shape = %v, want %v", dir, name, meta.Shape, want)
		}
	}
	p := layerPrefix(0) + "self_attn."
	expectShape(p+"q_a_proj.weight", []int{cfg.QLoraRank, cfg.HiddenSize})
	expectShape(p+"q_b_proj.weight", []int{cfg.NumHeads * cfg.HeadDim, cfg.QLoraRank})
	expectShape(p+"kv_a_proj_with_mqa.weight", []int{cfg.KVLoraRank + cfg.QKRopeHeadDim, cfg.HiddenSize})
	expectShape(p+"kv_b_proj.weight", []int{cfg.NumHeads * (cfg.QKNopeHeadDim + cfg.VHeadDim), cfg.KVLoraRank})
	expectShape(p+"o_proj.weight", []int{cfg.HiddenSize, cfg.NumHeads * cfg.VHeadDim})
	if _, ok := man[p+"q_proj.weight"]; ok {
		t.Fatalf("%s unexpectedly has standard q_proj tensor; DeepSeek V2 should use MLA q_a/q_b projections", dir)
	}
	if cfg.NumHeads*cfg.HeadDim == cfg.NumHeads*cfg.VHeadDim {
		t.Fatalf("%s standard o_proj input width unexpectedly matches MLA v width", dir)
	}
}

func TestOptionalGLMMoeDsaOracleExportMetadataCurrent(t *testing.T) {
	resolved, ok := resolveOracleDir(glmOracleDir)
	if !ok {
		t.Skip("no exported GLM oracle; run " + glmOracleExportHint)
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	var raw oracleRawDoc
	if err := readJSON(filepath.Join(resolved, "oracle.json"), &raw); err != nil {
		t.Fatalf("oracle: %v", err)
	}
	if raw.Model != glmOracleModel {
		t.Fatalf("%s oracle model = %q, want %q; re-export with: %s", glmOracleDir, raw.Model, glmOracleModel, glmOracleExportHint)
	}
	if len(raw.Config) == 0 {
		t.Fatalf("%s oracle.json has no embedded config; re-export with current export_oracle.py: %s", glmOracleDir, glmOracleExportHint)
	}
	var embedded Config
	if err := json.Unmarshal(raw.Config, &embedded); err != nil {
		t.Fatalf("%s oracle embedded config: %v", glmOracleDir, err)
	}
	for _, got := range []Config{cfg, embedded} {
		if got.ModelType != "glm_moe_dsa" || !got.isGLMMoeDsa() {
			t.Fatalf("%s config model_type=%q family=%q, want glm_moe_dsa; re-export with: %s",
				glmOracleDir, got.ModelType, got.archFamilyKey(), glmOracleExportHint)
		}
		if got.NumLayers == 0 || got.IndexTopK == 0 || got.QLoraRank == 0 || got.KVLoraRank == 0 {
			t.Fatalf("%s config is missing GLM DSA axes; re-export with: %s", glmOracleDir, glmOracleExportHint)
		}
	}
	if cfg.NumLayers != embedded.NumLayers || cfg.HiddenSize != embedded.HiddenSize || cfg.VocabSize != embedded.VocabSize {
		t.Fatalf("%s config.json and oracle.json embedded config diverge; re-export with: %s", glmOracleDir, glmOracleExportHint)
	}
	var wantIDs [][]int
	if err := json.Unmarshal([]byte(glmOraclePromptIDsJSON), &wantIDs); err != nil {
		t.Fatalf("bad glmOraclePromptIDsJSON: %v", err)
	}
	if len(raw.Prompts) != len(wantIDs) {
		t.Fatalf("%s prompts = %d, want %d; re-export with pinned prompt ids: %s",
			glmOracleDir, len(raw.Prompts), len(wantIDs), glmOracleExportHint)
	}
	for i, want := range wantIDs {
		if !sameInts(raw.Prompts[i].Ids, want) {
			t.Fatalf("%s prompt %d ids = %v, want pinned ids %v; re-export with: %s",
				glmOracleDir, i, raw.Prompts[i].Ids, want, glmOracleExportHint)
		}
	}
}

// TestOptionalGLMMoeDsaOracleDocumentsDSABoundary is the real-artifact boundary
// witness for the GLM-5.2 family (model_type "glm_moe_dsa": MoE + Dynamic Sparse
// Attention + IndexShare + MTP). It mirrors TestOptionalDeepSeekV2OracleDocumentsMLABoundary:
// it asserts the loader/shape boundary so a real artifact is not misread as the
// standard q/k/v path. Numeric DSA attention, cacheless Forward parity, and
// incremental Session cache parity are asserted by the tests below.
//
// When a tiny glm_moe_dsa oracle is exported to .cache/oracle-glm (see
// glmOracleExportHint), this test proves: the family is derived; the
// MoE config is preserved; GLM's DeepSeek-V3-style MLA projections and DSA indexer tensors
// are present; and dense-prefix MoE replacement is distinguished from routed MoE layers.
// Skipped until that artifact exists.
func TestOptionalGLMMoeDsaOracleDocumentsDSABoundary(t *testing.T) {
	const dir = glmOracleDir
	resolved, ok := resolveOracleDir(dir)
	if !ok {
		t.Skip("no exported GLM oracle; run " + glmOracleExportHint)
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	if !cfg.isGLM() {
		t.Fatalf("%s family = %q, want glm", dir, cfg.archFamilyKey())
	}
	if !cfg.isGLMMoeDsa() {
		t.Fatalf("%s family = %q, want the glm_moe_dsa (dsa) variant", dir, cfg.archFamilyKey())
	}
	if !cfg.IsMoE() {
		t.Fatalf("%s did not preserve MoE config: experts=%d topk=%d", dir, cfg.NumExperts, cfg.NumExpertsPerTok)
	}

	var doc oracleDoc
	if err := readJSON(filepath.Join(resolved, "oracle.json"), &doc); err != nil {
		t.Fatalf("oracle: %v", err)
	}
	if len(doc.Prompts) == 0 {
		t.Fatalf("%s oracle contains no prompts", dir)
	}
	var man map[string]tensorMeta
	if err := readJSON(filepath.Join(resolved, "manifest.json"), &man); err != nil {
		t.Fatalf("manifest: %v", err)
	}

	if cfg.QLoraRank == 0 || cfg.KVLoraRank == 0 || cfg.QKNopeHeadDim == 0 ||
		cfg.QKRopeHeadDim == 0 || cfg.VHeadDim == 0 {
		t.Fatalf("%s missing MLA config axes; re-export with current export_oracle.py", dir)
	}
	if cfg.IndexNHeads == 0 || cfg.IndexHeadDim == 0 || cfg.IndexTopK == 0 || len(cfg.IndexerTypes) == 0 {
		t.Fatalf("%s missing DSA indexer config axes; re-export with current export_oracle.py", dir)
	}
	if len(cfg.IndexerTypes) != cfg.NumLayers {
		t.Fatalf("%s indexer_types len = %d, want num_hidden_layers %d", dir, len(cfg.IndexerTypes), cfg.NumLayers)
	}
	expectShape := func(name string, want []int) {
		t.Helper()
		meta, ok := man[name]
		if !ok {
			t.Fatalf("%s missing GLM-MoE-DSA tensor %s", dir, name)
		}
		if !sameShape(meta.Shape, want) {
			t.Fatalf("%s tensor %s shape = %v, want %v", dir, name, meta.Shape, want)
		}
	}
	expectMissing := func(name string) {
		t.Helper()
		if _, ok := man[name]; ok {
			t.Fatalf("%s unexpectedly has standard-attention tensor %s; GLM-MoE-DSA uses MLA q_a/q_b and kv_a/kv_b projections", dir, name)
		}
	}

	p := layerPrefix(0) + "self_attn."
	expectShape(p+"q_a_proj.weight", []int{cfg.QLoraRank, cfg.HiddenSize})
	expectShape(p+"q_a_layernorm.weight", []int{cfg.QLoraRank})
	expectShape(p+"q_b_proj.weight", []int{cfg.NumHeads * (cfg.QKNopeHeadDim + cfg.QKRopeHeadDim), cfg.QLoraRank})
	expectShape(p+"kv_a_proj_with_mqa.weight", []int{cfg.KVLoraRank + cfg.QKRopeHeadDim, cfg.HiddenSize})
	expectShape(p+"kv_a_layernorm.weight", []int{cfg.KVLoraRank})
	expectShape(p+"kv_b_proj.weight", []int{cfg.NumHeads * (cfg.QKNopeHeadDim + cfg.VHeadDim), cfg.KVLoraRank})
	expectShape(p+"o_proj.weight", []int{cfg.HiddenSize, cfg.NumHeads * cfg.VHeadDim})
	for _, suffix := range []string{"q_proj.weight", "k_proj.weight", "v_proj.weight"} {
		expectMissing(p + suffix)
	}

	ip := p + "indexer."
	expectShape(ip+"wq_b.weight", []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.QLoraRank})
	expectShape(ip+"wk.weight", []int{cfg.IndexHeadDim, cfg.HiddenSize})
	expectShape(ip+"k_norm.weight", []int{cfg.IndexHeadDim})
	expectShape(ip+"k_norm.bias", []int{cfg.IndexHeadDim})
	expectShape(ip+"weights_proj.weight", []int{cfg.IndexNHeads, cfg.HiddenSize})

	moeHidden := cfg.MoEIntermediateSize
	if moeHidden == 0 {
		t.Fatalf("%s missing moe_intermediate_size; re-export with current export_oracle.py", dir)
	}
	if cfg.FirstKDenseReplace > 0 {
		expectShape(layerName(0, "mlp.gate_proj.weight"), []int{moeHidden, cfg.HiddenSize})
		expectShape(layerName(0, "mlp.up_proj.weight"), []int{moeHidden, cfg.HiddenSize})
		expectShape(layerName(0, "mlp.down_proj.weight"), []int{cfg.HiddenSize, moeHidden})
	}
	sparseLayer := cfg.FirstKDenseReplace
	if sparseLayer >= cfg.NumLayers {
		t.Fatalf("%s first_k_dense_replace=%d leaves no routed MoE layer in %d-layer oracle", dir, cfg.FirstKDenseReplace, cfg.NumLayers)
	}
	expectShape(layerName(sparseLayer, "mlp.gate.weight"), []int{cfg.NumExperts, cfg.HiddenSize})
	expectShape(layerName(sparseLayer, "mlp.experts.gate_up_proj"), []int{cfg.NumExperts, 2 * moeHidden, cfg.HiddenSize})
	expectShape(layerName(sparseLayer, "mlp.experts.down_proj"), []int{cfg.NumExperts, cfg.HiddenSize, moeHidden})
	if cfg.NSharedExperts > 0 {
		expectShape(layerName(sparseLayer, "mlp.shared_experts.gate_proj.weight"), []int{moeHidden, cfg.HiddenSize})
		expectShape(layerName(sparseLayer, "mlp.shared_experts.up_proj.weight"), []int{moeHidden, cfg.HiddenSize})
		expectShape(layerName(sparseLayer, "mlp.shared_experts.down_proj.weight"), []int{cfg.HiddenSize, moeHidden})
	}
	for _, prompt := range doc.Prompts {
		if len(prompt.DSATraces) != cfg.NumLayers {
			t.Fatalf("%s prompt %d DSA traces = %d, want one per layer (%d); re-export with current export_oracle.py",
				dir, prompt.Index, len(prompt.DSATraces), cfg.NumLayers)
		}
		seenLayer := map[int]bool{}
		for _, tr := range prompt.DSATraces {
			if tr.Source != "hf_forward_hook" {
				t.Fatalf("%s prompt %d layer %d DSA trace source = %q", dir, prompt.Index, tr.Layer, tr.Source)
			}
			if tr.Layer < 0 || tr.Layer >= cfg.NumLayers || seenLayer[tr.Layer] {
				t.Fatalf("%s prompt %d invalid/duplicate DSA trace layer %d", dir, prompt.Index, tr.Layer)
			}
			seenLayer[tr.Layer] = true
			if !sameShape(tr.TopKShape, []int{1, len(prompt.Ids), len(prompt.Ids)}) {
				t.Fatalf("%s prompt %d layer %d topk_shape = %v, want [1 %d %d]",
					dir, prompt.Index, tr.Layer, tr.TopKShape, len(prompt.Ids), len(prompt.Ids))
			}
			if len(tr.TopKIndices) != len(prompt.Ids) {
				t.Fatalf("%s prompt %d layer %d topk rows = %d, want seq %d",
					dir, prompt.Index, tr.Layer, len(tr.TopKIndices), len(prompt.Ids))
			}
			for qi, row := range tr.TopKIndices {
				if len(row) != len(prompt.Ids) {
					t.Fatalf("%s prompt %d layer %d topk[%d] len = %d, want %d",
						dir, prompt.Index, tr.Layer, qi, len(row), len(prompt.Ids))
				}
				for _, key := range row {
					if key < 0 || key >= len(prompt.Ids) {
						t.Fatalf("%s prompt %d layer %d topk[%d] has out-of-range key %d for seq %d",
							dir, prompt.Index, tr.Layer, qi, key, len(prompt.Ids))
					}
				}
			}
			if !sameShape(tr.AttnOutputShape, []int{1, len(prompt.Ids), cfg.HiddenSize}) {
				t.Fatalf("%s prompt %d layer %d attn_output_shape = %v, want [1 %d %d]",
					dir, prompt.Index, tr.Layer, tr.AttnOutputShape, len(prompt.Ids), cfg.HiddenSize)
			}
			st, err := os.Stat(filepath.Join(resolved, filepath.FromSlash(tr.AttnOutputFile)))
			if err != nil {
				t.Fatalf("%s prompt %d layer %d missing DSA attention output file %s: %v",
					dir, prompt.Index, tr.Layer, tr.AttnOutputFile, err)
			}
			if wantBytes := int64(4 * len(prompt.Ids) * cfg.HiddenSize); st.Size() != wantBytes {
				t.Fatalf("%s prompt %d layer %d DSA attention output bytes = %d, want %d",
					dir, prompt.Index, tr.Layer, st.Size(), wantBytes)
			}
		}
	}

	t.Logf("%s: glm_moe_dsa boundary witnessed — family+MLA+DSA-indexer+MoE geometry + HF DSA traces; "+
		"cacheless DSA forward and incremental Session cache are asserted separately", dir)
}

func TestOptionalGLMMoeDsaOracleReproducesDSAAttentionTrace(t *testing.T) {
	const dir = glmOracleDir
	m, doc := loadFixtureDir(t, dir, true)
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("%s family = %q, want glm_moe_dsa", dir, m.Cfg.archFamilyKey())
	}
	resolved, _ := resolveOracleDir(dir)
	cfg := m.Cfg
	H := cfg.HiddenSize
	for _, prompt := range doc.Prompts {
		seq := len(prompt.Ids)
		hid := readF32(t, filepath.Join(resolved, "oracle", itoa(prompt.Index)+".hidden.f32"))
		if len(hid) != (cfg.NumLayers+1)*seq*H {
			t.Fatalf("%s prompt %d hidden size %d != %d", dir, prompt.Index, len(hid), (cfg.NumLayers+1)*seq*H)
		}
		for _, tr := range prompt.DSATraces {
			layerInput := hid[tr.Layer*seq*H : (tr.Layer+1)*seq*H]
			topK, ok := glmDsaTopKIndices(m, tr.Layer, layerInput, seq)
			if !ok {
				t.Fatalf("%s prompt %d layer %d Go DSA top-k rejected trace input", dir, prompt.Index, tr.Layer)
			}
			for qi, row := range topK {
				want := causalTopKPrefix(tr.TopKIndices[qi], qi)
				if !sameInts(row, want) {
					t.Fatalf("%s prompt %d layer %d top-k[%d] = %v, want HF causal subset %v",
						dir, prompt.Index, tr.Layer, qi, row, want)
				}
			}
			got, ok := glmDsaAttentionOutputFromTopK(m, tr.Layer, layerInput, seq, topK)
			if !ok {
				t.Fatalf("%s prompt %d layer %d Go DSA attention reproduction rejected trace", dir, prompt.Index, tr.Layer)
			}
			ref := readF32(t, filepath.Join(resolved, filepath.FromSlash(tr.AttnOutputFile)))
			if len(ref) != seq*H {
				t.Fatalf("%s prompt %d layer %d DSA trace len = %d, want %d", dir, prompt.Index, tr.Layer, len(ref), seq*H)
			}
			cs := cosine(got, ref)
			d, at := maxAbsDiff(got, ref)
			t.Logf("%s prompt %d layer %d DSA attention trace cos=%.6f max|Δ|=%.3e at %d",
				dir, prompt.Index, tr.Layer, cs, d, at)
			if cs < 0.9999 || d > 1e-3 {
				t.Fatalf("%s prompt %d layer %d DSA attention trace mismatch: cos=%.6f max|Δ|=%.3e at %d",
					dir, prompt.Index, tr.Layer, cs, d, at)
			}
		}
	}
}

func TestOptionalGLMMoeDsaOracleReproducesDensePrefixLayer(t *testing.T) {
	const dir = glmOracleDir
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !cfg.isGLMMoeDsa() {
		t.Fatalf("%s family = %q, want glm_moe_dsa", dir, cfg.archFamilyKey())
	}
	if cfg.FirstKDenseReplace <= 0 {
		t.Fatalf("%s first_k_dense_replace=%d, want a dense prefix layer", dir, cfg.FirstKDenseReplace)
	}
	resolved, _ := resolveOracleDir(dir)
	H := cfg.HiddenSize
	mat := residentKernel{m}
	for _, prompt := range doc.Prompts {
		seq := len(prompt.Ids)
		hid := readF32(t, filepath.Join(resolved, "oracle", itoa(prompt.Index)+".hidden.f32"))
		layerInput := hid[:seq*H]
		topK, ok := glmDsaTopKIndices(m, 0, layerInput, seq)
		if !ok {
			t.Fatalf("%s prompt %d layer 0 Go DSA top-k failed", dir, prompt.Index)
		}
		attn, ok := glmDsaAttentionOutputFromTopK(m, 0, layerInput, seq, topK)
		if !ok {
			t.Fatalf("%s prompt %d layer 0 Go DSA attention failed", dir, prompt.Index)
		}
		got := make([]float32, seq*H)
		for pos := 0; pos < seq; pos++ {
			residual := layerInput[pos*H : (pos+1)*H]
			x := append([]float32(nil), residual...)
			for i := 0; i < H; i++ {
				x[i] += attn[pos*H+i]
			}
			xn := rmsnorm(x, m.tensor(layerName(0, "post_attention_layernorm.weight")), float32(cfg.RMSNormEps))
			mlp := m.ffnForLayer(0).apply(m, 0, mat.prep(xn), mat)
			for i := 0; i < H; i++ {
				got[pos*H+i] = x[i] + mlp[i]
			}
		}
		ref := hid[seq*H : 2*seq*H]
		cs := cosine(got, ref)
		d, at := maxAbsDiff(got, ref)
		t.Logf("%s prompt %d dense GLM DSA layer-0 cos=%.6f max|Δ|=%.3e at %d",
			dir, prompt.Index, cs, d, at)
		if cs < 0.9999 || d > 1e-3 {
			t.Fatalf("%s prompt %d dense GLM DSA layer-0 mismatch: cos=%.6f max|Δ|=%.3e at %d",
				dir, prompt.Index, cs, d, at)
		}
	}
}

func TestOptionalGLMMoeDsaOracleForwardMatchesHFCacheless(t *testing.T) {
	const dir = glmOracleDir
	m, doc := loadFixtureDir(t, dir, true)
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("%s family = %q, want glm_moe_dsa", dir, m.Cfg.archFamilyKey())
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracleMode(t, resolved, m, doc, false)
}

func TestOptionalGLMMoeDsaOracleSessionCacheMatchesHF(t *testing.T) {
	const dir = glmOracleDir
	m, doc := loadFixtureDir(t, dir, true)
	if !m.Cfg.isGLMMoeDsa() {
		t.Fatalf("%s family = %q, want glm_moe_dsa", dir, m.Cfg.archFamilyKey())
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
	for _, prompt := range doc.Prompts {
		if len(prompt.Ids) < 2 {
			continue
		}
		full := m.Forward(prompt.Ids).Logits[len(prompt.Ids)-1]
		split := m.NewSession()
		split.Prefill(prompt.Ids[:len(prompt.Ids)-1])
		got := split.Step(prompt.Ids[len(prompt.Ids)-1])
		if split.Cache.Len() != len(prompt.Ids) {
			t.Fatalf("%s prompt %d GLM DSA cache length = %d, want %d",
				dir, prompt.Index, split.Cache.Len(), len(prompt.Ids))
		}
		if d, at := maxAbsDiff(got, full); d > 1e-4 {
			t.Fatalf("%s prompt %d split Prefill/Step disagrees with cacheless Forward: max|delta|=%.3e at %d",
				dir, prompt.Index, d, at)
		}

		noLogits := m.NewSession()
		noLogits.PrefillNoLogits(prompt.Ids[:len(prompt.Ids)-1])
		got = noLogits.Step(prompt.Ids[len(prompt.Ids)-1])
		if noLogits.Cache.Len() != len(prompt.Ids) {
			t.Fatalf("%s prompt %d GLM DSA PrefillNoLogits cache length = %d, want %d",
				dir, prompt.Index, noLogits.Cache.Len(), len(prompt.Ids))
		}
		if d, at := maxAbsDiff(got, full); d > 1e-4 {
			t.Fatalf("%s prompt %d PrefillNoLogits/Step disagrees with cacheless Forward: max|delta|=%.3e at %d",
				dir, prompt.Index, d, at)
		}

		prefix := m.NewSession()
		prefix.PrefillNoLogits(prompt.Ids[:len(prompt.Ids)-1])
		reuse := m.SessionFromPrefix(prefix.Cache)
		got = reuse.Step(prompt.Ids[len(prompt.Ids)-1])
		if reuse.Cache.Len() != len(prompt.Ids) {
			t.Fatalf("%s prompt %d GLM DSA prefix-clone cache length = %d, want %d",
				dir, prompt.Index, reuse.Cache.Len(), len(prompt.Ids))
		}
		if d, at := maxAbsDiff(got, full); d > 1e-4 {
			t.Fatalf("%s prompt %d SessionFromPrefix/Step disagrees with cacheless Forward: max|delta|=%.3e at %d",
				dir, prompt.Index, d, at)
		}

		if len(prompt.GreedyIds) > 0 {
			gotIDs := m.NewSession().Generate(prompt.Ids, len(prompt.GreedyIds))
			if !sameInts(gotIDs, prompt.GreedyIds) {
				t.Fatalf("%s prompt %d GLM DSA Generate = %v, want HF greedy %v",
					dir, prompt.Index, gotIDs, prompt.GreedyIds)
			}
		}
	}
	ev := doc.Eviction
	if len(ev.NeverGreedy) > 0 {
		P, Q := len(ev.PrefixIds), len(ev.PoisonIds)
		s := m.NewSession()
		s.Prefill(ev.PrefixIds)
		s.Prefill(ev.PoisonIds)
		if removed := s.Cache.Evict(P, Q); removed != Q || s.Cache.Len() != P {
			t.Fatalf("%s GLM DSA tail evict removed %d (want %d), cache len %d (want %d)",
				dir, removed, Q, s.Cache.Len(), P)
		}
		logits := s.Prefill(ev.QueryIds)
		gotEvict := greedyContinue(s, logits, len(ev.NeverGreedy))
		if !eq(gotEvict, ev.NeverGreedy) {
			t.Fatalf("%s GLM DSA tail eviction != HF never-saw-poison\n  go=%v\n  hf=%v",
				dir, gotEvict, ev.NeverGreedy)
		}

		rep := m.NewSession()
		rep.Prefill(append(append(append([]int{}, ev.PrefixIds...), ev.PoisonIds...), ev.QueryIds...))
		rep.Cache.Evict(P, Q)
		assertGLMDsaCacheReroped(t, rep.Cache)

		poisoned := m.NewSession()
		poisonLogits := poisoned.Prefill(append(append(append([]int{}, ev.PrefixIds...), ev.PoisonIds...), ev.QueryIds...))
		gotPoison := greedyContinue(poisoned, poisonLogits, len(ev.PoisonedGreedy))
		if !eq(gotPoison, ev.PoisonedGreedy) {
			t.Fatalf("%s GLM DSA un-evicted continuation != HF poisoned\n  go=%v\n  hf=%v",
				dir, gotPoison, ev.PoisonedGreedy)
		}
	}
}

func assertGLMDsaCacheReroped(t *testing.T, c *KVCache) {
	t.Helper()
	cfg := c.cfg
	if !cfg.isGLMMoeDsa() || c.glm == nil {
		t.Fatalf("cache is not a GLM-MoE-DSA cache")
	}
	qkNope, qkRope := cfg.QKNopeHeadDim, cfg.QKRopeHeadDim
	qkHead := qkNope + qkRope
	kStride := glmDsaAttentionKStride(cfg)
	idxStride := cfg.IndexHeadDim
	for pos := 0; pos < c.Len(); pos++ {
		for l := 0; l < cfg.NumLayers; l++ {
			cos, sin := ropeRowForLayer(cfg, l, pos)
			for h := 0; h < cfg.NumHeads; h++ {
				off := pos*kStride + h*qkHead
				raw := c.glm.Kraw[l][off : off+qkHead]
				want := make([]float32, qkHead)
				copy(want[:qkNope], raw[:qkNope])
				copy(want[qkNope:], glmDsaApplyInterleavedRoPE(raw[qkNope:], cos, sin))
				got := c.glm.K[l][off : off+qkHead]
				if d, at := maxAbsDiff(want, got); d > fmaCrossPathTol {
					t.Fatalf("GLM DSA cache K not re-RoPEd at layer %d pos %d head %d: max|delta|=%.3e at %d",
						l, pos, h, d, at)
				}
			}
			if len(c.glm.IndexKraw[l]) == 0 {
				continue
			}
			off := pos * idxStride
			want := float64To32(c.glm.IndexKraw[l][off : off+idxStride])
			glmDsaApplyIndexerRoPE(want[:qkRope], cos, sin)
			got := float64To32(c.glm.IndexK[l][off : off+idxStride])
			if d, at := maxAbsDiff(want, got); d > fmaCrossPathTol {
				t.Fatalf("GLM DSA cache IndexK not re-RoPEd at layer %d pos %d: max|delta|=%.3e at %d",
					l, pos, d, at)
			}
		}
	}
}

func causalTopKPrefix(row []int, queryPos int) []int {
	out := make([]int, 0, len(row))
	for _, key := range row {
		if key <= queryPos {
			out = append(out, key)
		}
	}
	return out
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestOptionalMiniMaxM3OracleDocumentsMSABoundary is the real-artifact boundary witness
// for MiniMax-M3 (model_type minimax_m3: MoE + MiniMax Sparse Attention + SwiGLU-OAI). It
// mirrors the GLM-DSA / DeepSeek-MLA boundary tests: it asserts the family is derived, the
// MSA selector axes are preserved, the per-sparse-layer lightning-indexer tensors have the
// expected geometry, the attention backbone is real GQA (standard q/k/v, NOT MLA), and the
// MoE expert set is present. Skipped until a tiny minimax_m3 oracle is exported.
func TestOptionalMiniMaxM3OracleDocumentsMSABoundary(t *testing.T) {
	const dir = minimaxOracleDir
	resolved, ok := resolveOracleDir(dir)
	if !ok {
		t.Skip("no exported MiniMax-M3 oracle; run " + minimaxOracleExportHint)
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	if !cfg.isMiniMax() || !cfg.isMiniMaxSparseAttn() {
		t.Fatalf("%s family = %q, want minimax_m3 sparse", dir, cfg.archFamilyKey())
	}
	if cfg.IndexBlockSize == 0 || cfg.IndexTopKBlocks == 0 || cfg.IndexNHeads == 0 || cfg.IndexHeadDim == 0 {
		t.Fatalf("%s missing MSA axes: block=%d topk=%d nheads=%d headdim=%d; re-export: %s",
			dir, cfg.IndexBlockSize, cfg.IndexTopKBlocks, cfg.IndexNHeads, cfg.IndexHeadDim, minimaxOracleExportHint)
	}
	if cfg.IndexNHeads != cfg.NumKVHeads {
		t.Fatalf("%s index_n_heads=%d != num_key_value_heads=%d; the per-GQA-group selection wiring assumes one index head per group",
			dir, cfg.IndexNHeads, cfg.NumKVHeads)
	}
	if !cfg.IsMoE() {
		t.Fatalf("%s did not preserve MoE config: experts=%d topk=%d", dir, cfg.NumExperts, cfg.NumExpertsPerTok)
	}
	sparse := false
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isMSALayer(l) {
			sparse = true
			break
		}
	}
	if !sparse {
		t.Fatalf("%s has no minimax_m3_sparse layer; layer_types=%v", dir, cfg.LayerTypes)
	}
	var man map[string]tensorMeta
	if err := readJSON(filepath.Join(resolved, "manifest.json"), &man); err != nil {
		t.Fatalf("manifest: %v", err)
	}
	expectShape := func(name string, want []int) {
		t.Helper()
		meta, ok := man[name]
		if !ok {
			t.Fatalf("%s missing tensor %s", dir, name)
		}
		if !sameShape(meta.Shape, want) {
			t.Fatalf("%s tensor %s shape = %v, want %v", dir, name, meta.Shape, want)
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isMSALayer(l) {
			continue
		}
		ip := layerPrefix(l) + "self_attn.indexer."
		expectShape(ip+"q_proj.weight", []int{cfg.IndexNHeads * cfg.IndexHeadDim, cfg.HiddenSize})
		expectShape(ip+"k_proj.weight", []int{cfg.IndexHeadDim, cfg.HiddenSize})
		expectShape(ip+"q_norm.weight", []int{cfg.IndexHeadDim})
		expectShape(ip+"k_norm.weight", []int{cfg.IndexHeadDim})
		// MSA keeps the REAL uncompressed K/V (a standard GQA backbone), not MLA.
		expectShape(layerName(l, "self_attn.q_proj.weight"), []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize})
		expectShape(layerName(l, "self_attn.k_proj.weight"), []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize})
		if _, ok := man[layerName(l, "self_attn.kv_a_proj_with_mqa.weight")]; ok {
			t.Fatalf("%s sparse layer %d has MLA kv_a_proj; MSA uses standard GQA K/V", dir, l)
		}
	}
	t.Logf("%s: minimax_m3 MSA boundary witnessed — family + MSA axes + per-layer indexer geometry + GQA backbone + MoE", dir)
}

// TestOptionalMiniMaxM3OracleExportMetadataCurrent pins the exported oracle to a current
// minimax_m3 export: the family is derived, the MSA axes are present in both config.json
// and the embedded oracle config, and the two configs agree on the core dims. It is
// lenient on the exact model id (a tiny minimax_m3 fixture choice).
func TestOptionalMiniMaxM3OracleExportMetadataCurrent(t *testing.T) {
	resolved, ok := resolveOracleDir(minimaxOracleDir)
	if !ok {
		t.Skip("no exported MiniMax-M3 oracle; run " + minimaxOracleExportHint)
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	var raw oracleRawDoc
	if err := readJSON(filepath.Join(resolved, "oracle.json"), &raw); err != nil {
		t.Fatalf("oracle: %v", err)
	}
	if len(raw.Config) == 0 {
		t.Fatalf("%s oracle.json has no embedded config; re-export with current export_oracle.py: %s", minimaxOracleDir, minimaxOracleExportHint)
	}
	var embedded Config
	if err := json.Unmarshal(raw.Config, &embedded); err != nil {
		t.Fatalf("%s embedded config: %v", minimaxOracleDir, err)
	}
	for _, got := range []Config{cfg, embedded} {
		if !got.isMiniMaxSparseAttn() {
			t.Fatalf("%s config family=%q, want minimax_m3 sparse; re-export: %s", minimaxOracleDir, got.archFamilyKey(), minimaxOracleExportHint)
		}
		if got.IndexBlockSize == 0 || got.IndexTopKBlocks == 0 || got.IndexNHeads == 0 {
			t.Fatalf("%s config missing MSA axes; re-export: %s", minimaxOracleDir, minimaxOracleExportHint)
		}
	}
	if cfg.NumLayers != embedded.NumLayers || cfg.HiddenSize != embedded.HiddenSize || cfg.VocabSize != embedded.VocabSize {
		t.Fatalf("%s config.json and oracle.json embedded config diverge; re-export: %s", minimaxOracleDir, minimaxOracleExportHint)
	}
}

// TestOptionalMiniMaxM3OracleReproducesMSATrace proves the Go lightning-indexer selects
// the SAME key blocks per (index head == GQA group, query) as HF's MiniMaxM3VLIndexer,
// from the same per-layer hidden state HF authored. HF emits the selected block indices
// (-1 right-padded); the Go selector reproduces the ascending causal block set.
func TestOptionalMiniMaxM3OracleReproducesMSATrace(t *testing.T) {
	const dir = minimaxOracleDir
	m, doc := loadFixtureDir(t, dir, true)
	if !m.Cfg.isMiniMaxSparseAttn() {
		t.Fatalf("%s family = %q, want minimax_m3 sparse", dir, m.Cfg.archFamilyKey())
	}
	resolved, _ := resolveOracleDir(dir)
	cfg := m.Cfg
	H := cfg.HiddenSize
	for _, prompt := range doc.Prompts {
		seq := len(prompt.Ids)
		if len(prompt.MSATraces) == 0 {
			t.Fatalf("%s prompt %d has no MSA traces; re-export with current export_oracle.py: %s", dir, prompt.Index, minimaxOracleExportHint)
		}
		hid := readF32(t, filepath.Join(resolved, "oracle", itoa(prompt.Index)+".hidden.f32"))
		if len(hid) != (cfg.NumLayers+1)*seq*H {
			t.Fatalf("%s prompt %d hidden size %d != %d", dir, prompt.Index, len(hid), (cfg.NumLayers+1)*seq*H)
		}
		for _, tr := range prompt.MSATraces {
			if !cfg.isMSALayer(tr.Layer) {
				t.Fatalf("%s MSA trace at non-sparse layer %d", dir, tr.Layer)
			}
			// The HF indexer returns one block selection per query (heads pooled),
			// so the trace is [S_q][topk].
			if len(tr.BlockTopK) != seq {
				t.Fatalf("%s layer %d HF trace has %d query rows, want seq %d", dir, tr.Layer, len(tr.BlockTopK), seq)
			}
			layerInput := hid[tr.Layer*seq*H : (tr.Layer+1)*seq*H]
			goBlocks := m.minimaxIndexerNormalizedBlocks(tr.Layer, layerInput, seq)
			if len(goBlocks) != seq {
				t.Fatalf("%s layer %d go selection has %d query rows, want seq %d", dir, tr.Layer, len(goBlocks), seq)
			}
			for qpos := 0; qpos < seq; qpos++ {
				want := canonicalBlockSet(tr.BlockTopK[qpos], qpos, cfg.IndexBlockSize)
				if !sameInts(goBlocks[qpos], want) {
					t.Fatalf("%s layer %d query %d MSA blocks = %v, want HF %v",
						dir, tr.Layer, qpos, goBlocks[qpos], want)
				}
			}
		}
	}
}

// canonicalBlockSet normalizes an HF lightning-indexer top-k block row (block indices,
// -1 right-padded, in topk order) into the ascending set of causal block indices the Go
// selector produces: drop -1 padding and any non-causal block (> qpos/blockSize), dedup.
func canonicalBlockSet(row []int, qpos, blockSize int) []int {
	qb := qpos / blockSize
	seen := map[int]struct{}{}
	var out []int
	for _, b := range row {
		if b < 0 || b > qb {
			continue
		}
		if _, dup := seen[b]; dup {
			continue
		}
		seen[b] = struct{}{}
		out = append(out, b)
	}
	sort.Ints(out)
	return out
}

// TestOptionalMiniMaxM3OracleForwardMatchesHFCacheless is the rung-1 numeric witness for
// MiniMax-M3: the pure-Go cacheless Forward must reproduce HF's per-layer hidden states
// and final logits to f32 tolerance and pick the same next token at every position, with
// the MSA sparse layers, SwiGLU-OAI MoE, partial RoPE, and qk-norm all wired. Skipped
// until a tiny minimax_m3 oracle is exported (a GPU/artifact-node step).
func TestOptionalMiniMaxM3OracleForwardMatchesHFCacheless(t *testing.T) {
	const dir = minimaxOracleDir
	m, doc := loadFixtureDir(t, dir, true)
	if !m.Cfg.isMiniMaxSparseAttn() {
		t.Fatalf("%s family = %q, want minimax_m3 sparse", dir, m.Cfg.archFamilyKey())
	}
	resolved, _ := resolveOracleDir(dir)
	// checkCachedPrefill=false: only the cacheless Forward is wired for MiniMax-M3; the
	// incremental Session/KV cache MSA path is a separate gate (as GLM DSA staged it).
	assertForwardMatchesHFOracleMode(t, resolved, m, doc, false)
}

// TestOptionalQwen35HybridOracleForwardMatchesHF proves the pure-Go qwen35 forward —
// the 3 Gated-DeltaNet linear-attention layers plus the gated full-attention layer
// (output gate + per-head qk-norm + partial RoPE) — reproduces HF transformers to f32
// tolerance on the tiny fixture: per-layer hidden-state cosine >= 0.9999 and argmax
// parity at every position. It also asserts the loader DERIVED the qwen35 architectural
// knobs from the exported config (the (1+w) RMSNorm, qk-norm, the sigmoid output gate,
// partial rotary), that each layer maps to its OWN mixer tensor names (linear_attn.* on
// the Gated-DeltaNet layers, self_attn.{q,k,v,o}_proj on the gated full-attention layer),
// and that the layer mix is genuinely hybrid — so the witness is not vacuous. The fixture
// is gitignored; regenerate with qwen35OracleExportHint.
func TestOptionalQwen35HybridOracleForwardMatchesHF(t *testing.T) {
	const dir = qwen35OracleDir
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !cfg.IsQwen35Hybrid() {
		t.Fatalf("%s did not load as a qwen3_5 hybrid (layer_types=%v); regenerate: %s",
			dir, cfg.LayerTypes, qwen35OracleExportHint)
	}
	// The hybrid knobs are architectural, not optional, so the loader must derive them;
	// a missing one would silently change the math and make the parity check meaningless.
	if !cfg.AttnOutputGate || !cfg.QKNorm || !cfg.NormGain1p {
		t.Fatalf("%s qwen35 knobs not derived: attn_output_gate=%v qk_norm=%v norm_gain_1p=%v",
			dir, cfg.AttnOutputGate, cfg.QKNorm, cfg.NormGain1p)
	}
	if cfg.PartialRotaryFactor <= 0 || cfg.PartialRotaryFactor >= 1 {
		t.Fatalf("%s partial rotary factor not loaded from rope_parameters: %v", dir, cfg.PartialRotaryFactor)
	}
	// Tensor-name witness (#442): each layer's mixer must map to its OWN tensor set in
	// the loaded manifest — the Gated-DeltaNet linear_attn.* family on linear layers, the
	// standard self_attn.{q,k,v,o}_proj projections on full-attention layers — and never
	// the other mixer's tensors. Forward (asserted below) reads every one of these, so a
	// missing/misnamed tensor would already crash the parity pass; pinning the names here
	// makes the loader's hybrid layer_types -> tensor mapping an explicit, legible witness.
	nLinear := 0
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		has := func(suffix string) bool { _, ok := m.manifest[p+suffix]; return ok }
		if cfg.isLinearAttnLayer(l) {
			nLinear++
			for _, suffix := range []string{
				"linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
				"linear_attn.norm.weight", "linear_attn.in_proj_qkv.weight",
				"linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight",
				"linear_attn.in_proj_a.weight", "linear_attn.out_proj.weight",
			} {
				if !has(suffix) {
					t.Fatalf("%s linear-attn layer %d missing Gated-DeltaNet tensor %s%s", dir, l, p, suffix)
				}
			}
			if has("self_attn.q_proj.weight") {
				t.Fatalf("%s linear-attn layer %d unexpectedly has full-attention self_attn.q_proj (mixer leak)", dir, l)
			}
			continue
		}
		for _, suffix := range []string{"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"} {
			if !has("self_attn." + suffix) {
				t.Fatalf("%s full-attn layer %d missing self_attn.%s", dir, l, suffix)
			}
		}
		if has("linear_attn.conv1d.weight") {
			t.Fatalf("%s full-attn layer %d unexpectedly has Gated-DeltaNet linear_attn.conv1d (mixer leak)", dir, l)
		}
	}
	if nLinear == 0 || nLinear == cfg.NumLayers {
		t.Fatalf("%s layer mix is vacuous: %d/%d linear_attention (need both mixers)", dir, nLinear, cfg.NumLayers)
	}
	resolved, _ := resolveOracleDir(dir)
	// checkCachedPrefill=true: the qwen35 Session.Prefill recurrent path is fully wired
	// (see TestQwen35HybridSessionMatchesForwardAndPersistsState), so the cached prefill
	// must also reproduce the cacheless Forward last-position logits.
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalPhi3LongropeOracleCoversLongFactor(t *testing.T) {
	const dir = ".cache/oracle-phi3-longrope-local"
	m, doc := loadFixtureDir(t, dir, true)
	cfg := m.Cfg
	if !strings.Contains(cfg.archFamilyKey(), "phi3") {
		t.Fatalf("%s family = %q, want phi3", dir, cfg.archFamilyKey())
	}
	if !cfg.isLongrope() {
		t.Fatalf("%s did not load longrope config", dir)
	}
	if !longropeFactorPinned(cfg) {
		t.Fatalf("%s longrope factor is not pinned/well-formed: head_dim=%d short=%d long=%d",
			dir, cfg.HeadDim, len(cfg.LongRope.ShortFactor), len(cfg.LongRope.LongFactor))
	}
	if got := ropeLongFactor(cfg); !float64sEqual(got, cfg.LongRope.LongFactor) {
		t.Fatalf("%s selected factor = %v, want long factor %v", dir, got, cfg.LongRope.LongFactor)
	}
	if longropeAttnScaleMul(cfg) <= 1 {
		t.Fatalf("%s longrope attention scale multiplier = %v, want > 1", dir, longropeAttnScaleMul(cfg))
	}
	nonVacuous := false
	for _, p := range doc.Prompts {
		if len(p.Ids) > cfg.LongRope.OriginalMaxPositionEmbeddings {
			nonVacuous = true
			break
		}
	}
	if !nonVacuous {
		t.Fatalf("%s has no prompt longer than original_max_position_embeddings=%d",
			dir, cfg.LongRope.OriginalMaxPositionEmbeddings)
	}
	resolved, _ := resolveOracleDir(dir)
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func assertForwardMatchesHFOracle(t *testing.T, dir string, m *Model, doc oracleDoc) {
	t.Helper()
	assertForwardMatchesHFOracleMode(t, dir, m, doc, true)
}

func assertForwardMatchesHFOracleMode(t *testing.T, dir string, m *Model, doc oracleDoc, checkCachedPrefill bool) {
	t.Helper()
	cfg := m.Cfg
	H, V := cfg.HiddenSize, cfg.VocabSize
	nHS := cfg.NumLayers + 1

	for _, p := range doc.Prompts {
		seq := len(p.Ids)
		act := m.Forward(p.Ids)

		// ---- hidden states, layer by layer (localizes any bug) -------------
		hid := readF32(t, filepath.Join(dir, "oracle", itoa(p.Index)+".hidden.f32"))
		if len(hid) != nHS*seq*H {
			t.Fatalf("prompt %d hidden size %d != %d", p.Index, len(hid), nHS*seq*H)
		}
		// HF's hidden_states tuple is [embed, L0..last-layer-after-final-norm].
		// Our Hidden stores the last decoder output before final norm, so normalize
		// only that final slot before comparing.
		checkLayers := []int{0, 1, cfg.NumLayers / 2, cfg.NumLayers - 1, cfg.NumLayers}
		seenLayers := make(map[int]bool, len(checkLayers))
		for _, l := range checkLayers {
			if l < 0 || l > cfg.NumLayers || seenLayers[l] {
				continue
			}
			seenLayers[l] = true
			ref := hid[l*seq*H : (l+1)*seq*H]
			got := act.Hidden[l]
			tag := "layer " + itoa(l)
			switch l {
			case 0:
				tag = "embedding"
			case cfg.NumLayers:
				tag = "final norm"
				// normalize our last hidden, per-position, to match HF's post-norm entry
				normed := make([]float32, len(got))
				for t2 := 0; t2 < seq; t2++ {
					copy(normed[t2*H:], m.finalNorm(got[t2*H:(t2+1)*H]))
				}
				got = normed
			}
			cs := cosine(got, ref)
			d, _ := maxAbsDiff(got, ref)
			t.Logf("prompt %d %-16s cos=%.6f max|Δ|=%.3e", p.Index, tag, cs, d)
			if cs < 0.9999 {
				t.Errorf("prompt %d %s cosine %.6f < 0.9999", p.Index, tag, cs)
			}
		}

		// ---- logits: argmax must match at EVERY position -------------------
		lg := readF32(t, filepath.Join(dir, "oracle", itoa(p.Index)+".logits.f32"))
		for pos := 0; pos < seq; pos++ {
			ref := lg[pos*V : (pos+1)*V]
			gotAM := argmax(act.Logits[pos])
			refAM := argmax(ref)
			if gotAM != refAM || refAM != p.ArgmaxPos[pos] {
				t.Errorf("prompt %d pos %d argmax go=%d hf=%d oracle=%d",
					p.Index, pos, gotAM, refAM, p.ArgmaxPos[pos])
			}
		}
		// report logit fidelity on the last position
		last := seq - 1
		d, _ := maxAbsDiff(act.Logits[last], lg[last*V:(last+1)*V])
		t.Logf("prompt %d logits[last] max|Δ|=%.3e argmax=%d ✓", p.Index, d, p.ArgmaxPos[last])
		if d > 0.05 {
			t.Errorf("prompt %d last-pos logit max abs diff %.3e exceeds 0.05", p.Index, d)
		}
		if checkCachedPrefill {
			cached := m.NewSession().Prefill(p.Ids)
			if d, _ := maxAbsDiff(cached, act.Logits[last]); d > 1e-4 {
				t.Errorf("prompt %d cached Prefill disagrees with Forward last logits: max abs diff %.3e", p.Index, d)
			}
		}
	}
}

func TestOracleMatrixCoversRequiredFamilies(t *testing.T) {
	required := requiredOracleFamilies(os.Getenv(oracleRequiredFamiliesEnv))
	if len(required) == 0 {
		t.Skip("set " + oracleRequiredFamiliesEnv + " to require top-10 family coverage")
	}
	dirs, explicit := oracleMatrixDirs()
	if !explicit {
		t.Fatalf("%s requires explicit %s", oracleRequiredFamiliesEnv, oracleDirsEnv)
	}
	have := map[string]string{}
	for _, dir := range dirs {
		resolved, ok := resolveOracleDir(dir)
		if !ok {
			t.Fatalf("read %s: no exported weights", filepath.Join(dir, "weights.f32"))
		}
		var cfg Config
		if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
			t.Fatalf("read %s: %v", filepath.Join(resolved, "config.json"), err)
		}
		have[cfg.archFamilyKey()] = dir
	}
	missing := missingOracleFamilies(required, have)
	if len(missing) > 0 {
		keys := make([]string, 0, len(have))
		for key := range have {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		t.Fatalf("oracle matrix missing required families %v; have %v", missing, keys)
	}
}

func TestOracleMatrixDirsFromEnv(t *testing.T) {
	dirs, explicit := oracleMatrixDirsFromEnv("")
	if explicit || len(dirs) != 1 || dirs[0] != cacheDir {
		t.Fatalf("default dirs = %v explicit=%v, want [%s]/false", dirs, explicit, cacheDir)
	}

	raw := "a" + string(os.PathListSeparator) + "b" + string(os.PathListSeparator) + "a"
	dirs, explicit = oracleMatrixDirsFromEnv(raw)
	if !explicit || len(dirs) != 2 || dirs[0] != "a" || dirs[1] != "b" {
		t.Fatalf("explicit dirs = %v explicit=%v, want [a b]/true", dirs, explicit)
	}
}

func TestMissingOracleFamilies(t *testing.T) {
	have := map[string]string{
		"gemma3forcausallm":             "/tmp/gemma",
		"qwen25forcausallm":             "/tmp/qwen",
		"glmmoedsaglmmoedsaforcausallm": "/tmp/glm",
	}
	missing := missingOracleFamilies(requiredOracleFamilies("gemma3, qwen2_5, glm_moe_dsa, phi3"), have)
	if len(missing) != 1 || missing[0] != "phi3" {
		t.Fatalf("missing = %v, want [phi3]", missing)
	}
}
