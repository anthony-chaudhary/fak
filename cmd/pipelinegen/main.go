// Command pipelinegen runs fak's NATIVE engine generating tokens across
// pipeline-parallel stages — the runnable form of the cross-device transport
// contract (internal/model/pipeline.go). Each stage owns a contiguous band of
// transformer layers, is loaded standalone (only its weights resident), and the
// hidden state crosses a serialize->bytes->deserialize boundary between stages
// (the stand-in for a real NCCL/RPC worker hop). Greedy decode is routed through
// those stages via model.PipelineGenerate.
//
// This is the "actually running fak native backend for GLM-5.2" demonstrator that
// fits on hardware that exists: it runs a TINY GLM-MoE-DSA model end to end, not
// the real 753B checkpoint (which needs multi-node A100s + FP8/INT4-at-load). It
// is greedy (temperature 0) and re-forwards the growing sequence each step (O(n^2),
// no cross-stage incremental KV yet). See GLM-5.2-NATIVE-ENGINE-GAP-2026-06-20.md.
//
// Usage:
//
//	go run ./cmd/pipelinegen -selfcheck
//	    Build a tiny in-memory GLM-DSA model, generate through 2 stages, and assert
//	    the output equals monolithic Session.Generate. Zero external files.
//
//	go run ./cmd/pipelinegen -dir <glm-dsa-checkpoint> -stages 2 -prompt 3,1,4 -n 8
//	    Load a real GLM-DSA safetensors export as N separately-windowed stages and
//	    generate through them.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func main() {
	dir := flag.String("dir", "", "GLM-DSA safetensors checkpoint dir (config.json + shards); empty => -selfcheck")
	stages := flag.Int("stages", 2, "number of pipeline-parallel stages (contiguous layer bands)")
	promptCSV := flag.String("prompt", "3,1,4,1", "comma-separated prompt token ids")
	n := flag.Int("n", 16, "number of tokens to generate")
	selfcheck := flag.Bool("selfcheck", false, "build a tiny in-memory GLM-DSA model and assert pipeline==monolithic (no -dir needed)")
	incremental := flag.Bool("incremental", false, "use the incremental KV-resident decode path (per-stage band KV) instead of the O(n^2) re-forward")
	flag.Parse()

	if *dir == "" {
		*selfcheck = true
	}

	prompt, err := parseIDs(*promptCSV)
	if err != nil {
		fail("parse -prompt: %v", err)
	}
	if len(prompt) == 0 {
		fail("empty prompt")
	}

	if *selfcheck {
		runSelfcheck(prompt, *n, *stages, *incremental)
		return
	}
	runDir(*dir, prompt, *n, *stages, *incremental)
}

// runSelfcheck builds a tiny in-memory GLM-MoE-DSA model, runs greedy decode
// through the pipeline stages, and asserts the result is identical to the
// monolithic Session.Generate — a zero-dependency proof the native pipelined
// generate path runs and is correct. Stages share one in-memory model (the
// separately-windowed-load proof is the unit test + the -dir path).
func runSelfcheck(prompt []int, n, nStages int, incremental bool) {
	cfg := model.Config{
		HiddenSize: 32, NumLayers: 3, NumHeads: 4, NumKVHeads: 4, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 41, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "glm_moe_dsa", Architectures: []string{"GlmMoeDsaForCausalLM"},
		QLoraRank: 32, KVLoraRank: 32, QKNopeHeadDim: 4, QKRopeHeadDim: 4, VHeadDim: 8,
		IndexNHeads: 4, IndexHeadDim: 8, IndexTopK: 2,
		IndexerTypes: []string{"full", "shared", "full"},
		EOSTokenID:   -1, // never early-stop: decode exactly n
	}
	m := model.NewSyntheticGLMDsa(cfg)

	plan, specs, err := planStages(cfg, nStages)
	if err != nil {
		fail("partition: %v", err)
	}

	mode := "O(n^2) re-forward"
	if incremental {
		mode = "incremental KV-resident decode"
	}
	fmt.Printf("pipelinegen -selfcheck: tiny GLM-MoE-DSA, %d layers, indexers %v, %s\n",
		cfg.NumLayers, cfg.IndexerTypes, mode)
	printPlan(plan, specs)

	var got []int
	start := time.Now()
	if incremental {
		// Each stage gets its own Session over the shared model, so each holds its own
		// band KV — the real incremental path. (Separate windowed loads are the -dir path.)
		decoders := make([]*model.PipelineStageDecoder, len(specs))
		for i, s := range specs {
			decoders[i] = model.NewPipelineStageDecoder(s, m)
		}
		got, err = model.PipelineGenerateIncremental(prompt, n, decoders)
		if err != nil {
			fail("PipelineGenerateIncremental: %v", err)
		}
	} else {
		ps := make([]model.PipelineStage, len(specs))
		for i, s := range specs {
			ps[i] = model.PipelineStage{Spec: s, Model: m}
		}
		got, err = model.PipelineGenerate(prompt, n, ps)
		if err != nil {
			fail("PipelineGenerate: %v", err)
		}
	}
	elapsed := time.Since(start)

	want := m.NewSession().Generate(prompt, n)
	ok := len(got) == len(want)
	for i := range want {
		if i < len(got) && got[i] != want[i] {
			ok = false
		}
	}

	fmt.Printf("prompt:    %v\n", prompt)
	fmt.Printf("generated: %v  (%d tokens in %s, %.1f tok/s)\n",
		got, len(got), elapsed.Round(time.Millisecond), tokPerSec(len(got), elapsed))
	fmt.Printf("monolithic Session.Generate: %v\n", want)
	if !ok {
		fail("MISMATCH: pipeline generation != monolithic (got=%v want=%v)", got, want)
	}
	fmt.Println("MATCH: native pipelined generation == native monolithic generation (bit-exact greedy).")
	caveat := "greedy, O(n^2) re-forward"
	if incremental {
		caveat = "greedy, incremental per-stage KV"
	}
	fmt.Printf("note: tiny synthetic GLM-DSA, %s — NOT 753B serving (see GLM-5.2-NATIVE-ENGINE-GAP).\n", caveat)
}

// runDir loads a real GLM-DSA checkpoint and generates through N pipeline stages.
// Two on-disk formats are handled: safetensors (config.json + shards) loads each
// stage as a SEPARATE windowed model (only its band resident — the full
// cross-device proof); flat-f32 (config.json + manifest.json + weights.f32, the
// export_oracle.py format) loads one model and partitions it into bands, since the
// layer-window option is a safetensors-loader feature. With -incremental each stage
// keeps its band KV resident and advances one position per token (PipelineGenerateIncremental);
// otherwise it uses the O(n^2) re-forward (PipelineGenerate). Both cross the stage handoff.
func runDir(dir string, prompt []int, n, nStages int, incremental bool) {
	cfg, err := readConfig(dir)
	if err != nil {
		fail("read config: %v", err)
	}
	if !isGLMDsa(cfg) {
		fail("-dir %s is model_type %q, not a glm_moe_dsa checkpoint", dir, cfg.ModelType)
	}

	plan, specs, err := planStages(cfg, nStages)
	if err != nil {
		fail("partition: %v", err)
	}

	flat := fileExists(filepath.Join(dir, "weights.f32"))
	format := "safetensors (separate windowed loads)"
	if flat {
		format = "flat-f32 (one load, partitioned into bands)"
	}
	mode := "O(n^2) re-forward"
	if incremental {
		mode = "incremental KV-resident decode"
	}
	fmt.Printf("pipelinegen: %s — %d layers, %d stages, %s, %s\n", dir, cfg.NumLayers, len(specs), format, mode)
	printPlan(plan, specs)

	// Load each stage's model: standalone windowed (safetensors) or one shared model (flat-f32).
	models := make([]*model.Model, len(specs))
	if flat {
		m, err := model.Load(dir)
		if err != nil {
			fail("load flat-f32 model: %v", err)
		}
		for i := range specs {
			models[i] = m
		}
	} else {
		for i, s := range specs {
			stageModel, err := model.LoadSafetensorsQuantDir(dir, cfg, model.WithLayerWindow(s.Lo, s.Hi))
			if err != nil {
				fail("load stage %d [%d,%d): %v", i, s.Lo, s.Hi, err)
			}
			models[i] = stageModel
			fmt.Printf("  stage %d band [%d,%d) loaded standalone\n", i, s.Lo, s.Hi)
		}
	}

	var got []int
	start := time.Now()
	if incremental {
		decoders := make([]*model.PipelineStageDecoder, len(specs))
		for i, s := range specs {
			decoders[i] = model.NewPipelineStageDecoder(s, models[i])
		}
		got, err = model.PipelineGenerateIncremental(prompt, n, decoders)
		if err != nil {
			fail("PipelineGenerateIncremental: %v", err)
		}
	} else {
		ps := make([]model.PipelineStage, len(specs))
		for i, s := range specs {
			ps[i] = model.PipelineStage{Spec: s, Model: models[i]}
		}
		got, err = model.PipelineGenerate(prompt, n, ps)
		if err != nil {
			fail("PipelineGenerate: %v", err)
		}
	}
	elapsed := time.Since(start)

	caveat := "greedy, O(n^2) re-forward across pipeline stages"
	if incremental {
		caveat = "greedy, incremental per-stage KV across pipeline stages"
	}
	fmt.Printf("prompt:    %v\n", prompt)
	fmt.Printf("generated: %v  (%d tokens in %s, %.1f tok/s)\n",
		got, len(got), elapsed.Round(time.Millisecond), tokPerSec(len(got), elapsed))
	fmt.Printf("note: %s — correctness path, not a throughput path.\n", caveat)
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// planStages splits cfg.NumLayers into nStages contiguous bands. For a GLM-DSA
// model each interior cut is snapped FORWARD to the next full-indexer layer so the
// partition validator accepts it (a boundary may not fall on a shared layer). The
// returned cut set may yield fewer than nStages bands if snapping collapses cuts.
func planStages(cfg model.Config, nStages int) (model.PartitionPlan, []model.StageSpec, error) {
	N := cfg.NumLayers
	if nStages < 1 {
		nStages = 1
	}
	if nStages > N {
		nStages = N
	}
	cuts := make([]int, 0, nStages-1)
	seen := map[int]bool{}
	for i := 1; i < nStages; i++ {
		c := i * N / nStages
		c = snapToFullIndexer(cfg, c)
		if c <= 0 || c >= N || seen[c] {
			continue
		}
		seen[c] = true
		cuts = append(cuts, c)
	}
	plan, err := model.NewPartitionPlan(cfg, cuts)
	if err != nil {
		return model.PartitionPlan{}, nil, err
	}
	return plan, plan.Stages, nil
}

// snapToFullIndexer moves a cut forward to the first layer at-or-after it that is a
// full-indexer layer (so an IndexShare group is never split across a boundary). For
// a non-GLM model it returns c unchanged.
func snapToFullIndexer(cfg model.Config, c int) int {
	if !isGLMDsa(cfg) {
		return c
	}
	for c < cfg.NumLayers {
		if c < len(cfg.IndexerTypes) {
			kind := strings.ToLower(strings.TrimSpace(cfg.IndexerTypes[c]))
			if kind == "shared" || kind == "share" {
				c++
				continue
			}
		}
		return c
	}
	return c
}

func isGLMDsa(cfg model.Config) bool {
	key := strings.ToLower(cfg.ModelType)
	for _, a := range cfg.Architectures {
		key += " " + strings.ToLower(a)
	}
	return strings.Contains(key, "glm") && strings.Contains(key, "dsa")
}

func printPlan(plan model.PartitionPlan, specs []model.StageSpec) {
	parts := make([]string, len(specs))
	for i, s := range specs {
		role := ""
		if s.First {
			role += "embed"
		}
		if s.Last {
			if role != "" {
				role += "+"
			}
			role += "head"
		}
		if role == "" {
			role = "mid"
		}
		parts[i] = fmt.Sprintf("[%d,%d):%s", s.Lo, s.Hi, role)
	}
	fmt.Printf("plan: %d stages over %d layers -> %s\n", len(specs), plan.NumLayers, strings.Join(parts, " | "))
}

func parseIDs(csv string) ([]int, error) {
	fields := strings.Split(csv, ",")
	ids := make([]int, 0, len(fields))
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		v, err := strconv.Atoi(f)
		if err != nil {
			return nil, fmt.Errorf("token %q: %w", f, err)
		}
		ids = append(ids, v)
	}
	return ids, nil
}

// readConfig parses config.json from a checkpoint dir into a model.Config (the
// same mapping the existing loaders use), filling head_dim if HF omitted it.
func readConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

func tokPerSec(tokens int, d time.Duration) float64 {
	if d <= 0 {
		return 0
	}
	return float64(tokens) / d.Seconds()
}

func fail(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "pipelinegen: "+format+"\n", a...)
	os.Exit(1)
}
