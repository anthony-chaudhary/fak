// Command glmdsatput measures fak's NATIVE GLM-5.2 (glm_moe_dsa) decode throughput on a
// real compute backend (e.g. the CUDA A100 path), driving the in-kernel MLA + Dynamic
// Sparse Attention (DSA) indexer + sparse-attend + dense-FFN forward through fak's own
// kernels — NOT a third-party engine.
//
// HONEST SCOPE. This builds a SYNTHETIC glm_moe_dsa model (model.NewSyntheticGLMDsa: real
// architecture + real per-layer dims, but random weights and a reduced layer count so it
// fits one device). The tok/s it reports is therefore fak's GLM-5.2-architecture per-token
// device cost at the chosen scale — a real measurement of the native kernels on real
// hardware — and NOT the throughput of the full 753B checkpoint (which does not fit one
// GPU; its real measured number is the llama.cpp CPU-offload baseline). The dense-FFN form
// omits the MoE expert GEMMs, so this is an optimistic lower-bound on per-token work
// relative to the MoE 753B. Use it to witness that the native GLM-5.2 decode runs on the
// device and to track that path's speed, not to quote a 753B serving number.
//
// Build + run on a CUDA node (the backend only registers under -tags cuda):
//
//	go run -tags cuda ./cmd/glmdsatput -layers 8 -hidden 2048 -backend cuda -decode-steps 64
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func lcgIDs(n, vocab int) []int {
	ids := make([]int, n)
	state := uint64(2463534242)
	for i := 0; i < n; i++ {
		state = (state*1103515245 + 12345) & 0x7fffffff
		ids[i] = int(state % uint64(vocab))
	}
	return ids
}

func medianMS(ds []time.Duration) float64 {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[len(cp)/2].Nanoseconds()) / 1e6
}

// glmDims is the subset of the synthetic glm_moe_dsa config the throughput SWEEP varies —
// the dimensions that move the native per-token cost curve (depth, width, FFN, DSA selection
// size). It is the bisection unit: a "bad" config that triggers the device-kernel
// illegal-memory-access at the largest sweep points
// (docs/notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md §3 P0) and a known-good
// neighbour differ in one or more of these.
type glmDims struct {
	Layers int `json:"layers"`
	Hidden int `json:"hidden"`
	Heads  int `json:"heads"`
	Inter  int `json:"inter"`
	TopK   int `json:"index_topk"`
}

// bisectStep is one single-variable-reverted config the operator runs (in a FRESH process — a
// CUDA illegal access corrupts the context, so configs cannot share a process) to attribute the
// fault to one dimension: it is the failing `bad` config with exactly one dim reverted to the
// known-good `good` value. If `bad` faults but a step RUNS clean, that step's `dim` (held at
// bad's value) is the one carrying the out-of-bounds.
type bisectStep struct {
	Dim string `json:"dim"`
	glmDims
}

// bisectPlan emits the note's prescribed P0 next step — "single-variable on-box bisection (vary
// layers / hidden / heads / topk one at a time)" — as a deterministic, runnable plan instead of
// a manual discipline. For each dimension where good and bad differ, it yields `bad` with that
// one dimension reverted to good's value, in a stable order (layers, hidden, heads, inter, topk)
// so the plan is reproducible. Dims that already agree are skipped (reverting them is a no-op).
func bisectPlan(good, bad glmDims) []bisectStep {
	var steps []bisectStep
	add := func(dim string, d glmDims) { steps = append(steps, bisectStep{Dim: dim, glmDims: d}) }
	if good.Layers != bad.Layers {
		d := bad
		d.Layers = good.Layers
		add("layers", d)
	}
	if good.Hidden != bad.Hidden {
		d := bad
		d.Hidden = good.Hidden
		add("hidden", d)
	}
	if good.Heads != bad.Heads {
		d := bad
		d.Heads = good.Heads
		add("heads", d)
	}
	if good.Inter != bad.Inter {
		d := bad
		d.Inter = good.Inter
		add("inter", d)
	}
	if good.TopK != bad.TopK {
		d := bad
		d.TopK = good.TopK
		add("index_topk", d)
	}
	return steps
}

func main() {
	layers := flag.Int("layers", 8, "number of glm_moe_dsa layers (all full-indexer)")
	hidden := flag.Int("hidden", 2048, "hidden size H")
	heads := flag.Int("heads", 16, "attention heads")
	inter := flag.Int("inter", 8192, "dense FFN intermediate size")
	vocab := flag.Int("vocab", 8192, "vocab size (tied head)")
	qLora := flag.Int("q-lora", 1536, "MLA q_lora_rank")
	kvLora := flag.Int("kv-lora", 512, "MLA kv_lora_rank")
	qkNope := flag.Int("qk-nope", 128, "MLA qk_nope_head_dim")
	qkRope := flag.Int("qk-rope", 64, "MLA qk_rope_head_dim")
	vHead := flag.Int("v-head", 128, "MLA v_head_dim")
	idxHeads := flag.Int("index-heads", 16, "DSA indexer heads")
	idxDim := flag.Int("index-dim", 128, "DSA indexer head dim")
	idxTopK := flag.Int("index-topk", 256, "DSA indexer top-k selected keys")
	prompt := flag.Int("decode-prompt", 512, "prompt length before timed decode")
	steps := flag.Int("decode-steps", 64, "decode steps to time")
	reps := flag.Int("decode-reps", 5, "reps (median over per-token)")
	backendName := flag.String("backend", "cuda", "compute backend name (cuda); empty/legacy = host")
	quant := flag.Bool("quant", true, "Q8_0 quantized weight path (required for the device Q8 kernels)")
	emitJSON := flag.Bool("json", false, "emit one compact JSON record line (machine-readable) in addition to the human report")
	bisectBaseline := flag.String("bisect-baseline", "", "known-GOOD config \"layers hidden heads inter topk\"; with it set, do NOT benchmark — emit the single-variable bisection plan (this run's -layers/-hidden/-heads/-inter/-index-topk are the failing config) to pin the P0 device-kernel illegal-memory-access one dim at a time. GPU-free.")
	flag.Parse()

	// Bisection-plan mode: emit the deterministic single-variable sweep that pins the largest-
	// config device-kernel illegal-memory-access (the carried P0), then exit. This runs on ANY
	// host (no backend, no model build) — it is a plan the GPU runner executes, one fresh process
	// per step. See docs/notes/GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md §3.
	if *bisectBaseline != "" {
		var good glmDims
		if n, err := fmt.Sscan(*bisectBaseline, &good.Layers, &good.Hidden, &good.Heads, &good.Inter, &good.TopK); err != nil || n != 5 {
			fmt.Fprintf(os.Stderr, "-bisect-baseline must be 5 ints \"layers hidden heads inter topk\" (got %q)\n", *bisectBaseline)
			os.Exit(2)
		}
		bad := glmDims{Layers: *layers, Hidden: *hidden, Heads: *heads, Inter: *inter, TopK: *idxTopK}
		plan := bisectPlan(good, bad)
		fmt.Printf("=== glm_moe_dsa P0 single-variable bisection plan ===\n")
		fmt.Printf("good(no-fault): layers=%d hidden=%d heads=%d inter=%d topk=%d\n", good.Layers, good.Hidden, good.Heads, good.Inter, good.TopK)
		fmt.Printf("bad (faults)  : layers=%d hidden=%d heads=%d inter=%d topk=%d\n", bad.Layers, bad.Hidden, bad.Heads, bad.Inter, bad.TopK)
		if len(plan) == 0 {
			fmt.Printf("(no differing dims — good == bad; nothing to bisect)\n")
		}
		for _, s := range plan {
			fmt.Printf("  revert %-10s -> layers=%d hidden=%d heads=%d inter=%d topk=%d  (if THIS runs clean, dim %q at bad's value carries the OOB)\n",
				s.Dim, s.Layers, s.Hidden, s.Heads, s.Inter, s.TopK, s.Dim)
		}
		b, _ := json.Marshal(plan)
		fmt.Printf("GLMBISECT_JSON %s\n", b)
		return
	}

	indexerTypes := make([]string, *layers)
	for i := range indexerTypes {
		indexerTypes[i] = "full"
	}
	cfg := model.Config{
		HiddenSize:        *hidden,
		NumLayers:         *layers,
		NumHeads:          *heads,
		NumKVHeads:        *heads,
		HeadDim:           *qkNope + *qkRope,
		IntermediateSize:  *inter,
		VocabSize:         *vocab,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		EOSTokenID:        -1,
		ModelType:         "glm_moe_dsa",
		Architectures:     []string{"GlmMoeDsaForCausalLM"},
		QLoraRank:         *qLora,
		KVLoraRank:        *kvLora,
		QKNopeHeadDim:     *qkNope,
		QKRopeHeadDim:     *qkRope,
		VHeadDim:          *vHead,
		IndexNHeads:       *idxHeads,
		IndexHeadDim:      *idxDim,
		IndexTopK:         *idxTopK,
		IndexerTypes:      indexerTypes,
		TieWordEmbeddings: true,
	}

	var be compute.Backend
	if *backendName != "" && *backendName != "legacy" {
		var ok bool
		be, ok = compute.Lookup(*backendName)
		if !ok {
			fmt.Fprintf(os.Stderr, "backend %q not registered (registered: %v) — build with -tags cuda on a CUDA node\n", *backendName, compute.Registered())
			os.Exit(2)
		}
	}

	t0 := time.Now()
	m := model.NewSyntheticGLMDsa(cfg)
	if *quant {
		m.Quantize()
	}
	buildMS := float64(time.Since(t0).Nanoseconds()) / 1e6

	newSession := func() *model.Session {
		if be != nil {
			s := m.NewBackendSession(be)
			s.Quant = *quant
			return s
		}
		s := m.NewSession()
		s.Quant = *quant
		return s
	}

	// Warm up: page weights onto the device + JIT allocation paths.
	{
		s := newSession()
		s.Prefill(lcgIDs(16, *vocab))
		s.Step(7 % *vocab)
		s.Close()
	}

	// Prefill timing.
	pIDs := lcgIDs(*prompt, *vocab)
	pDs := make([]time.Duration, *reps)
	for r := 0; r < *reps; r++ {
		s := newSession()
		t := time.Now()
		s.Prefill(pIDs)
		pDs[r] = time.Since(t)
		s.Close()
	}
	prefillMS := medianMS(pDs)

	// Decode timing: prefill the prompt, then time `steps` incremental Step() calls.
	perTok := make([]time.Duration, 0, *reps)
	for r := 0; r < *reps; r++ {
		s := newSession()
		s.Prefill(pIDs)
		id := (r*131 + 7) % *vocab
		t := time.Now()
		for i := 0; i < *steps; i++ {
			logits := s.Step(id)
			id = (id*48271 + 1) % *vocab
			_ = logits
		}
		perTok = append(perTok, time.Since(t)/time.Duration(*steps))
		s.Close()
	}
	decodeMS := medianMS(perTok)

	backend := "host(legacy)"
	if be != nil {
		backend = fmt.Sprintf("%s (tier=%s class=%s)", be.Name(), be.Tier(), be.Class())
	}
	prec := "f32"
	if *quant {
		prec = "Q8_0"
	}
	fmt.Printf("=== fak NATIVE glm_moe_dsa decode throughput (SYNTHETIC weights; NOT the 753B) ===\n")
	fmt.Printf("backend       : %s  precision=%s\n", backend, prec)
	fmt.Printf("config        : layers=%d hidden=%d heads=%d inter=%d vocab=%d\n", *layers, *hidden, *heads, *inter, *vocab)
	fmt.Printf("MLA/DSA       : q_lora=%d kv_lora=%d qk_nope=%d qk_rope=%d v_head=%d | index_heads=%d index_dim=%d topk=%d\n",
		*qLora, *kvLora, *qkNope, *qkRope, *vHead, *idxHeads, *idxDim, *idxTopK)
	fmt.Printf("build+quant   : %.1f ms\n", buildMS)
	fmt.Printf("prefill       : P=%d  %.2f ms  (%.1f tok/s)\n", *prompt, prefillMS, float64(*prompt)/(prefillMS/1e3))
	fmt.Printf("DECODE        : %.3f ms/tok  (%.2f tok/s)  [median over %d reps x %d steps]\n",
		decodeMS, 1.0/(decodeMS/1e3), *reps, *steps)

	if *emitJSON {
		// One compact, machine-readable record per run. The `scope` field is load-bearing:
		// it travels with the number so no downstream reader can mistake this synthetic,
		// reduced-layer, dense-FFN lower-bound for the full 753B MoE serving throughput.
		rec := map[string]any{
			"schema":        "glm-throughput/1",
			"backend":       backend,
			"precision":     prec,
			"config":        map[string]int{"layers": *layers, "hidden": *hidden, "heads": *heads, "inter": *inter, "vocab": *vocab},
			"mla_dsa":       map[string]int{"q_lora": *qLora, "kv_lora": *kvLora, "qk_nope": *qkNope, "qk_rope": *qkRope, "v_head": *vHead, "index_heads": *idxHeads, "index_dim": *idxDim, "index_topk": *idxTopK},
			"prompt_len":    *prompt,
			"decode_steps":  *steps,
			"reps":          *reps,
			"build_ms":      round2(buildMS),
			"prefill_ms":    round2(prefillMS),
			"prefill_tok_s": round2(float64(*prompt) / (prefillMS / 1e3)),
			"decode_ms_tok": round3(decodeMS),
			"decode_tok_s":  round2(1.0 / (decodeMS / 1e3)),
			"model":         "glm_moe_dsa",
			"scope":         "synthetic-weights;reduced-layers;dense-FFN(no-MoE);optimistic-lower-bound;NOT-the-753B",
		}
		b, _ := json.Marshal(rec)
		fmt.Printf("GLMTPUT_JSON %s\n", b)
	}
}

func round2(f float64) float64 {
	return float64(int64(f*100+0.5)) / 100
}

func round3(f float64) float64 {
	return float64(int64(f*1000+0.5)) / 1000
}
