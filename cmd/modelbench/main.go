// Command modelbench measures the in-kernel pure-Go forward pass latency so the
// fusion lane has a HONEST throughput baseline to set against the next-best ways
// to run the same model (HF transformers; see bench_hf.py for the witness side).
//
// It measures BOTH the original naive serial core and the parity lane (parallel
// matmul + batched prefill GEMM + fdot ILP); pin FAK_WORKERS=1 to reproduce the
// serial baseline. MODEL-BASELINE-RESULTS.md sets the numbers against HF/llama.cpp.
//
// Core budget: the matmul worker count is resolved (in internal/model/budget.go) with
// precedence FAK_WORKERS=<n> (absolute) > FAK_BUDGET=<fraction> > all cores. FAK_BUDGET
// is the portable "use up to X% of this machine" knob — FAK_BUDGET=0.75 (or 75, or 75%)
// takes 75% of the logical cores on whatever box this is, so a bench can leave headroom
// for other agentic work without hardcoding a per-box core count. The -budget flag is
// the same knob on the command line. The emitted report records both the resolved
// "workers" count and the "budget" source so a number states the regime it was taken at.
//
// Apples-to-apples with bench_hf.py: both sides drive the SAME deterministic
// token-id sequences (an LCG, replicated bit-for-bit in Python) at the SAME sizes,
// feeding token IDS directly so no tokenizer enters the comparison. Token VALUES do
// not affect compute cost (matmul/attention cost depends only on sequence length),
// so synthetic ids measure the identical work a real prompt would.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// loadHF loads a HuggingFace snapshot directory (config.json + single-file or sharded
// safetensors) entirely in Go — the pure-Go safetensors reader + bf16->f32 decode in
// internal/model, no torch in the loop. It is what lets fak run any Llama/Qwen2-family
// checkpoint on this box without the export_oracle.py (torch) step: the generic config-driven
// forward pass already handles GQA, RoPE theta, SwiGLU, tied embeddings, and Qwen2 qkv-bias.
// Returns the model and a display name derived from model_type + parameter scale.
func readHFConfig(dir string) (model.Config, error) {
	var cfg model.Config
	cb, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, fmt.Errorf("config.json: %w", err)
	}
	if err := json.Unmarshal(cb, &cfg); err != nil {
		return cfg, fmt.Errorf("config.json parse: %w", err)
	}
	// HF Llama/Qwen2 configs omit head_dim (it is hidden_size/num_attention_heads).
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

func loadHF(dir string) (*model.Model, string, error) {
	cfg, err := readHFConfig(dir)
	if err != nil {
		return nil, "", err
	}
	m, err := model.LoadSafetensorsDir(dir, cfg)
	if err != nil {
		return nil, "", fmt.Errorf("safetensors: %w", err)
	}
	return m, hfName(cfg, dir), nil
}

// loadHFLean loads via the memory-lean quantize-at-load path (f32 of the big weights dropped),
// the loader that lets a 7B-class model fit on this box. Quant-only: the bench forces -quant.
func loadHFLean(dir string) (*model.Model, string, error) {
	cfg, err := readHFConfig(dir)
	if err != nil {
		return nil, "", err
	}
	m, err := model.LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		return nil, "", fmt.Errorf("safetensors(lean): %w", err)
	}
	return m, hfName(cfg, dir) + " [lean]", nil
}

func loadGGUF(path string) (*model.Model, string, error) {
	m, err := ggufload.LoadModel(path)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + " [gguf]", nil
}

func loadGGUFLean(path string, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	m, err := ggufload.LoadModelQuantProfile(path, lp)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + " [gguf-lean]", nil
}

// hfName builds a report label like "qwen2-1.5B" from the config (param count is approximated
// from the dominant weight shapes), falling back to the directory basename.
func hfName(cfg model.Config, dir string) string {
	base := filepath.Base(strings.TrimRight(dir, "/"))
	if cfg.ModelType == "" {
		return base
	}
	return fmt.Sprintf("%s (%s)", base, cfg.ModelType)
}

// argmax returns the index of the maximum logit (first on ties) — the greedy next token.
func argmax(v []float32) int {
	best, bi := float32(-math.MaxFloat32), 0
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// lcgIDs builds n deterministic token ids in [0,vocab) via a glibc LCG. The exact
// same recurrence is reproduced in bench_hf.py so both engines see identical input.
func lcgIDs(n, vocab int) []int {
	return lcgIDsSeed(n, vocab, 2463534242)
}

func lcgIDsSeed(n, vocab int, seed uint64) []int {
	ids := make([]int, n)
	state := seed
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

type prefillResult struct {
	Name           string  `json:"name,omitempty"`
	Source         string  `json:"source,omitempty"`
	Tokens         int     `json:"tokens"`
	RecordedTokens int     `json:"recorded_tokens,omitempty"`
	Reps           int     `json:"reps"`
	MedianMS       float64 `json:"median_ms"`
	TokPerSec      float64 `json:"tok_per_sec"`
}

type decodeResult struct {
	PromptTokens  int     `json:"prompt_tokens"`
	DecodeSteps   int     `json:"decode_steps"`
	Reps          int     `json:"reps"`
	PerTokenMedMS float64 `json:"per_token_median_ms"`
	TokPerSec     float64 `json:"tok_per_sec"`
}

type workloadDecodeResult struct {
	Name                 string  `json:"name"`
	Source               string  `json:"source,omitempty"`
	PromptTokens         int     `json:"prompt_tokens"`
	RecordedPromptTokens int     `json:"recorded_prompt_tokens,omitempty"`
	DecodeSteps          int     `json:"decode_steps"`
	RecordedDecodeTokens int     `json:"recorded_decode_tokens"`
	Reps                 int     `json:"reps"`
	PerTokenMedMS        float64 `json:"per_token_median_ms"`
	TokPerSec            float64 `json:"tok_per_sec"`
}

func capPositive(n, cap int) int {
	if cap > 0 && n > cap {
		return cap
	}
	if n < 1 {
		return 1
	}
	return n
}

// benchFlags holds every parsed command-line flag for the benchmark. Fields are
// pointers (as returned by the flag package) so the existing in-place mutations
// (e.g. -lean and -metal forcing -quant) keep working exactly as before.
type benchFlags struct {
	dir                   *string
	hf                    *string
	gguf                  *string
	lean                  *bool
	name                  *string
	out                   *string
	prefillSizesCSV       *string
	prefillReps           *int
	decodeReps            *int
	decodeSteps           *int
	decodePrompt          *int
	quant                 *bool
	metal                 *bool
	verify                *bool
	backendName           *string
	requireNonReference   *bool
	workloadPath          *string
	workloadPrefillCap    *int
	loadOnly              *bool
	loadProfile           *bool
	loadProfileTrace      *bool
	loadProfileTraceEvery *int
	phaseProfile          *bool
	budget                *float64
}

// parseFlags defines and parses the command-line flags, then expands a leading ~
// in the path flags (Go/PowerShell don't), so ~/... opens as intended.
func parseFlags() *benchFlags {
	f := &benchFlags{
		dir:                   flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir (fak format: config/manifest/weights.f32)"),
		hf:                    flag.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors, bf16/f32, loaded fully in Go); overrides -dir"),
		gguf:                  flag.String("gguf", "", "GGUF checkpoint path; default dequantizes to f32, -lean streams to Q8; overrides -hf and -dir"),
		lean:                  flag.Bool("lean", false, "memory-lean load: quantize big matmul weights at load and drop their f32 (with -hf or -gguf; implies -quant; fits much bigger models)"),
		name:                  flag.String("name", "", "model name for the report (default: derived from the source dir)"),
		out:                   flag.String("out", "", "write JSON result here (default stdout)"),
		prefillSizesCSV:       flag.String("prefill-sizes", "16,64,256", "comma-separated prompt lengths for prefill timings"),
		prefillReps:           flag.Int("prefill-reps", 5, "reps per prefill size (median)"),
		decodeReps:            flag.Int("decode-reps", 5, "reps for decode (median over per-token)"),
		decodeSteps:           flag.Int("decode-steps", 32, "tokens to decode"),
		decodePrompt:          flag.Int("decode-prompt", 16, "prompt length before decode"),
		quant:                 flag.Bool("quant", false, "use the Q8_0 quantized forward path (else f32)"),
		metal:                 flag.Bool("metal", false, "run prefill projections on the Metal GPU backend (requires -tags fakmetal; implies -quant for the weight store)"),
		verify:                flag.Bool("verify", false, "with -metal: cross-check the Metal prefill's last-token logits against the CPU Q8 path (argmax agreement + max|Δ|) and exit"),
		backendName:           flag.String("backend", "legacy", "execution backend: legacy or a compute backend name"),
		requireNonReference:   flag.Bool("require-non-reference", false, "fail unless -backend selects a non-reference compute backend"),
		workloadPath:          flag.String("workload", "", "optional recorded agent workload JSON; emits workload_prefill/workload_decode"),
		workloadPrefillCap:    flag.Int("workload-prefill-cap", 0, "cap recorded workload prompt lengths for smoke runs (0 = full recorded length)"),
		loadOnly:              flag.Bool("load-only", false, "load the model, emit load time + peak RSS JSON, and exit without running inference"),
		loadProfile:           flag.Bool("load-profile", false, "emit GGUF->Q8 quant-on-load phase profile (requires -gguf -lean; also enabled by -phase-profile)"),
		loadProfileTrace:      flag.Bool("load-profile-trace", false, "with GGUF load profiling, stream per-tensor load timings to stderr while loading"),
		loadProfileTraceEvery: flag.Int("load-profile-trace-every", 25, "tensor interval for -load-profile-trace after the first tensor"),
		phaseProfile:          flag.Bool("phase-profile", false, "emit one-shot coarse Session phase profiles for prefill/decode without perturbing median timings"),
		budget:                flag.Float64("budget", 0, "fractional core budget for this run: 0.75 = use up to 75% of the machine's logical cores (portable across box sizes; 75 or 0.75 both accepted). 0 = unset. FAK_WORKERS, if set, still overrides."),
	}
	flag.Parse()
	*f.dir = pathutil.ExpandTilde(*f.dir)
	*f.gguf = pathutil.ExpandTilde(*f.gguf)
	*f.hf = pathutil.ExpandTilde(*f.hf)
	return f
}

// applyBudget re-resolves the matmul worker count after init from a -budget flag (the
// env-driven default was already read at package load). FAK_WORKERS is an explicit
// absolute override, so honor it over a fractional -budget rather than silently ignoring it.
func applyBudget(budget float64) {
	if budget <= 0 {
		return
	}
	if os.Getenv("FAK_WORKERS") != "" {
		fmt.Fprintf(os.Stderr, "[fak] FAK_WORKERS is set; ignoring -budget %g (absolute override wins)\n", budget)
	} else if err := model.SetWorkerBudget(budget); err != nil {
		fmt.Fprintln(os.Stderr, "budget:", err)
		os.Exit(2)
	}
}

// validateFlags enforces the flag combinations that must hold before any load.
func validateFlags(f *benchFlags) {
	if *f.loadProfile && (*f.gguf == "" || !*f.lean) {
		fmt.Fprintln(os.Stderr, "-load-profile requires -gguf and -lean")
		os.Exit(2)
	}
	if *f.loadProfileTrace && (*f.gguf == "" || !*f.lean) {
		fmt.Fprintln(os.Stderr, "-load-profile-trace requires -gguf and -lean")
		os.Exit(2)
	}
}

// acquireMetalLease takes a machine-wide GPU lease so concurrent -metal runs QUEUE
// instead of stacking residency on the same unified-memory pool (a jetsam cascade and
// kernel watchdog panic on 2026-06-18). Default: wait for the lease; set
// FAK_GPU_LEASE_NOWAIT=1 to fail fast instead. The lease is held for the whole process
// and the OS drops it on exit, so an os.Exit path still frees it. Gate on Available()
// (cheap, model-independent) so a -metal run that will fall back to CPU does not
// needlessly serialize behind the GPU lease. Returns a release func to defer.
func acquireMetalLease(metal bool) func() {
	if !metal || !metalgemm.Available() {
		return func() {}
	}
	lease, lerr := gpulease.Acquire(gpulease.Options{NoWait: os.Getenv("FAK_GPU_LEASE_NOWAIT") != ""})
	if lerr != nil {
		fmt.Fprintln(os.Stderr, "metal:", lerr)
		os.Exit(1)
	}
	return lease.Release
}

// newGGUFLoadProfiler builds the GGUF->Q8 quant-on-load profiler when one of the
// profile flags is set on a -gguf -lean run, else returns nil.
func newGGUFLoadProfiler(f *benchFlags) *ggufload.LoadProfiler {
	wantLoadProfile := (*f.loadProfile || *f.loadProfileTrace || *f.phaseProfile) && *f.gguf != "" && *f.lean
	if !wantLoadProfile {
		return nil
	}
	lp := ggufload.NewLoadProfiler()
	if *f.loadProfileTrace {
		lp.Trace = os.Stderr
		lp.Every = *f.loadProfileTraceEvery
	}
	return lp
}

// loadModel selects the load path from the flags (lean GGUF/HF, plain GGUF/HF, or fak
// dir format) and returns the model plus its report label. May set *f.quant for -lean.
func loadModel(f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	if *f.lean {
		if *f.hf == "" && *f.gguf == "" {
			fmt.Fprintln(os.Stderr, "-lean requires -hf or -gguf")
			os.Exit(2)
		}
		*f.quant = true // the lean model holds no f32 for the big weights; the f32 path would panic
		if *f.gguf != "" {
			return loadGGUFLean(*f.gguf, lp)
		}
		return loadHFLean(*f.hf)
	}
	if *f.gguf != "" {
		return loadGGUF(*f.gguf)
	}
	if *f.hf != "" {
		return loadHF(*f.hf)
	}
	m, err := model.Load(*f.dir)
	return m, filepath.Base(*f.dir), err
}

// runLoadOnly emits the load-time + peak-RSS report and is the whole job for -load-only.
func runLoadOnly(f *benchFlags, modelName string, loadMS float64, ggufLoadProfile *ggufload.LoadProfile) {
	peakRSS, rssErr := peakRSSBytes()
	report := map[string]any{
		"app_version":          appversion.Current(),
		"engine":               "fak model load",
		"model":                modelName,
		"source":               loadSource(*f.hf, *f.gguf, *f.dir, *f.lean),
		"load_ms":              loadMS,
		"lean":                 *f.lean,
		"quantized_at_load":    *f.lean,
		"peak_rss_bytes":       peakRSS,
		"peak_rss_unavailable": rssErr != nil,
	}
	if rssErr != nil {
		report["peak_rss_error"] = rssErr.Error()
	}
	if ggufLoadProfile != nil {
		report["load_profile"] = ggufLoadProfile
	}
	writeReport(*f.out, report)
}

// resolveBackend looks up the named compute backend (nil for "legacy") and enforces the
// Q8-upload and non-reference gates. Returns the backend and the registered-backend list.
func resolveBackend(f *benchFlags) (compute.Backend, []string) {
	registeredBackends := compute.Registered()
	var be compute.Backend
	if *f.backendName != "legacy" {
		var ok bool
		be, ok = compute.Lookup(*f.backendName)
		if !ok {
			fmt.Fprintf(os.Stderr, "backend: unknown %q (registered: %v)\n", *f.backendName, registeredBackends)
			os.Exit(2)
		}
		// Q8 on a compute backend needs the device to accept quantized weight uploads
		// (the wired Q8 HAL path keys off Caps().UploadDtype). A backend that can't —
		// e.g. cpu-ref or an f32-only device — still refuses -quant rather than silently
		// running the f32 path under a Q8 flag.
		if *f.quant && !be.Caps().UploadDtype {
			fmt.Fprintf(os.Stderr, "backend: %q is f32-only (no Q8 upload support); omit -quant\n", be.Name())
			os.Exit(2)
		}
		if *f.requireNonReference && be.Class() == compute.Reference {
			fmt.Fprintf(os.Stderr, "backend: %q is %s; production Phase-1 gate requires a non-reference backend\n", be.Name(), be.Class())
			os.Exit(2)
		}
	} else if *f.requireNonReference {
		fmt.Fprintln(os.Stderr, "backend: -require-non-reference needs -backend to name a compute backend")
		os.Exit(2)
	}
	return be, registeredBackends
}

// resolveMetal validates the Metal prefill path: it needs the Q8 weight store (it
// dequantizes its f16 GPU copies from it) and a live device. Resolve availability before
// the quantize step so the report is honest, falling back to CPU Q8 when unavailable.
func resolveMetal(f *benchFlags) {
	if !*f.metal {
		return
	}
	*f.quant = true
	if !metalgemm.Available() {
		if metalgemm.Compiled() {
			fmt.Fprintln(os.Stderr, "metal: no usable Metal device; falling back to CPU Q8 prefill")
		} else {
			fmt.Fprintln(os.Stderr, "metal: backend not compiled in (rebuild with -tags fakmetal); falling back to CPU Q8 prefill")
		}
		*f.metal = false
	}
}

// runVerify proves the Metal prefill is numerically faithful before trusting its speed.
// Compare its last-token logits to the CPU Q8 path (already argmax-validated vs the HF
// oracle) on several prompt lengths: argmax must agree, and the logit max|Δ| should sit
// at the f16-vs-Q8 noise floor, not diverge. It is terminal (exits the process).
func runVerify(f *benchFlags, m *model.Model, vocab int) {
	if !*f.metal {
		fmt.Fprintln(os.Stderr, "-verify requires -metal")
		os.Exit(2)
	}
	allOK := true
	for _, P := range []int{8, 32, 128, 256} {
		ids := lcgIDs(P, vocab)
		sc := m.NewSession()
		sc.Quant = true
		lc := sc.Prefill(ids)
		sg := m.NewSession()
		sg.Metal = true
		lg := sg.Prefill(ids)
		var maxAbs float64
		ac, ag := argmax(lc), argmax(lg)
		for i := range lc {
			if d := math.Abs(float64(lc[i] - lg[i])); d > maxAbs {
				maxAbs = d
			}
		}
		ok := ac == ag
		allOK = allOK && ok
		fmt.Printf("P=%-4d argmax cpu=%-7d metal=%-7d agree=%-5v  max|Δlogit|=%.4f\n", P, ac, ag, ok, maxAbs)
	}
	if allOK {
		fmt.Println("VERIFY OK — Metal prefill argmax-matches the CPU Q8 path on all lengths")
	} else {
		fmt.Println("VERIFY FAIL — Metal prefill diverges from the CPU Q8 path")
		os.Exit(1)
	}
}

// describeEngine derives the human-readable engine string, precision label, and the
// backend sub-report from the resolved flags and backend.
func describeEngine(f *benchFlags, be compute.Backend, registeredBackends []string) (engine, precision string, backendReport map[string]any) {
	engine = "fak-in-kernel (pure-Go, parallel matmul + batched prefill GEMM + fdot ILP)"
	precision = "f32"
	if *f.quant {
		engine = "fak-in-kernel Q8_0 (pure-Go, quantized weights+activations, int8×int8→int32 dot)"
		precision = "Q8_0"
	}
	if *f.metal {
		engine = "fak-in-kernel Metal prefill (MPS f16 GEMM on GPU; CPU Q8 decode)"
		precision = "Q8_0 weights / f16 GPU GEMM"
	}
	backendReport = map[string]any{
		"selected":            "legacy",
		"registered_backends": registeredBackends,
	}
	if be != nil {
		engine = fmt.Sprintf("fak-in-kernel via compute HAL backend %q", be.Name())
		backendReport = map[string]any{
			"selected":            be.Name(),
			"tier":                be.Tier(),
			"class":               be.Class().String(),
			"caps":                be.Caps(),
			"registered_backends": registeredBackends,
		}
	}
	return engine, precision, backendReport
}

func modelConfigReport(cfg model.Config) map[string]any {
	return map[string]any{
		"model_type":              cfg.ModelType,
		"architectures":           cfg.Architectures,
		"hidden_size":             cfg.HiddenSize,
		"num_hidden_layers":       cfg.NumLayers,
		"num_attention_heads":     cfg.NumHeads,
		"num_key_value_heads":     cfg.NumKVHeads,
		"head_dim":                cfg.HeadDim,
		"intermediate_size":       cfg.IntermediateSize,
		"vocab_size":              cfg.VocabSize,
		"is_moe":                  cfg.IsMoE(),
		"num_local_experts":       cfg.NumExperts,
		"num_experts_per_tok":     cfg.NumExpertsPerTok,
		"q_lora_rank":             cfg.QLoraRank,
		"kv_lora_rank":            cfg.KVLoraRank,
		"qk_nope_head_dim":        cfg.QKNopeHeadDim,
		"qk_rope_head_dim":        cfg.QKRopeHeadDim,
		"v_head_dim":              cfg.VHeadDim,
		"index_n_heads":           cfg.IndexNHeads,
		"index_head_dim":          cfg.IndexHeadDim,
		"index_topk":              cfg.IndexTopK,
		"indexer_types":           cfg.IndexerTypes,
		"max_position_embeddings": cfg.MaxPositionEmbeddings,
	}
}

// runPrefill times Session.Prefill over each P in prefillSizes (builds KV cache, last
// logits) and records the median timings and any phase profiles into the report maps.
func runPrefill(f *benchFlags, newSession func() *model.Session, vocab int, prefillSizes []int, report, phaseReport map[string]any) {
	var prefills []prefillResult
	var prefillPhases []*model.PhaseProfile
	for _, p := range prefillSizes {
		ids := lcgIDs(p, vocab)
		ds := make([]time.Duration, *f.prefillReps)
		for r := 0; r < *f.prefillReps; r++ {
			s := newSession()
			t := time.Now()
			s.Prefill(ids)
			ds[r] = time.Since(t)
			s.Close()
		}
		med := medianMS(ds)
		prefills = append(prefills, prefillResult{
			Tokens: p, Reps: *f.prefillReps, MedianMS: med,
			TokPerSec: float64(p) / (med / 1e3),
		})
		fmt.Fprintf(os.Stderr, "[fak] prefill P=%d: %.1f ms (%.1f tok/s)\n", p, med, float64(p)/(med/1e3))
		if *f.phaseProfile {
			s := newSession()
			pp := model.NewPhaseProfiler()
			s.PhaseProfiler = pp
			t := time.Now()
			s.Prefill(ids)
			total := time.Since(t)
			snap := pp.Snapshot("prefill", p, 0, total.Nanoseconds())
			prefillPhases = append(prefillPhases, snap)
			fmt.Fprint(os.Stderr, phaseTable(snap))
			s.Close()
		}
	}
	report["prefill"] = prefills
	if *f.phaseProfile {
		phaseReport["prefill"] = prefillPhases
	}
}

// runDecode prefills a short prompt then times D incremental Step() calls, recording the
// per-token median and any phase profile into the report maps.
func runDecode(f *benchFlags, newSession func() *model.Session, vocab int, report, phaseReport map[string]any) {
	prompt := lcgIDs(*f.decodePrompt, vocab)
	perTok := make([]time.Duration, 0, *f.decodeReps)
	for r := 0; r < *f.decodeReps; r++ {
		s := newSession()
		s.Prefill(prompt)
		id := int(uint64(r*131+7) % uint64(vocab))
		t := time.Now()
		for i := 0; i < *f.decodeSteps; i++ {
			logits := s.Step(id)
			// pick next deterministically (argmax-free, value-irrelevant to cost)
			id = (id*48271 + 1) % vocab
			_ = logits
		}
		perTok = append(perTok, time.Since(t)/time.Duration(*f.decodeSteps))
		s.Close()
	}
	med := medianMS(perTok)
	report["decode"] = decodeResult{
		PromptTokens: *f.decodePrompt, DecodeSteps: *f.decodeSteps, Reps: *f.decodeReps,
		PerTokenMedMS: med, TokPerSec: 1.0 / (med / 1e3),
	}
	fmt.Fprintf(os.Stderr, "[fak] decode: %.1f ms/tok (%.1f tok/s)\n", med, 1.0/(med/1e3))
	if *f.phaseProfile {
		s := newSession()
		s.Prefill(prompt)
		pp := model.NewPhaseProfiler()
		s.PhaseProfiler = pp
		id := 7
		t := time.Now()
		for i := 0; i < *f.decodeSteps; i++ {
			logits := s.Step(id)
			id = (id*48271 + 1) % vocab
			_ = logits
		}
		total := time.Since(t)
		snap := pp.Snapshot("decode", *f.decodePrompt, *f.decodeSteps, total.Nanoseconds())
		phaseReport["decode"] = snap
		fmt.Fprint(os.Stderr, phaseTable(snap))
		s.Close()
	}
}

// runWorkload replays the recorded agent workload cases: a prefill timing and a decode
// timing per case, at the recorded (capped) prompt/decode lengths, into the report.
func runWorkload(f *benchFlags, newSession func() *model.Session, vocab int, workload *model.BenchWorkload, report map[string]any) {
	var wp []prefillResult
	for i, c := range workload.Cases {
		n := capPositive(c.PromptTokens, *f.workloadPrefillCap)
		ids := lcgIDsSeed(n, vocab, 0xC0FFEE+uint64(i)*977)
		ds := make([]time.Duration, *f.prefillReps)
		for r := 0; r < *f.prefillReps; r++ {
			s := newSession()
			t := time.Now()
			s.Prefill(ids)
			ds[r] = time.Since(t)
			s.Close()
		}
		med := medianMS(ds)
		wp = append(wp, prefillResult{
			Name: c.Name, Source: c.Source, Tokens: n, RecordedTokens: c.PromptTokens,
			Reps: *f.prefillReps, MedianMS: med, TokPerSec: float64(n) / (med / 1e3),
		})
		fmt.Fprintf(os.Stderr, "[fak workload] prefill %s P=%d recorded=%d: %.1f ms\n", c.Name, n, c.PromptTokens, med)
	}
	report["workload_prefill"] = wp

	var wd []workloadDecodeResult
	for i, c := range workload.Cases {
		promptN := capPositive(c.PromptTokens, *f.workloadPrefillCap)
		steps := capPositive(c.CompletionTokens, *f.decodeSteps)
		prompt := lcgIDsSeed(promptN, vocab, 0xA11CE+uint64(i)*131)
		perTok := make([]time.Duration, 0, *f.decodeReps)
		for r := 0; r < *f.decodeReps; r++ {
			s := newSession()
			s.Prefill(prompt)
			id := int((uint64(r+1)*2654435761 + uint64(i)) % uint64(vocab))
			t := time.Now()
			for j := 0; j < steps; j++ {
				logits := s.Step(id)
				id = (id*48271 + 1) % vocab
				_ = logits
			}
			perTok = append(perTok, time.Since(t)/time.Duration(steps))
			s.Close()
		}
		med := medianMS(perTok)
		wd = append(wd, workloadDecodeResult{
			Name: c.Name, Source: c.Source,
			PromptTokens: promptN, RecordedPromptTokens: c.PromptTokens,
			DecodeSteps: steps, RecordedDecodeTokens: c.CompletionTokens,
			Reps: *f.decodeReps, PerTokenMedMS: med, TokPerSec: 1.0 / (med / 1e3),
		})
		fmt.Fprintf(os.Stderr, "[fak workload] decode %s prompt=%d recorded=%d steps=%d/%d: %.1f ms/tok\n",
			c.Name, promptN, c.PromptTokens, steps, c.CompletionTokens, med)
	}
	report["workload_decode"] = wd
}

func main() {
	f := parseFlags()
	applyBudget(*f.budget)
	validateFlags(f)

	prefillSizes, parseErr := parsePositiveInts(*f.prefillSizesCSV)
	if parseErr != nil {
		fmt.Fprintln(os.Stderr, "prefill-sizes:", parseErr)
		os.Exit(2)
	}

	var workload *model.BenchWorkload
	if *f.workloadPath != "" {
		var err error
		workload, err = model.LoadBenchWorkload(*f.workloadPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "workload:", err)
			os.Exit(1)
		}
	}

	defer acquireMetalLease(*f.metal)()

	ggufLoadProfiler := newGGUFLoadProfiler(f)

	t0 := time.Now()
	m, modelName, err := loadModel(f, ggufLoadProfiler)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	if *f.name != "" {
		modelName = *f.name
	}
	loadNanos := time.Since(t0).Nanoseconds()
	loadMS := float64(loadNanos) / 1e6
	var ggufLoadProfile *ggufload.LoadProfile
	if ggufLoadProfiler != nil {
		ggufLoadProfile = ggufLoadProfiler.Snapshot("gguf-lean-q8", *f.gguf, loadNanos)
	}
	if *f.loadOnly {
		runLoadOnly(f, modelName, loadMS, ggufLoadProfile)
		return
	}
	vocab := m.Cfg.VocabSize
	be, registeredBackends := resolveBackend(f)
	resolveMetal(f)

	// Quantize once up front (off the timed path) when in Q8 mode. newSession stamps the
	// Quant flag onto every session the benchmark creates so prefill+decode use it.
	var quantMS float64
	if *f.quant {
		tq := time.Now()
		m.Quantize()
		quantMS = float64(time.Since(tq).Nanoseconds()) / 1e6
	}
	newSession := func() *model.Session {
		if be != nil {
			s := m.NewBackendSession(be)
			s.Quant = *f.quant // routes the HAL through the Q8 weight path when the backend advertises UploadDtype
			return s
		}
		s := m.NewSession()
		s.Quant = *f.quant
		s.Metal = *f.metal
		return s
	}

	if *f.verify {
		runVerify(f, m, vocab)
		return
	}

	// Warm up: first forward pages in all the weights + JITs allocation paths.
	// Time only steady-state, matching the HF side which also warms up.
	{
		s := newSession()
		s.Prefill(lcgIDs(8, vocab))
		s.Step(s.Cache.Len() % vocab)
		s.Close()
	}

	engine, precision, backendReport := describeEngine(f, be, registeredBackends)
	report := map[string]any{
		"app_version":       appversion.Current(),
		"engine":            engine,
		"model":             modelName,
		"model_config":      modelConfigReport(m.Cfg),
		"source":            loadSource(*f.hf, *f.gguf, *f.dir, *f.lean),
		"precision":         precision,
		"backend":           backendReport,
		"load_ms":           loadMS,
		"quant_ms":          quantMS,
		"lean":              *f.lean,
		"quantized_at_load": *f.lean,
		"workers":           model.NumWorkers(),   // actual matmul parallelism these numbers were taken at
		"budget":            model.WorkerBudget(), // how the worker count was resolved (FAK_WORKERS / FAK_BUDGET / -budget / default)
		"go_threads":        fmt.Sprintf("GOMAXPROCS=%d, matmul workers=%d (FAK_WORKERS / FAK_BUDGET / -budget to pin)", runtime.GOMAXPROCS(0), model.NumWorkers()),
	}
	if ggufLoadProfile != nil {
		report["load_profile"] = ggufLoadProfile
	}
	phaseReport := map[string]any{}
	if workload != nil {
		report["workload"] = map[string]any{
			"path":             *f.workloadPath,
			"schema":           workload.Schema,
			"name":             workload.Name,
			"source":           workload.Source,
			"cases":            len(workload.Cases),
			"prefill_cap":      *f.workloadPrefillCap,
			"decode_steps_cap": *f.decodeSteps,
			"token_ids":        "deterministic LCG IDs at recorded prompt/decode lengths; token values are cost-irrelevant for this compute benchmark",
		}
	}

	runPrefill(f, newSession, vocab, prefillSizes, report, phaseReport)
	runDecode(f, newSession, vocab, report, phaseReport)
	if workload != nil {
		runWorkload(f, newSession, vocab, workload, report)
	}
	if *f.phaseProfile {
		report["phase_profile"] = phaseReport
	}

	writeReport(*f.out, report)
}

func loadSource(hf, gguf, dir string, lean bool) string {
	if gguf != "" {
		return gguf
	}
	if hf == "" {
		return dir
	}
	if lean {
		return filepath.Join(hf, "model.safetensors") + " (quantize-at-load)"
	}
	return filepath.Join(hf, "model.safetensors")
}

func parsePositiveInts(csv string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(csv, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := strconv.Atoi(part)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("invalid positive integer %q", part)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one size is required")
	}
	return out, nil
}

func writeReport(out string, report map[string]any) {
	b, _ := json.MarshalIndent(report, "", "  ")
	if out != "" {
		if err := os.WriteFile(out, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "wrote", out)
		return
	}
	fmt.Println(string(b))
}

func phaseTable(p *model.PhaseProfile) string {
	if p == nil {
		return ""
	}
	n := 8
	if len(p.Phases) < n {
		n = len(p.Phases)
	}
	s := fmt.Sprintf("[fak phase] %s tokens=%d steps=%d total=%.1f ms bottleneck=%s\n",
		p.Mode, p.Tokens, p.Steps, p.TotalMS, p.Bottleneck)
	for i := 0; i < n; i++ {
		ph := p.Phases[i]
		s += fmt.Sprintf("  %-28s %7.1f ms %5.1f%% calls=%d\n", ph.Phase, ph.MS, ph.TimePct, ph.Calls)
	}
	return s
}
