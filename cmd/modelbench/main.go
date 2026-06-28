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
	"github.com/anthony-chaudhary/fak/internal/benchcli"
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
func loadHF(dir string) (*model.Model, string, error) {
	return loadHFWith(dir, "safetensors", "", model.LoadSafetensorsDir)
}

// loadHFLean loads via the memory-lean quantize-at-load path (f32 of the big weights dropped),
// the loader that lets a 7B-class model fit on this box. Quant-only: the bench forces -quant.
func loadHFLean(dir string) (*model.Model, string, error) {
	return loadHFWith(dir, "safetensors(lean)", " [lean]", func(d string, c model.Config) (*model.Model, error) {
		return model.LoadSafetensorsQuantDir(d, c)
	})
}

// loadHFWith reads the HF config from dir and loads the model via load, wrapping a load
// failure as "<label>: <err>" and returning the hfName display string with nameSuffix
// appended. It is the shared body of loadHF (full) and loadHFLean (memory-lean) which
// differ only by loader, error label, and name suffix.
func loadHFWith(dir, label, nameSuffix string, load func(string, model.Config) (*model.Model, error)) (*model.Model, string, error) {
	cfg, err := benchcli.ReadHFConfig(dir)
	if err != nil {
		return nil, "", err
	}
	m, err := load(dir, cfg)
	if err != nil {
		return nil, "", fmt.Errorf("%s: %w", label, err)
	}
	return m, hfName(cfg, dir) + nameSuffix, nil
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

func loadGGUFQ4K(path string) (*model.Model, string, error) {
	m, err := ggufload.LoadModelQ4K(path)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + " [gguf-q4k]", nil
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

func logitTop2(v []float32) (top1Idx int, top1, top2 float32) {
	top1, top2 = float32(-math.MaxFloat32), float32(-math.MaxFloat32)
	for i, x := range v {
		if x > top1 {
			top2 = top1
			top1, top1Idx = x, i
		} else if x > top2 {
			top2 = x
		}
	}
	return top1Idx, top1, top2
}

func cosineF32(a, b []float32) float64 {
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}

func tryF32Prefill(m *model.Model, ids []int) (logits []float32, ok bool) {
	defer func() {
		if recover() != nil {
			logits, ok = nil, false
		}
	}()
	s := m.NewSession()
	return s.Prefill(ids), true
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
	q4k                   *bool
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
	preflight             *bool
	smoke                 *bool
	smokeDeadline         *time.Duration
	fitCheck              *bool
	loadProgress          *bool
}

// parseFlags defines and parses the command-line flags, then expands a leading ~
// in the path flags (Go/PowerShell don't), so ~/... opens as intended.
func parseFlags() *benchFlags {
	f := &benchFlags{
		dir:                   flag.String("dir", "internal/model/.cache/smollm2-135m", "model export dir (fak format: config/manifest/weights.f32)"),
		hf:                    flag.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors, bf16/f32, loaded fully in Go); overrides -dir"),
		gguf:                  flag.String("gguf", "", "GGUF checkpoint path; default dequantizes to f32, -lean streams to Q8; overrides -hf and -dir"),
		lean:                  flag.Bool("lean", false, "memory-lean load: quantize big matmul weights at load and drop their f32 (with -hf or -gguf; implies -quant; fits much bigger models)"),
		q4k:                   flag.Bool("q4k", false, "with -gguf, load eligible Q4_K tensors as resident raw Q4_K and run the Q4_K session path"),
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
		preflight:             flag.Bool("preflight", false, "FAIL FAST: read only the GGUF header (no tensor load), report arch/est-size/device-fit/ETA, and exit in seconds. Refuses a bad-arch / too-big / bad-header model before the multi-minute load. Requires -gguf."),
		smoke:                 flag.Bool("smoke", false, "header preflight, then load (under -smoke-deadline) and decode ONE token to prove the forward runs, then exit — before the full prefill/decode/workload grid. Requires -gguf."),
		smokeDeadline:         flag.Duration("smoke-deadline", 90*time.Second, "hard wall-clock cap on the -smoke load: if the load exceeds it, abort and report SMOKE_LOAD_TIMEOUT with the last progress line instead of hanging"),
		fitCheck:              flag.Bool("fit-check", true, "before a normal load, refuse a model that a capacity-reporting -backend KNOWS won't fit (typed refusal instead of a mid-load OOM panic). Fail-open on legacy/cpu-ref. -fit-check=false for deliberate stress runs."),
		loadProgress:          flag.Bool("load-progress", true, "stream throttled load progress (percent / GB / elapsed / GB-per-s) to stderr on lean/q4k GGUF loads so a multi-minute load is not silent; -load-progress=false silences it"),
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
	if *f.q4k {
		switch {
		case *f.gguf == "":
			fmt.Fprintln(os.Stderr, "-q4k requires -gguf")
			os.Exit(2)
		case *f.hf != "":
			fmt.Fprintln(os.Stderr, "-q4k cannot be combined with -hf")
			os.Exit(2)
		case *f.lean:
			fmt.Fprintln(os.Stderr, "-q4k is its own GGUF resident-quant load path; omit -lean")
			os.Exit(2)
		case *f.backendName != "legacy":
			fmt.Fprintln(os.Stderr, "-q4k currently runs through the legacy resident-Q4_K session path; omit -backend")
			os.Exit(2)
		case *f.metal || *f.verify:
			fmt.Fprintln(os.Stderr, "-q4k modelbench scoring is CPU-only for now; omit -metal/-verify")
			os.Exit(2)
		}
	}
	if *f.loadProfile && (*f.gguf == "" || !*f.lean) {
		fmt.Fprintln(os.Stderr, "-load-profile requires -gguf and -lean")
		os.Exit(2)
	}
	if *f.loadProfileTrace && (*f.gguf == "" || !*f.lean) {
		fmt.Fprintln(os.Stderr, "-load-profile-trace requires -gguf and -lean")
		os.Exit(2)
	}
	// -preflight / -smoke read the header (and -smoke loads) of a GGUF; the estimators cover
	// the f32 path too, so do NOT also require -lean (that would block a plain-GGUF preflight).
	if *f.preflight && *f.gguf == "" {
		fmt.Fprintln(os.Stderr, "-preflight requires -gguf")
		os.Exit(2)
	}
	if *f.smoke && *f.gguf == "" {
		fmt.Fprintln(os.Stderr, "-smoke requires -gguf")
		os.Exit(2)
	}
	if *f.preflight && *f.smoke {
		fmt.Fprintln(os.Stderr, "-preflight and -smoke are mutually exclusive (preflight is header-only; smoke also loads)")
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

// newGGUFLoadProfiler builds the GGUF->Q8 quant-on-load profiler. It is created when either a
// -load-profile* flag is set (which attaches the machine-readable load_profile to the report) OR
// -load-progress is on for a lean GGUF load (the default) — so a multi-minute load streams a
// throttled percent/GB/elapsed/GB-per-s status to stderr instead of being a silent black box.
// Returns nil when neither applies (e.g. the f32 path, which does not Tick) so the loader keeps
// its existing no-bookkeeping behavior.
func newGGUFLoadProfiler(f *benchFlags) *ggufload.LoadProfiler {
	leanGGUF := *f.gguf != "" && *f.lean
	wantLoadProfile := (*f.loadProfile || *f.loadProfileTrace || *f.phaseProfile) && leanGGUF
	wantProgress := *f.loadProgress && leanGGUF
	if !wantLoadProfile && !wantProgress {
		return nil
	}
	lp := ggufload.NewLoadProfiler()
	if wantProgress {
		lp.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
	}
	if *f.loadProfileTrace {
		lp.Trace = os.Stderr
		lp.Every = *f.loadProfileTraceEvery
	}
	return lp
}

// loadModel selects the load path from the flags (lean GGUF/HF, plain GGUF/HF, or fak
// dir format) and returns the model plus its report label. May set *f.quant for -lean.
func loadModel(f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	if *f.q4k {
		return loadGGUFQ4K(*f.gguf)
	}
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
		"source":               loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k),
		"load_ms":              loadMS,
		"lean":                 *f.lean,
		"q4k":                  *f.q4k,
		"quantized_at_load":    *f.lean || *f.q4k,
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

// q8UploadUnsupported reports whether -quant was requested against a backend that cannot
// accept quantized weight uploads. The wired Q8 HAL path routes matmul weights through
// compute.Q8_0 only when the backend advertises Caps().UploadDtype (#472); a backend that
// can't — cpu-ref or an f32-only device — must refuse -quant rather than silently run the
// f32 path under a Q8 flag. When quant is false the f32 path is unchanged, so the gate never
// fires regardless of the backend's caps.
func q8UploadUnsupported(quant bool, caps compute.Caps) bool {
	return quant && !caps.UploadDtype
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
		if q8UploadUnsupported(*f.quant, be.Caps()) {
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
// Prefer the f32 path as the reference when the model still holds f32 weights; CPU Q8 is a
// useful speed baseline, but it can flip a greedy token on tiny margins. A f16 GPU path is
// accepted when cosine stays high and every decisive f32 argmax (top1-top2 >= margin) agrees.
// It is terminal (exits the process).
func runVerify(f *benchFlags, m *model.Model, vocab int) {
	if !*f.metal {
		fmt.Fprintln(os.Stderr, "-verify requires -metal")
		os.Exit(2)
	}
	const minMetalVerifyCosine = 0.999
	const decisiveLogitMargin = 0.02
	lengths := []int{8, 32, 128, 256}
	if extra, err := parsePositiveInts(*f.prefillSizesCSV); err == nil {
		seen := make(map[int]bool, len(lengths)+len(extra))
		for _, n := range lengths {
			seen[n] = true
		}
		for _, n := range extra {
			if !seen[n] {
				lengths = append(lengths, n)
				seen[n] = true
			}
		}
	}
	allOK := true
	for _, P := range lengths {
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
		_, c1, c2 := logitTop2(lc)
		_, g1, g2 := logitTop2(lg)
		cpuMetalCos := cosineF32(lc, lg)
		ok := ac == ag && cpuMetalCos >= minMetalVerifyCosine
		status := "cpu-q8"
		line := fmt.Sprintf("P=%-4d argmax cpu=%-7d metal=%-7d agree=%-5v  max|Δlogit|=%.4f  cosine=%.8f  margin cpu=%.4f metal=%.4f",
			P, ac, ag, ac == ag, maxAbs, cpuMetalCos, c1-c2, g1-g2)
		if lf, hasF32 := tryF32Prefill(m, ids); hasF32 {
			af, f1, f2 := logitTop2(lf)
			f32Margin := f1 - f2
			f32MetalCos := cosineF32(lf, lg)
			decisive := f32Margin >= decisiveLogitMargin
			ok = f32MetalCos >= minMetalVerifyCosine && (!decisive || af == ag)
			status = "f32"
			if !decisive && af != ag {
				status = "f32-near-tie"
			}
			line += fmt.Sprintf("  f32_argmax=%-7d f32_margin=%.4f f32_cpu_cos=%.8f f32_metal_cos=%.8f",
				af, f32Margin, cosineF32(lf, lc), f32MetalCos)
		}
		allOK = allOK && ok
		fmt.Printf("%s  status=%s ok=%v\n", line, status, ok)
	}
	if allOK {
		fmt.Println("VERIFY OK — Metal prefill matches the f32 reference on decisive margins")
	} else {
		fmt.Println("VERIFY FAIL — Metal prefill diverges from the available reference")
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
	if *f.q4k {
		engine = "fak-in-kernel resident Q4_K/Q8 hybrid (raw GGUF Q4_K majority + Q8 minority)"
		precision = "Q4_K/Q8 resident hybrid"
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

// timePrefillReps runs reps fresh-session prefills of ids (a new Session per rep, closed
// after timing) and returns the median wall time in ms.
func timePrefillReps(newSession func() *model.Session, ids []int, reps int) float64 {
	ds := make([]time.Duration, reps)
	for r := 0; r < reps; r++ {
		s := newSession()
		t := time.Now()
		s.Prefill(ids)
		ds[r] = time.Since(t)
		s.Close()
	}
	return medianMS(ds)
}

// medDecodeReps runs reps fresh-session decodes of prompt (a new Session per rep, closed
// after timing), seeding each rep's first token via seedID(r), and returns the median
// per-token time in ms over steps decode steps.
func medDecodeReps(newSession func() *model.Session, prompt []int, reps, steps, vocab int, seedID func(r int) int) float64 {
	perTok := make([]time.Duration, 0, reps)
	for r := 0; r < reps; r++ {
		s := newSession()
		s.Prefill(prompt)
		id := seedID(r)
		perTok = append(perTok, stepDecode(s, id, steps, vocab)/time.Duration(steps))
		s.Close()
	}
	return medianMS(perTok)
}

// stepDecode runs steps incremental Step() calls from the seed id, advancing the id
// deterministically (argmax-free, value-irrelevant to cost), and returns the elapsed time.
func stepDecode(s *model.Session, id, steps, vocab int) time.Duration {
	t := time.Now()
	for i := 0; i < steps; i++ {
		logits := s.Step(id)
		id = (id*48271 + 1) % vocab
		_ = logits
	}
	return time.Since(t)
}

// runPrefill times Session.Prefill over each P in prefillSizes (builds KV cache, last
// logits) and records the median timings and any phase profiles into the report maps.
func runPrefill(f *benchFlags, newSession func() *model.Session, vocab int, prefillSizes []int, report, phaseReport map[string]any) {
	var prefills []prefillResult
	var prefillPhases []*model.PhaseProfile
	for _, p := range prefillSizes {
		ids := lcgIDs(p, vocab)
		med := timePrefillReps(newSession, ids, *f.prefillReps)
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
	med := medDecodeReps(newSession, prompt, *f.decodeReps, *f.decodeSteps, vocab, func(r int) int {
		return int(uint64(r*131+7) % uint64(vocab))
	})
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
		total := stepDecode(s, id, *f.decodeSteps, vocab)
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
		med := timePrefillReps(newSession, ids, *f.prefillReps)
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
		med := medDecodeReps(newSession, prompt, *f.decodeReps, steps, vocab, func(r int) int {
			return int((uint64(r+1)*2654435761 + uint64(i)) % uint64(vocab))
		})
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

	// FAIL FAST: -preflight reads only the GGUF header and exits in seconds, never loading a
	// tensor. It is the answer to "load something for 20 min just to learn a small thing".
	if *f.preflight {
		runPreflight(f)
		return
	}

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

	// FAIL FAST before the load: when a capacity-reporting -backend is named and the model is
	// known too big, refuse with a typed sizing message instead of OOM-panicking mid-load.
	// Header-only and fail-open (legacy/cpu-ref never refused). -fit-check=false to override.
	if *f.fitCheck && *f.gguf != "" {
		runFitGate(f)
	}

	ggufLoadProfiler := newGGUFLoadProfiler(f)

	t0 := time.Now()
	m, modelName, err := loadModelMaybeDeadline(f, ggufLoadProfiler)
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
	// Attach the machine-readable load_profile to the report only when a -load-profile* flag
	// asked for it. A profiler created solely for default-on -load-progress streams to stderr
	// but must not bloat every report's JSON with a phase breakdown nobody requested.
	if ggufLoadProfiler != nil && (*f.loadProfile || *f.loadProfileTrace || *f.phaseProfile) {
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

	// -smoke: prove the forward actually runs (load + one decode) and exit BEFORE the full
	// prefill/decode/workload grid, so a broken forward is caught in one token, not after the
	// whole grid is set up. The load already happened under -smoke-deadline above.
	if *f.smoke {
		runSmoke(f, m, modelName, loadMS, vocab)
		return
	}
	newSession := func() *model.Session {
		if be != nil {
			s := m.NewBackendSession(be)
			s.Quant = *f.quant // routes the HAL through the Q8 weight path when the backend advertises UploadDtype
			return s
		}
		s := m.NewSession()
		s.Quant = *f.quant
		s.Q4K = *f.q4k
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
		"source":            loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k),
		"precision":         precision,
		"backend":           backendReport,
		"load_ms":           loadMS,
		"quant_ms":          quantMS,
		"lean":              *f.lean,
		"q4k":               *f.q4k,
		"quantized_at_load": *f.lean || *f.q4k,
		"workers":           model.NumWorkers(),   // global matmul worker budget (prefill and explicit paths)
		"budget":            model.WorkerBudget(), // how the worker count was resolved (FAK_WORKERS / FAK_BUDGET / -budget / default)
		"q8_decode_workers": model.Q8DecodeWorkers(),
		"q8_decode_budget":  model.Q8DecodeWorkerBudget(),
		"go_threads":        fmt.Sprintf("GOMAXPROCS=%d, matmul workers=%d, q8 decode workers=%d (FAK_WORKERS / FAK_BUDGET / -budget to pin)", runtime.GOMAXPROCS(0), model.NumWorkers(), model.Q8DecodeWorkers()),
	}
	if ggufLoadProfile != nil {
		report["load_profile"] = ggufLoadProfile
	}
	if *f.q4k {
		rep := m.ResidentReport()
		report["resident"] = rep
		report["resident_summary"] = model.FormatResidentReport(rep)
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

// modelbenchDeviceHeadroom matches serve's device-fit headroom (serveGGUFDeviceHeadroom): the
// fraction of the device budget reserved for KV/scratch/activations not in the weight estimate.
const modelbenchDeviceHeadroom = 0.15

// preflightInputFor opens the GGUF header (no tensor read) and builds the classifier input for
// the resolved backend and load regime. Returns the input ready for BuildModelPreflight; the
// OpenErr field carries any header-open failure so the classifier reports REFUSE_BAD_HEADER.
func preflightInputFor(f *benchFlags, be compute.Backend) ggufload.PreflightInput {
	ws, err := ggufload.OpenWeights(*f.gguf)
	return ggufload.PreflightInput{
		Path:     *f.gguf,
		OpenErr:  err,
		Source:   ws,
		Backend:  be,
		Headroom: modelbenchDeviceHeadroom,
		Lean:     *f.lean,
		Q4K:      *f.q4k,
	}
}

// runPreflight is the -preflight entry: resolve the backend, open only the header, classify, emit
// the JSON report + a human summary to stderr, and exit non-zero on any REFUSE_*. It never calls
// loadModel, so no tensor byte is read — the whole job finishes in seconds.
func runPreflight(f *benchFlags) {
	be, _ := resolveBackend(f)
	in := preflightInputFor(f, be)
	if in.Source != nil {
		defer in.Source.Close()
	}
	pf := ggufload.BuildModelPreflight(in)
	fmt.Fprint(os.Stderr, pf.Render())
	writeReport(*f.out, map[string]any{
		"app_version": appversion.Current(),
		"engine":      "fak modelbench preflight",
		"preflight":   pf,
	})
	if pf.Refused() {
		os.Exit(1)
	}
}

// runFitGate is the default pre-load device-fit refusal for a normal (non-preflight) run. It runs
// the same header-only classifier and exits non-zero ONLY on REFUSE_TOO_BIG / REFUSE_BAD_HEADER /
// REFUSE_BAD_ARCH — turning a would-be mid-load OOM into a typed refusal. Fail-open: with no
// capacity-reporting backend it returns READY/FIT_UNKNOWN and the load proceeds unchanged.
func runFitGate(f *benchFlags) {
	be, _ := resolveBackend(f)
	in := preflightInputFor(f, be)
	if in.Source != nil {
		defer in.Source.Close()
	}
	pf := ggufload.BuildModelPreflight(in)
	if pf.Refused() {
		fmt.Fprint(os.Stderr, pf.Render())
		fmt.Fprintln(os.Stderr, "fak: refusing the load (pass -fit-check=false to override the fit gate, or -preflight to inspect)")
		os.Exit(1)
	}
}

// Closed-vocabulary -smoke statuses.
const (
	smokeStatusLoaded        = "SMOKE_LOADED"         // load finished within the deadline
	smokeStatusTimeout       = "SMOKE_LOAD_TIMEOUT"   // load exceeded -smoke-deadline (aborted)
	smokeStatusOK            = "SMOKE_OK"             // forward ran and produced finite logits
	smokeStatusForwardFailed = "SMOKE_FORWARD_FAILED" // forward panicked or produced NaN/Inf
)

// smokeOutcome is the PURE deadline decision for the -smoke load: given whether the load finished
// and how long it took against the deadline, it returns the closed status. Factored out so the
// timeout logic is unit-testable without a real multi-minute load.
func smokeOutcome(done bool, elapsed, deadline time.Duration) string {
	if !done {
		return smokeStatusTimeout
	}
	if deadline > 0 && elapsed > deadline {
		return smokeStatusTimeout
	}
	return smokeStatusLoaded
}

// loadModelMaybeDeadline loads the model. Under -smoke it runs the load in a goroutine and races
// it against -smoke-deadline: on timeout it reports SMOKE_LOAD_TIMEOUT with the elapsed time (the
// load goroutine is abandoned; the process exits) so a load that would have run for an hour is
// bounded. Without -smoke it loads synchronously, exactly as before.
func loadModelMaybeDeadline(f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	if !*f.smoke || *f.smokeDeadline <= 0 {
		return loadModel(f, lp)
	}
	type loadRes struct {
		m    *model.Model
		name string
		err  error
	}
	ch := make(chan loadRes, 1)
	start := time.Now()
	go func() {
		m, name, err := loadModel(f, lp)
		ch <- loadRes{m, name, err}
	}()
	select {
	case r := <-ch:
		// Won the race within the deadline window.
		return r.m, r.name, r.err
	case <-time.After(*f.smokeDeadline):
		// The deadline fired first. smokeOutcome (the pure, tested classifier) names this
		// SMOKE_LOAD_TIMEOUT; report it and exit. The load goroutine is abandoned (the process
		// exits), so a load that would have run for an hour is bounded by -smoke-deadline.
		elapsed := time.Since(start)
		if smokeOutcome(false, elapsed, *f.smokeDeadline) == smokeStatusTimeout {
			reportSmokeTimeout(f, elapsed)
		}
		return nil, "", nil // unreachable: reportSmokeTimeout exits
	}
}

// reportSmokeTimeout emits the SMOKE_LOAD_TIMEOUT artifact (with the last progress visible on
// stderr from the load profiler) and exits non-zero.
func reportSmokeTimeout(f *benchFlags, elapsed time.Duration) {
	fmt.Fprintf(os.Stderr, "fak: -smoke load exceeded -smoke-deadline %s (%.0fs elapsed) — aborting\n", *f.smokeDeadline, elapsed.Seconds())
	writeReport(*f.out, map[string]any{
		"app_version":     appversion.Current(),
		"engine":          "fak modelbench smoke",
		"smoke_status":    smokeStatusTimeout,
		"source":          *f.gguf,
		"elapsed_seconds": elapsed.Seconds(),
		"deadline":        f.smokeDeadline.String(),
	})
	os.Exit(1)
}

// allFinite reports whether every logit is a finite number — the cheapest proof a forward pass
// produced real output rather than NaN/Inf (a broken kernel or a config mismatch).
func allFinite(logits []float32) bool {
	if len(logits) == 0 {
		return false
	}
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

// runSmoke is the -smoke entry after a successful (deadline-bounded) load: it decodes ONE token
// and asserts the logits are finite, emitting SMOKE_OK or SMOKE_FORWARD_FAILED and exiting. This
// proves the forward runs before committing to the full prefill/decode/workload grid. It reuses
// the recover-guarded prefill pattern so a panicking forward becomes a clean SMOKE_FORWARD_FAILED.
func runSmoke(f *benchFlags, m *model.Model, modelName string, loadMS float64, vocab int) {
	status := smokeStatusOK
	var detail string
	func() {
		defer func() {
			if r := recover(); r != nil {
				status = smokeStatusForwardFailed
				detail = fmt.Sprintf("forward panicked: %v", r)
			}
		}()
		s := m.NewSession()
		s.Quant = *f.quant
		s.Q4K = *f.q4k
		s.Metal = *f.metal
		defer s.Close()
		logits := s.Prefill(lcgIDs(*f.decodePrompt, vocab))
		if !allFinite(logits) {
			status = smokeStatusForwardFailed
			detail = "prefill produced non-finite logits (NaN/Inf)"
		}
	}()
	fmt.Fprintf(os.Stderr, "fak modelbench smoke: %s (%s, loaded in %.1fs)\n", status, modelName, loadMS/1000)
	if detail != "" {
		fmt.Fprintf(os.Stderr, "  detail: %s\n", detail)
	}
	writeReport(*f.out, map[string]any{
		"app_version":  appversion.Current(),
		"engine":       "fak modelbench smoke",
		"model":        modelName,
		"source":       loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k),
		"smoke_status": status,
		"load_ms":      loadMS,
		"smoke_detail": detail,
	})
	if status != smokeStatusOK {
		os.Exit(1)
	}
}

func loadSource(hf, gguf, dir string, lean, q4k bool) string {
	if gguf != "" {
		if q4k {
			return gguf + " (resident Q4_K)"
		}
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
