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
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchckpt"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
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

type q4kModelLoader func(string, *ggufload.LoadProfiler) (*model.Model, error)

var (
	loadResidentQ4K q4kModelLoader = func(path string, _ *ggufload.LoadProfiler) (*model.Model, error) {
		return ggufload.LoadModelQ4K(path)
	}
	loadStreamedDenseQ4K q4kModelLoader = func(path string, p *ggufload.LoadProfiler) (*model.Model, error) {
		return ggufload.LoadModelQ4KStreamedDense(path, p)
	}
)

func loadGGUFQ4K(path string, lp *ggufload.LoadProfiler, streamed bool) (*model.Model, string, error) {
	loader := loadResidentQ4K
	label := " [gguf-q4k]"
	if streamed {
		loader = loadStreamedDenseQ4K
		label = " [gguf-q4k-streamed-dense]"
	}
	m, err := loader(path, lp)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + label, nil
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
	streamQ4K             *bool
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
	checkpoint            *string
	resume                *string
	nativeProfileOut      *string
	nativeProfileReadback *string

	processExit     func(int)
	weightCloser    func() error
	weightCloseOnce sync.Once
	weightCloseErr  error
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
		streamQ4K:             flag.Bool("stream-q4k", false, "benchmark-only: with -gguf -q4k, stream eligible dense Q4_K tensors from the checkpoint instead of keeping their raw bytes resident"),
		name:                  flag.String("name", "", "model name for the report (default: derived from the source dir)"),
		out:                   flag.String("out", "", "write JSON result here (default stdout)"),
		prefillSizesCSV:       flag.String("prefill-sizes", "16,64,256", "comma-separated prompt lengths for prefill timings"),
		prefillReps:           flag.Int("prefill-reps", 5, "reps per prefill size (median)"),
		decodeReps:            flag.Int("decode-reps", 5, "reps for decode (median over per-token)"),
		decodeSteps:           flag.Int("decode-steps", 32, "tokens to decode"),
		decodePrompt:          flag.Int("decode-prompt", 16, "prompt length before decode"),
		quant:                 flag.Bool("quant", false, "use the Q8_0 quantized forward path (else f32)"),
		metal:                 flag.Bool("metal", false, "run prefill projections on the Metal GPU backend (auto-compiled on darwin/arm64+cgo, no build tag needed; implies -quant for the Q8 weight store; with -q4k, routes Q4_K tensors through MetalQ4K)"),
		verify:                flag.Bool("verify", false, "with -metal: cross-check the Metal prefill's last-token logits against the CPU Q8 path (argmax agreement + max|Δ|) and exit"),
		backendName:           flag.String("backend", "legacy", "execution backend: legacy or a compute backend name"),
		requireNonReference:   flag.Bool("require-non-reference", false, "fail unless -backend selects a non-reference compute backend"),
		workloadPath:          flag.String("workload", "", "optional recorded agent workload JSON; emits workload_prefill/workload_decode"),
		workloadPrefillCap:    flag.Int("workload-prefill-cap", 0, "cap recorded workload prompt lengths for smoke runs (0 = full recorded length)"),
		loadOnly:              flag.Bool("load-only", false, "load the model, emit load time + peak RSS JSON, and exit without running inference"),
		loadProfile:           flag.Bool("load-profile", false, "emit a GGUF load-phase profile (requires -gguf and either -lean or -stream-q4k; also enabled by -phase-profile)"),
		loadProfileTrace:      flag.Bool("load-profile-trace", false, "with profiled GGUF loading, stream per-tensor load timings to stderr while loading"),
		loadProfileTraceEvery: flag.Int("load-profile-trace-every", 25, "tensor interval for -load-profile-trace after the first tensor"),
		phaseProfile:          flag.Bool("phase-profile", false, "emit one-shot coarse Session phase profiles for prefill/decode without perturbing median timings"),
		budget:                flag.Float64("budget", 0, "fractional core budget for this run: 0.75 = use up to 75% of the machine's logical cores (portable across box sizes; 75 or 0.75 both accepted). 0 = unset. FAK_WORKERS, if set, still overrides."),
		preflight:             flag.Bool("preflight", false, "FAIL FAST: read only the GGUF header (no tensor load), report arch/est-size/device-fit/ETA, and exit in seconds. Refuses a bad-arch / too-big / bad-header model before the multi-minute load. Requires -gguf."),
		smoke:                 flag.Bool("smoke", false, "header preflight, then load (under -smoke-deadline) and decode ONE token to prove the forward runs, then exit — before the full prefill/decode/workload grid. Requires -gguf."),
		smokeDeadline:         flag.Duration("smoke-deadline", 90*time.Second, "hard wall-clock cap on the -smoke load: if the load exceeds it, abort and report SMOKE_LOAD_TIMEOUT with the last progress line instead of hanging"),
		fitCheck:              flag.Bool("fit-check", true, "before a normal load, refuse a model that a capacity-reporting -backend KNOWS won't fit (typed refusal instead of a mid-load OOM panic). Fail-open on legacy/cpu-ref. -fit-check=false for deliberate stress runs."),
		loadProgress:          flag.Bool("load-progress", true, "stream throttled load progress (percent / GB / elapsed / GB-per-s) to stderr on lean/q4k GGUF loads so a multi-minute load is not silent; -load-progress=false silences it"),
		checkpoint:            flag.String("checkpoint", "", "per-cell write-ahead checkpoint path (#2382): each grid cell (prefill size / decode / workload case) is appended as it completes, so a crash mid-grid keeps the cells already measured instead of discarding the whole sweep"),
		resume:                flag.String("resume", "", "resume from an existing checkpoint (alias for -checkpoint on an existing file): reuse the recorded cells and measure only the missing ones; refuses if the file was built for a different model/precision/grid"),
		nativeProfileOut:      flag.String("native-performance-profile", "", "write one fak-native Metal P=32/T=64 session capture in the existing native-performance v1 schema, then exit"),
		nativeProfileReadback: flag.String("native-performance-readback", "", "validate a native-performance profile and its companion raw-event receipt without loading a model"),
	}
	flag.Parse()
	*f.dir = pathutil.ExpandTilde(*f.dir)
	*f.gguf = pathutil.ExpandTilde(*f.gguf)
	*f.hf = pathutil.ExpandTilde(*f.hf)
	return f
}

// bindWeightCloser transfers a successful streamed-loader checkpoint lifetime into the
// command's terminal-status guard. Both normal returns and explicit exit paths call the same
// once-only close seam, so the checkpoint cannot leak or be closed twice.
func (f *benchFlags) bindWeightCloser(close func() error) {
	f.weightCloser = close
}

func (f *benchFlags) closeTransferredWeights() error {
	f.weightCloseOnce.Do(func() {
		if f.weightCloser != nil {
			f.weightCloseErr = f.weightCloser()
		}
	})
	return f.weightCloseErr
}

// exit closes transferred streamed weights synchronously before terminating. os.Exit skips
// defers, so every post-load terminal helper routes through this method; processExit is
// injectable solely so lifecycle tests can observe the ordering without ending the test binary.
func (f *benchFlags) exit(code int) {
	if err := f.closeTransferredWeights(); err != nil {
		fmt.Fprintln(os.Stderr, "close streamed Q4_K checkpoint:", err)
		code = 1
	}
	if f.processExit != nil {
		f.processExit(code)
		return
	}
	os.Exit(code)
}

func streamQ4KEnabled(f *benchFlags) bool {
	return f.streamQ4K != nil && *f.streamQ4K
}

func bindLoadedModelWeights(f *benchFlags, m *model.Model) bool {
	if !streamQ4KEnabled(f) {
		return false
	}
	f.bindWeightCloser(m.CloseWeights)
	return true
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
func validateFlagCombinations(f *benchFlags) error {
	if streamQ4KEnabled(f) && (*f.gguf == "" || !*f.q4k) {
		return fmt.Errorf("-stream-q4k requires exact -gguf and -q4k")
	}
	if streamQ4KEnabled(f) && *f.nativeProfileOut != "" {
		return fmt.Errorf("-stream-q4k is not yet admitted into the strict -native-performance-profile control envelope")
	}
	if *f.nativeProfileReadback != "" && *f.nativeProfileOut != "" {
		return fmt.Errorf("-native-performance-readback and -native-performance-profile are mutually exclusive")
	}
	if *f.nativeProfileOut != "" && (*f.gguf == "" || !*f.q4k || !*f.metal || *f.decodePrompt != 32 || *f.decodeSteps != 64) {
		return fmt.Errorf("-native-performance-profile requires -gguf, -q4k, -metal, -decode-prompt=32, and -decode-steps=64")
	}
	if *f.q4k {
		switch {
		case *f.gguf == "":
			return fmt.Errorf("-q4k requires -gguf")
		case *f.hf != "":
			return fmt.Errorf("-q4k cannot be combined with -hf")
		case *f.lean:
			return fmt.Errorf("-q4k is its own GGUF resident-quant load path; omit -lean")
		case *f.backendName != "legacy":
			return fmt.Errorf("-q4k currently runs through the legacy resident-Q4_K session path; omit -backend")
		case *f.verify:
			return fmt.Errorf("-q4k -verify is not wired; use go test ./internal/model -run MetalQ4K for the parity gate (darwin/arm64+cgo auto-compiles Metal, no build tag needed)")
		}
	}
	profiledGGUF := *f.gguf != "" && (*f.lean || streamQ4KEnabled(f))
	if *f.loadProfile && !profiledGGUF {
		return fmt.Errorf("-load-profile requires -gguf and either -lean or -stream-q4k")
	}
	if *f.loadProfileTrace && !profiledGGUF {
		return fmt.Errorf("-load-profile-trace requires -gguf and either -lean or -stream-q4k")
	}
	// -preflight / -smoke read the header (and -smoke loads) of a GGUF; the estimators cover
	// the f32 path too, so do NOT also require -lean (that would block a plain-GGUF preflight).
	if *f.preflight && *f.gguf == "" {
		return fmt.Errorf("-preflight requires -gguf")
	}
	if *f.smoke && *f.gguf == "" {
		return fmt.Errorf("-smoke requires -gguf")
	}
	if *f.preflight && *f.smoke {
		return fmt.Errorf("-preflight and -smoke are mutually exclusive (preflight is header-only; smoke also loads)")
	}
	return nil
}

func validateFlags(f *benchFlags) {
	if err := validateFlagCombinations(f); err != nil {
		fmt.Fprintln(os.Stderr, err)
		f.exit(2)
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

// newGGUFLoadProfiler builds the lean-Q8 or streamed-Q4_K GGUF load profiler. It is created when either a
// -load-profile* flag is set (which attaches the machine-readable load_profile to the report) OR
// -load-progress is on for one of those profiled paths (the default) — so a multi-minute load streams a
// throttled percent/GB/elapsed/GB-per-s status to stderr instead of being a silent black box.
// Returns nil when neither applies (e.g. the f32 path, which does not Tick) so the loader keeps
// its existing no-bookkeeping behavior.
func newGGUFLoadProfiler(f *benchFlags) *ggufload.LoadProfiler {
	profiledGGUF := *f.gguf != "" && (*f.lean || streamQ4KEnabled(f))
	wantLoadProfile := (*f.loadProfile || *f.loadProfileTrace || *f.phaseProfile) && profiledGGUF
	wantProgress := *f.loadProgress && profiledGGUF
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
		return loadGGUFQ4K(*f.gguf, lp, streamQ4KEnabled(f))
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

// loadWorkerControl is the exact, non-inferred readback of GGUF loader concurrency.
// Effective is present only when the public environment contract supplies a valid explicit
// override; the unexported ggufload default remains owned by that package and is not duplicated.
type loadWorkerControl struct {
	FAKGGUFLoadWorkers string `json:"fak_gguf_load_workers"`
	Source             string `json:"source"`
	GOMAXPROCS         int    `json:"gomaxprocs"`
	EffectiveCount     *int   `json:"effective_count,omitempty"`
}

func readLoadWorkerControl(lookup func(string) (string, bool), gomaxprocs int) loadWorkerControl {
	literal, explicit := lookup("FAK_GGUF_LOAD_WORKERS")
	control := loadWorkerControl{
		FAKGGUFLoadWorkers: literal,
		Source:             "unset",
		GOMAXPROCS:         gomaxprocs,
	}
	if !explicit {
		return control
	}
	control.Source = "explicit"
	if n, err := strconv.Atoi(strings.TrimSpace(literal)); err == nil && n >= 1 {
		control.EffectiveCount = &n
	}
	return control
}

func currentLoadWorkerControl() loadWorkerControl {
	return readLoadWorkerControl(os.LookupEnv, runtime.GOMAXPROCS(0))
}

func loadReportIdentity(f *benchFlags) map[string]any {
	return map[string]any{
		"source":              loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, streamQ4KEnabled(f)),
		"stream_q4k":          streamQ4KEnabled(f),
		"load_worker_control": currentLoadWorkerControl(),
	}
}

func ggufLoadProfileIdentity(f *benchFlags) (mode, source string) {
	if streamQ4KEnabled(f) {
		return "gguf-streamed-dense-q4k", loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, true)
	}
	return "gguf-lean-q8", *f.gguf
}

// runLoadOnly emits the load-time + peak-RSS report and is the whole job for -load-only.
func runLoadOnly(f *benchFlags, modelName string, loadMS float64, ggufLoadProfile *ggufload.LoadProfile) {
	peakRSS, rssErr := peakRSSBytes()
	report := map[string]any{
		"app_version":          appversion.Current(),
		"engine":               "fak model load",
		"model":                modelName,
		"load_ms":              loadMS,
		"lean":                 *f.lean,
		"q4k":                  *f.q4k,
		"quantized_at_load":    *f.lean || *f.q4k,
		"peak_rss_bytes":       peakRSS,
		"peak_rss_unavailable": rssErr != nil,
	}
	for key, value := range loadReportIdentity(f) {
		report[key] = value
	}
	if rssErr != nil {
		report["peak_rss_error"] = rssErr.Error()
	}
	if ggufLoadProfile != nil {
		report["load_profile"] = ggufLoadProfile
	}
	writeReport(f, report)
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
			f.exit(2)
		}
		// Q8 on a compute backend needs the device to accept quantized weight uploads
		// (the wired Q8 HAL path keys off Caps().UploadDtype). A backend that can't —
		// e.g. cpu-ref or an f32-only device — still refuses -quant rather than silently
		// running the f32 path under a Q8 flag.
		if q8UploadUnsupported(*f.quant, be.Caps()) {
			fmt.Fprintf(os.Stderr, "backend: %q is f32-only (no Q8 upload support); omit -quant\n", be.Name())
			f.exit(2)
		}
		if *f.requireNonReference && be.Class() == compute.Reference {
			fmt.Fprintf(os.Stderr, "backend: %q is %s; production Phase-1 gate requires a non-reference backend\n", be.Name(), be.Class())
			f.exit(2)
		}
	} else if *f.requireNonReference {
		fmt.Fprintln(os.Stderr, "backend: -require-non-reference needs -backend to name a compute backend")
		f.exit(2)
	}
	return be, registeredBackends
}

// resolveMetal validates the requested Metal path and falls back to CPU when the backend is
// unavailable. The Q8/f16 lane needs Quantize(); the resident Q4_K lane already carries its
// mixed q4_k/Q8 store from LoadModelQ4K, so -q4k -metal must not force a second Q8_0 copy.
func resolveMetal(f *benchFlags) {
	if !*f.metal {
		return
	}
	if *f.q4k {
		if !metalgemm.Available() {
			if metalgemm.Compiled() {
				fmt.Fprintln(os.Stderr, "metal: no usable Metal device; falling back to CPU resident Q4_K")
			} else {
				fmt.Fprintln(os.Stderr, "metal: backend not compiled in (requires darwin/arm64 with cgo — auto-compiled, no build tag needed); falling back to CPU resident Q4_K")
			}
			*f.metal = false
		}
		return
	}
	*f.quant = true
	if !metalgemm.Available() {
		if metalgemm.Compiled() {
			fmt.Fprintln(os.Stderr, "metal: no usable Metal device; falling back to CPU Q8 prefill")
		} else {
			fmt.Fprintln(os.Stderr, "metal: backend not compiled in (requires darwin/arm64 with cgo — auto-compiled, no build tag needed); falling back to CPU Q8 prefill")
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
		f.exit(2)
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
		ac, ag := mathx.ArgmaxF32(lc), mathx.ArgmaxF32(lg)
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
		f.exit(1)
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
		if *f.metal {
			engine = "fak-in-kernel Metal Q4_K/Q8 hybrid (raw GGUF Q4_K majority through MetalQ4K; Q8 minority on CPU)"
			precision = "Q4_K/Q8 resident hybrid + MetalQ4K"
		}
	} else if *f.metal {
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

func applyLegacySessionFlags(s *model.Session, f *benchFlags) {
	s.Quant = *f.quant
	s.Q4K = *f.q4k
	if *f.q4k {
		s.MetalQ4K = *f.metal
		return
	}
	s.Metal = *f.metal
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

func nativeModelConfigIdentity(cfg model.Config) (map[string]any, error) {
	raw, err := json.Marshal(cfg)
	if err != nil {
		return nil, err
	}
	var identity map[string]any
	if err := json.Unmarshal(raw, &identity); err != nil {
		return nil, err
	}
	// Config deliberately excludes loader-only identity fields from its ordinary JSON shape.
	// The capture receipt needs the exact loaded identity, so bind those fields alongside every
	// exported config axis rather than relying on modelConfigReport's display subset.
	identity["name"] = cfg.Name
	identity["eos_token_id"] = cfg.EOSTokenID
	identity["eos_token_ids"] = cfg.EOSTokenIDs
	return identity, nil
}

const (
	nativeProfileReceiptSchema = "fak-native-metal-profile-receipt/v1"
	nativeProfileArtifactBytes = int64(17106775008)
	nativeProfileUnset         = "<unset>"
)

var nativeProfileRequiredEnvironment = map[string]string{
	"FAK_METAL_STREAM_Q4K": "1",
	"FAK_Q4K":              "1",
}

// nativeProfileDeniedEnvironment is the complete set of process controls read by the Qwen3.8
// resident-Q4_K forward, its attention/KV helpers, and its worker scheduler. The profile contract
// pins their default behavior by requiring them to be absent. FAK_Q4K_FREE_CPU and
// FAK_METAL_Q8_UPLOAD are deliberately denied: either one changes residency and must never
// masquerade as the documented all-Metal, no-override control run.
var nativeProfileDeniedEnvironment = []string{
	"FAK_ARM_TILE",
	"FAK_BUDGET",
	"FAK_FDOT3_SIMD",
	"FAK_FDOT3_SIMD_MINB",
	"FAK_GDN_BATCHED",
	"FAK_HIDDEN_STEER",
	"FAK_HIDDEN_STEER_ALPHA",
	"FAK_HIDDEN_STEER_LAYER",
	"FAK_HIDDEN_STEER_POS",
	"FAK_HIDDEN_TAP",
	"FAK_HIDDEN_TAP_OPS",
	"FAK_HIDDEN_TAP_POS",
	"FAK_HYBRID_KV",
	"FAK_KQ_INT8",
	"FAK_METAL_DECODE",
	"FAK_METAL_Q8_UPLOAD",
	"FAK_METAL_RESIDENT",
	"FAK_PAGED_KV",
	"FAK_PAGED_KV_BLOCK_TOKENS",
	"FAK_PAR_SHARDS",
	"FAK_PAR_SPIN",
	"FAK_PREFILL_NO_ATTN",
	"FAK_PREFILL_NO_GDN",
	"FAK_Q4K_FREE_CPU",
	"FAK_Q4K_MM",
	"FAK_Q8_GEMM_GROUP",
	"FAK_QATTN_GQA",
	"FAK_QGEMM",
	"FAK_QGEMM_GROUP",
	"FAK_QGEMM_GROUP_MAXP",
	"FAK_QKERNEL",
	"FAK_QPROFILE",
	"FAK_QWEN35_PREFILL_TOKEN_LOOP",
	"FAK_SAXPY3_SIMD_MINB",
	"FAK_SAXPY3_SIMD_MINPOS",
	"FAK_WORKERS",
	"GOMAXPROCS",
}

const (
	nativeControlFlagBudget   = "flag:-budget"
	nativeControlLogicalCPUs  = "runtime:logical_cpus"
	nativeControlGOMAXPROCS   = "runtime:gomaxprocs"
	nativeControlWorkers      = "runtime:matmul_workers"
	nativeControlQ8Workers    = "runtime:q8_decode_workers"
	nativeControlWorkerBudget = "runtime:worker_budget"
)

func nativeProfileControlEnvironment(lookup func(string) (string, bool), environ []string, budget float64) (map[string]string, error) {
	if budget != 0 {
		return nil, fmt.Errorf("native performance profile unavailable: -budget must be 0, got %g", budget)
	}
	controls := make(map[string]string, len(nativeProfileRequiredEnvironment)+len(nativeProfileDeniedEnvironment)+6)
	for key, want := range nativeProfileRequiredEnvironment {
		got, ok := lookup(key)
		if !ok || got != want {
			return nil, fmt.Errorf("native performance profile unavailable: %s must equal %s", key, want)
		}
		controls[key] = got
	}
	for _, key := range nativeProfileDeniedEnvironment {
		if got, ok := lookup(key); ok {
			return nil, fmt.Errorf("native performance profile unavailable: %s override is not allowed (got %q)", key, got)
		}
		controls[key] = nativeProfileUnset
	}
	// Refuse unknown FAK_* declarations too: otherwise a newly added process knob could alter a
	// run before the receipt schema learns to bind it. This is intentionally fail-closed.
	for _, declaration := range environ {
		key, _, _ := strings.Cut(declaration, "=")
		if !strings.HasPrefix(key, "FAK_") {
			continue
		}
		if _, ok := nativeProfileRequiredEnvironment[key]; ok {
			continue
		}
		return nil, fmt.Errorf("native performance profile unavailable: unrecognized %s override is not allowed", key)
	}
	controls[nativeControlFlagBudget] = "0"
	controls[nativeControlLogicalCPUs] = strconv.Itoa(runtime.NumCPU())
	controls[nativeControlGOMAXPROCS] = strconv.Itoa(runtime.GOMAXPROCS(0))
	controls[nativeControlWorkers] = strconv.Itoa(model.NumWorkers())
	controls[nativeControlQ8Workers] = strconv.Itoa(model.Q8DecodeWorkers())
	controls[nativeControlWorkerBudget] = model.WorkerBudget()
	if err := validateNativeProfileControls(controls); err != nil {
		return nil, err
	}
	return controls, nil
}

func validateNativeProfileControls(controls map[string]string) error {
	wantLen := len(nativeProfileRequiredEnvironment) + len(nativeProfileDeniedEnvironment) + 6
	if len(controls) != wantLen {
		return fmt.Errorf("native performance control receipt has %d fields, want %d", len(controls), wantLen)
	}
	for key, want := range nativeProfileRequiredEnvironment {
		if controls[key] != want {
			return fmt.Errorf("required control %s was not captured as %s", key, want)
		}
	}
	for _, key := range nativeProfileDeniedEnvironment {
		if controls[key] != nativeProfileUnset {
			return fmt.Errorf("denied control %s was not captured as unset", key)
		}
	}
	if controls[nativeControlFlagBudget] != "0" || controls[nativeControlWorkerBudget] != "default(GOMAXPROCS)" {
		return fmt.Errorf("worker budget controls are not the default envelope")
	}
	for _, key := range []string{nativeControlLogicalCPUs, nativeControlGOMAXPROCS, nativeControlWorkers, nativeControlQ8Workers} {
		value, err := strconv.Atoi(controls[key])
		if err != nil || value <= 0 {
			return fmt.Errorf("control %s is not a positive integer", key)
		}
	}
	return nil
}

type nativeFileIdentity struct {
	Bytes  int64  `json:"bytes"`
	SHA256 string `json:"sha256"`
}

type nativeArtifactIdentity struct {
	nativeFileIdentity
	Model         string `json:"model"`
	ModelRevision string `json:"model_revision"`
}

type nativeHostIdentity struct {
	GOOS                 string `json:"goos"`
	GOARCH               string `json:"goarch"`
	CPU                  string `json:"cpu"`
	MetalDevice          string `json:"metal_device"`
	GPUCores             int    `json:"gpu_cores"`
	MemoryBytes          uint64 `json:"memory_bytes"`
	MetalWorkingSetBytes uint64 `json:"metal_working_set_bytes"`
}

type nativeSourceIdentity struct {
	Revision   string `json:"revision"`
	Modified   bool   `json:"modified"`
	DiffBytes  int    `json:"diff_bytes,omitempty"`
	DiffSHA256 string `json:"diff_sha256,omitempty"`
}

type nativeProfileReceipt struct {
	Schema            string                     `json:"schema"`
	BindingSHA256     string                     `json:"binding_sha256"`
	ProfileSHA256     string                     `json:"profile_sha256"`
	EnvelopeID        string                     `json:"envelope_id"`
	Artifact          nativeArtifactIdentity     `json:"artifact"`
	ModelConfig       map[string]any             `json:"model_config"`
	ModelConfigSHA256 string                     `json:"model_config_sha256"`
	Host              nativeHostIdentity         `json:"host"`
	Source            nativeSourceIdentity       `json:"source"`
	Binary            nativeFileIdentity         `json:"binary"`
	Controls          map[string]string          `json:"controls"`
	Execution         metalgemm.ExecutionReceipt `json:"execution"`
	Fallbacks         model.MetalFallbackReceipt `json:"fallbacks"`
}

func sha256JSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(b)
	return fmt.Sprintf("%x", sum), nil
}

func fileIdentity(path string) (nativeFileIdentity, error) {
	f, err := os.Open(path)
	if err != nil {
		return nativeFileIdentity{}, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nativeFileIdentity{}, err
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nativeFileIdentity{}, err
	}
	return nativeFileIdentity{Bytes: info.Size(), SHA256: fmt.Sprintf("%x", h.Sum(nil))}, nil
}

func profileEnvelopeByID(graph nativeperf.Graph, id string) (nativeperf.Envelope, error) {
	for _, envelope := range graph.Envelopes {
		if envelope.ID == id {
			return envelope, nil
		}
	}
	return nativeperf.Envelope{}, fmt.Errorf("unknown native-performance envelope %q", id)
}

func exactMetalProfileEnvelope(graph nativeperf.Graph, artifact nativeFileIdentity) (nativeperf.Envelope, error) {
	for _, envelope := range graph.Envelopes {
		if envelope.Backend == "metal" && envelope.ArtifactSHA256 == artifact.SHA256 {
			if artifact.Bytes != nativeProfileArtifactBytes {
				return nativeperf.Envelope{}, fmt.Errorf("artifact bytes = %d, want %d", artifact.Bytes, nativeProfileArtifactBytes)
			}
			return envelope, nil
		}
	}
	return nativeperf.Envelope{}, fmt.Errorf("artifact SHA-256 %q is not a pinned Metal envelope", artifact.SHA256)
}

func commandOutput(name string, args ...string) (string, error) {
	b, err := exec.Command(name, args...).Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(b)), nil
}

func captureNativeHost() (nativeHostIdentity, error) {
	host := nativeHostIdentity{GOOS: runtime.GOOS, GOARCH: runtime.GOARCH, MetalDevice: metalgemm.DeviceName()}
	if host.GOOS != "darwin" || host.GOARCH != "arm64" || host.MetalDevice == "" {
		return nativeHostIdentity{}, fmt.Errorf("native Metal host unavailable: %s/%s device=%q", host.GOOS, host.GOARCH, host.MetalDevice)
	}
	var err error
	if host.CPU, err = commandOutput("sysctl", "-n", "machdep.cpu.brand_string"); err != nil || host.CPU == "" {
		return nativeHostIdentity{}, fmt.Errorf("CPU identity: %w", err)
	}
	memory, err := commandOutput("sysctl", "-n", "hw.memsize")
	if err != nil {
		return nativeHostIdentity{}, fmt.Errorf("host memory: %w", err)
	}
	if host.MemoryBytes, err = strconv.ParseUint(memory, 10, 64); err != nil || host.MemoryBytes == 0 {
		return nativeHostIdentity{}, fmt.Errorf("host memory value %q: %w", memory, err)
	}
	var displays struct {
		GPUs []struct {
			Name  string `json:"_name"`
			Model string `json:"sppci_model"`
			Cores string `json:"sppci_cores"`
		} `json:"SPDisplaysDataType"`
	}
	displayJSON, err := exec.Command("system_profiler", "SPDisplaysDataType", "-json").Output()
	if err != nil {
		return nativeHostIdentity{}, fmt.Errorf("GPU identity unavailable: %w", err)
	}
	if err := json.Unmarshal(displayJSON, &displays); err != nil {
		return nativeHostIdentity{}, fmt.Errorf("GPU identity decode: %w", err)
	}
	for _, gpu := range displays.GPUs {
		if gpu.Name == host.MetalDevice || gpu.Model == host.MetalDevice {
			host.GPUCores, err = strconv.Atoi(gpu.Cores)
			break
		}
	}
	if err != nil || host.GPUCores <= 0 {
		return nativeHostIdentity{}, fmt.Errorf("GPU core identity unavailable for %q", host.MetalDevice)
	}
	if host.MetalWorkingSetBytes, _ = metalgemm.DeviceMemoryTotal(); host.MetalWorkingSetBytes == 0 {
		return nativeHostIdentity{}, fmt.Errorf("Metal working-set capacity unavailable")
	}
	return host, nil
}

func captureNativeBuild() (nativeSourceIdentity, nativeFileIdentity, error) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return nativeSourceIdentity{}, nativeFileIdentity{}, fmt.Errorf("Go build identity unavailable")
	}
	var source nativeSourceIdentity
	revisionFound, modifiedFound := false, false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			source.Revision, revisionFound = setting.Value, setting.Value != ""
		case "vcs.modified":
			source.Modified, modifiedFound = setting.Value == "true", true
		}
	}
	if !revisionFound || !modifiedFound {
		return nativeSourceIdentity{}, nativeFileIdentity{}, fmt.Errorf("binary lacks vcs.revision or vcs.modified build settings")
	}
	if source.Modified {
		diff, err := exec.Command("git", "diff", "--no-ext-diff", "--binary", source.Revision, "--").Output()
		if err != nil || len(diff) == 0 {
			return nativeSourceIdentity{}, nativeFileIdentity{}, fmt.Errorf("modified source diff identity unavailable: %w", err)
		}
		diffSum := sha256.Sum256(diff)
		source.DiffBytes = len(diff)
		source.DiffSHA256 = fmt.Sprintf("%x", diffSum)
	}
	executable, err := os.Executable()
	if err != nil {
		return nativeSourceIdentity{}, nativeFileIdentity{}, err
	}
	binary, err := fileIdentity(executable)
	return source, binary, err
}

func validateNativeHost(envelope nativeperf.Envelope, host nativeHostIdentity) error {
	wantHardware := fmt.Sprintf("%s, %d-GPU-core", host.CPU, host.GPUCores)
	if host.GOOS != "darwin" || host.GOARCH != "arm64" || host.CPU == "" || host.MetalDevice != host.CPU || host.GPUCores <= 0 || wantHardware != envelope.Hardware {
		return fmt.Errorf("host identity %+v does not match envelope hardware %q", host, envelope.Hardware)
	}
	if host.MemoryBytes != uint64(envelope.MemoryGiB)*(1<<30) || host.MetalWorkingSetBytes == 0 {
		return fmt.Errorf("host memory identity physical=%d Metal=%d does not match %d GiB envelope", host.MemoryBytes, host.MetalWorkingSetBytes, envelope.MemoryGiB)
	}
	return nil
}

func nativeReceiptBinding(receipt nativeProfileReceipt) (string, error) {
	receipt.BindingSHA256 = ""
	return sha256JSON(receipt)
}

func validateNativeProfileReceipt(profileBytes []byte, profile nativeperf.ProfileBundle, receipt nativeProfileReceipt) error {
	if receipt.Schema != nativeProfileReceiptSchema {
		return fmt.Errorf("unexpected native profile receipt schema %q", receipt.Schema)
	}
	if err := nativeperf.ValidateProfile(nativeperf.ActiveGraph(), profile); err != nil {
		return err
	}
	envelope, err := profileEnvelopeByID(nativeperf.ActiveGraph(), profile.EnvelopeID)
	if err != nil {
		return err
	}
	profileSum := sha256.Sum256(profileBytes)
	if receipt.ProfileSHA256 != fmt.Sprintf("%x", profileSum) || receipt.EnvelopeID != profile.EnvelopeID {
		return fmt.Errorf("profile digest or envelope binding mismatch")
	}
	if receipt.Artifact.SHA256 != envelope.ArtifactSHA256 || receipt.Artifact.Bytes != nativeProfileArtifactBytes || receipt.Artifact.Model != envelope.Model || receipt.Artifact.ModelRevision != envelope.ModelRevision {
		return fmt.Errorf("artifact identity does not match envelope")
	}
	if len(receipt.ModelConfig) == 0 {
		return fmt.Errorf("model config is empty")
	}
	configSHA, err := sha256JSON(receipt.ModelConfig)
	if err != nil || configSHA != receipt.ModelConfigSHA256 {
		return fmt.Errorf("model config digest mismatch")
	}
	if err := validateNativeHost(envelope, receipt.Host); err != nil {
		return err
	}
	if receipt.Source.Revision == "" || receipt.Binary.Bytes <= 0 || len(receipt.Binary.SHA256) != 64 {
		return fmt.Errorf("source revision or binary identity incomplete")
	}
	if receipt.Source.Modified && (receipt.Source.DiffBytes <= 0 || len(receipt.Source.DiffSHA256) != 64) {
		return fmt.Errorf("modified source lacks an exact diff identity")
	}
	if !receipt.Source.Modified && (receipt.Source.DiffBytes != 0 || receipt.Source.DiffSHA256 != "") {
		return fmt.Errorf("clean source carries a contradictory diff identity")
	}
	if err := validateNativeProfileControls(receipt.Controls); err != nil {
		return err
	}
	if err := metalgemm.ValidateExecutionReceipt(receipt.Execution); err != nil {
		return err
	}
	if err := model.ValidateMetalFallbackReceipt(receipt.Fallbacks); err != nil {
		return err
	}
	if receipt.Fallbacks.PromisedCPUFallbacks != profile.Execution.FallbackCount {
		return fmt.Errorf("raw fallback total does not match v1 execution identity")
	}
	if profile.Metal == nil || receipt.Execution.Counters.CommandBuffers != profile.Metal.CommandBuffers || receipt.Execution.Counters.Encoders != profile.Metal.Encoders || receipt.Execution.Counters.DispatchMilliseconds != profile.Metal.DispatchMilliseconds || receipt.Execution.Counters.WaitMilliseconds != profile.Metal.WaitMilliseconds {
		return fmt.Errorf("raw execution totals do not match v1 Metal counter block")
	}
	binding, err := nativeReceiptBinding(receipt)
	if err != nil || binding != receipt.BindingSHA256 {
		return fmt.Errorf("source/binary/artifact/event binding digest mismatch")
	}
	return nil
}

func nativeReceiptPath(profilePath string) string {
	ext := filepath.Ext(profilePath)
	return strings.TrimSuffix(profilePath, ext) + ".receipt.json"
}

func runNativeProfileReadback(profilePath string) error {
	profileBytes, err := os.ReadFile(profilePath)
	if err != nil {
		return err
	}
	profile, err := nativeperf.DecodeProfile(profileBytes)
	if err != nil {
		return err
	}
	receiptBytes, err := os.ReadFile(nativeReceiptPath(profilePath))
	if err != nil {
		return err
	}
	var receipt nativeProfileReceipt
	if err := json.Unmarshal(receiptBytes, &receipt); err != nil {
		return err
	}
	return validateNativeProfileReceipt(profileBytes, profile, receipt)
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

func requireNoMetalFallbacks(count int) error {
	if count != 0 {
		return fmt.Errorf("native performance profile unavailable: session observed %d promised Metal dispatch fallbacks", count)
	}
	return nil
}

func appendNativeProfilePhase(phases []nativeperf.ProfilePhase, name string, duration time.Duration) []nativeperf.ProfilePhase {
	startMilliseconds := 0.0
	if len(phases) > 0 {
		previous := phases[len(phases)-1]
		startMilliseconds = previous.StartMilliseconds + previous.DurationMilliseconds
	}
	return append(phases, nativeperf.ProfilePhase{
		Name:                 name,
		StartMilliseconds:    startMilliseconds,
		DurationMilliseconds: float64(duration.Nanoseconds()) / 1e6,
	})
}

func runNativePerformanceProfile(f *benchFlags, m *model.Model, loadNanos int64, vocab int, controls map[string]string, newSession func() *model.Session) error {
	loadDuration := time.Duration(loadNanos)
	phases := appendNativeProfilePhase(nil, "load-setup", loadDuration)

	s := newSession()
	profiler := model.NewPhaseProfiler()
	s.PhaseProfiler = profiler
	prompt := lcgIDs(32, vocab)
	t := time.Now()
	logits := s.Prefill(prompt)
	d := time.Since(t)
	phases = appendNativeProfilePhase(phases, "prefill", d)

	id := 7 % vocab
	t = time.Now()
	logits = s.Step(id)
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "first-token", d)

	t = time.Now()
	for i := 1; i < 64; i++ {
		id = (id*48271 + 1) % vocab
		logits = s.Step(id)
	}
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "steady-decode", d)

	t = time.Now()
	for _, value := range logits {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("native performance verification: non-finite logits")
		}
	}
	executionReceipt, err := profiler.MetalExecutionReceipt()
	if err != nil {
		return fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if err := metalgemm.ValidateExecutionReceipt(executionReceipt); err != nil {
		return fmt.Errorf("native performance profile unavailable: %w", err)
	}
	counters := executionReceipt.Counters
	fallbackReceipt, err := profiler.MetalFallbackReceipt()
	if err != nil {
		return fmt.Errorf("native performance fallback receipt unavailable: %w", err)
	}
	fallbackCount := fallbackReceipt.PromisedCPUFallbacks
	if err := requireNoMetalFallbacks(fallbackCount); err != nil {
		return err
	}
	resident := m.ResidentReport().TotalResidentBytes
	workingSet, err := peakRSSBytes()
	if err != nil || resident <= 0 || workingSet == 0 {
		return fmt.Errorf("native performance profile unavailable: memory capture resident=%d working_set=%d: %w", resident, workingSet, err)
	}
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "verification", d)

	t = time.Now()
	s.Close()
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "teardown", d)

	artifactFile, err := fileIdentity(*f.gguf)
	if err != nil {
		return fmt.Errorf("native performance artifact identity: %w", err)
	}
	envelope, err := exactMetalProfileEnvelope(nativeperf.ActiveGraph(), artifactFile)
	if err != nil {
		return fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if envelope.PromptTokens != 32 || envelope.DecodeTokens != 64 || *f.decodePrompt != envelope.PromptTokens || *f.decodeSteps != envelope.DecodeTokens {
		return fmt.Errorf("native performance P/T controls do not match envelope: got P=%d T=%d, want P=%d T=%d", *f.decodePrompt, *f.decodeSteps, envelope.PromptTokens, envelope.DecodeTokens)
	}
	if *f.name != "qwen38:27b" {
		return fmt.Errorf("native performance model name %q is not the pinned qwen38:27b identity", *f.name)
	}
	host, err := captureNativeHost()
	if err != nil {
		return fmt.Errorf("native performance host identity: %w", err)
	}
	if err := validateNativeHost(envelope, host); err != nil {
		return fmt.Errorf("native performance profile unavailable: %w", err)
	}
	source, binary, err := captureNativeBuild()
	if err != nil {
		return fmt.Errorf("native performance build identity: %w", err)
	}
	weights, err := ggufload.OpenWeights(*f.gguf)
	if err != nil {
		return fmt.Errorf("native performance artifact config: %w", err)
	}
	headerConfig, configErr := weights.File.Config()
	weights.Close()
	if configErr != nil {
		return fmt.Errorf("native performance artifact config: %w", configErr)
	}
	loadedConfig, err := nativeModelConfigIdentity(m.Cfg)
	if err != nil {
		return err
	}
	headerConfigReport, err := nativeModelConfigIdentity(headerConfig)
	if err != nil {
		return err
	}
	loadedConfigSHA, err := sha256JSON(loadedConfig)
	if err != nil {
		return err
	}
	headerConfigSHA, err := sha256JSON(headerConfigReport)
	if err != nil || headerConfigSHA != loadedConfigSHA {
		return fmt.Errorf("native performance loaded model config does not match exact artifact header")
	}

	profile := nativeperf.ProfileBundle{
		Schema:                 nativeperf.ProfileSchema,
		EnvelopeID:             envelope.ID,
		Execution:              nativeperf.ExecutionIdentity{Engine: envelope.Engine, ForwardPath: envelope.ForwardPath, FallbackCount: fallbackCount},
		Phases:                 phases,
		Metal:                  &nativeperf.MetalCounters{CommandBuffers: counters.CommandBuffers, Encoders: counters.Encoders, DispatchMilliseconds: counters.DispatchMilliseconds, WaitMilliseconds: counters.WaitMilliseconds, ResidentBytes: uint64(resident), WorkingSetBytes: workingSet},
		AttributionUnavailable: &nativeperf.AttributionUnavailable{Reason: nativeperf.AttributionUnavailableCapture, Detail: "fak-native session lifecycle capture does not export per-lever dispatch attribution"},
	}
	if err := nativeperf.ValidateProfile(nativeperf.ActiveGraph(), profile); err != nil {
		return err
	}
	b, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	profileSum := sha256.Sum256(b)
	receipt := nativeProfileReceipt{
		Schema:            nativeProfileReceiptSchema,
		ProfileSHA256:     fmt.Sprintf("%x", profileSum),
		EnvelopeID:        envelope.ID,
		Artifact:          nativeArtifactIdentity{nativeFileIdentity: artifactFile, Model: envelope.Model, ModelRevision: envelope.ModelRevision},
		ModelConfig:       loadedConfig,
		ModelConfigSHA256: loadedConfigSHA,
		Host:              host,
		Source:            source,
		Binary:            binary,
		Controls:          controls,
		Execution:         executionReceipt,
		Fallbacks:         fallbackReceipt,
	}
	receipt.BindingSHA256, err = nativeReceiptBinding(receipt)
	if err != nil {
		return err
	}
	if err := validateNativeProfileReceipt(b, profile, receipt); err != nil {
		return fmt.Errorf("native performance receipt self-check: %w", err)
	}
	receiptBytes, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	receiptBytes = append(receiptBytes, '\n')
	if err := os.WriteFile(*f.nativeProfileOut, b, 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(nativeReceiptPath(*f.nativeProfileOut), receiptBytes, 0o644); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "wrote", *f.nativeProfileOut)
	fmt.Fprintln(os.Stderr, "wrote", nativeReceiptPath(*f.nativeProfileOut))
	return nil
}

// runPrefill times Session.Prefill over each P in prefillSizes (builds KV cache, last
// logits) and records the median timings and any phase profiles into the report maps. With a
// checkpoint (ck != nil), each per-size cell is write-ahead persisted as it completes and reused
// verbatim on resume, so a crash mid-grid keeps the sizes already measured (#2382).
func runPrefill(f *benchFlags, ck *benchckpt.Ledger, newSession func() *model.Session, vocab int, prefillSizes []int, report, phaseReport map[string]any) {
	var prefills []prefillResult
	var prefillPhases []*model.PhaseProfile
	for _, p := range prefillSizes {
		ids := lcgIDs(p, vocab)
		res, reused := checkpointCell(ck, fmt.Sprintf("prefill:P=%d", p), func() prefillResult {
			med := timePrefillReps(newSession, ids, *f.prefillReps)
			return prefillResult{
				Tokens: p, Reps: *f.prefillReps, MedianMS: med,
				TokPerSec: float64(p) / (med / 1e3),
			}
		}, f)
		prefills = append(prefills, res)
		fmt.Fprintf(os.Stderr, "[fak] prefill P=%d: %.1f ms (%.1f tok/s)%s\n", p, res.MedianMS, res.TokPerSec, resumedTag(reused))
		// Phase profiling reruns the forward with a profiler attached; a cell reused from the
		// checkpoint has no live measurement to profile, so skip it (the timing grid resumes;
		// the -phase-profile deep-dive covers only the freshly-measured sizes).
		if *f.phaseProfile && !reused {
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
// per-token median and any phase profile into the report maps. The single decode cell is
// checkpointed under the "decode" key, so a resume reuses it rather than re-measuring (#2382).
func runDecode(f *benchFlags, ck *benchckpt.Ledger, newSession func() *model.Session, vocab int, report, phaseReport map[string]any) {
	prompt := lcgIDs(*f.decodePrompt, vocab)
	res, reused := checkpointCell(ck, "decode", func() decodeResult {
		med := medDecodeReps(newSession, prompt, *f.decodeReps, *f.decodeSteps, vocab, func(r int) int {
			return int(uint64(r*131+7) % uint64(vocab))
		})
		return decodeResult{
			PromptTokens: *f.decodePrompt, DecodeSteps: *f.decodeSteps, Reps: *f.decodeReps,
			PerTokenMedMS: med, TokPerSec: 1.0 / (med / 1e3),
		}
	}, f)
	report["decode"] = res
	fmt.Fprintf(os.Stderr, "[fak] decode: %.1f ms/tok (%.1f tok/s)%s\n", res.PerTokenMedMS, res.TokPerSec, resumedTag(reused))
	if *f.phaseProfile && !reused {
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
// timing per case, at the recorded (capped) prompt/decode lengths, into the report. Each case's
// prefill and decode cell is checkpointed under a per-case key, so a crash partway through the
// workload keeps the cases already measured and a resume runs only the missing ones (#2382).
func runWorkload(f *benchFlags, ck *benchckpt.Ledger, newSession func() *model.Session, vocab int, workload *model.BenchWorkload, report map[string]any) {
	var wp []prefillResult
	for i, c := range workload.Cases {
		n := capPositive(c.PromptTokens, *f.workloadPrefillCap)
		res, reused := checkpointCell(ck, fmt.Sprintf("wprefill:%d", i), func() prefillResult {
			ids := lcgIDsSeed(n, vocab, 0xC0FFEE+uint64(i)*977)
			med := timePrefillReps(newSession, ids, *f.prefillReps)
			return prefillResult{
				Name: c.Name, Source: c.Source, Tokens: n, RecordedTokens: c.PromptTokens,
				Reps: *f.prefillReps, MedianMS: med, TokPerSec: float64(n) / (med / 1e3),
			}
		}, f)
		wp = append(wp, res)
		fmt.Fprintf(os.Stderr, "[fak workload] prefill %s P=%d recorded=%d: %.1f ms%s\n", c.Name, n, c.PromptTokens, res.MedianMS, resumedTag(reused))
	}
	report["workload_prefill"] = wp

	var wd []workloadDecodeResult
	for i, c := range workload.Cases {
		promptN := capPositive(c.PromptTokens, *f.workloadPrefillCap)
		steps := capPositive(c.CompletionTokens, *f.decodeSteps)
		res, reused := checkpointCell(ck, fmt.Sprintf("wdecode:%d", i), func() workloadDecodeResult {
			prompt := lcgIDsSeed(promptN, vocab, 0xA11CE+uint64(i)*131)
			med := medDecodeReps(newSession, prompt, *f.decodeReps, steps, vocab, func(r int) int {
				return int((uint64(r+1)*2654435761 + uint64(i)) % uint64(vocab))
			})
			return workloadDecodeResult{
				Name: c.Name, Source: c.Source,
				PromptTokens: promptN, RecordedPromptTokens: c.PromptTokens,
				DecodeSteps: steps, RecordedDecodeTokens: c.CompletionTokens,
				Reps: *f.decodeReps, PerTokenMedMS: med, TokPerSec: 1.0 / (med / 1e3),
			}
		}, f)
		wd = append(wd, res)
		fmt.Fprintf(os.Stderr, "[fak workload] decode %s prompt=%d recorded=%d steps=%d/%d: %.1f ms/tok%s\n",
			c.Name, promptN, c.PromptTokens, steps, c.CompletionTokens, res.PerTokenMedMS, resumedTag(reused))
	}
	report["workload_decode"] = wd
}

// resumedTag annotates a stderr progress line when a cell was reused from a checkpoint rather
// than freshly measured, so a resume's log makes plain which cells were re-run.
func resumedTag(reused bool) string {
	if reused {
		return " [resumed]"
	}
	return ""
}

// checkpointCell returns the cell recorded under key in ck (reused verbatim, reused=true) or
// measures a fresh one via measure, write-ahead appends it, and returns it (reused=false). A nil
// ledger always measures — the historical no-checkpoint path, byte-for-byte unchanged. This is
// the per-cell seam issue #2382 asks for: every completed grid cell is persisted before the next
// begins, so a crash keeps cells 1..N-1 and a -resume reuses them instead of re-measuring.
func checkpointCell[T any](ck *benchckpt.Ledger, key string, measure func() T, terminal ...*benchFlags) (cell T, reused bool) {
	if ck != nil {
		var cached T
		if ok, err := ck.Cell(key, &cached); err == nil && ok {
			return cached, true
		}
	}
	cell = measure()
	if ck != nil {
		if err := ck.Append(key, cell); err != nil {
			fmt.Fprintf(os.Stderr, "modelbench: checkpoint append %q: %v\n", key, err)
			if len(terminal) > 0 && terminal[0] != nil {
				terminal[0].exit(1)
				return cell, false
			}
			os.Exit(1)
		}
	}
	return cell, false
}

// modelbenchFingerprint is the grid identity a checkpoint is bound to (#2382). It captures
// everything that makes a measured cell VALID to reuse — the model source/name, the precision
// lane, the backend, the resolved matmul worker regime, and the per-cell rep/step counts a median
// is taken over — but deliberately NOT the prefill-size LIST: a cell for P=16 means the same thing
// regardless of which sibling sizes are swept, which is exactly what lets a crash-then-resume over
// the same (or a superset) grid reuse recorded cells. A resume whose model/precision/counts differ
// refuses with benchckpt.ErrFingerprintMismatch rather than blending incompatible cells.
func modelbenchFingerprint(f *benchFlags, modelName string) benchckpt.Fingerprint {
	return benchckpt.Fingerprint{
		"schema":              "modelbench-grid/1",
		"source":              loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, streamQ4KEnabled(f)),
		"name":                modelName,
		"quant":               *f.quant,
		"metal":               *f.metal,
		"q4k":                 *f.q4k,
		"stream_q4k":          streamQ4KEnabled(f),
		"lean":                *f.lean,
		"backend":             *f.backendName,
		"prefill_reps":        *f.prefillReps,
		"decode_reps":         *f.decodeReps,
		"decode_steps":        *f.decodeSteps,
		"decode_prompt":       *f.decodePrompt,
		"workload":            *f.workloadPath,
		"workload_cap":        *f.workloadPrefillCap,
		"workers":             model.NumWorkers(),
		"load_worker_control": currentLoadWorkerControl(),
	}
}

// openCheckpoint opens the write-ahead checkpoint for a full grid run when -checkpoint or
// -resume names a path. -resume is an alias for -checkpoint on an existing file: both feed the
// one path that both writes ahead and, if it already matches this grid's fingerprint, resumes
// from it. A resume whose fingerprint differs refuses here (exit 2) rather than silently mixing
// incompatible cells. Returns a nil ledger (and nil closer) when no checkpoint was requested.
func openCheckpoint(f *benchFlags, modelName string) *benchckpt.Ledger {
	ckPath := *f.checkpoint
	if strings.TrimSpace(*f.resume) != "" {
		ckPath = *f.resume
	}
	if ckPath == "" {
		return nil
	}
	l, err := benchckpt.Open(ckPath, modelbenchFingerprint(f, modelName))
	if err != nil {
		fmt.Fprintf(os.Stderr, "modelbench: refusing to resume %s: %v\n", ckPath, err)
		f.exit(2)
	}
	return l
}

func main() {
	f := parseFlags()
	validateFlags(f)
	var nativeControls map[string]string
	if *f.nativeProfileOut != "" {
		var err error
		nativeControls, err = nativeProfileControlEnvironment(os.LookupEnv, os.Environ(), *f.budget)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	applyBudget(*f.budget)
	if *f.nativeProfileReadback != "" {
		if err := runNativeProfileReadback(*f.nativeProfileReadback); err != nil {
			fmt.Fprintln(os.Stderr, "native performance readback:", err)
			os.Exit(1)
		}
		fmt.Fprintln(os.Stderr, "native performance readback: PASS", *f.nativeProfileReadback)
		return
	}

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
	if bindLoadedModelWeights(f, m) {
		defer f.closeTransferredWeights()
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
		mode, source := ggufLoadProfileIdentity(f)
		ggufLoadProfile = ggufLoadProfiler.Snapshot(mode, source, loadNanos)
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
		applyLegacySessionFlags(s, f)
		return s
	}
	if *f.nativeProfileOut != "" {
		if err := runNativePerformanceProfile(f, m, loadNanos, vocab, nativeControls, newSession); err != nil {
			fmt.Fprintln(os.Stderr, err)
			f.exit(1)
		}
		return
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
		"app_version":         appversion.Current(),
		"engine":              engine,
		"model":               modelName,
		"model_config":        modelConfigReport(m.Cfg),
		"precision":           precision,
		"backend":             backendReport,
		"load_ms":             loadMS,
		"quant_ms":            quantMS,
		"lean":                *f.lean,
		"q4k":                 *f.q4k,
		"stream_q4k":          streamQ4KEnabled(f),
		"metal":               *f.metal,
		"metal_q4k":           *f.q4k && *f.metal,
		"quantized_at_load":   *f.lean || *f.q4k,
		"workers":             model.NumWorkers(),   // global matmul worker budget (prefill and explicit paths)
		"budget":              model.WorkerBudget(), // how the worker count was resolved (FAK_WORKERS / FAK_BUDGET / -budget / default)
		"q8_decode_workers":   model.Q8DecodeWorkers(),
		"q8_decode_budget":    model.Q8DecodeWorkerBudget(),
		"go_threads":          fmt.Sprintf("GOMAXPROCS=%d, matmul workers=%d, q8 decode workers=%d (FAK_WORKERS / FAK_BUDGET / -budget to pin)", runtime.GOMAXPROCS(0), model.NumWorkers(), model.Q8DecodeWorkers()),
		"load_worker_control": currentLoadWorkerControl(),
	}
	report["source"] = loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, streamQ4KEnabled(f))
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

	// Open the per-cell write-ahead checkpoint (if -checkpoint/-resume was passed) now that the
	// model/precision regime is finalized — its fingerprint binds the recorded cells to this
	// grid. Only the grid AFTER load is protected; the -preflight/-smoke/-load-only/-verify
	// early exits above never reach here, which is exactly the gap #2382 names.
	ck := openCheckpoint(f, modelName)
	if ck != nil {
		defer ck.Close()
	}

	runPrefill(f, ck, newSession, vocab, prefillSizes, report, phaseReport)
	runDecode(f, ck, newSession, vocab, report, phaseReport)
	if workload != nil {
		runWorkload(f, ck, newSession, vocab, workload, report)
	}
	if *f.phaseProfile {
		report["phase_profile"] = phaseReport
	}

	writeReport(f, report)
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
	writeReport(f, map[string]any{
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
	report := map[string]any{
		"app_version":     appversion.Current(),
		"engine":          "fak modelbench smoke",
		"smoke_status":    smokeStatusTimeout,
		"elapsed_seconds": elapsed.Seconds(),
		"deadline":        f.smokeDeadline.String(),
	}
	for key, value := range loadReportIdentity(f) {
		report[key] = value
	}
	writeReport(f, report)
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
		applyLegacySessionFlags(s, f)
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
	writeReport(f, map[string]any{
		"app_version":         appversion.Current(),
		"engine":              "fak modelbench smoke",
		"model":               modelName,
		"source":              loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, streamQ4KEnabled(f)),
		"stream_q4k":          streamQ4KEnabled(f),
		"load_worker_control": currentLoadWorkerControl(),
		"smoke_status":        status,
		"load_ms":             loadMS,
		"smoke_detail":        detail,
	})
	if status != smokeStatusOK {
		f.exit(1)
	}
}

func loadSource(hf, gguf, dir string, lean, q4k, streamQ4K bool) string {
	if gguf != "" {
		if q4k {
			if streamQ4K {
				return gguf + " (streamed dense Q4_K)"
			}
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

func writeReport(f *benchFlags, report map[string]any) {
	b, _ := benchcli.MarshalReport(report)
	if *f.out != "" {
		if err := os.WriteFile(*f.out, b, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write:", err)
			f.exit(1)
		}
		fmt.Fprintln(os.Stderr, "wrote", *f.out)
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
