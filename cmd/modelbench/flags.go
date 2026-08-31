package main

import (
	"flag"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

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
	q4kGateUpSlab         *bool
	vulkanQ4KProfile      *bool
	vulkanStageQ4K        *bool
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
	nativeProfileCompare  *string
	nativeDecodeHandoff   *model.Qwen35DecodeHandoffMode
	qwenSwapOut           *string
	qwenSwapReadback      *string

	processExit     func(int)
	weightCloser    func() error
	weightCloseOnce sync.Once
	weightCloseErr  error
}

// parseFlags defines and parses the command-line flags, then expands a leading ~
// in the path flags (Go/PowerShell don't), so ~/... opens as intended.
func parseFlags() *benchFlags {
	nativeProfileComparisonPhaseSelection = profileComparisonPhasePrefill
	nativeProfileComparisonAxisSelection = profileComparisonAxisSequence
	nativeDecodeHandoff := model.Qwen35DecodeHandoffAuto
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
		q4kGateUpSlab:         flag.Bool("q4k-gateup-slab", false, "reuse the bounded Q4_K gate/up output slab within each benchmark session"),
		vulkanQ4KProfile:      flag.Bool("vulkan-q4k-profile", false, "enable Vulkan Q4_K timing profiles (requires -backend vulkan)"),
		vulkanStageQ4K:        flag.Bool("vulkan-stage-q4k", false, "use Vulkan host-visible Q4_K staging (requires -backend vulkan)"),
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
		nativeProfileCompare:  flag.String("native-performance-compare", "", "compare exactly six comma-separated canonical profile paths in order: 3 selector OFF controls, then 3 selector ON candidates; requires every candidate below the control median and at least 15% median improvement; companion .receipt.json paths are derived"),
		nativeDecodeHandoff:   &nativeDecodeHandoff,
		qwenSwapOut:           flag.String("qwen38-paged-swap", "", "write one exact Qwen3.8 fak-native Metal NativeScheduler OFF/ON paged-swap receipt, then exit"),
		qwenSwapReadback:      flag.String("qwen38-paged-swap-readback", "", "validate an exact Qwen3.8 NativeScheduler paged-swap receipt without loading a model"),
	}
	flag.Var(&nativeProfileComparisonPhaseSelection, "native-performance-compare-phase", "stable comparison phase: prefill (default), steady-decode, or end-to-end (full contiguous capture including setup, verification, and teardown)")
	flag.Var(&nativeProfileComparisonAxisSelection, "native-performance-compare-axis", "typed comparison axis: sequence (default) or m3-decode-handoff")
	flag.Var(f.nativeDecodeHandoff, "native-performance-qwen35-decode-handoff", "benchmark-only Qwen3.8 decode route: AUTO (compatible default), CONTROL, or MIXER; CONTROL/MIXER require the sequence selector ON")
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

// runWithTransferredWeightLifetime keeps streamed checkpoint ownership live for a terminal
// operation such as the native-performance capture, then closes it before returning. main also
// retains its process-wide deferred guard; weightCloseOnce makes both the success return and an
// ensuing f.exit error path converge on exactly one Model.CloseWeights call.
func runWithTransferredWeightLifetime(f *benchFlags, run func() error) (err error) {
	defer func() {
		if closeErr := f.closeTransferredWeights(); closeErr != nil {
			if err == nil {
				err = fmt.Errorf("close streamed Q4_K checkpoint: %w", closeErr)
			} else {
				err = fmt.Errorf("%w; close streamed Q4_K checkpoint: %v", err, closeErr)
			}
		}
	}()
	return run()
}

// onceFinishNativeProfile lets the success path time teardown explicitly while the deferred
// error-path guard uses the same operation. The execution state must be gone before the caller
// returns to runWithTransferredWeightLifetime, which owns the mapped checkpoint close.
func onceFinishNativeProfile(finish func()) func() {
	var once sync.Once
	return func() { once.Do(finish) }
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
	qwenSwapOut := f.qwenSwapOut != nil && *f.qwenSwapOut != ""
	qwenSwapReadback := f.qwenSwapReadback != nil && *f.qwenSwapReadback != ""
	nativeModes := 0
	for _, selected := range []bool{*f.nativeProfileReadback != "", *f.nativeProfileOut != "", *f.nativeProfileCompare != "", qwenSwapOut, qwenSwapReadback} {
		if selected {
			nativeModes++
		}
	}
	if nativeModes > 1 {
		return fmt.Errorf("native profile and qwen38 paged-swap terminal modes are mutually exclusive")
	}
	if qwenSwapOut && (*f.gguf == "" || !*f.q4k || !*f.metal || *f.name != "qwen38:27b" || *f.backendName != "legacy") {
		return fmt.Errorf("-qwen38-paged-swap requires -gguf, -q4k, -metal, -name=qwen38:27b, and the fak-native legacy engine path")
	}
	if *f.nativeDecodeHandoff != model.Qwen35DecodeHandoffAuto && *f.nativeProfileOut == "" {
		return fmt.Errorf("-native-performance-qwen35-decode-handoff=%s requires -native-performance-profile", *f.nativeDecodeHandoff)
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
