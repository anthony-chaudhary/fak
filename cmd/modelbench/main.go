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
	"context"
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/benchckpt"
	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/macbench"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

// loadModel selects the load path from the flags (lean GGUF/HF, plain GGUF/HF, or fak
// dir format) and returns the model plus its report label. May set *f.quant for -lean.
func loadModel(f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	return loadModelContext(context.Background(), f, lp)
}

func loadModelContext(ctx context.Context, f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	if *f.q4k {
		if *f.backendName != "legacy" {
			// Dense formats without a HAL path must remain reachable through Q8.
			// Vulkan also retains eligible Q2_K payloads for its native kernel.
			opts := []ggufload.Q4KLoadOption{ggufload.WithDenseKQuantResident(false)}
			if *f.backendName == "vulkan" {
				opts = append(opts, ggufload.WithDenseQ2KResident(true))
				if q2kEmbeddingEligible(*f.gguf) {
					opts = append(opts, ggufload.WithQ2KEmbeddingResident(true))
				}
			}
			loader := ggufload.LoadModelQ4KProfileOptionsContext
			label := " [gguf-q4k]"
			if streamQ4KEnabled(f) {
				loader = ggufload.LoadModelQ4KStreamedDenseContext
				label = " [gguf-q4k-streamed-dense]"
			}
			m, err := loader(ctx, *f.gguf, lp, opts...)
			return m, filepath.Base(*f.gguf) + label, err
		}
		return loadGGUFQ4KContext(ctx, *f.gguf, lp, streamQ4KEnabled(f))
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

func q2kEmbeddingEligible(path string) bool {
	if path == "" {
		return false
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		return false
	}
	defer ws.Close()
	return q2kEmbeddingSourceEligible(ws)
}

func q2kEmbeddingSourceEligible(ws *ggufload.WeightSource) bool {
	if ws == nil {
		return false
	}
	cfg, err := ws.File.Config()
	if err != nil || !cfg.IsQwen35Hybrid() || cfg.IsMoE() || cfg.TieWordEmbeddings {
		return false
	}
	var hasQ2KEmbd, hasOutput bool
	for _, t := range ws.File.Tensors {
		if t.Name == "token_embd.weight" && t.Type == ggufload.TensorQ2_K {
			hasQ2KEmbd = true
		}
		if t.Name == "output.weight" {
			hasOutput = true
		}
	}
	return hasQ2KEmbd && hasOutput
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
		if q8UploadUnsupported(*f.quant || *f.q4k, be.Caps()) {
			fmt.Fprintf(os.Stderr, "backend: %q is f32-only (no quantized upload support); omit -quant/-q4k\n", be.Name())
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
	if (*f.vulkanQ4KProfile || *f.vulkanStageQ4K) && !compute.ConfigureVulkanQ4K(be, *f.vulkanQ4KProfile, *f.vulkanStageQ4K) {
		fmt.Fprintln(os.Stderr, "backend: -vulkan-q4k-profile/-vulkan-stage-q4k require -backend vulkan")
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
		sc.Q4KGateUpOutputSlab = *f.q4kGateUpSlab
		lc := sc.Prefill(ids)
		sg := m.NewSession()
		sg.Metal = true
		sg.Q4KGateUpOutputSlab = *f.q4kGateUpSlab
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
		engine = "fak-in-kernel GGUF resident mixed quantization"
		precision = "GGUF resident mixed quantization"
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
			"native_host_stages":  "permitted",
			"device_coverage":     "unmeasured",
			"tier":                be.Tier(),
			"class":               be.Class().String(),
			"caps":                be.Caps(),
			"registered_backends": registeredBackends,
		}
		if *f.q4k {
			if *f.backendName == "vulkan" {
				precision = "resident Q4_K/Q2_K + unsupported dense formats converted to Q8"
				backendReport["dense_non_q4k_load"] = "retain eligible Q2_K; dequantize other dense formats then quantize to Q8"
			} else {
				precision = "resident Q4_K + dense non-Q4_K converted to Q8"
				backendReport["dense_non_q4k_load"] = "dequantize then quantize to Q8; source encoding not retained"
			}
			backendReport["fit_estimate"] = "conservative full-F32 upper bound; not actual resident bytes"
		}
	}
	return engine, precision, backendReport
}

func applyLegacySessionFlags(s *model.Session, f *benchFlags) {
	if f == nil {
		return
	}
	if f.quant != nil {
		s.Quant = *f.quant
	}
	if f.q4k != nil {
		s.Q4K = *f.q4k
	}
	if f.q4kGateUpSlab != nil {
		s.Q4KGateUpOutputSlab = *f.q4kGateUpSlab
	}
	if f.q4k != nil && *f.q4k {
		if f.metal != nil {
			s.MetalQ4K = *f.metal
		}
		return
	}
	if f.metal != nil {
		s.Metal = *f.metal
	}
}

// Smoke and timed execution must use the same backend and resident-weight flags.
// A HAL session may intentionally execute unsupported quant formats on native CPU.
func newBenchSession(m *model.Model, f *benchFlags, be compute.Backend) *model.Session {
	if be == nil {
		s := m.NewSession()
		applyLegacySessionFlags(s, f)
		return s
	}
	s := m.NewBackendSession(be)
	if f != nil {
		quant := f.quant != nil && *f.quant
		q4k := f.q4k != nil && *f.q4k
		s.Quant = quant || q4k
		if f.q4k != nil {
			s.Q4K = *f.q4k
		}
		if f.q4kGateUpSlab != nil {
			s.Q4KGateUpOutputSlab = *f.q4kGateUpSlab
		}
	}
	return s
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
	var cachePhaseLatency modelperfobs.CachePhaseLatencyRecorder

	s := newSession()
	finishProfile := onceFinishNativeProfile(s.Close)
	defer finishProfile()
	sequenceSelector := controls[nativeProfileSequenceSelector]
	profiler, err := configureNativeProfileSession(s, controls)
	if err != nil {
		return err
	}
	logits, sequenceExecuted, phases, err := runNativeProfileForward(s, vocab, sequenceSelector, phases)
	if err != nil {
		return err
	}

	t := time.Now()
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
	d := time.Since(t)
	phases = appendNativeProfilePhase(phases, "verification", d)
	handoffReceipt := s.Qwen35DecodeHandoffReceipt()
	if err := model.ValidateQwen35DecodeHandoffReceipt(handoffReceipt); err != nil {
		return fmt.Errorf("native performance decode handoff receipt: %w", err)
	}

	t = time.Now()
	finishProfile()
	d = time.Since(t)
	phases = appendNativeProfilePhase(phases, "teardown", d)

	identity, err := captureNativeProfileIdentity(f, m, sequenceSelector, sequenceExecuted)
	if err != nil {
		return err
	}

	profile := nativeperf.ProfileBundle{
		Schema:                 nativeperf.ProfileSchema,
		EnvelopeID:             identity.envelope.ID,
		Execution:              nativeperf.ExecutionIdentity{Engine: identity.envelope.Engine, ForwardPath: identity.forwardPath, FallbackCount: fallbackCount},
		Phases:                 phases,
		Metal:                  &nativeperf.MetalCounters{CommandBuffers: counters.CommandBuffers, Encoders: counters.Encoders, DispatchMilliseconds: counters.DispatchMilliseconds, WaitMilliseconds: counters.WaitMilliseconds, ResidentBytes: uint64(resident), WorkingSetBytes: workingSet},
		AttributionUnavailable: &nativeperf.AttributionUnavailable{Reason: nativeperf.AttributionUnavailableCapture, Detail: "fak-native session lifecycle capture does not export per-lever dispatch attribution"},
	}
	if err := validateNativeProfileForControls(profile, controls); err != nil {
		return err
	}
	b, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	profileSum := sha256.Sum256(b)
	q4kResidency := m.Q4KResidencyReceipt()
	cacheLatencyReceipt := cachePhaseLatency.Receipt()
	receipt := nativeProfileReceipt{
		Schema:              nativeProfileReceiptSchema,
		ProfileSHA256:       fmt.Sprintf("%x", profileSum),
		EnvelopeID:          identity.envelope.ID,
		Artifact:            nativeArtifactIdentity{nativeFileIdentity: identity.artifactFile, Model: identity.envelope.Model, ModelRevision: identity.envelope.ModelRevision},
		ModelConfig:         identity.loadedConfig,
		ModelConfigSHA256:   identity.loadedConfigSHA,
		Host:                identity.host,
		Source:              identity.source,
		Binary:              identity.binary,
		Controls:            controls,
		Execution:           executionReceipt,
		Fallbacks:           fallbackReceipt,
		Q4KResidency:        &q4kResidency,
		Qwen35DecodeHandoff: &handoffReceipt,
		CachePhaseLatency:   &cacheLatencyReceipt,
	}
	receipt.BindingSHA256, err = nativeReceiptBinding(receipt)
	if err != nil {
		return err
	}
	if err := validateNativeProfileReceipt(b, profile, receipt); err != nil {
		return fmt.Errorf("native performance receipt self-check: %w", err)
	}
	receiptBytes, err := benchcli.MarshalReport(receipt)
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

// configureNativeProfileSession applies the profile's session controls (candidate
// sequence route, decode handoff) and attaches the phase profiler.
func configureNativeProfileSession(s *model.Session, controls map[string]string) (*model.PhaseProfiler, error) {
	if controls[nativeProfileSequenceSelector] == nativeProfileSelectorOn {
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			return nil, fmt.Errorf("native performance candidate route unavailable: %w", err)
		}
	}
	var handoffMode model.Qwen35DecodeHandoffMode
	if err := handoffMode.Set(controls[nativeProfileDecodeHandoffControl]); err != nil {
		return nil, fmt.Errorf("native performance decode handoff: %w", err)
	}
	if err := s.SetQwen35DecodeHandoffMode(handoffMode); err != nil {
		return nil, fmt.Errorf("native performance decode handoff unavailable: %w", err)
	}
	profiler := model.NewPhaseProfiler()
	s.PhaseProfiler = profiler
	return profiler, nil
}

// runNativeProfileForward runs the measured forward pass (prefill with the candidate
// route finalized inside the timed window, first-token, 63 steady steps), appending
// the "prefill"/"first-token"/"steady-decode" phases; returns final logits + executed.
func runNativeProfileForward(s *model.Session, vocab int, sequenceSelector string, phases []nativeperf.ProfilePhase) ([]float32, bool, []nativeperf.ProfilePhase, error) {
	prompt := lcgIDs(32, vocab)
	t := time.Now()
	logits := s.Prefill(prompt)
	sequenceExecuted := false
	if sequenceSelector == nativeProfileSelectorOn {
		var err error
		sequenceExecuted, err = s.FinalizeQwen35MetalGDNPreprojectedSequence()
		if err != nil {
			return nil, false, phases, fmt.Errorf("native performance candidate route failed: %w", err)
		}
	} else {
		_, sequenceExecuted = s.Qwen35GDNDecodePath()
	}
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
	return logits, sequenceExecuted, phases, nil
}

// nativeProfileIdentity is every identity witness the native profile + receipt bind to.
type nativeProfileIdentity struct {
	artifactFile    nativeFileIdentity
	envelope        nativeperf.Envelope
	forwardPath     string
	host            nativeHostIdentity
	source          nativeSourceIdentity
	binary          nativeFileIdentity
	loadedConfig    map[string]any
	loadedConfigSHA string
}

// captureNativeProfileIdentity collects and cross-checks every identity the native
// profile attests: artifact bytes, pinned envelope + P/T controls, executed forward
// path, pinned model name, host + build identities, and loaded-vs-header model config.
func captureNativeProfileIdentity(f *benchFlags, m *model.Model, sequenceSelector string, sequenceExecuted bool) (nativeProfileIdentity, error) {
	artifactFile, err := fileIdentity(*f.gguf)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance artifact identity: %w", err)
	}
	envelope, err := exactMetalProfileEnvelope(nativeperf.ActiveGraph(), artifactFile)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if envelope.PromptTokens != 32 || envelope.DecodeTokens != 64 || *f.decodePrompt != envelope.PromptTokens || *f.decodeSteps != envelope.DecodeTokens {
		return nativeProfileIdentity{}, fmt.Errorf("native performance P/T controls do not match envelope: got P=%d T=%d, want P=%d T=%d", *f.decodePrompt, *f.decodeSteps, envelope.PromptTokens, envelope.DecodeTokens)
	}
	forwardPath, err := nativeProfileExecutedForwardPath(envelope.ForwardPath, sequenceSelector, sequenceExecuted)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance profile unavailable: %w", err)
	}
	if *f.name != "qwen38:27b" {
		return nativeProfileIdentity{}, fmt.Errorf("native performance model name %q is not the pinned qwen38:27b identity", *f.name)
	}
	host, err := captureNativeHost()
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance host identity: %w", err)
	}
	if err := validateNativeHost(envelope, host); err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance profile unavailable: %w", err)
	}
	source, binary, err := captureNativeBuild()
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance build identity: %w", err)
	}
	weights, err := ggufload.OpenWeights(*f.gguf)
	if err != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance artifact config: %w", err)
	}
	headerConfig, configErr := weights.File.Config()
	weights.Close()
	if configErr != nil {
		return nativeProfileIdentity{}, fmt.Errorf("native performance artifact config: %w", configErr)
	}
	loadedConfig, err := nativeModelConfigIdentity(m.Cfg)
	if err != nil {
		return nativeProfileIdentity{}, err
	}
	headerConfigReport, err := nativeModelConfigIdentity(headerConfig)
	if err != nil {
		return nativeProfileIdentity{}, err
	}
	loadedConfigSHA, err := sha256JSON(loadedConfig)
	if err != nil {
		return nativeProfileIdentity{}, err
	}
	headerConfigSHA, err := sha256JSON(headerConfigReport)
	if err != nil || headerConfigSHA != loadedConfigSHA {
		return nativeProfileIdentity{}, fmt.Errorf("native performance loaded model config does not match exact artifact header")
	}
	return nativeProfileIdentity{
		artifactFile:    artifactFile,
		envelope:        envelope,
		forwardPath:     forwardPath,
		host:            host,
		source:          source,
		binary:          binary,
		loadedConfig:    loadedConfig,
		loadedConfigSHA: loadedConfigSHA,
	}, nil
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

// maybeCompareNativeProfiles handles -native-profile-compare: true = process consumed.
func maybeCompareNativeProfiles(f *benchFlags) bool {
	if *f.nativeProfileCompare == "" {
		return false
	}
	comparison := compareNativeProfileCampaign(*f.nativeProfileCompare)
	if err := writeProfileComparison(f, comparison); err != nil {
		fmt.Fprintln(os.Stderr, "native performance comparison:", err)
		f.exit(1)
	}
	return true
}

// maybeRunQwenSwapReadback handles -qwen-swap-readback: true = process consumed.
func maybeRunQwenSwapReadback(f *benchFlags) bool {
	if *f.qwenSwapReadback == "" {
		return false
	}
	if err := runQwen38PagedSwapReadback(*f.qwenSwapReadback); err != nil {
		fmt.Fprintln(os.Stderr, "qwen38 paged-swap readback:", err)
		f.exit(1)
	}
	fmt.Fprintln(os.Stderr, "qwen38 paged-swap readback: PASS", *f.qwenSwapReadback)
	return true
}

// resolveBenchControls resolves the controlled environments for the native
// performance profile and the qwen-swap probe; a control error exits 2.
func resolveBenchControls(f *benchFlags) (nativeControls map[string]string, qwenSwapControls map[string]string, qwenSwapMaxBlocks int) {
	if *f.nativeProfileOut != "" {
		var err error
		nativeControls, err = nativeProfileControlEnvironment(os.LookupEnv, os.Environ(), *f.budget, *f.nativeDecodeHandoff)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	if *f.qwenSwapOut != "" {
		var err error
		qwenSwapControls, qwenSwapMaxBlocks, err = qwen38PagedSwapControls(os.LookupEnv, os.Environ(), *f.budget)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	return nativeControls, qwenSwapControls, qwenSwapMaxBlocks
}

// maybeRunNativeProfileReadback handles -native-profile-readback: true = consumed.
func maybeRunNativeProfileReadback(f *benchFlags) bool {
	if *f.nativeProfileReadback == "" {
		return false
	}
	if err := runNativeProfileReadback(*f.nativeProfileReadback); err != nil {
		fmt.Fprintln(os.Stderr, "native performance readback:", err)
		os.Exit(1)
	}
	fmt.Fprintln(os.Stderr, "native performance readback: PASS", *f.nativeProfileReadback)
	return true
}

// loadWorkloadIfAny loads the -workload file, or nil; a bad file exits 1.
func loadWorkloadIfAny(f *benchFlags) *model.BenchWorkload {
	if *f.workloadPath == "" {
		return nil
	}
	workload, err := model.LoadBenchWorkload(*f.workloadPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "workload:", err)
		os.Exit(1)
	}
	return workload
}

// loadBenchModel loads the model (deadline-bounded under -smoke). The caller measures
// loadNanos only AFTER binding transferred weights, so that cost stays in load_ms.
func loadBenchModel(f *benchFlags, lp *ggufload.LoadProfiler) (*model.Model, string, time.Time) {
	t0 := time.Now()
	m, modelName, err := loadModelMaybeDeadline(f, lp)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	return m, modelName, t0
}

// snapshotGGUFLoadProfile attaches load_profile to the report ONLY when a -load-profile*
// flag asked for it — a profiler created for default-on -load-progress streams to stderr
// but must not bloat every report's JSON.
func snapshotGGUFLoadProfile(f *benchFlags, ggufLoadProfiler *ggufload.LoadProfiler, loadNanos int64) *ggufload.LoadProfile {
	// Resident Q4_K progress does not provide a detailed conversion-phase profile.
	profiledGGUF := *f.gguf != "" && (*f.lean || streamQ4KEnabled(f))
	if profiledGGUF && ggufLoadProfiler != nil && (*f.loadProfile || *f.loadProfileTrace || *f.phaseProfile) {
		mode, source := ggufLoadProfileIdentity(f)
		return ggufLoadProfiler.Snapshot(mode, source, loadNanos)
	}
	return nil
}

// quantizeIfNeeded quantizes off the timed path when in Q8 mode; returns ms.
func quantizeIfNeeded(f *benchFlags, m *model.Model) float64 {
	if !*f.quant {
		return 0
	}
	tq := time.Now()
	m.Quantize()
	return float64(time.Since(tq).Nanoseconds()) / 1e6
}

// warmupBenchSession pages in weights + JITs allocation paths so only steady-state is timed.
func warmupBenchSession(newSession func() *model.Session, vocab int) {
	s := newSession()
	s.Prefill(lcgIDs(8, vocab))
	s.Step(s.Cache.Len() % vocab)
	s.Close()
}

// assembleBenchReport builds the report's identity half (engine, precision, workers,
// load/quant timings, workload metadata); the run* helpers append the measured cells.
func assembleBenchReport(f *benchFlags, be compute.Backend, registeredBackends []string, m *model.Model, modelName string, loadMS, quantMS float64, workload *model.BenchWorkload, ggufLoadProfile *ggufload.LoadProfile) map[string]any {
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
	if m != nil && m.NUMAReplicasLabel() != "" && m.NUMAReplicasLabel() != "numa_replicas=unrun" {
		report["numa_replicas"] = m.NUMAReplicasLabel()
	}
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
	return report
}

var (
	macbenchMTP            = flag.Bool("macbench-mtp", false, "run 4-way Apple Silicon MTP comparative benchmark (fak-native vs AX Engine vs MTPLX vs llama.cpp)")
	macbenchMTPAlt         = flag.Bool("mtp-comparison", false, "alias for -macbench-mtp")
	macbenchMTPOut         = flag.String("macbench-mtp-out", "", "write 4-way MTP comparative benchmark packet to this JSON path")
	macbenchMTPOutAlt      = flag.String("mtp-comparison-out", "", "alias for -macbench-mtp-out")
	macbenchMTPReadback    = flag.String("macbench-mtp-readback", "", "validate an MTP comparative benchmark packet without loading a model")
	macbenchMTPReadbackAlt = flag.String("mtp-comparison-readback", "", "alias for -macbench-mtp-readback")
	macbenchMTPDryRun      = flag.Bool("macbench-mtp-dry-run", false, "validate 4-way Apple Silicon MTP runner configuration without executing adapters")
	macbenchMTPDryRunAlt   = flag.Bool("mtp-comparison-dry-run", false, "alias for -macbench-mtp-dry-run")
)

func maybeRunMTPComparison(f *benchFlags) bool {
	readbackPath := *macbenchMTPReadback
	if readbackPath == "" {
		readbackPath = *macbenchMTPReadbackAlt
	}
	if readbackPath != "" {
		data, err := os.ReadFile(readbackPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp readback: read file: %v\n", err)
			f.exit(1)
		}
		var packet macbench.MTPComparisonPacket
		if err := json.Unmarshal(data, &packet); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp readback: decode packet: %v\n", err)
			f.exit(1)
		}
		if err := macbench.ValidateMTPComparisonEvidence(packet, readbackPath); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp readback: invalid packet: %v\n", err)
			f.exit(1)
		}
		fmt.Printf("VALID mtp_comparison_packet schema=%s campaign=%s host=%s fak_native_decode=%.2f tok/s\n",
			packet.Schema, packet.CampaignID, packet.HostID, packet.Summary.FakNativeDecodeTokS)
		return true
	}

	dryRun := *macbenchMTPDryRun || *macbenchMTPDryRunAlt
	runMTP := *macbenchMTP || *macbenchMTPAlt || dryRun
	if !runMTP {
		return false
	}

	opts := macbench.DefaultMTPRunnerOptions()
	opts.Adapters = macbench.DefaultMTPAdapters()

	if dryRun {
		if err := macbench.ValidateMTPRunnerEnvelope(opts); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp dry-run invalid: %v\n", err)
			f.exit(1)
		}
		fmt.Printf("DRY_RUN_PLAN_VALID campaign=%s host=%s model=%s draft_depth=%d\n",
			opts.CampaignID, opts.HostID, opts.Model.ID, opts.SpeculativeConfig.DraftDepth)
		return true
	}

	runner := macbench.NewMTPRunner(opts)
	packet, err := runner.Run(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "macbench mtp run: %v\n", err)
		f.exit(1)
	}

	outPath := *macbenchMTPOut
	if outPath == "" {
		outPath = *macbenchMTPOutAlt
	}
	if outPath == "" && f.out != nil && *f.out != "" {
		outPath = *f.out
	}

	b, err := json.MarshalIndent(packet, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "macbench mtp marshal: %v\n", err)
		f.exit(1)
	}

	if outPath != "" {
		if err := os.WriteFile(outPath, b, 0644); err != nil {
			fmt.Fprintf(os.Stderr, "macbench mtp write %s: %v\n", outPath, err)
			f.exit(1)
		}
		fmt.Printf("WROTE %s (%.2f tok/s sustained fak-native decode)\n", outPath, packet.Summary.FakNativeDecodeTokS)
	} else {
		fmt.Println(string(b))
	}
	return true
}

func main() {
	f := parseFlags()
	validateFlags(f)
	if maybeRunMTPComparison(f) {
		return
	}
	if rawDecodeEnabled() {
		if err := validateRawDecodeFlags(f); err != nil {
			fmt.Fprintln(os.Stderr, "flags:", err)
			f.exit(2)
		}
	}
	if maybeCompareNativeProfiles(f) {
		return
	}
	if maybeRunQwenSwapReadback(f) {
		return
	}
	nativeControls, qwenSwapControls, qwenSwapMaxBlocks := resolveBenchControls(f)
	applyBudget(*f.budget)
	if maybeRunNativeProfileReadback(f) {
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

	workload := loadWorkloadIfAny(f)

	defer acquireMetalLease(*f.metal)()

	// FAIL FAST before the load: when a capacity-reporting -backend is named and the model is
	// known too big, refuse with a typed sizing message instead of OOM-panicking mid-load.
	// Header-only and fail-open (legacy/cpu-ref never refused). -fit-check=false to override.
	if *f.fitCheck && *f.gguf != "" {
		runFitGate(f)
	}

	ggufLoadProfiler := newGGUFLoadProfiler(f)
	if rawDecodeEnabled() {
		resolveMetal(f)
		if err := runRawDecodeArtifact(f, ggufLoadProfiler); err != nil {
			fmt.Fprintln(os.Stderr, "raw decode:", err)
			f.exit(1)
		}
		return
	}
	m, modelName, loadStart := loadBenchModel(f, ggufLoadProfiler)
	if bindLoadedModelWeights(f, m) {
		defer f.closeTransferredWeights()
	}
	numaReq := ""
	if f.numaReplicas != nil && *f.numaReplicas != "" {
		numaReq = *f.numaReplicas
	}
	if numaReq != "" || os.Getenv("FAK_NUMA_REPLICAS") != "" {
		lbl := m.ApplyNUMAWeightReplicas(numaReq)
		fmt.Fprintf(os.Stderr, "numa_replicas: %s\n", lbl)
	}
	if *f.name != "" {
		modelName = *f.name
	}
	loadNanos := time.Since(loadStart).Nanoseconds()
	loadMS := float64(loadNanos) / 1e6
	ggufLoadProfile := snapshotGGUFLoadProfile(f, ggufLoadProfiler, loadNanos)
	if *f.loadOnly {
		runLoadOnly(f, modelName, loadMS, ggufLoadProfile)
		return
	}
	vocab := m.Cfg.VocabSize
	be, registeredBackends := resolveBackend(f)
	resolveMetal(f)
	if *f.qwenSwapOut != "" {
		if err := runWithTransferredWeightLifetime(f, func() error {
			return runQwen38PagedSwap(f, m, qwenSwapControls, qwenSwapMaxBlocks)
		}); err != nil {
			fmt.Fprintln(os.Stderr, "qwen38 paged-swap:", err)
			f.exit(1)
		}
		return
	}

	quantMS := quantizeIfNeeded(f, m)

	// -smoke: prove the forward actually runs (load + one decode) and exit BEFORE the full
	// prefill/decode/workload grid, so a broken forward is caught in one token, not after the
	// whole grid is set up. The load already happened under -smoke-deadline above.
	if *f.smoke {
		runSmoke(f, m, modelName, loadMS, vocab)
		return
	}
	newSession := func() *model.Session {
		return newBenchSession(m, f, be)
	}
	if *f.nativeProfileOut != "" {
		if err := runWithTransferredWeightLifetime(f, func() error {
			return runNativePerformanceProfile(f, m, loadNanos, vocab, nativeControls, newSession)
		}); err != nil {
			fmt.Fprintln(os.Stderr, err)
			f.exit(1)
		}
		return
	}

	if *f.verify {
		runVerify(f, m, vocab)
		return
	}

	warmupBenchSession(newSession, vocab)

	report := assembleBenchReport(f, be, registeredBackends, m, modelName, loadMS, quantMS, workload, ggufLoadProfile)
	phaseReport := map[string]any{}

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
	// Backed Vulkan Q4_K uses mixed packed/Q8/F32 storage. Streaming changes the
	// eligible Q4_K host/staging lifecycle, but not its device representation.
	convertedDense := *f.q4k && be != nil
	vulkanMixed := convertedDense && *f.backendName == "vulkan"
	streamedDenseQ4K := vulkanMixed && streamQ4KEnabled(f)
	residentQ2KEmbedding := vulkanMixed && q2kEmbeddingSourceEligible(ws)
	return ggufload.PreflightInput{
		Path:                 *f.gguf,
		OpenErr:              err,
		Source:               ws,
		Backend:              be,
		Headroom:             modelbenchDeviceHeadroom,
		Lean:                 *f.lean && !convertedDense,
		Q4K:                  *f.q4k && !convertedDense,
		VulkanMixedQ4K:       vulkanMixed,
		StreamedDenseQ4K:     streamedDenseQ4K,
		ResidentQ2KEmbedding: residentQ2KEmbedding,
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

// runSmoke is the -smoke entry after a successful (deadline-bounded) load: it decodes multiple tokens
// (controlled by -smoke-decode-steps, defaulting to 16) and asserts the logits are finite, emitting SMOKE_OK
// or SMOKE_FORWARD_FAILED and exiting. This proves the forward and autoregressive decode loop run before committing
// to the full prefill/decode/workload grid. It reuses the recover-guarded pattern so a panic becomes SMOKE_FORWARD_FAILED.
func runSmoke(f *benchFlags, m *model.Model, modelName string, loadMS float64, vocab int) {
	status := smokeStatusOK
	var detail string
	be, registeredBackends := resolveBackend(f)
	engine, precision, backendReport := describeEngine(f, be, registeredBackends)
	smokeSteps := 16
	if f.smokeDecodeSteps != nil && *f.smokeDecodeSteps > 0 {
		smokeSteps = *f.smokeDecodeSteps
	}
	var decodeMS float64
	var decodeTokS float64
	func() {
		defer func() {
			if r := recover(); r != nil {
				status = smokeStatusForwardFailed
				detail = fmt.Sprintf("forward panicked: %v", r)
			}
		}()
		s := newBenchSession(m, f, be)
		defer s.Close()
		logits := s.Prefill(lcgIDs(*f.decodePrompt, vocab))
		if !allFinite(logits) {
			status = smokeStatusForwardFailed
			detail = "prefill produced non-finite logits (NaN/Inf)"
			return
		}
		tDecodeStart := time.Now()
		for step := 0; step < smokeSteps; step++ {
			logits = s.Step(mathx.ArgmaxF32(logits))
			if !allFinite(logits) {
				status = smokeStatusForwardFailed
				detail = fmt.Sprintf("decode step %d produced non-finite logits (NaN/Inf)", step+1)
				return
			}
		}
		elapsed := time.Since(tDecodeStart)
		decodeMS = float64(elapsed.Nanoseconds()) / 1e6
		if elapsed.Seconds() > 0 {
			decodeTokS = float64(smokeSteps) / elapsed.Seconds()
		}
	}()
	if decodeTokS > 0 {
		fmt.Fprintf(os.Stderr, "fak modelbench smoke: %s (%s, loaded in %.1fs, %d decode steps @ %.2f tok/s)\n",
			status, modelName, loadMS/1000, smokeSteps, decodeTokS)
	} else {
		fmt.Fprintf(os.Stderr, "fak modelbench smoke: %s (%s, loaded in %.1fs)\n", status, modelName, loadMS/1000)
	}
	if detail != "" {
		fmt.Fprintf(os.Stderr, "  detail: %s\n", detail)
	}
	writeReport(f, map[string]any{
		"app_version":              appversion.Current(),
		"engine":                   engine,
		"precision":                precision,
		"backend":                  backendReport,
		"model":                    modelName,
		"source":                   loadSource(*f.hf, *f.gguf, *f.dir, *f.lean, *f.q4k, streamQ4KEnabled(f)),
		"stream_q4k":               streamQ4KEnabled(f),
		"load_worker_control":      currentLoadWorkerControl(),
		"smoke_status":             status,
		"smoke_decode_steps":       smokeSteps,
		"smoke_decode_ms":          decodeMS,
		"smoke_decode_tok_per_sec": decodeTokS,
		"load_ms":                  loadMS,
		"smoke_detail":             detail,
	})
	if status != smokeStatusOK {
		f.exit(1)
	}
}

func parseTokenSize(s string) (int, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty token size")
	}
	lower := strings.ToLower(s)
	if strings.HasSuffix(lower, "k") {
		numStr := strings.TrimSuffix(lower, "k")
		f, err := strconv.ParseFloat(numStr, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid token size %q: %w", s, err)
		}
		if f <= 0 {
			return 0, fmt.Errorf("token size must be positive")
		}
		return int(math.Round(f * 1024)), nil
	}
	val, err := strconv.Atoi(s)
	if err != nil || val <= 0 {
		return 0, fmt.Errorf("invalid positive integer %q", s)
	}
	return val, nil
}

func parsePositiveInts(csv string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(csv, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		n, err := parseTokenSize(part)
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
