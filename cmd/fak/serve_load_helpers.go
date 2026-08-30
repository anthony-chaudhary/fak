package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func serveDenseKQuantOptions(backend compute.Backend) []ggufload.Q4KLoadOption {
	if backend == nil {
		return nil
	}
	return []ggufload.Q4KLoadOption{ggufload.WithDenseKQuantResident(false)}
}

func serveDeviceResidentQ4K(backend compute.Backend) bool {
	if backend == nil || !backend.Caps().UploadDtype {
		return false
	}
	// Resident Q4_K is the efficient device representation; FAK_Q4K=0 remains the
	// explicit rollback to the legacy Q8 staging path.
	return os.Getenv("FAK_Q4K") != "0"
}

// serveArtifactResidentQ4K gates the runtime load path on both device capability and
// the encodings in the artifact itself. Backend capability alone must never relabel a
// Q8_0 checkpoint as resident Q4_K.
func serveArtifactResidentQ4K(backend compute.Backend, artifact ggufload.ArtifactQuant) bool {
	return artifact.Q4KResident && serveDeviceResidentQ4K(backend)
}

func serveQuantProvenance(artifact ggufload.ArtifactQuant, residentQ4K bool) gateway.StartupMessage {
	resident, session := "Q8_0", "Q8_0"
	if residentQ4K {
		resident, session = "Q4_K", "Q4_K"
	}
	return serveStartupMessage("quant-provenance", "info", fmt.Sprintf(
		"artifact_quant=%s artifact_inventory=%s resident_quant=%s session_quant=%s",
		artifact.Name, artifact.Inventory, resident, session))
}
func serveStartupMessage(kind, level, text string) gateway.StartupMessage {
	return newServeStartupMessage("model-load", kind, level, text)
}

func withServeStartupMessages(p *gateway.ModelLoadProfile, messages ...gateway.StartupMessage) *gateway.ModelLoadProfile {
	if p != nil {
		p.Messages = append(p.Messages, messages...)
	}
	return p
}

func newServeLoadProfiler() *ggufload.LoadProfiler {
	p := ggufload.NewLoadProfiler()
	p.AlertWriter = os.Stderr
	return p
}

func loadServeInKernelModel(modelPath string, backend compute.Backend, cpuOffloadExperts bool, contextBudgetTokens int, expertShard *ggufload.ExpertShard, expertRanks int) (inKernelModel *fakmodel.Model, inKernelQ4K bool, loadProfile *gateway.ModelLoadProfile, phase gateway.StartupPhase) {
	if modelPath == "" {
		return nil, false, nil, gateway.StartupPhase{}
	}
	tLoad := time.Now()
	if info, err := os.Stat(modelPath); err == nil && info.IsDir() {
		m, err := fakmodel.LoadSafetensorsQuantConfigDir(modelPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: safetensors load:", err)
			must(err)
		}
		loadDur := time.Since(tLoad)
		profile := &gateway.ModelLoadProfile{
			Source:       modelPath,
			Mode:         "safetensors",
			TotalSeconds: loadDur.Seconds(),
			Bottleneck:   "safetensors-load",
			Phases:       []gateway.ModelLoadPhase{{Phase: "safetensors-load", Seconds: loadDur.Seconds()}},
			Messages: []gateway.StartupMessage{serveStartupMessage("load-complete", "info",
				fmt.Sprintf("model_type=%s layers=%d hidden=%d %s", m.Cfg.ModelType, m.Cfg.NumLayers, m.Cfg.HiddenSize, fakmodel.FormatResidentReport(m.ResidentReport())))},
		}
		return m, false, profile, gateway.StartupPhase{Name: "model-load", Dur: loadDur}
	}
	ggufPath := modelPath
	artifactQuant, err := ggufload.ClassifyArtifactQuant(ggufPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve: GGUF quant inventory:", err)
		must(err)
	}
	residentQ4K := serveArtifactResidentQ4K(backend, artifactQuant)
	var loadMessages []gateway.StartupMessage
	// A sharded expert-parallel rank (expertShard != nil) admits ONLY its routed-expert band into
	// the resident store — the residency that fits GLM-5.2 across the fleet (#971). It rides ONLY
	// the resident-Q4_K arms (cpu-offload, device FAK_Q4K, pure-CPU FAK_Q4K): those carry the
	// WithExpertShard seam (the raw-super-block splitter filters the batched routed experts). The
	// Q8/f32 arms have no shard seam, so a sharded serve that would land on one is REFUSED here —
	// loading a full model on a rank sized only for its band would OOM or silently defeat the shard.
	var q4kOpts []ggufload.Q4KLoadOption
	// Device Q4_K sessions currently stage Q4_K and Q8 matrices, but not dense Q5_K/Q6_K/IQ
	// tensors. Route those mixed-quant dense weights through the loader's dequant-to-Q8 arm;
	// otherwise they are stranded in the host-only k-quant store and warmup cannot resolve them.
	q4kOpts = append(q4kOpts, serveDenseKQuantOptions(backend)...)
	if expertShard != nil {
		q4kOpts = append(q4kOpts, ggufload.WithExpertShard(expertShard.Lo, expertShard.Hi))
		must(serveShardSeamRefusal(backend, cpuOffloadExperts, serveShardSeamEnvQ4K() && artifactQuant.Q4KResident))
	}
	// How many ranks this process's WEIGHTS are actually split across. A rank is sharded only when
	// it was handed a band: expertRanks alone must never select a per-rank plan, or an unsharded
	// process would under-account the full model it really loads and pass a fit it should fail.
	residentRanks := 1
	if expertShard != nil && expertRanks > 1 {
		residentRanks = expertRanks
	}
	// #1062 pre-launch load-path check: warn (don't refuse) before a large GGUF load when the
	// weights sit on a network filesystem. NFS/CIFS read at network speed — the ~50-100x
	// time-to-ready tax a CPU server hit loading GLM-5.2 off /projects (NFS, ~82 min) vs a local NVMe
	// (minutes). Probed once here so it covers every serve arm (device + CPU); fail-open, so a
	// local or unclassifiable weights path prints nothing and loads exactly as before.
	if w := compute.WarnSlowLoadPath(compute.ProbeLoadPath(ggufPath)); w != "" {
		loadMessages = append(loadMessages, serveStartupMessage("slow-load-path", "warning", w))
	}
	switch {
	case backend != nil && cpuOffloadExperts && artifactQuant.Q4KResident:
		loadMessages = append(loadMessages, serveQuantProvenance(artifactQuant, true))
		if !backend.Caps().UploadDtype {
			must(fmt.Errorf("fak serve: --cpu-offload-experts requires backend %q to advertise quantized UploadDtype (Q8_0 upload); use a quantized-upload backend or omit --cpu-offload-experts", backend.Name()))
		}
		loadMessages = append(loadMessages, serveStartupMessage("load-mode", "info", fmt.Sprintf("GGUF device load -> direct-resident Q4_K on backend %q (raw super-blocks copied to VRAM/host, dequant fused into the GEMM tile, no f32/Q8 round-trip; experts host-resident)", backend.Name())))
		// Device backend + CPU expert-offload: the DIRECT-RESIDENT-Q4_K path. Q4_K matmul weights
		// are copied to VRAM (dense) / host RAM (experts) as raw super-blocks and served with the
		// dequant-fused k_q4k_gemm tile (#485) — skipping the lean path's Q4_K->f32->Q8 round-trip
		// entirely. That round-trip was the load bottleneck on the 466 GB GLM-5.2 (every tensor
		// decompressed to f32 then re-quantized); the resident path is I/O-bound only, cutting the
		// load from ~100 min to minutes. The per-request session decodes Q4_K (s.Q4K=true) on both
		// the device (dense) and host (offloaded experts). The fit check still uses the dense-vs-
		// expert split so experts dwarfing VRAM stay host-scoped while the dense side must fit.
		// #4952: a sharded rank admits only its expert band (WithExpertShard above), so the host
		// expert pool must be charged that band and not the whole routed set — otherwise the
		// host-scope refusal below over-refuses ~ranks-fold, and it does so BEFORE the authoritative
		// rank-local gate (refuseEPPlanIfUnfit) ever runs. residentRanks is 1 for every unsharded
		// serve, which plans exactly as before.
		memPlan, err := fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath, backend, residentRanks, contextBudgetTokens)
		must(err)
		// #971 blocker 3: the dense weights fit-checked above land in VRAM, but the routed MoE
		// experts (~424 GiB for GLM-5.2 Q4_K) are pinned in HOST RAM — and a device backend does not
		// advertise HostCapacity, so the device fit check fails OPEN on them. Guard the host expert
		// pool against the box's real MemAvailable so a load that would OOM-kill the host (or a second
		// concurrent large load on a contended box) refuses cleanly here instead of wedging the box.
		must(compute.RefuseHostScopedPlanIfTooBigForHost(memPlan, serveGGUFHostHeadroom))
		return loadResidentQ4KDevice(ggufPath, tLoad, memPlan, backend, loadMessages, q4kOpts...)
	case residentQ4K:
		// Standard-arch device serve: hold raw Q4_K matmul tensors RESIDENT on the
		// device (dequant fused into the GEMM tile, no Q4_K->f32->Q8 round-trip), instead of the
		// legacy Q8 rollback below. The resident path loads ~0.56 B/param instead of Q8's
		// ~1 B/param, nearly halving weight VRAM (#949). No expert offload here (that is the
		// cpuOffloadExperts arm) — all weights are device-resident, so the fit uses the
		// non-offload device plan (EstimateLoadMemoryPlan, quant-aware), same helper the Q8 arm
		// uses; only the loader differs. A backend without UploadDtype falls through to the Q8/
		// f32 arms unchanged (the device Q4_K GEMM needs the quantized-upload seam).
		loadMessages = append(loadMessages, serveStartupMessage("load-mode", "info", fmt.Sprintf("GGUF device load -> resident Q4_K on backend %q (raw super-blocks, dequant-fused GEMM, ~0.56 B/param vs Q8 ~1 B/param)", backend.Name())))
		var memPlan compute.MemoryPlan
		var err error
		if residentRanks > 1 {
			memPlan, err = fitAndPlanServeGGUFExpertParallelPathOnDevice(ggufPath, backend, residentRanks, contextBudgetTokens)
		} else {
			memPlan, err = fitAndPlanServeGGUFPathOnDevice(ggufPath, backend, false, contextBudgetTokens)
		}
		must(err)
		return loadResidentQ4KDevice(ggufPath, tLoad, memPlan, backend, loadMessages, q4kOpts...)
	case backend != nil:
		loadMessages = append(loadMessages, serveQuantProvenance(artifactQuant, false))
		if backend.Caps().UploadDtype {
			// A device backend that can consume Q8_0 uploads should not be forced through
			// the f32 resident path. The served planner runs Session.Quant=true, so this
			// is the memory-lean representation it will actually execute.
			loadMessages = append(loadMessages, serveStartupMessage("load-mode", "info", fmt.Sprintf("GGUF device load -> mixed precision on backend %q (Q8 resident weights, f32 activations/KV)", backend.Name())))
			memPlan, err := fitAndPlanServeGGUFPathOnDevice(ggufPath, backend, false, contextBudgetTokens)
			must(err)
			prof := newServeLoadProfiler()
			mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
			must(err)
			modelengine.Preload(mm)
			loadNanos := time.Since(tLoad).Nanoseconds()
			profile := withServeStartupMessages(withServeGGUFMemoryProfile(toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8-device", ggufPath, loadNanos)), memPlan, backend), loadMessages...)
			return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
		}
		// Backends without quantized upload still need f32-resident weights; a lean-Q8
		// model would drop the f32 matmul weights they fall back to.
		loadMessages = append(loadMessages, serveStartupMessage("load-mode", "info", fmt.Sprintf("GGUF device load -> f32 resident weights on backend %q (backend has no quantized UploadDtype)", backend.Name())))
		memPlan, err := fitAndPlanServeGGUFPathOnDevice(ggufPath, backend, true, contextBudgetTokens)
		must(err)
		mm, err := ggufload.LoadModel(ggufPath)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := withServeStartupMessages(withServeGGUFMemoryProfile(toGatewayLoadProfile(&ggufload.LoadProfile{
			Mode:       "gguf-f32-device",
			Source:     ggufPath,
			TotalNanos: loadNanos,
			TotalMS:    float64(loadNanos) / 1e6,
			Phases:     []ggufload.LoadPhaseStat{{Phase: "f32-load", Calls: 1, Nanos: loadNanos, MS: float64(loadNanos) / 1e6, TimePct: 100}},
			Bottleneck: "f32-load",
		}), memPlan, backend), loadMessages...)
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	case os.Getenv("FAK_Q4K") != "" && artifactQuant.Q4KResident:
		// CPU-path memory-fit pre-flight (#974): refuse cleanly with a typed FitTooBig BEFORE the
		// all-resident load can drive MemAvailable to ~0 and OOM-wedge the host (parity with the
		// device path's fit plan). Fail-open where host RAM is not probeable.
		must(fitServeGGUFPathOnHost(ggufPath, false, contextBudgetTokens))
		// Pure-CPU reference serve via the direct-resident-Q4_K loader. That loader already
		// routes the mixed Q5_K/Q6_K experts (GLM-5.2's ~417 GB bulk) to a raw-resident byte
		// copy keyed on the GGUF quant type — the SAME resident-K-quant lever the device
		// cpu-offload case above uses (internal/ggufload/quant_q4k_loader.go). The only thing
		// missing on this path was the WITNESS: the old call threaded no LoadProfiler, so a
		// multi-minute GLM-5.2 load ran silent AND emitted no per-quant-type load-path summary,
		// leaving an operator unable to SEE whether the expert bulk took the raw-resident path
		// (dequant≈0) or the slow f32 round-trip. Thread a real profiler here too (parity with
		// the device cpu-offload case) so both the streamed summary and the gateway /metrics
		// profile carry the resident-vs-dequant breakdown — the witness #975 needs.
		mm, prof, loadNanos := loadResidentQ4KProfiled(ggufPath, tLoad, q4kOpts...)
		loadMessages = append(loadMessages, serveStartupMessage("resident-layout", "info", fakmodel.FormatResidentReport(mm.ResidentReport())))
		profile := withServeStartupMessages(toGatewayLoadProfile(prof.Snapshot("gguf-resident-q4k", ggufPath, loadNanos)), loadMessages...)
		return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	default:
		// CPU-path memory-fit pre-flight (#974): same clean FitTooBig refusal as the FAK_Q4K arm
		// above, so the default lean CPU serve cannot OOM-wedge the host either.
		must(fitServeGGUFPathOnHost(ggufPath, false, contextBudgetTokens))
		prof := newServeLoadProfiler()
		mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := withServeStartupMessages(toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8", ggufPath, loadNanos)), loadMessages...)
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	}
}

// loadResidentQ4KProfiled runs the profiled raw-Q4_K resident load shared by the three Q4_K
// serve arms (device cpu-offload, device-resident, pure-CPU). It keeps ordinary progress
// off the launch terminal, preserves safety alerts before readiness, and returns the model,
// profiler, and elapsed load nanos for the gateway's durable startup surface.

func loadResidentQ4KProfiled(ggufPath string, tLoad time.Time, opts ...ggufload.Q4KLoadOption) (*fakmodel.Model, *ggufload.LoadProfiler, int64) {
	prof := newServeLoadProfiler()
	// opts carries the per-rank expert shard (ggufload.WithExpertShard) for a sharded expert-
	// parallel serve: this process admits ONLY its band's routed experts into the resident store,
	// so its footprint is the replicated remainder + one band (≈ model/ranks), not the full model.
	// Empty opts (the default, every non-EP serve) is byte-identical to the old LoadModelQ4KProfile.
	var mm *fakmodel.Model
	var err error
	if os.Getenv("FAK_STREAM_Q4K") == "1" || os.Getenv("FAK_METAL_STREAM_Q4K") == "1" {
		mm, err = ggufload.LoadModelQ4KStreamedDense(ggufPath, prof, opts...)
	} else {
		mm, err = ggufload.LoadModelQ4KProfileOptions(ggufPath, prof, opts...)
	}
	must(err)
	modelengine.PreloadQ4K(mm)
	loadNanos := time.Since(tLoad).Nanoseconds()
	return mm, prof, loadNanos
}

// loadResidentQ4KDevice is the device Q4_K arm's shared tail: it runs the profiled resident
// load and folds the host/device memory plan into the retained profile. The cpu-offload and
// device-resident arms differ only in how memPlan is derived upstream. opts threads the per-rank
// expert shard (see loadResidentQ4KProfiled).
func loadResidentQ4KDevice(ggufPath string, tLoad time.Time, memPlan compute.MemoryPlan, backend compute.Backend, messages []gateway.StartupMessage, opts ...ggufload.Q4KLoadOption) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase) {
	mm, prof, loadNanos := loadResidentQ4KProfiled(ggufPath, tLoad, opts...)
	messages = append(messages, serveStartupMessage("resident-layout", "info", fakmodel.FormatResidentReport(mm.ResidentReport())))
	profile := withServeStartupMessages(withServeGGUFMemoryProfile(toGatewayLoadProfile(prof.Snapshot("gguf-resident-q4k-device", ggufPath, loadNanos)), memPlan, backend), messages...)
	return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
}

// resolveServeTokenizer picks the in-kernel chat planner's tokenizer: an explicit
// --tokenizer (a tokenizer.json or its directory, matching cmd/fakchat's resolution) wins;
// otherwise, with --gguf set, the GGUF's EMBEDDED tokenizer is used so /v1/chat/completions
// and /v1/messages serve real in-kernel chat by default (like cmd/simpledemo); otherwise it
// returns nil, leaving the gateway's offline MockPlanner fallback. The bool reports whether
// a tokenizer-load startup phase should be recorded.
//
// On a real load it ALSO arms the in-kernel engine's detokenizer (modelengine.SetTokenizer),
// symmetric with how loadServeInKernelModel preloads the weights: that closes #463's named
// gap — the lower-level /v1/fak/syscall route then NL-tokenizes a call's arguments and
// returns decoded TEXT (generated_text) instead of raw token ids. With no real tokenizer the
// engine keeps its byte-level default, so the CI/no-export path is unchanged.
func resolveServeTokenizer(tokPath, ggufPath string, sinks ...func(gateway.StartupMessage)) (*tokenizer.Tokenizer, bool) {
	emit := func(message gateway.StartupMessage) {
		for _, sink := range sinks {
			if sink != nil {
				sink(message)
			}
		}
	}
	if tokPath != "" {
		tokFile := tokPath
		if info, err := os.Stat(tokFile); err == nil && info.IsDir() {
			tokFile = filepath.Join(tokFile, "tokenizer.json")
		}
		tok, err := tokenizer.LoadJSON(tokFile)
		must(err)
		modelengine.SetTokenizer(tok)
		return tok, true
	}
	if ggufPath != "" {
		if info, err := os.Stat(ggufPath); err == nil && info.IsDir() {
			tok, err := tokenizer.LoadJSON(filepath.Join(ggufPath, "tokenizer.json"))
			if err != nil {
				emit(serveStartupMessage("tokenizer-fallback", "warning",
					fmt.Sprintf("safetensors directory has no usable tokenizer.json (%v); chat uses the offline mock planner", err)))
				return nil, false
			}
			modelengine.SetTokenizer(tok)
			return tok, true
		}
		// No explicit --tokenizer: fall back to the tokenizer EMBEDDED in the GGUF. Virtually
		// every GGUF carries its full vocab+merges, so `fak serve --gguf X` (no --base-url)
		// serves real in-kernel chat out of the box instead of silently dropping
		// /v1/chat/completions to the offline MockPlanner. If the GGUF embeds no usable BPE
		// tokenizer (e.g. an SPM-only checkpoint), we keep the MockPlanner fallback — pass
		// --tokenizer to override.
		if tok, err := embeddedGGUFTokenizer(ggufPath); err == nil {
			modelengine.SetTokenizer(tok)
			return tok, true
		} else {
			emit(serveStartupMessage("tokenizer-fallback", "warning",
				fmt.Sprintf("--gguf has no usable embedded BPE tokenizer (%v); chat uses the offline mock planner; pass --tokenizer <dir|file> for real chat", err)))
		}
	}
	return nil, false
}

// resetOnBudgetHook gates the human-like auto-reset behind the --reset-on-budget flag.
// When the flag is off it returns nil, so the gateway keeps the historical 409 + reset
// directive verbatim (the reset is strictly opt-in). When on, it returns the host hook
// that distills a carryover seed and re-arms the continuation trace with a fresh context
// budget. The flag is validated to require --context-budget-tokens, so freshContextTokens
// is positive here whenever enabled is true.
func resetOnBudgetHook(enabled bool, freshContextTokens int) gateway.ResetOnBudgetFunc {
	if !enabled {
		return nil
	}
	return resetServedSessionOnBudget(freshContextTokens)
}

// The 3.5x bound is the observed native-fak startup high-water mark from the
// Qwen3.8-27B Q4_K_M campaign on an M3 Pro, not a GGUF steady-state estimate.
const metalGGUFObservedPeakMultiplier = 3.5

const (
	streamedQ4KModeRetainedCPU = "retained-cpu-backing"
	streamedQ4KModeFreeCPU     = "free-cpu-after-upload"

	// streamedQ4KMeasuredPeakBytes is the /usr/bin/time maximum resident set from
	// the canonical no-FAK_Q4K_FREE_CPU receipt. The 36 GiB control reached
	// readiness but grew swap by 7,681,930,691 bytes, so it is a refusal witness,
	// not an admission receipt.
	streamedQ4KMeasuredPeakBytes int64 = 22754885632

	// The receipt derives the fail-closed bound by adding the observed peak swap
	// delta to the 36 GiB host and rounding up to the next whole GiB. See
	// docs/_witnesses/issue-8971-streamed-q4k-capacity/canonical-no-free-cpu.json.
	streamedQ4KMetalCapacityBytes int64 = 44 << 30

	// streamedQ4KFreeCPUMetalCapacityBytes is the host size of the immutable #8964
	// FAK_Q4K_FREE_CPU=1 control, not its old 18 GiB maximum-RSS sample. The exact
	// 36 GiB M3 Pro reached native Metal readiness after 64,915,847,712 bytes of
	// cache displacement and reduced swap in both recorded runs. This bound applies
	// only when the operator explicitly declares release-after-upload mode.
	streamedQ4KFreeCPUMetalCapacityBytes int64 = 36 << 30
)

func metalGGUFPeakCapacity(metal bool, steady, total int64, known bool) (peak int64, refuse bool) {
	if !metal || steady <= 0 || total <= 0 || !known {
		return 0, false
	}
	peakFloat := float64(steady) * metalGGUFObservedPeakMultiplier
	if peakFloat >= float64(^uint64(0)>>1) {
		return 0, false
	}
	peak = int64(peakFloat)
	return peak, peak > total
}

func streamedQ4KMetalCapacity(total int64, known, freeCPU bool) (required int64, refuse bool, mode string) {
	mode = streamedQ4KModeRetainedCPU
	required = streamedQ4KMetalCapacityBytes
	if freeCPU {
		mode = streamedQ4KModeFreeCPU
		required = streamedQ4KFreeCPUMetalCapacityBytes
	}
	if total <= 0 || !known {
		return 0, false, mode
	}
	return required, required > total, mode
}

func refuseStreamedQ4KMetalCapacity(total int64, known, freeCPU bool) error {
	required, refuse, mode := streamedQ4KMetalCapacity(total, known, freeCPU)
	if !refuse {
		return nil
	}
	return fmt.Errorf("fak serve: METAL_STREAM_Q4K_PEAK_TOO_BIG: mode=%s native streamed Q4_K startup requires %d bytes (%.2f GiB), host has %d bytes (%.2f GiB); use a larger-memory Mac", mode, required, float64(required)/(1<<30), total, float64(total)/(1<<30))
}

func refuseOversubscribedMetalGGUF(path string) error {
	if os.Getenv("FAK_STREAM_Q4K") == "1" || os.Getenv("FAK_METAL_STREAM_Q4K") == "1" {
		total, _, known := compute.HostSystemMemoryInfo()
		return refuseStreamedQ4KMetalCapacity(total, known, os.Getenv("FAK_Q4K_FREE_CPU") == "1")
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		return err
	}
	defer ws.Close()
	plan, err := ws.EstimateLoadMemoryPlan()
	if err != nil {
		return err
	}
	steady := plan.Total()
	total, _, known := compute.HostSystemMemoryInfo()
	peak, refuse := metalGGUFPeakCapacity(true, steady, total, known)
	if !refuse {
		return nil
	}
	return fmt.Errorf("fak serve: METAL_GGUF_PEAK_TOO_BIG: estimated steady weights %.2f GiB, startup peak %.2f GiB, host has %.2f GiB; native Metal startup would overcommit unified memory before listener readiness; use a larger-memory Mac or a delegated quantized Metal engine", float64(steady)/(1<<30), float64(peak)/(1<<30), float64(total)/(1<<30))
}
