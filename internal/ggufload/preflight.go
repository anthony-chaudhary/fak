package ggufload

import (
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// preflight.go — the FAIL-FAST, header-only pre-check that lets a benchmark or test loop
// learn whether a GGUF will load BEFORE it pays the multi-minute (for GLM-5.2, ~100 min)
// tensor load. The expensive load is "load then discover": a wrong architecture, a model too
// big for the device (which OOM-panics mid-load), or a bad header are all only found at the
// end. This classifier turns the COMMON failures into a sub-second refusal off the parsed
// header alone — no tensor byte is read — by composing the pieces the loader already owns:
// OpenWeights (header parse), File.Config (arch + required-key validity), EstimateLoadBytes
// (the byte footprint the loader will demand), and FitOnDevice (the typed, fail-open device-fit
// refusal). It mirrors internal/terminalbench/preflight.go: a PURE classifier (no os/flag/I/O —
// the caller opens the header and hands the facts in) so it is fully unit-testable without a
// real multi-GB checkpoint or a GPU.

// ModelPreflightSchema identifies the modelbench preflight artifact.
const ModelPreflightSchema = "fak.modelbench-preflight.v1"

// Closed-vocabulary preflight verdicts. They describe whether this model can be loaded on this
// host, never a benchmark result.
const (
	PreflightReady        = "READY"             // header parsed, arch known, fits (or fit unknown) — safe to load
	PreflightRefuseTooBig = "REFUSE_TOO_BIG"    // a capacity-reporting backend KNOWS the model exceeds its ceiling
	PreflightRefuseArch   = "REFUSE_BAD_ARCH"   // header parsed but the architecture/required config keys are missing
	PreflightRefuseHeader = "REFUSE_BAD_HEADER" // the GGUF header could not be opened/parsed, or the estimate failed
)

// Closed-vocabulary device-fit sub-states, orthogonal to the verdict (a READY can be FIT_OK or
// FIT_UNKNOWN; a REFUSE_TOO_BIG is always FIT_TOO_BIG).
const (
	FitOK          = "FIT_OK"      // the named backend reports the model fits within its headroom-adjusted budget
	FitUnknown     = "FIT_UNKNOWN" // no backend, or a backend that cannot probe capacity (the portable floor) — fail open
	FitTooBigState = "FIT_TOO_BIG" // the backend KNOWS the model exceeds its ceiling
)

// defaultAssumedGiBPerSec is the conservative load-throughput assumption behind the ROUGH ETA
// when the caller does not supply one. It is deliberately pessimistic (slow CPU-bound quant-on-
// load) so the ETA over- rather than under-states; it is always rendered as an estimate, never a
// witnessed number.
const defaultAssumedGiBPerSec = 0.5

// PreflightInput carries the already-parsed header facts plus the load regime. The caller does
// the single header-only OpenWeights and passes the result (or the open error) in, so this
// classifier stays pure.
type PreflightInput struct {
	Path     string          // the checkpoint path, for the report
	OpenErr  error           // non-nil iff OpenWeights failed; Source must then be nil
	Source   *WeightSource   // the header-only weight source (nil iff OpenErr != nil)
	Backend  compute.Backend // the resolved compute backend (may be nil for the legacy/cpu-ref floor)
	Headroom float64         // fraction of the device budget reserved for KV/scratch (passed to FitOnDevice)

	// Load-regime flags select which byte estimate to use, matching loadModel's dispatch:
	// lean/q4k load the raw quantized payload (EstimateLoadBytes); the default GGUF path
	// dequantizes to f32 resident (EstimateF32LoadBytes); --cpu-offload-experts splits expert
	// bytes to the host (EstimateCPUOffloadExpertsMemoryPlan).
	Lean           bool
	Q4K            bool
	OffloadExperts bool
	// VulkanMixedQ4K selects the backed Vulkan -q4k loader's actual storage policy:
	// eligible Q4_K/Q2_K matrices remain packed while unsupported dense formats make
	// the bounded f32 -> Q8 round trip.
	VulkanMixedQ4K bool

	// AssumedGiBPerSec drives the ROUGH ETA; 0 uses defaultAssumedGiBPerSec.
	AssumedGiBPerSec float64
}

// ModelPreflight is the header-only readiness artifact: a verdict plus the cheap facts an
// operator (or a smoke arm) needs to decide whether to pay the full load.
type ModelPreflight struct {
	Schema                 string  `json:"schema"`
	Path                   string  `json:"path,omitempty"`
	Verdict                string  `json:"verdict"`
	Arch                   string  `json:"arch,omitempty"`
	TensorCount            int     `json:"tensor_count,omitempty"`
	EstLoadBytes           int64   `json:"est_load_bytes,omitempty"`
	EstLoadGiB             float64 `json:"est_load_gib,omitempty"`
	EstReadBytes           int64   `json:"est_read_bytes,omitempty"`
	EstHostResidentBytes   int64   `json:"est_host_resident_bytes,omitempty"`
	EstDeviceResidentBytes int64   `json:"est_device_resident_bytes,omitempty"`
	EstLoadStagingBytes    int64   `json:"est_load_staging_bytes,omitempty"`
	FitState               string  `json:"fit_state"`
	FitScope               string  `json:"fit_scope,omitempty"`
	DeviceAvailBytes       int64   `json:"device_avail_bytes,omitempty"`
	HostAvailBytes         int64   `json:"host_avail_bytes,omitempty"`
	ETASecondsEst          float64 `json:"eta_seconds_est,omitempty"` // bytes / assumed GiB/s — an ESTIMATE, never witnessed
	Reason                 string  `json:"reason,omitempty"`
	NextAction             string  `json:"next_action,omitempty"`
}

// Refused reports whether the verdict is any REFUSE_* (the caller exits non-zero on true).
func (p ModelPreflight) Refused() bool {
	return p.Verdict == PreflightRefuseTooBig || p.Verdict == PreflightRefuseArch || p.Verdict == PreflightRefuseHeader
}

// BuildModelPreflight classifies the header facts into the readiness artifact. It short-circuits
// in cheapest-failure-first order — bad header, bad arch, then the byte estimate, then the
// device-fit check — so the most common reasons a load would have been wasted are reported off
// the header without ever reading a tensor. The fit check is fail-open: a nil or non-probing
// backend yields FIT_UNKNOWN and a READY verdict, so the portable floor is never falsely refused.
func BuildModelPreflight(in PreflightInput) ModelPreflight {
	out := ModelPreflight{
		Schema:   ModelPreflightSchema,
		Path:     strings.TrimSpace(in.Path),
		FitState: FitUnknown,
	}

	// Rung 1: the header must open/parse.
	if in.OpenErr != nil || in.Source == nil {
		out.Verdict = PreflightRefuseHeader
		out.Reason = headerReason(in.OpenErr)
		out.NextAction = "confirm the path is a readable GGUF checkpoint (and all shards present for a split checkpoint), then re-run the preflight"
		return out
	}

	// Rung 2: the architecture + required config keys must be present.
	cfg, cfgErr := in.Source.File.Config()
	if cfgErr != nil {
		out.Verdict = PreflightRefuseArch
		out.Reason = cfgErr.Error()
		out.NextAction = "the GGUF header is missing the architecture or a required config key fak's loader needs; this checkpoint is not loadable as-is"
		return out
	}
	out.Arch = cfg.ModelType
	out.TensorCount = len(in.Source.File.Tensors)

	// Rung 3: estimate the simultaneous load demands off the header. Ordinary regimes keep
	// their historical single-demand estimate; Vulkan mixed Q4_K additionally separates the
	// retained host store, device copy, and bounded worker staging peak.
	est, estErr := estimateLoadFor(in)
	if estErr != nil {
		out.Verdict = PreflightRefuseHeader
		out.Reason = estErr.Error()
		out.NextAction = "the tensor directory could not be sized from the header; the checkpoint may be malformed"
		return out
	}
	out.EstLoadBytes = est.plan.Total()
	out.EstLoadGiB = float64(out.EstLoadBytes) / (1 << 30)
	out.EstReadBytes = est.readBytes
	out.EstHostResidentBytes = est.hostResidentBytes
	out.EstDeviceResidentBytes = est.deviceResidentBytes
	out.EstLoadStagingBytes = est.stagingBytes
	etaGiB := out.EstLoadGiB
	if est.readBytes > 0 {
		etaGiB = float64(est.readBytes) / (1 << 30)
	}
	out.ETASecondsEst = etaSeconds(etaGiB, in.AssumedGiBPerSec)

	// Rung 4: the device-fit check (fail-open). REFUSE only when a capacity-reporting backend
	// KNOWS the model exceeds its ceiling.
	if fitErr := compute.RefuseMemoryPlanIfTooBig(in.Backend, est.plan, in.Headroom); fitErr != nil {
		var fe *compute.FitError
		if errors.As(fitErr, &fe) {
			out.Verdict = PreflightRefuseTooBig
			out.FitState = FitTooBigState
			out.FitScope = string(fe.Scope)
			if fe.Scope == compute.MemoryScopeHost {
				out.HostAvailBytes = fe.Avail
			} else {
				out.DeviceAvailBytes = fe.Avail
			}
			out.Reason = fe.Error()
			out.NextAction = "this model does not fit the named device; use a bigger device, --cpu-offload-experts, a smaller quant, or omit -backend to run on the portable floor"
			return out
		}
		// A non-FitError fit failure (e.g. a header re-walk error) is a header-class refusal.
		out.Verdict = PreflightRefuseHeader
		out.Reason = fitErr.Error()
		out.NextAction = "the device-fit estimate failed; re-check the checkpoint"
		return out
	}

	out.Verdict = PreflightReady
	if memoryPlanProbes(in.Backend, est.plan) {
		out.FitState = FitOK
		out.DeviceAvailBytes = deviceAvailBytes(in.Backend, in.Headroom)
		out.HostAvailBytes = hostAvailBytes(in.Backend, in.Headroom)
	} else {
		out.FitState = FitUnknown
	}
	out.NextAction = "header check passed — safe to load (or run -smoke for a 1-token forward proof before the full bench)"
	return out
}

type preflightEstimate struct {
	plan                compute.MemoryPlan
	readBytes           int64
	hostResidentBytes   int64
	deviceResidentBytes int64
	stagingBytes        int64
}

// estimateLoadFor selects the memory plan matching loadModel's dispatch. The Vulkan mixed
// regime is the only multi-pool plan; all older regimes retain their prior plan and total.
func estimateLoadFor(in PreflightInput) (preflightEstimate, error) {
	var plan compute.MemoryPlan
	var err error
	switch {
	case in.VulkanMixedQ4K:
		return estimateVulkanMixedQ4K(in.Source)
	case in.OffloadExperts:
		plan, err = in.Source.EstimateCPUOffloadExpertsMemoryPlan()
	case in.Lean || in.Q4K:
		plan, err = in.Source.EstimateLoadMemoryPlan()
	default:
		plan, err = in.Source.EstimateF32LoadMemoryPlan()
	}
	if err != nil {
		return preflightEstimate{}, err
	}
	return preflightEstimate{
		plan:                plan,
		readBytes:           plan.Total(),
		hostResidentBytes:   plan.HostTotal(),
		deviceResidentBytes: plan.DeviceTotal(),
	}, nil
}

// estimateVulkanMixedQ4K mirrors modelbench's dense Vulkan loader: eligible Q4_K and Q2_K
// matmul weights remain packed, while unsupported formats are dequantized, canonicalized,
// and stored as Q8 or f32. The header-only plan conservatively includes page-aligned host
// backing, the device copy, and the largest W raw+two-f32 worker windows. W is loadWorkers(),
// the exact runtime concurrency including FAK_GGUF_LOAD_WORKERS. Split/MoE/unknown layouts
// fail closed until they share their exact transform and sharding contract with this estimator.
func estimateVulkanMixedQ4K(s *WeightSource) (preflightEstimate, error) {
	if s == nil || s.File == nil {
		return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate has no weight source")
	}
	if len(s.readerFor) > 0 || len(s.closers) > 1 {
		return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate does not yet support split checkpoints")
	}
	cfg, err := s.File.Config()
	if err != nil {
		return preflightEstimate{}, err
	}
	if cfg.IsMoE() {
		return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate does not yet support MoE tensor splitting")
	}

	var readBytes, hostPacked, hostQ8, hostF32Logical, deviceBytes int64
	staging := make([]int64, 0, len(s.File.Tensors))
	for _, info := range s.File.Tensors {
		payload, err := tensorPayloadBytes(info)
		if err != nil {
			return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate tensor %s: %w", info.Name, err)
		}
		payloadBytes, err := checkedEstimateUint64(payload, "payload", info.Name)
		if err != nil {
			return preflightEstimate{}, err
		}
		if readBytes, err = checkedEstimateAdd(readBytes, payloadBytes, "read bytes"); err != nil {
			return preflightEstimate{}, err
		}

		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate has no canonical mapping for tensor %s", info.Name)
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return preflightEstimate{}, err
		}
		elems, err := tensorElems(info)
		if err != nil {
			return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate tensor %s: %w", info.Name, err)
		}
		f32Bytes, err := checkedEstimateMulUint64(elems, 4, "f32 bytes", info.Name)
		if err != nil {
			return preflightEstimate{}, err
		}

		retained := (info.Type == TensorQ4_K && model.ResidentQ4KEligible(cfg, canon)) ||
			(info.Type == TensorQ2_K && model.ResidentKQuantEligible(cfg, canon))
		if retained {
			alloc, err := conservativePageAllocation(payloadBytes)
			if err != nil {
				return preflightEstimate{}, fmt.Errorf("gguf: mixed Vulkan estimate tensor %s: %w", info.Name, err)
			}
			if hostPacked, err = checkedEstimateAdd(hostPacked, alloc, "host packed bytes"); err != nil {
				return preflightEstimate{}, err
			}
			if deviceBytes, err = checkedEstimateAdd(deviceBytes, payloadBytes, "device packed bytes"); err != nil {
				return preflightEstimate{}, err
			}
			staging = append(staging, payloadBytes)
			continue
		}

		stage, err := checkedEstimateAdd(payloadBytes, f32Bytes, "load staging bytes")
		if err == nil {
			stage, err = checkedEstimateAdd(stage, f32Bytes, "load staging bytes")
		}
		if err != nil {
			return preflightEstimate{}, err
		}
		staging = append(staging, stage)

		q8Weight := model.IsQuantWeight(canon) && len(shape) == 2
		tiedEmbedding := cfg.TieWordEmbeddings && canon == "model.embed_tokens.weight" && len(shape) == 2
		if q8Weight || tiedEmbedding {
			q8Logical, q8Host, err := estimateQ8Allocation(info.Name, shape)
			if err != nil {
				return preflightEstimate{}, err
			}
			if hostQ8, err = checkedEstimateAdd(hostQ8, q8Host, "host Q8 bytes"); err != nil {
				return preflightEstimate{}, err
			}
			if deviceBytes, err = checkedEstimateAdd(deviceBytes, q8Logical, "device Q8 bytes"); err != nil {
				return preflightEstimate{}, err
			}
		}
		if !q8Weight || tiedEmbedding {
			if hostF32Logical, err = checkedEstimateAdd(hostF32Logical, f32Bytes, "host f32 bytes"); err != nil {
				return preflightEstimate{}, err
			}
			// These tensors remain in the builder's F32 store and may be materialized as
			// F32 by the backed session. A tied embedding needs this copy and its Q8 head.
			if deviceBytes, err = checkedEstimateAdd(deviceBytes, f32Bytes, "device f32 bytes"); err != nil {
				return preflightEstimate{}, err
			}
		}
	}

	hostF32Reserved, err := checkedEstimateAdd(hostF32Logical, hostF32Logical, "host f32 growth reserve")
	if err != nil {
		return preflightEstimate{}, err
	}
	hostResident, err := checkedEstimateAdd(hostPacked, hostQ8, "host resident bytes")
	if err == nil {
		hostResident, err = checkedEstimateAdd(hostResident, hostF32Reserved, "host resident bytes")
	}
	if err != nil {
		return preflightEstimate{}, err
	}

	sort.Slice(staging, func(i, j int) bool { return staging[i] > staging[j] })
	workers := loadWorkers()
	if workers > len(staging) {
		workers = len(staging)
	}
	var stagingBytes int64
	for _, n := range staging[:workers] {
		if stagingBytes, err = checkedEstimateAdd(stagingBytes, n, "load staging bytes"); err != nil {
			return preflightEstimate{}, err
		}
	}

	plan := make(compute.MemoryPlan, 0, 3)
	if deviceBytes > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceBytes, Detail: "gguf-vulkan-mixed-device-resident", DType: "mixed"})
	}
	if hostResident > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryWeights, Scope: compute.MemoryScopeHost, Bytes: hostResident, Detail: "gguf-vulkan-mixed-host-resident", DType: "mixed"})
	}
	if stagingBytes > 0 {
		plan = append(plan, compute.MemoryDemand{Class: compute.MemoryScratchpad, Scope: compute.MemoryScopeHost, Bytes: stagingBytes, Detail: fmt.Sprintf("gguf-vulkan-mixed-load-staging-w%d", workers), DType: compute.F32.String()})
	}
	return preflightEstimate{
		plan:                plan,
		readBytes:           readBytes,
		hostResidentBytes:   hostResident,
		deviceResidentBytes: deviceBytes,
		stagingBytes:        stagingBytes,
	}, nil
}

func estimateQ8Allocation(name string, shape []int) (logical, host int64, err error) {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%32 != 0 {
		return 0, 0, fmt.Errorf("gguf: mixed Vulkan estimate tensor %s has unsupported Q8 shape %v", name, shape)
	}
	codes, err := checkedEstimateMul(int64(shape[0]), int64(shape[1]), "Q8 code bytes")
	if err != nil {
		return 0, 0, err
	}
	scales, err := checkedEstimateMul(int64(shape[0]), int64(shape[1]/32), "Q8 scale count")
	if err == nil {
		scales, err = checkedEstimateMul(scales, 4, "Q8 scale bytes")
	}
	if err != nil {
		return 0, 0, err
	}
	logical, err = checkedEstimateAdd(codes, scales, "Q8 logical bytes")
	if err != nil {
		return 0, 0, err
	}
	codeAlloc, err := conservativePageAllocation(codes)
	if err != nil {
		return 0, 0, err
	}
	scaleAlloc, err := conservativePageAllocation(scales)
	if err != nil {
		return 0, 0, err
	}
	host, err = checkedEstimateAdd(codeAlloc, scaleAlloc, "Q8 host allocation")
	return logical, host, err
}

func conservativePageAllocation(n int64) (int64, error) {
	if n <= 0 {
		return 0, nil
	}
	page := int64(os.Getpagesize())
	if page <= 1 {
		return n, nil
	}
	if n > math.MaxInt64-(page-1) {
		return 0, fmt.Errorf("page-rounded allocation overflows int64")
	}
	pages := (n + page - 1) / page
	if pages > math.MaxInt64/page {
		return 0, fmt.Errorf("page-rounded allocation overflows int64")
	}
	rounded := pages * page
	return checkedEstimateAdd(rounded, page, "page-aligned allocation")
}

func checkedEstimateUint64(n uint64, what, name string) (int64, error) {
	if n > math.MaxInt64 {
		return 0, fmt.Errorf("gguf: mixed Vulkan estimate %s for tensor %s overflows int64", what, name)
	}
	return int64(n), nil
}

func checkedEstimateMulUint64(a, b uint64, what, name string) (int64, error) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, fmt.Errorf("gguf: mixed Vulkan estimate %s for tensor %s overflows uint64", what, name)
	}
	return checkedEstimateUint64(a*b, what, name)
}

func checkedEstimateMul(a, b int64, what string) (int64, error) {
	if a < 0 || b < 0 || (a != 0 && b > math.MaxInt64/a) {
		return 0, fmt.Errorf("gguf: mixed Vulkan estimate %s overflows int64", what)
	}
	return a * b, nil
}

func checkedEstimateAdd(a, b int64, what string) (int64, error) {
	if a < 0 || b < 0 || a > math.MaxInt64-b {
		return 0, fmt.Errorf("gguf: mixed Vulkan estimate %s overflows int64", what)
	}
	return a + b, nil
}

// memoryPlanProbes requires every non-empty scope to have a real capacity source. A missing
// host probe for a mixed plan remains READY/FIT_UNKNOWN (fail open), never falsely FIT_OK.
func memoryPlanProbes(be compute.Backend, plan compute.MemoryPlan) bool {
	if plan.DeviceTotal() > 0 {
		if _, _, known := compute.DeviceMemoryInfo(be); !known {
			return false
		}
	}
	if plan.HostTotal() > 0 {
		if _, _, known := compute.HostMemoryInfo(be); !known {
			return false
		}
	}
	return len(plan) > 0
}

// deviceAvailBytes reports the backend's known free device bytes for the READY/FIT_OK report,
// 0 when the backend cannot probe.
func deviceAvailBytes(be compute.Backend, headroom float64) int64 {
	total, free, known := compute.DeviceMemoryInfo(be)
	if !known {
		return 0
	}
	if free < 0 {
		free = total
	}
	return compute.BudgetAfterHeadroom(free, headroom)
}

// hostAvailBytes reports the backend's headroom-adjusted host-memory budget for the
// READY/FIT_OK report, or 0 when the backend cannot probe it.
func hostAvailBytes(be compute.Backend, headroom float64) int64 {
	total, free, known := compute.HostMemoryInfo(be)
	if !known {
		return 0
	}
	if free < 0 {
		free = total
	}
	return compute.BudgetAfterHeadroom(free, headroom)
}

// etaSeconds is the ROUGH load-time estimate: GiB / assumed GiB-per-second. It is always an
// estimate — labeled as such in the report and the renderer — never a witnessed throughput.
func etaSeconds(gib, assumedGiBPerSec float64) float64 {
	rate := assumedGiBPerSec
	if rate <= 0 {
		rate = defaultAssumedGiBPerSec
	}
	if gib <= 0 {
		return 0
	}
	return gib / rate
}

// headerReason renders the rung-1 reason from the open error (or a generic message when the
// caller passed a nil source without an error).
func headerReason(openErr error) string {
	if openErr != nil {
		return openErr.Error()
	}
	return "no GGUF weight source (header not opened)"
}

// Render returns a human-readable one-block summary of the preflight for stderr.
func (p ModelPreflight) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak modelbench preflight: %s\n", p.Verdict)
	if p.Path != "" {
		fmt.Fprintf(&b, "  path:   %s\n", p.Path)
	}
	if p.Arch != "" {
		fmt.Fprintf(&b, "  arch:   %s (%d tensors)\n", p.Arch, p.TensorCount)
	}
	if p.EstLoadBytes > 0 {
		etaGiB := p.EstLoadGiB
		if p.EstReadBytes > 0 {
			etaGiB = float64(p.EstReadBytes) / (1 << 30)
		}
		fmt.Fprintf(&b, "  load:   ~%.2f GiB peak estimated, ~%.0fs ETA (estimate: %.2f GiB read / %.2f GiB/s)\n",
			p.EstLoadGiB, p.ETASecondsEst, etaGiB, gibPerSecFromETA(etaGiB, p.ETASecondsEst))
		if p.EstHostResidentBytes > 0 || p.EstDeviceResidentBytes > 0 || p.EstLoadStagingBytes > 0 {
			fmt.Fprintf(&b, "  memory: ~%.2f GiB host resident + ~%.2f GiB device resident + ~%.2f GiB load staging\n",
				float64(p.EstHostResidentBytes)/(1<<30), float64(p.EstDeviceResidentBytes)/(1<<30), float64(p.EstLoadStagingBytes)/(1<<30))
		}
	}
	fmt.Fprintf(&b, "  fit:    %s", p.FitState)
	if p.FitScope != "" {
		fmt.Fprintf(&b, " (%s)", p.FitScope)
	}
	if p.DeviceAvailBytes > 0 {
		fmt.Fprintf(&b, " (device budget ~%.2f GiB)", float64(p.DeviceAvailBytes)/(1<<30))
	}
	if p.HostAvailBytes > 0 {
		fmt.Fprintf(&b, " (host budget ~%.2f GiB)", float64(p.HostAvailBytes)/(1<<30))
	}
	b.WriteString("\n")
	if p.Reason != "" {
		fmt.Fprintf(&b, "  reason: %s\n", p.Reason)
	}
	if p.NextAction != "" {
		fmt.Fprintf(&b, "  next:   %s\n", p.NextAction)
	}
	return b.String()
}

// gibPerSecFromETA back-derives the assumed rate for the render line so it shows the assumption
// behind the ETA. Returns the default when the ETA is unset.
func gibPerSecFromETA(gib, etaSec float64) float64 {
	if etaSec <= 0 || gib <= 0 {
		return defaultAssumedGiBPerSec
	}
	return gib / etaSec
}
