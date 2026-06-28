package ggufload

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
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

	// AssumedGiBPerSec drives the ROUGH ETA; 0 uses defaultAssumedGiBPerSec.
	AssumedGiBPerSec float64
}

// ModelPreflight is the header-only readiness artifact: a verdict plus the cheap facts an
// operator (or a smoke arm) needs to decide whether to pay the full load.
type ModelPreflight struct {
	Schema           string  `json:"schema"`
	Path             string  `json:"path,omitempty"`
	Verdict          string  `json:"verdict"`
	Arch             string  `json:"arch,omitempty"`
	TensorCount      int     `json:"tensor_count,omitempty"`
	EstLoadBytes     int64   `json:"est_load_bytes,omitempty"`
	EstLoadGiB       float64 `json:"est_load_gib,omitempty"`
	FitState         string  `json:"fit_state"`
	DeviceAvailBytes int64   `json:"device_avail_bytes,omitempty"`
	ETASecondsEst    float64 `json:"eta_seconds_est,omitempty"` // bytes / assumed GiB/s — an ESTIMATE, never witnessed
	Reason           string  `json:"reason,omitempty"`
	NextAction       string  `json:"next_action,omitempty"`
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

	// Rung 3: estimate the load bytes off the header (the make/append the loader will demand).
	estBytes, estErr := estimateLoadBytesFor(in)
	if estErr != nil {
		out.Verdict = PreflightRefuseHeader
		out.Reason = estErr.Error()
		out.NextAction = "the tensor directory could not be sized from the header; the checkpoint may be malformed"
		return out
	}
	out.EstLoadBytes = estBytes
	out.EstLoadGiB = float64(estBytes) / (1 << 30)
	out.ETASecondsEst = etaSeconds(out.EstLoadGiB, in.AssumedGiBPerSec)

	// Rung 4: the device-fit check (fail-open). REFUSE only when a capacity-reporting backend
	// KNOWS the model exceeds its ceiling.
	if fitErr := fitOnDeviceFor(in); fitErr != nil {
		var fe *compute.FitError
		if errors.As(fitErr, &fe) {
			out.Verdict = PreflightRefuseTooBig
			out.FitState = FitTooBigState
			out.DeviceAvailBytes = fe.Avail
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
	if deviceProbes(in.Backend) {
		out.FitState = FitOK
		out.DeviceAvailBytes = deviceAvailBytes(in.Backend, in.Headroom)
	} else {
		out.FitState = FitUnknown
	}
	out.NextAction = "header check passed — safe to load (or run -smoke for a 1-token forward proof before the full bench)"
	return out
}

// estimateLoadBytesFor selects the byte estimate matching the load regime, mirroring loadModel's
// dispatch: lean/q4k read the raw quantized payload; --cpu-offload-experts sums the offload plan;
// the default GGUF path dequantizes to f32 resident.
func estimateLoadBytesFor(in PreflightInput) (int64, error) {
	switch {
	case in.OffloadExperts:
		plan, err := in.Source.EstimateCPUOffloadExpertsMemoryPlan()
		if err != nil {
			return 0, err
		}
		return plan.Total(), nil
	case in.Lean || in.Q4K:
		return in.Source.EstimateLoadBytes()
	default:
		return in.Source.EstimateF32LoadBytes()
	}
}

// fitOnDeviceFor runs the device-fit refusal matching the load regime. Each underlying
// Fit*OnDevice is fail-open (nil for an unprobeable/nil backend), so this returns nil unless a
// capacity-reporting backend knows the model is too big.
func fitOnDeviceFor(in PreflightInput) error {
	switch {
	case in.OffloadExperts:
		return in.Source.FitCPUOffloadExpertsOnDevice(in.Backend, in.Headroom)
	case in.Lean || in.Q4K:
		return in.Source.FitOnDevice(in.Backend, in.Headroom)
	default:
		return in.Source.FitF32OnDevice(in.Backend, in.Headroom)
	}
}

// deviceProbes reports whether the backend can report its capacity (so FIT_OK is meaningful).
// A nil or non-probing backend cannot, so the fit state stays FIT_UNKNOWN.
func deviceProbes(be compute.Backend) bool {
	_, _, known := compute.DeviceMemoryInfo(be)
	return known
}

// deviceAvailBytes reports the backend's known free device bytes for the READY/FIT_OK report,
// 0 when the backend cannot probe.
func deviceAvailBytes(be compute.Backend, headroom float64) int64 {
	_, free, known := compute.DeviceMemoryInfo(be)
	if !known {
		return 0
	}
	return free
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
		fmt.Fprintf(&b, "  load:   ~%.2f GiB estimated, ~%.0fs ETA (estimate: %.2f GiB / %.2f GiB/s)\n",
			p.EstLoadGiB, p.ETASecondsEst, p.EstLoadGiB, gibPerSecFromETA(p.EstLoadGiB, p.ETASecondsEst))
	}
	fmt.Fprintf(&b, "  fit:    %s", p.FitState)
	if p.DeviceAvailBytes > 0 {
		fmt.Fprintf(&b, " (device has ~%.2f GiB free)", float64(p.DeviceAvailBytes)/(1<<30))
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
