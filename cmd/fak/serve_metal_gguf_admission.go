package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// The 3.5x bound is the observed native-fak startup high-water mark from the
// Qwen3.8-27B Q4_K_M campaign on an M3 Pro, not a GGUF steady-state estimate.
// It applies to the legacy Q8-quant-profile/F32 arms, where weights are staged
// and duplicated before the resident copy settles.
const metalGGUFObservedPeakMultiplier = 3.5

// metalResidentQ4KStagingScratchBytes is the load-staging margin for the
// resident-Q4K arm, where weights load directly into resident Metal buffers
// rather than being duplicated into Q8 staging. Startup peak is therefore
// steady + this scratch, not the legacy 3.5x. The refusal path and the
// admission plan share metalServeStartupPeakBytes so they cannot drift.
const metalResidentQ4KStagingScratchBytes int64 = 1 << 30

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

// metalServeStartupPeakBytes is the single arm-aware startup-peak estimator
// shared by the admission REFUSAL path (refuseOversubscribedMetalGGUFForHost)
// and the admission PLAN path (estimateMetalWeightsMemoryBounds). Keeping both
// on one function is what prevents the plan from admitting an artifact the
// refusal path rejects (or vice versa).
//
// The resident-Q4K arm loads weights directly into resident Metal buffers, so
// its startup peak is steady plus metalResidentQ4KStagingScratchBytes staging
// scratch. Every legacy arm (quant-profile-q8, f32) keeps the historical
// metalGGUFObservedPeakMultiplier blanket bound. refuse is only meaningful when
// total is known and positive; an unknown host keeps the historic pass-through
// (no refusal) while still reporting the estimate for planning.
func metalServeStartupPeakBytes(arm serveLoadArm, steady, total int64, known bool) (peak int64, refuse bool) {
	if arm == serveLoadArmResidentQ4K {
		if steady <= 0 {
			return 0, false
		}
		if steady > (1<<63-1)-metalResidentQ4KStagingScratchBytes {
			peak = 1<<63 - 1 // saturate: an unrepresentable steady is certainly oversized
		} else {
			peak = steady + metalResidentQ4KStagingScratchBytes
		}
		if total > 0 && known {
			return peak, peak > total
		}
		return peak, false
	}
	peak, refuse = metalGGUFPeakCapacity(true, steady, total, known)
	if peak <= steady {
		peak = int64(float64(steady) * metalGGUFObservedPeakMultiplier)
	}
	return peak, refuse
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

type serveLoadArm string

const (
	serveLoadArmResidentQ4K       serveLoadArm = "resident-q4k"
	serveLoadArmQuantProfileQ8    serveLoadArm = "quant-profile-q8"
	serveLoadArmCPUOffloadExperts serveLoadArm = "cpu-offload-experts"
	serveLoadArmF32               serveLoadArm = "f32"
)

func resolveMetalServeLoadArm(ws *ggufload.WeightSource) serveLoadArm {
	if ws == nil {
		return serveLoadArmResidentQ4K
	}
	quant := ggufload.ClassifyTensorQuant(ws.File.Tensors)
	if (quant.Q4KResident || quant.Recipe == "UD-Q2_K_XL" || servePQ2Artifact(quant)) && os.Getenv("FAK_Q4K") != "0" {
		return serveLoadArmResidentQ4K
	}
	return serveLoadArmQuantProfileQ8
}
