package agent

import (
	"fmt"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Host-stage vocabulary of a device-backed inkernel_chat line: host= lists the
// stages that ran on the host in the turn, or names one of these instead.
const (
	// inKernelHostStageSample is a token selected on the host from logits read
	// back off a device-memory backend.
	inKernelHostStageSample = "sample"
	// inKernelHostStagesNone means no host stage advanced during the turn.
	inKernelHostStagesNone = "none"
	// inKernelHostStagesUnobserved means the turn opened no host-stage window.
	inKernelHostStagesUnobserved = "unobserved"
)

// inKernelHostSampledTokens is process-wide, like compute's host-fallback
// counters, so a turn reads both as one window delta.
var inKernelHostSampledTokens atomic.Uint64

// observeHostSampledToken counts one host token selection when be keeps its
// tensors in device memory; a CPU-session selection is not a host fallback.
func observeHostSampledToken(be compute.Backend) {
	if be != nil && be.Caps().DeviceMemory {
		inKernelHostSampledTokens.Add(1)
	}
}

type inKernelHostStageSnapshot struct {
	sampled   uint64
	fallbacks []compute.HostFallbackCount
}

func readInKernelHostStageSnapshot() *inKernelHostStageSnapshot {
	return &inKernelHostStageSnapshot{sampled: inKernelHostSampledTokens.Load(), fallbacks: compute.HostFallbackCounts()}
}

// inKernelHostStageWindow is the host-stage counter base of the planner's
// current turn. Device turns hold devMu across the window, so its delta is
// that turn's host work; a coalesced cohort shares one forward and one delta.
type inKernelHostStageWindow struct {
	base atomic.Pointer[inKernelHostStageSnapshot]
}

func (w *inKernelHostStageWindow) begin() { w.base.Store(readInKernelHostStageSnapshot()) }

// stages lists the host stages that advanced since begin: sample first, then
// compute's host-fallback labels in label order. ok is false before any begin.
func (w *inKernelHostStageWindow) stages() (stages []string, ok bool) {
	base := w.base.Load()
	if base == nil {
		return nil, false
	}
	now := readInKernelHostStageSnapshot()
	if now.sampled > base.sampled {
		stages = append(stages, inKernelHostStageSample)
	}
	before := make(map[string]uint64, len(base.fallbacks))
	for _, c := range base.fallbacks {
		before[c.Stage] = c.Count
	}
	for _, c := range now.fallbacks {
		if c.Count > before[c.Stage] {
			stages = append(stages, c.Stage)
		}
	}
	return stages, true
}

// hostStagesLabel is the host= value of a device-backed summary line.
func (p *InKernelPlanner) hostStagesLabel() string {
	stages, ok := p.hostStages.stages()
	switch {
	case !ok:
		return inKernelHostStagesUnobserved
	case len(stages) == 0:
		return inKernelHostStagesNone
	}
	return strings.Join(stages, ",")
}

// executionPlacementClauses split the summary's placement keys by who ran the
// decode. A CPU session (cpu-ref, or Metal's CPU-session seam) keeps q8dec, its
// own SIMD decode tier. A compute backend drops it: q8dec is a CPUID reading,
// not an executed path. That turn names its executed host stages in host= instead.
func (p *InKernelPlanner) executionPlacementClauses(q8kern, q8fusedMark string) (hostClause, q8Clause string) {
	if p == nil || p.backend == nil {
		return "", fmt.Sprintf(" q8dec=%s%s/%dw", q8kern, q8fusedMark, model.Q8DecodeWorkers())
	}
	return " host=" + p.hostStagesLabel(), ""
}
