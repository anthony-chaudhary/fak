package fleet

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ErrNoGPUSignal is the typed answer when the probe cannot observe any GPU
// utilization: nvidia-smi is missing or failed, timed out, or its output carried
// no parseable per-GPU row. The producer then leaves Report.GPU nil —
// unknown-utilization, NEVER a zero-filled "0% idle" reading on a box that
// simply could not measure.
var ErrNoGPUSignal = errors.New("no gpu utilization signal")

// gpuProbeBusyThreshold is the per-GPU utilization percent at which a GPU counts
// toward Busy in the probe's reading. Below it a GPU is idle for waste-crit
// purposes; the fold's own waste floor is a separate, fleet-level knob.
const gpuProbeBusyThreshold = 5

// gpuProbeTimeout bounds the nvidia-smi sample: a hung driver must not stall a
// self-report, so the probe fails soft instead of waiting.
const gpuProbeTimeout = 5 * time.Second

// gpuProbeRunner runs one external command and returns its stdout. It is an
// unexported package var so tests stub it without touching a real binary; the
// default shells out to the named program under the caller's context.
var gpuProbeRunner = func(ctx context.Context, name string, args ...string) (stdout []byte, err error) {
	return exec.CommandContext(ctx, name, args...).Output()
}

// ProbeGPUUtilization samples nvidia-smi NOW and folds the per-GPU utilization
// percentages into a GPUStats: Total = number of GPU rows, Busy = rows at or
// above the busy threshold, UtilPct = the mean rounded to an int. It is
// fail-soft by contract: a missing nvidia-smi, a timeout, or unparseable output
// returns ErrNoGPUSignal (wrapped with the cause) — never a panic, never log
// spam, never a fabricated reading.
func ProbeGPUUtilization(ctx context.Context) (stats GPUStats, err error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ctx, cancel := context.WithTimeout(ctx, gpuProbeTimeout)
	defer cancel()
	out, runErr := gpuProbeRunner(ctx, "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
	if runErr != nil {
		return GPUStats{}, fmt.Errorf("%w: nvidia-smi: %v", ErrNoGPUSignal, rootErr(runErr))
	}
	stats, ok := ParseGPUUtilizationCSV(string(out))
	if !ok {
		return GPUStats{}, fmt.Errorf("%w: unparseable nvidia-smi utilization output", ErrNoGPUSignal)
	}
	return stats, nil
}

// ParseGPUUtilizationCSV parses `nvidia-smi --query-gpu=utilization.gpu
// --format=csv,noheader,nounits` output: one integer percent per GPU line.
// Lines outside 0-100 (NVML warnings, error text, blank lines) are skipped;
// when no valid row remains, ok=false. CRLF endings are accepted.
func ParseGPUUtilizationCSV(out string) (stats GPUStats, ok bool) {
	return parseGPUUtilizationCSV(out, gpuProbeBusyThreshold)
}

// CapacityLearner is the traffic-learned capacity signal: a per-replica EWMA of
// observed output-token rate (tokens/second) over served completions. Unlike
// ProbeGPUUtilization — which reads a hardware label from nvidia-smi — the
// estimate derives entirely from served traffic, so a mixed heterogeneous pool
// self-calibrates with no device metadata. Borrowed shape: vllm-project/aibrix
// pkg/plugins/gateway/algorithms/load_balance.go:316-339@a89b94a574b2
// (Apache-2.0, ADAPT — re-expressed in Go, not copied).
type CapacityLearner struct {
	mu    sync.Mutex
	decay float64
	rates map[string]float64
}

// NewCapacityLearner builds a learner whose EWMA gain is decay: each new
// observation moves the estimate decay-fractions of the way toward the
// observed rate (decay 0.2 keeps roughly a five-sample memory, so a degraded or
// upgraded replica re-prices itself within seconds of traffic). decay must lie
// in (0, 1].
func NewCapacityLearner(decay float64) (*CapacityLearner, error) {
	if !(decay > 0 && decay <= 1) {
		return nil, fmt.Errorf("fleet: capacity learner decay %g out of range (0,1]", decay)
	}
	return &CapacityLearner{decay: decay, rates: make(map[string]float64)}, nil
}

// ObserveOutput folds one served completion into the named replica's learned
// output rate. A zero-token or zero-duration observation carries no rate
// information and is ignored; the first observation for a replica seeds its
// estimate directly instead of blending with an absent prior.
func (l *CapacityLearner) ObserveOutput(name string, outputTokens int, elapsed time.Duration) {
	if l == nil || name == "" || outputTokens <= 0 || elapsed <= 0 {
		return
	}
	rate := float64(outputTokens) / elapsed.Seconds()
	l.mu.Lock()
	defer l.mu.Unlock()
	prev, ok := l.rates[name]
	if !ok {
		l.rates[name] = rate
		return
	}
	l.rates[name] = l.decay*rate + (1-l.decay)*prev
}

// OutputRate reports the replica's current learned output-token rate. ok=false
// means the learner has no observation for that replica yet — unknown, never a
// fabricated zero.
func (l *CapacityLearner) OutputRate(name string) (float64, bool) {
	if l == nil {
		return 0, false
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	rate, ok := l.rates[name]
	return rate, ok
}

func parseGPUUtilizationCSV(out string, busyThreshold int) (stats GPUStats, ok bool) {
	sum := 0
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		util, err := strconv.Atoi(line)
		if err != nil || util < 0 || util > 100 {
			continue
		}
		stats.Total++
		sum += util
		if util >= busyThreshold {
			stats.Busy++
		}
	}
	if stats.Total == 0 {
		return GPUStats{}, false
	}
	stats.UtilPct = int(math.Round(float64(sum) / float64(stats.Total)))
	return stats, true
}
