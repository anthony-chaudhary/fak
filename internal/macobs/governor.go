package macobs

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// Default36GBWiredCeilingBytes is the 75% wired memory limit on a 36GB Apple Silicon system (27 GB).
const Default36GBWiredCeilingBytes uint64 = 27 * 1024 * 1024 * 1024

// Closed reason tokens emitted by MemoryGovernor.
const (
	ReasonHeadroomOK                = "HEADROOM_OK"
	ReasonNonSharedPreambleSwapRisk = "NON_SHARED_PREAMBLE_SWAP_RISK"
	ReasonWiredCeilingExceeded      = "WIRED_MEMORY_CEILING_EXCEEDED"
	ReasonSwapDeltaDetected         = "SWAP_DELTA_DETECTED"
	ReasonPressureCritical          = "MEMORY_PRESSURE_CRITICAL"
	ReasonPressureWarn              = "MEMORY_PRESSURE_WARN"
	ReasonSwapRisk                  = "SWAP_RISK"
)

// GovernorOption configures MemoryGovernor instances.
type GovernorOption func(*MemoryGovernor)

// WithGovernorCeilingBytes overrides the wired memory ceiling in bytes.
func WithGovernorCeilingBytes(ceiling uint64) GovernorOption {
	return func(g *MemoryGovernor) {
		if ceiling > 0 {
			g.wiredCeilingBytes = ceiling
		}
	}
}

// WithGovernorBaselineSwap sets custom baseline swap and pageout counters.
func WithGovernorBaselineSwap(swapUsed, pageouts uint64) GovernorOption {
	return func(g *MemoryGovernor) {
		g.baselineSwapUsed = swapUsed
		g.baselinePageOuts = pageouts
		g.currentSwapUsed = swapUsed
		g.currentPageOuts = pageouts
	}
}

// WithGovernorPressureSubscriber attaches a memory pressure subscriber to the governor.
func WithGovernorPressureSubscriber(sub MemoryPressureSubscriber) GovernorOption {
	return func(g *MemoryGovernor) {
		g.pressureSub = sub
	}
}

// MemoryGovernor enforces zero-swap admission bounds for concurrent agents on Apple Silicon
// by tracking 3:1 GDN recurrent state, RadixAttention shared prefix pinning, and wired memory headroom.
type MemoryGovernor struct {
	mu                   sync.RWMutex
	cfg                  HeadroomConfig
	hw                   HardwareTelemetry
	wiredCeilingBytes    uint64
	baselineSwapUsed     uint64
	baselinePageOuts     uint64
	currentSwapUsed      uint64
	currentPageOuts      uint64
	sharedPreamblePinned bool
	sharedPreambleTokens uint64
	sharedPreambleBytes  uint64
	kvBytesPerToken      uint64
	activeAgents         map[string]AgentSpec
	activeOrder          []string
	peakResidentBytes    uint64
	pressureSub          MemoryPressureSubscriber
}

// NewMemoryGovernor initializes an admission memory governor under hardware limits.
func NewMemoryGovernor(hw HardwareTelemetry, cfg HeadroomConfig, opts ...GovernorOption) *MemoryGovernor {
	if cfg.Layers == 0 {
		cfg = DefaultHeadroomConfig()
	}

	effLayers := cfg.EffectiveKVLayers()
	kvHeads := cfg.KVHeads
	if kvHeads == 0 {
		kvHeads = 8
	}
	headDim := cfg.HeadDim
	if headDim == 0 {
		headDim = 128
	}
	kvBytesPerElem := cfg.KVBytesPerElement
	if kvBytesPerElem == 0 {
		kvBytesPerElem = 2
	}
	if cfg.PrivateTailTokens == 0 {
		cfg.PrivateTailTokens = 1024
	}
	if cfg.SharedPrefixTokens == 0 {
		cfg.SharedPrefixTokens = 4096
	}
	if cfg.RecurrentStateBytes == 0 {
		cfg.RecurrentStateBytes = DefaultQwen38RecurrentStatePerAgentBytes
	}

	kvBytesPerToken := 2 * effLayers * kvHeads * headDim * kvBytesPerElem
	sharedPreambleBytes := cfg.SharedPrefixTokens * kvBytesPerToken

	wiredLimit := hw.WiredMemoryLimitBytes
	if wiredLimit == 0 && hw.TotalSystemMemoryBytes > 0 {
		wiredLimit = (hw.TotalSystemMemoryBytes * 3) / 4
	}
	if wiredLimit == 0 {
		wiredLimit = Default36GBWiredCeilingBytes
	}

	g := &MemoryGovernor{
		cfg:                  cfg,
		hw:                   hw,
		wiredCeilingBytes:    wiredLimit,
		baselineSwapUsed:     hw.SwapUsedBytes,
		baselinePageOuts:     hw.PageOuts,
		currentSwapUsed:      hw.SwapUsedBytes,
		currentPageOuts:      hw.PageOuts,
		sharedPreamblePinned: true,
		sharedPreambleTokens: cfg.SharedPrefixTokens,
		sharedPreambleBytes:  sharedPreambleBytes,
		kvBytesPerToken:      kvBytesPerToken,
		activeAgents:         make(map[string]AgentSpec),
		activeOrder:          make([]string, 0, 32),
	}

	for _, opt := range opts {
		opt(g)
	}

	// Calculate initial resident bytes (base weights + OS reserve + pinned preamble)
	base := g.cfg.ModelWeightBytes + g.cfg.OSReserveBytes
	initResident := base
	if g.sharedPreamblePinned {
		initResident += g.sharedPreambleBytes
	}
	g.peakResidentBytes = initResident

	return g
}

// CanAdmit evaluates whether an agent launch request can be admitted without disk swap.
func (g *MemoryGovernor) CanAdmit(spec AgentSpec) (bool, string) {
	decision := g.EvaluateAdmission(spec)
	return decision.Admitted, decision.Reason
}

// EvaluateAdmission evaluates an incoming AgentSpec against current memory headroom and returns a structured decision.
func (g *MemoryGovernor) EvaluateAdmission(spec AgentSpec) AdmissionDecision {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return g.evaluateAdmissionLocked(spec)
}

func (g *MemoryGovernor) evaluateAdmissionLocked(spec AgentSpec) AdmissionDecision {
	// Apply defaults for unspecified fields
	if spec.PreambleTokens == 0 {
		spec.PreambleTokens = g.sharedPreambleTokens
	}
	if spec.TailTokens == 0 {
		spec.TailTokens = g.cfg.PrivateTailTokens
	}
	if spec.RecurrentStateBytes == 0 {
		spec.RecurrentStateBytes = g.cfg.RecurrentStateBytes
	}

	// 1. Check live swap delta assertion if counters have moved
	if g.currentSwapUsed > g.baselineSwapUsed || g.currentPageOuts > g.baselinePageOuts {
		swapDelta := g.currentSwapUsed - g.baselineSwapUsed
		pageDelta := g.currentPageOuts - g.baselinePageOuts
		return AdmissionDecision{
			Admitted:               false,
			Verdict:                AdmissionVerdictRejected,
			Reason:                 fmt.Sprintf("SWAP_DELTA_DETECTED: observed swap delta of %d bytes, pageouts delta of %d; zero-swap bound violated", swapDelta, pageDelta),
			ReasonToken:            ReasonSwapDeltaDetected,
			ProjectedResidentBytes: g.currentResidentLocked(),
			WiredCeilingBytes:      g.wiredCeilingBytes,
			AvailableHeadroomBytes: 0,
			ActiveAgents:           len(g.activeAgents),
			MaxSharedAgents:        g.computeMaxSharedLocked(),
			MaxIsolatedAgents:      g.computeMaxIsolatedLocked(),
		}
	}

	// 2. Check OS memory pressure if subscriber attached
	if g.pressureSub != nil {
		lvl := g.pressureSub.CurrentLevel()
		if lvl == PressureCritical {
			return AdmissionDecision{
				Admitted:               false,
				Verdict:                AdmissionVerdictQueued,
				Reason:                 "MEMORY_PRESSURE_CRITICAL: macOS memory pressure is critical; agent launch requests queued to prevent OOM",
				ReasonToken:            ReasonPressureCritical,
				ProjectedResidentBytes: g.currentResidentLocked(),
				WiredCeilingBytes:      g.wiredCeilingBytes,
				AvailableHeadroomBytes: 0,
				ActiveAgents:           len(g.activeAgents),
				MaxSharedAgents:        g.computeMaxSharedLocked(),
				MaxIsolatedAgents:      g.computeMaxIsolatedLocked(),
			}
		}
		if lvl == PressureWarn && !spec.SharedPreamble {
			return AdmissionDecision{
				Admitted:               false,
				Verdict:                AdmissionVerdictRejected,
				Reason:                 "MEMORY_PRESSURE_WARN: non-shared preamble rejected under OS memory pressure",
				ReasonToken:            ReasonPressureWarn,
				ProjectedResidentBytes: g.currentResidentLocked(),
				WiredCeilingBytes:      g.wiredCeilingBytes,
				AvailableHeadroomBytes: 0,
				ActiveAgents:           len(g.activeAgents),
				MaxSharedAgents:        g.computeMaxSharedLocked(),
				MaxIsolatedAgents:      g.computeMaxIsolatedLocked(),
			}
		}
	}

	// 3. Compute memory requirements
	currentResident := g.currentResidentLocked()
	var incBytes uint64
	if spec.SharedPreamble {
		// Private reasoning tail + O(1) recurrent state
		incBytes = spec.TailTokens*g.kvBytesPerToken + spec.RecurrentStateBytes
		// If preamble was not pinned, add preamble footprint
		if !g.sharedPreamblePinned {
			incBytes += spec.PreambleTokens * g.kvBytesPerToken
		}
	} else {
		// Non-shared preamble: requires full preamble allocation + private tail + recurrent state
		incBytes = (spec.PreambleTokens+spec.TailTokens)*g.kvBytesPerToken + spec.RecurrentStateBytes
	}

	projectedResident := currentResident + incBytes
	var availableHeadroom uint64
	if g.wiredCeilingBytes > currentResident {
		availableHeadroom = g.wiredCeilingBytes - currentResident
	}

	// 4. Enforce wired ceiling bounds (zero-swap protection)
	if projectedResident > g.wiredCeilingBytes {
		if !spec.SharedPreamble {
			isoMB := incBytes / (1024 * 1024)
			projectedMB := projectedResident / (1024 * 1024)
			ceilingMB := g.wiredCeilingBytes / (1024 * 1024)
			swapRiskMB := (projectedResident - g.wiredCeilingBytes) / (1024 * 1024)
			return AdmissionDecision{
				Admitted:               false,
				Verdict:                AdmissionVerdictRejected,
				Reason:                 fmt.Sprintf("NON_SHARED_PREAMBLE_SWAP_RISK: non-shared preamble of %d tokens requires %d MB KV, projecting %d MB total memory exceeding %d MB wired ceiling (%d MB swap risk); disk swap would occur", spec.PreambleTokens, isoMB, projectedMB, ceilingMB, swapRiskMB),
				ReasonToken:            ReasonNonSharedPreambleSwapRisk,
				ProjectedResidentBytes: projectedResident,
				WiredCeilingBytes:      g.wiredCeilingBytes,
				AvailableHeadroomBytes: availableHeadroom,
				ActiveAgents:           len(g.activeAgents),
				MaxSharedAgents:        g.computeMaxSharedLocked(),
				MaxIsolatedAgents:      g.computeMaxIsolatedLocked(),
			}
		}

		return AdmissionDecision{
			Admitted:               false,
			Verdict:                AdmissionVerdictRejected,
			Reason:                 fmt.Sprintf("WIRED_MEMORY_CEILING_EXCEEDED: projected resident memory %d MB exceeds %d MB wired ceiling (%d active agents); disk swap risk (SWAP_RISK)", projectedResident/(1024*1024), g.wiredCeilingBytes/(1024*1024), len(g.activeAgents)),
			ReasonToken:            ReasonWiredCeilingExceeded,
			ProjectedResidentBytes: projectedResident,
			WiredCeilingBytes:      g.wiredCeilingBytes,
			AvailableHeadroomBytes: availableHeadroom,
			ActiveAgents:           len(g.activeAgents),
			MaxSharedAgents:        g.computeMaxSharedLocked(),
			MaxIsolatedAgents:      g.computeMaxIsolatedLocked(),
		}
	}

	// 5. Admission granted with zero-swap guarantee
	var remainingHeadroom uint64
	if g.wiredCeilingBytes > projectedResident {
		remainingHeadroom = g.wiredCeilingBytes - projectedResident
	}

	return AdmissionDecision{
		Admitted:               true,
		Verdict:                AdmissionVerdictAdmitted,
		Reason:                 fmt.Sprintf("HEADROOM_OK: admitted with zero-swap guarantee (projected %d MB / %d MB wired ceiling, active: %d)", projectedResident/(1024*1024), g.wiredCeilingBytes/(1024*1024), len(g.activeAgents)+1),
		ReasonToken:            ReasonHeadroomOK,
		ProjectedResidentBytes: projectedResident,
		WiredCeilingBytes:      g.wiredCeilingBytes,
		AvailableHeadroomBytes: remainingHeadroom,
		ActiveAgents:           len(g.activeAgents),
		MaxSharedAgents:        g.computeMaxSharedLocked(),
		MaxIsolatedAgents:      g.computeMaxIsolatedLocked(),
	}
}

// Admit reserves memory for an agent if headroom permits.
func (g *MemoryGovernor) Admit(spec AgentSpec) (bool, string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	decision := g.evaluateAdmissionLocked(spec)
	if !decision.Admitted {
		return false, decision.Reason
	}

	if spec.ID == "" {
		spec.ID = fmt.Sprintf("agent-%d", len(g.activeAgents)+1)
	}
	if spec.PreambleTokens == 0 {
		spec.PreambleTokens = g.sharedPreambleTokens
	}
	if spec.TailTokens == 0 {
		spec.TailTokens = g.cfg.PrivateTailTokens
	}
	if spec.RecurrentStateBytes == 0 {
		spec.RecurrentStateBytes = g.cfg.RecurrentStateBytes
	}

	g.activeAgents[spec.ID] = spec
	g.activeOrder = append(g.activeOrder, spec.ID)

	if decision.ProjectedResidentBytes > g.peakResidentBytes {
		g.peakResidentBytes = decision.ProjectedResidentBytes
	}

	return true, decision.Reason
}

// AdmitWithQueue attempts to admit an agent, queuing up to timeout if headroom is temporarily constrained.
func (g *MemoryGovernor) AdmitWithQueue(ctx context.Context, spec AgentSpec, timeout time.Duration) (bool, string) {
	if timeout <= 0 {
		return g.Admit(spec)
	}
	if ctx == nil {
		ctx = context.Background()
	}

	deadline := time.Now().Add(timeout)
	for {
		g.mu.Lock()
		decision := g.evaluateAdmissionLocked(spec)
		if decision.Admitted {
			if spec.ID == "" {
				spec.ID = fmt.Sprintf("agent-%d", len(g.activeAgents)+1)
			}
			if spec.PreambleTokens == 0 {
				spec.PreambleTokens = g.sharedPreambleTokens
			}
			if spec.TailTokens == 0 {
				spec.TailTokens = g.cfg.PrivateTailTokens
			}
			if spec.RecurrentStateBytes == 0 {
				spec.RecurrentStateBytes = g.cfg.RecurrentStateBytes
			}
			g.activeAgents[spec.ID] = spec
			g.activeOrder = append(g.activeOrder, spec.ID)
			if decision.ProjectedResidentBytes > g.peakResidentBytes {
				g.peakResidentBytes = decision.ProjectedResidentBytes
			}
			g.mu.Unlock()
			return true, decision.Reason
		}
		g.mu.Unlock()

		remaining := time.Until(deadline)
		if remaining <= 0 {
			return false, decision.Reason
		}

		select {
		case <-ctx.Done():
			return false, fmt.Sprintf("admission canceled while queued: %v", ctx.Err())
		case <-time.After(minDuration(remaining, 10*time.Millisecond)):
		}
	}
}

func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

// IsSwapRisk reports whether the admission decision indicates disk swap risk.
func (d AdmissionDecision) IsSwapRisk() bool {
	return !d.Admitted && (d.ReasonToken == ReasonSwapRisk ||
		d.ReasonToken == ReasonNonSharedPreambleSwapRisk ||
		d.ReasonToken == ReasonWiredCeilingExceeded ||
		d.ReasonToken == ReasonSwapDeltaDetected ||
		strings.Contains(d.Reason, "SWAP_RISK") ||
		strings.Contains(d.Reason, "swap"))
}

// Release frees an active agent's reservation.
func (g *MemoryGovernor) Release(agentID string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, exists := g.activeAgents[agentID]; !exists {
		return false
	}
	delete(g.activeAgents, agentID)

	for i, id := range g.activeOrder {
		if id == agentID {
			g.activeOrder = append(g.activeOrder[:i], g.activeOrder[i+1:]...)
			break
		}
	}
	return true
}

// Reset clears all active agent allocations and restores governor to initial state.
func (g *MemoryGovernor) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.activeAgents = make(map[string]AgentSpec)
	g.activeOrder = g.activeOrder[:0]

	base := g.cfg.ModelWeightBytes + g.cfg.OSReserveBytes
	if g.sharedPreamblePinned {
		base += g.sharedPreambleBytes
	}
	g.peakResidentBytes = base
}

// ActiveCount returns the number of currently resident agents.
func (g *MemoryGovernor) ActiveCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.activeAgents)
}

// ActiveAgents returns a copy of currently resident agent specifications.
func (g *MemoryGovernor) ActiveAgents() []AgentSpec {
	g.mu.RLock()
	defer g.mu.RUnlock()

	out := make([]AgentSpec, 0, len(g.activeOrder))
	for _, id := range g.activeOrder {
		if spec, ok := g.activeAgents[id]; ok {
			out = append(out, spec)
		}
	}
	return out
}

// SetPreamblePinned configures whether the shared preamble is kept pinned in global KV cache.
func (g *MemoryGovernor) SetPreamblePinned(pinned bool, tokens uint64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.sharedPreamblePinned = pinned
	if tokens > 0 {
		g.sharedPreambleTokens = tokens
		g.sharedPreambleBytes = tokens * g.kvBytesPerToken
	}
}

// UpdateHardwareTelemetry refreshes live swap, pageout, and wired limit values.
func (g *MemoryGovernor) UpdateHardwareTelemetry(hw HardwareTelemetry) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.hw = hw
	g.currentSwapUsed = hw.SwapUsedBytes
	g.currentPageOuts = hw.PageOuts
	if hw.WiredMemoryLimitBytes > 0 {
		g.wiredCeilingBytes = hw.WiredMemoryLimitBytes
	}
}

// AssertZeroSwapDelta verifies that no disk swap or pageout activity occurred since baseline.
func (g *MemoryGovernor) AssertZeroSwapDelta() error {
	g.mu.RLock()
	defer g.mu.RUnlock()

	swapDelta := int64(g.currentSwapUsed) - int64(g.baselineSwapUsed)
	pageoutsDelta := int64(g.currentPageOuts) - int64(g.baselinePageOuts)
	if swapDelta > 0 || pageoutsDelta > 0 {
		return fmt.Errorf("zero-swap bound violated: swap_used_delta=%d bytes, pageouts_delta=%d", swapDelta, pageoutsDelta)
	}
	return nil
}

func (g *MemoryGovernor) currentResidentLocked() uint64 {
	resident := g.cfg.ModelWeightBytes + g.cfg.OSReserveBytes
	if g.sharedPreamblePinned {
		resident += g.sharedPreambleBytes
	}
	for _, a := range g.activeAgents {
		if a.SharedPreamble {
			resident += a.TailTokens*g.kvBytesPerToken + a.RecurrentStateBytes
		} else {
			resident += (a.PreambleTokens+a.TailTokens)*g.kvBytesPerToken + a.RecurrentStateBytes
		}
	}
	return resident
}

func (g *MemoryGovernor) computeMaxSharedLocked() int {
	base := g.cfg.ModelWeightBytes + g.cfg.OSReserveBytes
	if g.wiredCeilingBytes <= base {
		return 0
	}
	pool := g.wiredCeilingBytes - base
	if pool < g.sharedPreambleBytes {
		return 0
	}
	rem := pool - g.sharedPreambleBytes
	tailPerAgent := g.cfg.PrivateTailTokens*g.kvBytesPerToken + g.cfg.RecurrentStateBytes
	if tailPerAgent == 0 {
		return 0
	}
	return int(rem / tailPerAgent)
}

func (g *MemoryGovernor) computeMaxIsolatedLocked() int {
	base := g.cfg.ModelWeightBytes + g.cfg.OSReserveBytes
	if g.wiredCeilingBytes <= base {
		return 0
	}
	pool := g.wiredCeilingBytes - base
	isoPerAgent := g.cfg.ContextTokens*g.kvBytesPerToken + g.cfg.RecurrentStateBytes
	if isoPerAgent == 0 {
		return 0
	}
	return int(pool / isoPerAgent)
}

// Telemetry snapshots the real-time status of the memory governor.
func (g *MemoryGovernor) Telemetry() GovernorTelemetry {
	g.mu.RLock()
	defer g.mu.RUnlock()

	resident := g.currentResidentLocked()
	var remainingHeadroom uint64
	if g.wiredCeilingBytes > resident {
		remainingHeadroom = g.wiredCeilingBytes - resident
	}

	maxShared := g.computeMaxSharedLocked()
	maxIsolated := g.computeMaxIsolatedLocked()
	active := len(g.activeAgents)
	availSlots := maxShared - active
	if availSlots < 0 {
		availSlots = 0
	}

	swapDelta := uint64(0)
	if g.currentSwapUsed > g.baselineSwapUsed {
		swapDelta = g.currentSwapUsed - g.baselineSwapUsed
	}
	pageDelta := uint64(0)
	if g.currentPageOuts > g.baselinePageOuts {
		pageDelta = g.currentPageOuts - g.baselinePageOuts
	}

	zeroSwapGuaranteed := resident <= g.wiredCeilingBytes && swapDelta == 0 && pageDelta == 0

	status := "NOMINAL"
	if swapDelta > 0 || pageDelta > 0 {
		status = "SWAP_VIOLATION"
	} else if active >= 24 {
		status = "CAPACITY_OPTIMAL_24_AGENTS"
	} else if active > 0 {
		status = "CONCURRENCY_OK"
	}

	peak := g.peakResidentBytes
	if resident > peak {
		peak = resident
	}

	return GovernorTelemetry{
		ActiveAgents:                active,
		AvailableAgentSlots:         availSlots,
		MaxSharedAgents:             maxShared,
		MaxIsolatedAgents:           maxIsolated,
		WiredMemoryCeilingBytes:     g.wiredCeilingBytes,
		ResidentMemoryBytes:         resident,
		PeakMemoryBytes:             peak,
		RemainingHeadroomBytes:      remainingHeadroom,
		SharedPreamblePinned:        g.sharedPreamblePinned,
		SharedPreambleTokens:        g.sharedPreambleTokens,
		SharedPreambleBytes:         g.sharedPreambleBytes,
		RecurrentStateBytesPerAgent: g.cfg.RecurrentStateBytes,
		SwapUsedDeltaBytes:          swapDelta,
		PageoutsDelta:               pageDelta,
		ZeroSwapGuaranteed:          zeroSwapGuaranteed,
		ConcurrencyStatus:           status,
		Available:                   true,
	}
}
