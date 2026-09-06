package agentsched

import (
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

const (
	// DefaultMaxWorkers is the default worker concurrency ceiling.
	DefaultMaxWorkers = 16

	// MaxCPUPctThreshold is the host CPU percentage threshold where GateHostEnvelope trips.
	MaxCPUPctThreshold = 85.0
	// EarlyWarningCPUThreshold is the host CPU threshold triggering pre-emptive turn pacing.
	EarlyWarningCPUThreshold = 75.0
	// MinFreeRAMBytesThreshold is the minimum free system RAM required before GateHostEnvelope trips.
	MinFreeRAMBytesThreshold = uint64(4 * 1024 * 1024 * 1024) // 4 GB
	// MaxOpenHandlesThreshold is the maximum open file handles allowed before GateHostEnvelope trips.
	MaxOpenHandlesThreshold = 130000 // 130k handles

	// DefaultPacingModerateMS is the turn pacing delay injected under moderate host stress.
	DefaultPacingModerateMS int64 = 100

	// DefaultPacingHighMS is the turn pacing delay injected under severe host stress.
	DefaultPacingHighMS int64 = 250
)

// AdmissionGate identifies one of the four gates in the admission governor.
type AdmissionGate uint8

const (
	GateWorkerConcurrency AdmissionGate = 1
	GateHostEnvelope      AdmissionGate = 2
	GateProviderHeadroom  AdmissionGate = 3
	GateLaneClearance     AdmissionGate = 4
)

// String returns the canonical human-readable identifier for the admission gate.
func (g AdmissionGate) String() string {
	switch g {
	case GateWorkerConcurrency:
		return "GATE_1_WORKER_CONCURRENCY"
	case GateHostEnvelope:
		return "GATE_2_HOST_ENVELOPE"
	case GateProviderHeadroom:
		return "GATE_3_PROVIDER_HEADROOM"
	case GateLaneClearance:
		return "GATE_4_LANE_CLEARANCE"
	default:
		return fmt.Sprintf("GATE_%d", g)
	}
}

// HostTelemetry represents current host CPU, memory, handles, and thermal state.
type HostTelemetry struct {
	CPUPct          float64   `json:"cpu_pct"`
	FreeRAMBytes    uint64    `json:"free_ram_bytes"`
	OpenHandles     int       `json:"open_handles"`
	ThermalPressure bool      `json:"thermal_pressure"`
	PowerSag        bool      `json:"power_sag"`
	ObservedAt      time.Time `json:"observed_at"`
}

// ProviderHeadroom checks whether an account has token budget and is not throttled.
type ProviderHeadroom interface {
	HasTokenBudget(accountID string, tokens int64) bool
	IsThrottled(accountID string) bool
}

// MemoryProviderHeadroom is an in-memory headroom tracker suitable for production & tests.
type MemoryProviderHeadroom struct {
	mu           sync.RWMutex
	tokenBudgets map[string]int64
	throttled    map[string]bool
}

// NewMemoryProviderHeadroom creates a new MemoryProviderHeadroom.
func NewMemoryProviderHeadroom() *MemoryProviderHeadroom {
	return &MemoryProviderHeadroom{
		tokenBudgets: make(map[string]int64),
		throttled:    make(map[string]bool),
	}
}

// SetBudget sets the available token budget for an account.
func (m *MemoryProviderHeadroom) SetBudget(accountID string, tokens int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.tokenBudgets[accountID] = tokens
}

// SetThrottled marks an account as in active rate-limit throttle.
func (m *MemoryProviderHeadroom) SetThrottled(accountID string, throttled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if throttled {
		m.throttled[accountID] = true
	} else {
		delete(m.throttled, accountID)
	}
}

// HasTokenBudget reports whether the account has sufficient tokens.
func (m *MemoryProviderHeadroom) HasTokenBudget(accountID string, tokens int64) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if tokens <= 0 {
		return true
	}
	avail, exists := m.tokenBudgets[accountID]
	if !exists {
		return true // unconstrained if not explicitly registered
	}
	return avail >= tokens
}

// IsThrottled reports whether the account is currently throttled.
func (m *MemoryProviderHeadroom) IsThrottled(accountID string) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.throttled[accountID]
}

// AdmissionVerdict is the evaluated result of the 4-gate check.
type AdmissionVerdict struct {
	Admitted     bool          `json:"admitted"`
	FailedGate   AdmissionGate `json:"failed_gate,omitempty"`
	Reason       string        `json:"reason,omitempty"`
	RetryAfterMS int64         `json:"retry_after_ms,omitempty"`
	PacingMS     int64         `json:"pacing_ms,omitempty"`
}

// GovernorConfig configures the 4-gate admission governor and priority queue.
type GovernorConfig struct {
	BaseConcurrency int
	QueueCapacity   int
	Taxonomy        laneadmit.Taxonomy
	Headroom        ProviderHeadroom
	DropP3OnStress  bool
}

// Governor manages agent task scheduling, 4-gate admission, and dynamic thermal/power load shedding.
type Governor struct {
	mu sync.Mutex

	baseConcurrency      int
	effectiveConcurrency int
	inFlight             int
	queue                *PriorityQueue

	telemetry       HostTelemetry
	p3Paused        bool
	dropP3OnStress  bool
	pacingMS        int64
	stableTickCount int

	headroom   ProviderHeadroom
	taxonomy   laneadmit.Taxonomy
	heldLeases map[string]laneadmit.Lease
}

// NewGovernor creates an initialized Governor.
func NewGovernor(cfg GovernorConfig) *Governor {
	baseK := cfg.BaseConcurrency
	if baseK <= 0 {
		baseK = DefaultMaxWorkers
	}
	qCap := cfg.QueueCapacity
	if qCap <= 0 {
		qCap = abi.MaxQueueCapacity
	}

	headroom := cfg.Headroom
	if headroom == nil {
		headroom = NewMemoryProviderHeadroom()
	}

	return &Governor{
		baseConcurrency:      baseK,
		effectiveConcurrency: baseK,
		queue:                NewPriorityQueue(qCap),
		headroom:             headroom,
		taxonomy:             cfg.Taxonomy,
		heldLeases:           make(map[string]laneadmit.Lease),
		dropP3OnStress:       cfg.DropP3OnStress,
		telemetry: HostTelemetry{
			FreeRAMBytes: MinFreeRAMBytesThreshold + 1024*1024*1024,
			ObservedAt:   time.Now(),
		},
	}
}

// Queue returns the underlying PriorityQueue.
func (g *Governor) Queue() *PriorityQueue {
	return g.queue
}

// Submit enqueues a task into the priority queue.
func (g *Governor) Submit(task *Task) error {
	return g.queue.Enqueue(task)
}

// PacingMS returns the current turn pacing interval in milliseconds.
func (g *Governor) PacingMS() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.pacingMS
}

// EffectiveConcurrency returns the dynamically adjusted worker concurrency ceiling.
func (g *Governor) EffectiveConcurrency() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.effectiveConcurrency
}

// InFlight returns the number of currently executing worker tasks.
func (g *Governor) InFlight() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.inFlight
}

// IsP3Paused reports whether speculative tasks are paused.
func (g *Governor) IsP3Paused() bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.p3Paused
}

// UpdateTelemetry updates host telemetry and triggers dynamic load shedding and turn pacing.
func (g *Governor) UpdateTelemetry(t HostTelemetry) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.telemetry = t
	if g.telemetry.ObservedAt.IsZero() {
		g.telemetry.ObservedAt = time.Now()
	}

	// 1. Check severe stress: CPU >= 85% or thermal/power pressure flagged.
	severeStress := t.CPUPct >= MaxCPUPctThreshold || t.ThermalPressure || t.PowerSag
	// 2. Check early warning: CPU >= 75% or free RAM < 4GB or handles > 130k.
	earlyWarning := severeStress || t.CPUPct >= EarlyWarningCPUThreshold || (t.FreeRAMBytes > 0 && t.FreeRAMBytes < MinFreeRAMBytesThreshold) || (t.OpenHandles >= MaxOpenHandlesThreshold)

	if severeStress {
		g.stableTickCount = 0
		// Dynamically downscale active worker concurrency K -> max(1, K/2)
		newK := g.effectiveConcurrency / 2
		if newK < 1 {
			newK = 1
		}
		g.effectiveConcurrency = newK
		g.p3Paused = true
		g.pacingMS = DefaultPacingHighMS

		if g.dropP3OnStress {
			g.queue.DropP3()
		}
	} else if earlyWarning {
		g.stableTickCount = 0
		// Early warning: enforce turn pacing and pause P3 workloads
		g.p3Paused = true
		if t.CPUPct >= EarlyWarningCPUThreshold {
			g.pacingMS = DefaultPacingModerateMS
		}
		if g.dropP3OnStress {
			g.queue.DropP3()
		}
	} else {
		// Telemetry confirms normal envelope; gradually restore concurrency.
		g.stableTickCount++
		if g.stableTickCount >= 2 {
			if g.effectiveConcurrency < g.baseConcurrency {
				g.effectiveConcurrency++
			}
			if g.effectiveConcurrency >= g.baseConcurrency {
				g.p3Paused = false
				g.pacingMS = 0
			}
		}
	}
}

// CheckAdmission evaluates the 4 admission gates for a candidate task without state mutation.
func (g *Governor) CheckAdmission(task *Task) *AdmissionVerdict {
	g.mu.Lock()
	defer g.mu.Unlock()

	return g.checkAdmissionLocked(task)
}

func (g *Governor) checkAdmissionLocked(task *Task) *AdmissionVerdict {
	// Gate 1: worker concurrency ceiling.
	if g.inFlight >= g.effectiveConcurrency {
		return &AdmissionVerdict{
			Admitted:     false,
			FailedGate:   GateWorkerConcurrency,
			Reason:       fmt.Sprintf("worker concurrency ceiling reached (%d/%d active)", g.inFlight, g.effectiveConcurrency),
			RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
			PacingMS:     g.pacingMS,
		}
	}

	// Gate 2: host telemetry envelope.
	if g.telemetry.CPUPct >= MaxCPUPctThreshold {
		return &AdmissionVerdict{
			Admitted:     false,
			FailedGate:   GateHostEnvelope,
			Reason:       fmt.Sprintf("host CPU saturated (%.1f%% >= %.1f%%)", g.telemetry.CPUPct, MaxCPUPctThreshold),
			RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
			PacingMS:     g.pacingMS,
		}
	}
	if g.telemetry.ThermalPressure || g.telemetry.PowerSag {
		return &AdmissionVerdict{
			Admitted:     false,
			FailedGate:   GateHostEnvelope,
			Reason:       "host thermal or power-sag pressure flagged",
			RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
			PacingMS:     g.pacingMS,
		}
	}
	if g.telemetry.FreeRAMBytes > 0 && g.telemetry.FreeRAMBytes < MinFreeRAMBytesThreshold {
		return &AdmissionVerdict{
			Admitted:     false,
			FailedGate:   GateHostEnvelope,
			Reason:       fmt.Sprintf("insufficient free RAM (%d bytes < %d bytes)", g.telemetry.FreeRAMBytes, MinFreeRAMBytesThreshold),
			RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
			PacingMS:     g.pacingMS,
		}
	}
	if g.telemetry.OpenHandles >= MaxOpenHandlesThreshold {
		return &AdmissionVerdict{
			Admitted:     false,
			FailedGate:   GateHostEnvelope,
			Reason:       fmt.Sprintf("open handle limit exceeded (%d >= %d)", g.telemetry.OpenHandles, MaxOpenHandlesThreshold),
			RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
			PacingMS:     g.pacingMS,
		}
	}

	// Gate 3: provider headroom and throttle status.
	if task != nil && task.AccountID != "" {
		if g.headroom.IsThrottled(task.AccountID) {
			return &AdmissionVerdict{
				Admitted:     false,
				FailedGate:   GateProviderHeadroom,
				Reason:       fmt.Sprintf("account %s is in active rate-limit throttle", task.AccountID),
				RetryAfterMS: abi.DefaultRetryAfterQueueFullMS,
				PacingMS:     g.pacingMS,
			}
		}
		if task.TokensNeeded > 0 && !g.headroom.HasTokenBudget(task.AccountID, task.TokensNeeded) {
			return &AdmissionVerdict{
				Admitted:     false,
				FailedGate:   GateProviderHeadroom,
				Reason:       fmt.Sprintf("account %s has insufficient token budget for %d tokens", task.AccountID, task.TokensNeeded),
				RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
				PacingMS:     g.pacingMS,
			}
		}
	}

	// Gate 4: lane clearance against active leases.
	if task != nil && (task.Lane != "" || len(task.Tree) > 0) {
		req := laneadmit.Request{
			Surface: laneadmit.SurfaceDispatch,
			Lane:    task.Lane,
			Tree:    task.Tree,
			Holder:  task.ID,
		}
		var liveLeases []laneadmit.Lease
		for _, l := range g.heldLeases {
			liveLeases = append(liveLeases, l)
		}

		verdict := laneadmit.Decide(req, liveLeases, g.taxonomy)
		if !verdict.Admit {
			return &AdmissionVerdict{
				Admitted:     false,
				FailedGate:   GateLaneClearance,
				Reason:       fmt.Sprintf("lane clearance conflict: %s (%s)", verdict.Reason, verdict.Detail),
				RetryAfterMS: abi.DefaultRetryAfterResourceConstrainedMS,
				PacingMS:     g.pacingMS,
			}
		}
	}

	return &AdmissionVerdict{
		Admitted: true,
		PacingMS: g.pacingMS,
	}
}

// TryAdmit attempts to admit the highest priority task currently ready.
// If admitted, the task is dequeued, inFlight is incremented, and its lease is registered.
// Returns (task, verdict, nil) if admitted; (nil, verdict, nil) if waiting or blocked; or an error.
func (g *Governor) TryAdmit() (*Task, *AdmissionVerdict, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	allowP3 := !g.p3Paused
	candidates := g.queue.Candidates(allowP3)
	if len(candidates) == 0 {
		return nil, &AdmissionVerdict{Admitted: false, Reason: "queue empty"}, nil
	}

	var firstVerdict *AdmissionVerdict
	var admittedTask *Task
	var admittedVerdict *AdmissionVerdict

	for _, candidate := range candidates {
		verdict := g.checkAdmissionLocked(candidate)
		if verdict.Admitted {
			admittedTask = candidate
			admittedVerdict = verdict
			break
		}
		if firstVerdict == nil {
			firstVerdict = verdict
		}
	}

	if admittedTask == nil {
		return nil, firstVerdict, nil
	}

	// Dequeue the admitted candidate from the queue.
	if !g.queue.RemoveTask(admittedTask) {
		return nil, &AdmissionVerdict{Admitted: false, Reason: "concurrent dequeue"}, nil
	}
	dequeued := admittedTask

	g.inFlight++

	// Register held lane lease if task declared a lane or tree
	if dequeued.Lane != "" || len(dequeued.Tree) > 0 {
		if dequeued.ID == "" {
			dequeued.ID = fmt.Sprintf("lease-%d", time.Now().UnixNano())
		}
		leaseID := dequeued.ID
		g.heldLeases[leaseID] = laneadmit.Lease{
			ID:     leaseID,
			Lane:   dequeued.Lane,
			Tree:   dequeued.Tree,
			Holder: dequeued.ID,
		}
	}

	return dequeued, admittedVerdict, nil
}

// Release completes execution for task: decrements in-flight worker count and releases lane leases.
func (g *Governor) Release(task *Task) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.inFlight > 0 {
		g.inFlight--
	}
	if task != nil {
		delete(g.heldLeases, task.ID)
	}
}

// Reset clears all in-flight workers, held leases, and resets concurrency to base.
func (g *Governor) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.inFlight = 0
	g.effectiveConcurrency = g.baseConcurrency
	g.p3Paused = false
	g.pacingMS = 0
	g.stableTickCount = 0
	g.heldLeases = make(map[string]laneadmit.Lease)
}

// ErrAdmitBlocked is returned when immediate turn admission is refused.
var ErrAdmitBlocked = errors.New("agentsched: task admission blocked by governor")
