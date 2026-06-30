package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// ScaleMode selects who owns replica-count changes for a P/D serving fleet.
type ScaleMode string

const (
	ScaleModeNative          ScaleMode = "native"
	ScaleModeDynamoDelegated ScaleMode = "defer_to_dynamo"
)

// ScaleAction is the structured action recorded for each planner tick.
type ScaleAction string

const (
	ScaleActionHold      ScaleAction = "hold"
	ScaleActionScaleUp   ScaleAction = "scale_up"
	ScaleActionScaleDown ScaleAction = "scale_down"
	ScaleActionDeferred  ScaleAction = "deferred"
	ScaleActionError     ScaleAction = "error"
)

// PoolSignals is the L2 serving signal set the P/D planner consumes for one pool.
type PoolSignals struct {
	Goodput       float64       `json:"goodput"`
	QueueDepth    int           `json:"queue_depth"`
	KVUtilization float64       `json:"kv_utilization"`
	TTFT          time.Duration `json:"ttft_ns"`
	TPOT          time.Duration `json:"tpot_ns"`
}

// WorkerSignals carries the same L2 signal shape at per-worker granularity for
// replay and for operators that want to explain a pool aggregate.
type WorkerSignals struct {
	WorkerID      string        `json:"worker_id"`
	Role          WorkerRole    `json:"role"`
	Goodput       float64       `json:"goodput"`
	QueueDepth    int           `json:"queue_depth"`
	KVUtilization float64       `json:"kv_utilization"`
	TTFT          time.Duration `json:"ttft_ns"`
	TPOT          time.Duration `json:"tpot_ns"`
}

// ServingSignals is one planner tick over the split prefill/decode fleet.
type ServingSignals struct {
	Prefill PoolSignals     `json:"prefill"`
	Decode  PoolSignals     `json:"decode"`
	Workers []WorkerSignals `json:"workers,omitempty"`
}

// ServingSignalsFromMetricRows folds normalized serving telemetry rows into the
// split-pool signal shape the autoscaler plans over. Worker roles come from the
// membership registry so scraped vLLM/SGLang/native metric rows do not need to
// grow a fak-only role label.
func ServingSignalsFromMetricRows(m *FleetMembership, rows []ServingMetricRow) ServingSignals {
	var out ServingSignals
	if m == nil || len(rows) == 0 {
		return out
	}
	roles := map[string]WorkerRole{}
	for _, st := range m.Snapshot() {
		roles[st.Spec.ID] = st.Spec.Role
	}
	for _, row := range rows {
		workerID := normalizeServingLabels(row.Labels).Worker
		role := roles[workerID]
		if role != RolePrefill && role != RoleDecode {
			continue
		}
		ws := WorkerSignals{
			WorkerID:      workerID,
			Role:          role,
			Goodput:       gaugeOrZero(row.Goodput),
			QueueDepth:    int(gaugeOrZero(row.Waiting)),
			KVUtilization: gaugeOrZero(row.KVUtilization),
			TTFT:          meanServingHistogramDuration(row.TTFT),
			TPOT:          meanServingHistogramDuration(row.TPOT),
		}
		out.Workers = append(out.Workers, ws)
	}
	out.Prefill = aggregateWorkerSignals(out.WorkerPool(RolePrefill))
	out.Decode = aggregateWorkerSignals(out.WorkerPool(RoleDecode))
	return out
}

func (s ServingSignals) Pool(role WorkerRole) PoolSignals {
	var sig PoolSignals
	switch role {
	case RolePrefill:
		sig = s.Prefill
	case RoleDecode:
		sig = s.Decode
	default:
		return PoolSignals{}
	}
	if !sig.zero() {
		return sig
	}
	return aggregateWorkerSignals(s.WorkerPool(role))
}

func (s ServingSignals) WorkerPool(role WorkerRole) []WorkerSignals {
	if len(s.Workers) == 0 {
		return nil
	}
	out := make([]WorkerSignals, 0, len(s.Workers))
	for _, w := range s.Workers {
		if w.Role == role {
			out = append(out, w)
		}
	}
	return out
}

func (s PoolSignals) zero() bool {
	return s.Goodput == 0 && s.QueueDepth == 0 && s.KVUtilization == 0 && s.TTFT == 0 && s.TPOT == 0
}

func aggregateWorkerSignals(workers []WorkerSignals) PoolSignals {
	var out PoolSignals
	for _, w := range workers {
		out.Goodput += w.Goodput
		out.QueueDepth += w.QueueDepth
		if w.KVUtilization > out.KVUtilization {
			out.KVUtilization = w.KVUtilization
		}
		if w.TTFT > out.TTFT {
			out.TTFT = w.TTFT
		}
		if w.TPOT > out.TPOT {
			out.TPOT = w.TPOT
		}
	}
	return out
}

func gaugeOrZero(g ServingGauge) float64 {
	if !g.Set {
		return 0
	}
	return g.Value
}

func meanServingHistogramDuration(h ServingHistogram) time.Duration {
	if !h.Sum.Set || !h.Count.Set || h.Count.Value <= 0 {
		return 0
	}
	return time.Duration((h.Sum.Value / h.Count.Value) * float64(time.Second))
}

// PoolObjective is the bounded SLO/goodput target for one pool.
type PoolObjective struct {
	MinReplicas   int
	MaxReplicas   int
	ScaleStep     int
	GoodputTarget float64
	QueueHigh     int
	QueueLow      int
	KVUtilHigh    float64
	KVUtilLow     float64
	TTFTSLO       time.Duration
	TPOTSLO       time.Duration
}

// ServingAutoscalerConfig configures the split-pool scale controller.
type ServingAutoscalerConfig struct {
	Mode            ScaleMode
	Prefill         PoolObjective
	Decode          PoolObjective
	Cooldown        time.Duration
	HysteresisTicks int
	TraceSink       ScaleTraceSink
}

// ScaleActuator starts new replicas. Scale-down is deliberately driven through
// FleetMembership.Drain so in-flight work is allowed to quiesce before removal.
type ScaleActuator interface {
	Start(ctx context.Context, role WorkerRole, count int) ([]WorkerSpec, error)
}

// NativeWorkerFactory mints a worker spec for a native fak worker launch request.
type NativeWorkerFactory func(role WorkerRole, ordinal int) WorkerSpec

// NativePoolActuator is the native fak start seam. It does not provision GPUs or
// kill processes; it mints the worker specs the host start loop registers.
type NativePoolActuator struct {
	mu      sync.Mutex
	next    map[WorkerRole]int
	factory NativeWorkerFactory
}

func NewNativePoolActuator(factory NativeWorkerFactory) *NativePoolActuator {
	return &NativePoolActuator{next: map[WorkerRole]int{}, factory: factory}
}

func (a *NativePoolActuator) Start(ctx context.Context, role WorkerRole, count int) ([]WorkerSpec, error) {
	if count < 0 {
		return nil, fmt.Errorf("gateway: negative scale-up count %d", count)
	}
	if count == 0 {
		return nil, nil
	}
	if a == nil {
		return nil, errors.New("gateway: nil native actuator")
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.next == nil {
		a.next = map[WorkerRole]int{}
	}
	out := make([]WorkerSpec, 0, count)
	for i := 0; i < count; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		a.next[role]++
		ordinal := a.next[role]
		spec := WorkerSpec{
			ID:     fmt.Sprintf("%s-%d", role, ordinal),
			Role:   role,
			Engine: EngineNative,
		}
		if a.factory != nil {
			minted := a.factory(role, ordinal)
			if minted.ID != "" {
				spec.ID = minted.ID
			}
			if minted.Role != "" {
				spec.Role = minted.Role
			}
			if minted.Engine != "" {
				spec.Engine = minted.Engine
			}
			if minted.Endpoint != "" {
				spec.Endpoint = minted.Endpoint
			}
		}
		if spec.Role == "" {
			spec.Role = role
		}
		if spec.Engine == "" {
			spec.Engine = EngineNative
		}
		out = append(out, spec)
	}
	return out, nil
}

// ScaleDecision is the structured, replayable trace emitted for every role on
// every autoscaler tick.
type ScaleDecision struct {
	At              time.Time       `json:"at"`
	Mode            ScaleMode       `json:"mode"`
	Role            WorkerRole      `json:"role"`
	Action          ScaleAction     `json:"action"`
	Reason          string          `json:"reason"`
	CurrentReplicas int             `json:"current_replicas"`
	DesiredReplicas int             `json:"desired_replicas"`
	AppliedReplicas int             `json:"applied_replicas"`
	MinReplicas     int             `json:"min_replicas"`
	MaxReplicas     int             `json:"max_replicas"`
	Signals         PoolSignals     `json:"signals"`
	WorkerSignals   []WorkerSignals `json:"worker_signals,omitempty"`
	Workers         []string        `json:"workers,omitempty"`
}

type ScaleTraceSink interface {
	ObserveScaleDecision(ScaleDecision)
}

// ScaleDecisionJournal is an append-only in-memory trace sink. Hosts can render
// JSONLines() to logs or persist Records() for replay.
type ScaleDecisionJournal struct {
	mu      sync.Mutex
	records []ScaleDecision
}

func NewScaleDecisionJournal() *ScaleDecisionJournal { return &ScaleDecisionJournal{} }

func (j *ScaleDecisionJournal) ObserveScaleDecision(d ScaleDecision) {
	if j == nil {
		return
	}
	j.mu.Lock()
	j.records = append(j.records, d)
	j.mu.Unlock()
}

func (j *ScaleDecisionJournal) Records() []ScaleDecision {
	if j == nil {
		return nil
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]ScaleDecision, len(j.records))
	copy(out, j.records)
	return out
}

func (j *ScaleDecisionJournal) JSONLines() string {
	records := j.Records()
	var b strings.Builder
	for _, record := range records {
		line, err := json.Marshal(record)
		if err != nil {
			continue
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	return b.String()
}

type roleScaleState struct {
	pendingDir   int
	pendingTicks int
	lastAction   time.Time
}

// ServingAutoscaler computes and applies bounded replica targets for the split
// prefill/decode pools.
type ServingAutoscaler struct {
	membership *FleetMembership
	actuator   ScaleActuator
	cfg        ServingAutoscalerConfig
	journal    *ScaleDecisionJournal
	trace      ScaleTraceSink

	mu    sync.Mutex
	state map[WorkerRole]roleScaleState
}

func NewServingAutoscaler(m *FleetMembership, actuator ScaleActuator, cfg ServingAutoscalerConfig) *ServingAutoscaler {
	if cfg.Mode == "" {
		cfg.Mode = ScaleModeNative
	}
	if cfg.HysteresisTicks < 1 {
		cfg.HysteresisTicks = 1
	}
	journal := NewScaleDecisionJournal()
	trace := cfg.TraceSink
	return &ServingAutoscaler{
		membership: m,
		actuator:   actuator,
		cfg:        cfg,
		journal:    journal,
		trace:      trace,
		state:      map[WorkerRole]roleScaleState{},
	}
}

func (a *ServingAutoscaler) Decisions() []ScaleDecision {
	if a == nil || a.journal == nil {
		return nil
	}
	return a.journal.Records()
}

func (a *ServingAutoscaler) DecisionJSONLines() string {
	if a == nil || a.journal == nil {
		return ""
	}
	return a.journal.JSONLines()
}

func (a *ServingAutoscaler) Reconcile(ctx context.Context, now time.Time, signals ServingSignals) ([]ScaleDecision, error) {
	if a == nil {
		return nil, errors.New("gateway: nil serving autoscaler")
	}
	if a.membership == nil {
		return nil, errors.New("gateway: serving autoscaler has no membership registry")
	}
	if now.IsZero() {
		now = time.Now()
	}
	roles := []struct {
		role    WorkerRole
		signals PoolSignals
		workers []WorkerSignals
		obj     PoolObjective
	}{
		{role: RolePrefill, signals: signals.Pool(RolePrefill), workers: signals.WorkerPool(RolePrefill), obj: a.cfg.Prefill},
		{role: RoleDecode, signals: signals.Pool(RoleDecode), workers: signals.WorkerPool(RoleDecode), obj: a.cfg.Decode},
	}
	decisions := make([]ScaleDecision, 0, len(roles))
	var errs []error
	for _, r := range roles {
		d, err := a.reconcileRole(ctx, now, r.role, r.signals, r.workers, r.obj)
		decisions = append(decisions, d)
		a.emit(d)
		if err != nil {
			errs = append(errs, err)
		}
	}
	return decisions, errors.Join(errs...)
}

func (a *ServingAutoscaler) reconcileRole(ctx context.Context, now time.Time, role WorkerRole, sig PoolSignals, workers []WorkerSignals, obj PoolObjective) (ScaleDecision, error) {
	obj = obj.normalized()
	current := a.activeReplicas(role)
	target, reason := obj.target(current, sig)
	d := ScaleDecision{
		At:              now,
		Mode:            a.cfg.Mode,
		Role:            role,
		Action:          ScaleActionHold,
		Reason:          reason,
		CurrentReplicas: current,
		DesiredReplicas: target,
		AppliedReplicas: current,
		MinReplicas:     obj.MinReplicas,
		MaxReplicas:     obj.MaxReplicas,
		Signals:         sig,
		WorkerSignals:   workers,
	}
	if a.cfg.Mode == ScaleModeDynamoDelegated {
		d.Action = ScaleActionDeferred
		d.Reason = appendReason(reason, "dynamo_planner_delegated")
		return d, nil
	}
	dir := direction(target, current)
	if dir == 0 {
		a.resetPending(role)
		return d, nil
	}
	if ok, gate := a.readyToAct(role, now, dir); !ok {
		d.Reason = appendReason(reason, gate)
		return d, nil
	}

	var err error
	if dir > 0 {
		d.Action = ScaleActionScaleUp
		d.Workers, err = a.scaleUp(ctx, role, target-current)
		d.AppliedReplicas = current + len(d.Workers)
	} else {
		d.Action = ScaleActionScaleDown
		d.Workers, err = a.scaleDown(role, current-target)
		d.AppliedReplicas = current - len(d.Workers)
	}
	if err != nil {
		d.Action = ScaleActionError
		d.Reason = appendReason(reason, err.Error())
		return d, err
	}
	a.noteAction(role, now)
	return d, nil
}

func (a *ServingAutoscaler) emit(d ScaleDecision) {
	if a == nil {
		return
	}
	a.journal.ObserveScaleDecision(d)
	if a.trace != nil {
		a.trace.ObserveScaleDecision(d)
	}
}

func (a *ServingAutoscaler) activeReplicas(role WorkerRole) int {
	n := 0
	for _, st := range a.membership.Snapshot() {
		if st.Spec.Role == role && !st.Draining {
			n++
		}
	}
	return n
}

func (a *ServingAutoscaler) scaleUp(ctx context.Context, role WorkerRole, count int) ([]string, error) {
	if count <= 0 {
		return nil, nil
	}
	if a.actuator == nil {
		return nil, errors.New("gateway: serving autoscaler has no scale actuator")
	}
	specs, err := a.actuator.Start(ctx, role, count)
	if err != nil {
		return nil, err
	}
	if len(specs) != count {
		return nil, fmt.Errorf("gateway: actuator returned %d workers, want %d", len(specs), count)
	}
	ids := make([]string, 0, len(specs))
	for _, spec := range specs {
		if spec.Role == "" {
			spec.Role = role
		}
		if spec.Role != role {
			return ids, fmt.Errorf("gateway: actuator returned %s worker for %s pool", spec.Role, role)
		}
		if spec.Engine == "" {
			spec.Engine = EngineNative
		}
		if err := a.membership.Add(spec); err != nil {
			return ids, err
		}
		ids = append(ids, spec.ID)
	}
	return ids, nil
}

func (a *ServingAutoscaler) scaleDown(role WorkerRole, count int) ([]string, error) {
	if count <= 0 {
		return nil, nil
	}
	snap := a.membership.Snapshot()
	candidates := make([]WorkerStatus, 0, len(snap))
	for _, st := range snap {
		if st.Spec.Role == role && !st.Draining {
			candidates = append(candidates, st)
		}
	}
	if count > len(candidates) {
		count = len(candidates)
	}
	ids := make([]string, 0, count)
	for i := 0; i < count; i++ {
		st := candidates[len(candidates)-1-i]
		if err := a.membership.Drain(st.Spec.ID); err != nil {
			return ids, err
		}
		ids = append(ids, st.Spec.ID)
	}
	return ids, nil
}

func (a *ServingAutoscaler) readyToAct(role WorkerRole, now time.Time, dir int) (bool, string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	st := a.state[role]
	if a.cfg.Cooldown > 0 && !st.lastAction.IsZero() && now.Sub(st.lastAction) < a.cfg.Cooldown {
		a.state[role] = st
		return false, "cooldown"
	}
	if st.pendingDir != dir {
		st.pendingDir = dir
		st.pendingTicks = 0
	}
	st.pendingTicks++
	a.state[role] = st
	if st.pendingTicks < a.cfg.HysteresisTicks {
		return false, "hysteresis"
	}
	return true, ""
}

func (a *ServingAutoscaler) resetPending(role WorkerRole) {
	a.mu.Lock()
	st := a.state[role]
	st.pendingDir = 0
	st.pendingTicks = 0
	a.state[role] = st
	a.mu.Unlock()
}

func (a *ServingAutoscaler) noteAction(role WorkerRole, now time.Time) {
	a.mu.Lock()
	a.state[role] = roleScaleState{lastAction: now}
	a.mu.Unlock()
}

func (o PoolObjective) normalized() PoolObjective {
	if o.MinReplicas < 0 {
		o.MinReplicas = 0
	}
	if o.MaxReplicas < o.MinReplicas {
		o.MaxReplicas = o.MinReplicas
	}
	if o.ScaleStep < 1 {
		o.ScaleStep = 1
	}
	return o
}

func (o PoolObjective) target(current int, sig PoolSignals) (int, string) {
	if current < o.MinReplicas {
		return o.MinReplicas, "min_replicas"
	}
	if current > o.MaxReplicas {
		return o.MaxReplicas, "max_replicas"
	}
	reasons := o.highReasons(sig)
	if len(reasons) > 0 {
		return clampInt(current+o.ScaleStep, o.MinReplicas, o.MaxReplicas), strings.Join(reasons, ",")
	}
	if current > o.MinReplicas && o.lowEnough(sig) {
		return clampInt(current-o.ScaleStep, o.MinReplicas, o.MaxReplicas), "below_low_water"
	}
	return current, "within_band"
}

func (o PoolObjective) highReasons(sig PoolSignals) []string {
	var reasons []string
	if o.QueueHigh > 0 && sig.QueueDepth > o.QueueHigh {
		reasons = append(reasons, "queue_high")
	}
	if o.GoodputTarget > 0 && sig.Goodput < o.GoodputTarget {
		reasons = append(reasons, "goodput_below_target")
	}
	if o.KVUtilHigh > 0 && sig.KVUtilization > o.KVUtilHigh {
		reasons = append(reasons, "kv_util_high")
	}
	if o.TTFTSLO > 0 && sig.TTFT > o.TTFTSLO {
		reasons = append(reasons, "ttft_slo")
	}
	if o.TPOTSLO > 0 && sig.TPOT > o.TPOTSLO {
		reasons = append(reasons, "tpot_slo")
	}
	return reasons
}

func (o PoolObjective) lowEnough(sig PoolSignals) bool {
	if sig.QueueDepth > o.QueueLow {
		return false
	}
	if o.GoodputTarget > 0 && sig.Goodput < o.GoodputTarget {
		return false
	}
	if o.KVUtilLow > 0 && sig.KVUtilization > o.KVUtilLow {
		return false
	}
	if o.TTFTSLO > 0 && sig.TTFT > o.TTFTSLO {
		return false
	}
	if o.TPOTSLO > 0 && sig.TPOT > o.TPOTSLO {
		return false
	}
	return true
}

func direction(target, current int) int {
	switch {
	case target > current:
		return 1
	case target < current:
		return -1
	default:
		return 0
	}
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func appendReason(base, extra string) string {
	if base == "" {
		return extra
	}
	if extra == "" {
		return base
	}
	return base + "," + extra
}
