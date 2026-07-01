package taskmgr

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	rmetrics "runtime/metrics"
	"sort"
	"sync"
	"time"
)

const SchemaSnapshot = "fak.task-manager-snapshot.v1"

type State string

const (
	StateRunning  State = "running"
	StateDone     State = "done"
	StateFailed   State = "failed"
	StateCanceled State = "canceled"
)

const DefaultLivenessTimeout = 30 * time.Second

type LivenessClass string

const (
	LivenessUnknown LivenessClass = ""
	LivenessIdle    LivenessClass = "idle"
	LivenessLive    LivenessClass = "live"
	LivenessStalled LivenessClass = "stalled"
)

// Sampler reads this process' resource state. The default sampler uses the Go
// runtime: memory stats, goroutine count, and runtime CPU-class seconds when the
// Go toolchain exposes that metric. The clock is injectable so tests can prove ETA
// and elapsed-time math without sleeping.
type Sampler func(processStart, now time.Time) ResourceSample

type Option func(*Manager)

func WithClock(clock func() time.Time) Option {
	return func(m *Manager) {
		if clock != nil {
			m.clock = clock
		}
	}
}

func WithSampler(sampler Sampler) Option {
	return func(m *Manager) {
		if sampler != nil {
			m.sampler = sampler
		}
	}
}

func WithLivenessTimeout(timeout time.Duration) Option {
	return func(m *Manager) {
		if timeout > 0 {
			m.livenessTimeout = timeout
		}
	}
}

// WithOriginWitness runs w immediately when a task or step is started with
// EvidenceRefs. This moves the quality check to the origin record instead of
// relying on an after-the-fact scorecard pass to discover missing or bad evidence.
func WithOriginWitness(w Witness) Option {
	return func(m *Manager) {
		m.originWitness = w
	}
}

type TaskSpec struct {
	TaskID       string            `json:"task_id"`
	Title        string            `json:"title,omitempty"`
	Total        float64           `json:"total,omitempty"`
	Unit         string            `json:"unit,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
	EvidenceRefs []EvidenceRef     `json:"evidence_refs,omitempty"`
}

type StepSpec struct {
	StepID       string            `json:"step_id"`
	Title        string            `json:"title,omitempty"`
	Concept      string            `json:"concept,omitempty"`
	Total        float64           `json:"total,omitempty"`
	Unit         string            `json:"unit,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
	EvidenceRefs []EvidenceRef     `json:"evidence_refs,omitempty"`
}

type Snapshot struct {
	Schema          string            `json:"schema"`
	ProcessID       int               `json:"process_id"`
	GoOS            string            `json:"goos"`
	GoArch          string            `json:"goarch"`
	GoVersion       string            `json:"go_version"`
	StartedUnixNano int64             `json:"started_unix_nano"`
	TSUnixNano      int64             `json:"ts_unix_nano"`
	UptimeSeconds   float64           `json:"uptime_s"`
	Resource        ResourceSample    `json:"resource"`
	ResourceDelta   ResourceDelta     `json:"resource_delta"`
	Tasks           []TaskSnapshot    `json:"tasks"`
	Concepts        []ConceptUsage    `json:"concepts,omitempty"`
	Labels          map[string]string `json:"labels,omitempty"`
}

type TaskSnapshot struct {
	TaskID             string            `json:"task_id"`
	Title              string            `json:"title,omitempty"`
	State              State             `json:"state"`
	Reason             string            `json:"reason,omitempty"`
	LivenessClass      LivenessClass     `json:"liveness_class,omitempty"`
	BeatsSeen          int64             `json:"beats_seen,omitempty"`
	LastBeatUnixNano   int64             `json:"last_beat_unix_nano,omitempty"`
	LastBeatAgeSeconds *float64          `json:"last_beat_age_s,omitempty"`
	StartedUnixNano    int64             `json:"started_unix_nano"`
	EndedUnixNano      int64             `json:"ended_unix_nano,omitempty"`
	RuntimeSeconds     float64           `json:"runtime_s"`
	Progress           Progress          `json:"progress"`
	ETASeconds         *float64          `json:"eta_s,omitempty"`
	ETAUnixNano        *int64            `json:"estimated_completion_unix_nano,omitempty"`
	CurrentStep        string            `json:"current_step,omitempty"`
	Resource           ResourceWindow    `json:"resource"`
	Steps              []StepSnapshot    `json:"steps,omitempty"`
	Concepts           []ConceptUsage    `json:"concepts,omitempty"`
	Labels             map[string]string `json:"labels,omitempty"`
	EvidenceRefs       []EvidenceRef     `json:"evidence_refs,omitempty"`
	// Witness is the optional, independently-attested completion rung. It is nil
	// for a claimed-only task; the claimed State above is never overwritten.
	Witness *WitnessRecord `json:"witness,omitempty"`
}

type StepSnapshot struct {
	StepID             string            `json:"step_id"`
	Title              string            `json:"title,omitempty"`
	Concept            string            `json:"concept,omitempty"`
	State              State             `json:"state"`
	Reason             string            `json:"reason,omitempty"`
	LivenessClass      LivenessClass     `json:"liveness_class,omitempty"`
	BeatsSeen          int64             `json:"beats_seen,omitempty"`
	LastBeatUnixNano   int64             `json:"last_beat_unix_nano,omitempty"`
	LastBeatAgeSeconds *float64          `json:"last_beat_age_s,omitempty"`
	StartedUnixNano    int64             `json:"started_unix_nano"`
	EndedUnixNano      int64             `json:"ended_unix_nano,omitempty"`
	RuntimeSeconds     float64           `json:"runtime_s"`
	Progress           Progress          `json:"progress"`
	ETASeconds         *float64          `json:"eta_s,omitempty"`
	ETAUnixNano        *int64            `json:"estimated_completion_unix_nano,omitempty"`
	Resource           ResourceWindow    `json:"resource"`
	Labels             map[string]string `json:"labels,omitempty"`
	EvidenceRefs       []EvidenceRef     `json:"evidence_refs,omitempty"`
	// Witness is the optional, independently-attested completion rung for this
	// step. Nil means claimed-only; the claimed State above is never overwritten.
	Witness *WitnessRecord `json:"witness,omitempty"`
}

type Progress struct {
	Done    float64  `json:"done,omitempty"`
	Total   float64  `json:"total,omitempty"`
	Unit    string   `json:"unit,omitempty"`
	Percent *float64 `json:"percent,omitempty"`
}

type ResourceSample struct {
	TSUnixNano     int64   `json:"ts_unix_nano"`
	WallSeconds    float64 `json:"wall_s"`
	CPUSeconds     float64 `json:"cpu_s"`
	HeapAllocBytes uint64  `json:"heap_alloc_bytes,omitempty"`
	HeapInuseBytes uint64  `json:"heap_inuse_bytes,omitempty"`
	HeapSysBytes   uint64  `json:"heap_sys_bytes,omitempty"`
	SysBytes       uint64  `json:"sys_bytes,omitempty"`
	Goroutines     int     `json:"goroutines,omitempty"`
}

type ResourceDelta struct {
	WallSeconds    float64 `json:"wall_s"`
	CPUSeconds     float64 `json:"cpu_s"`
	HeapAllocBytes int64   `json:"heap_alloc_bytes,omitempty"`
	HeapInuseBytes int64   `json:"heap_inuse_bytes,omitempty"`
	HeapSysBytes   int64   `json:"heap_sys_bytes,omitempty"`
	SysBytes       int64   `json:"sys_bytes,omitempty"`
	Goroutines     int     `json:"goroutines,omitempty"`
}

type ResourceWindow struct {
	Start   ResourceSample `json:"start"`
	Current ResourceSample `json:"current"`
	Delta   ResourceDelta  `json:"delta"`
}

type ConceptUsage struct {
	Concept        string  `json:"concept"`
	Steps          int     `json:"steps"`
	RunningSteps   int     `json:"running_steps,omitempty"`
	RuntimeSeconds float64 `json:"runtime_s"`
	CPUSeconds     float64 `json:"cpu_s,omitempty"`
}

type Manager struct {
	mu              sync.Mutex
	clock           func() time.Time
	sampler         Sampler
	livenessTimeout time.Duration
	originWitness   Witness
	started         time.Time
	startResource   ResourceSample
	tasks           map[string]*taskState
	order           []string
	labels          map[string]string
}

type Task struct {
	manager *Manager
	id      string
}

type Step struct {
	manager *Manager
	taskID  string
	stepID  string
}

type taskState struct {
	spec      TaskSpec
	state     State
	reason    string
	started   time.Time
	ended     time.Time
	start     ResourceSample
	end       ResourceSample
	progress  progressState
	heartbeat heartbeatState
	steps     map[string]*stepState
	stepOrder []string
	witness   *WitnessRecord
}

type stepState struct {
	spec      StepSpec
	state     State
	reason    string
	started   time.Time
	ended     time.Time
	start     ResourceSample
	end       ResourceSample
	progress  progressState
	heartbeat heartbeatState
	witness   *WitnessRecord
}

type progressState struct {
	done  float64
	total float64
	unit  string
}

type heartbeatState struct {
	beats int64
	last  time.Time
}

func NewManager(opts ...Option) *Manager {
	m := &Manager{
		clock:           time.Now,
		sampler:         SampleRuntime,
		livenessTimeout: DefaultLivenessTimeout,
		tasks:           map[string]*taskState{},
	}
	for _, opt := range opts {
		opt(m)
	}
	m.started = m.clock()
	m.startResource = m.sampleAt(m.started)
	return m
}

func (m *Manager) StartTask(spec TaskSpec) (*Task, error) {
	if err := validateTaskSpec(spec); err != nil {
		return nil, err
	}
	witness, err := m.originWitnessRecord(Claim{
		TaskID: spec.TaskID,
		State:  StateRunning,
		Refs:   cloneEvidenceRefs(spec.EvidenceRefs),
	})
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.tasks[spec.TaskID]; exists {
		return nil, fmt.Errorf("taskmgr: task %q already exists", spec.TaskID)
	}
	now := m.clock()
	if spec.Unit == "" && spec.Total > 0 {
		spec.Unit = "work"
	}
	st := &taskState{
		spec:     cloneTaskSpec(spec),
		state:    StateRunning,
		started:  now,
		start:    m.sampleAt(now),
		progress: progressState{total: spec.Total, unit: spec.Unit},
		steps:    map[string]*stepState{},
		witness:  witness,
	}
	m.tasks[spec.TaskID] = st
	m.order = append(m.order, spec.TaskID)
	return &Task{manager: m, id: spec.TaskID}, nil
}

func (m *Manager) Task(id string) (*Task, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.tasks[id]; !ok {
		return nil, false
	}
	return &Task{manager: m, id: id}, true
}

func (t *Task) StartStep(spec StepSpec) (*Step, error) {
	return t.manager.StartStep(t.id, spec)
}

func (t *Task) SetProgress(done, total float64, unit string) error {
	return t.manager.SetTaskProgress(t.id, done, total, unit)
}

func (t *Task) Beat() error { return t.manager.BeatTask(t.id) }

func (t *Task) Finish() error              { return t.manager.FinishTask(t.id) }
func (t *Task) Fail(reason string) error   { return t.manager.FailTask(t.id, reason) }
func (t *Task) Cancel(reason string) error { return t.manager.CancelTask(t.id, reason) }

func (s *Step) SetProgress(done, total float64, unit string) error {
	return s.manager.SetStepProgress(s.taskID, s.stepID, done, total, unit)
}

func (s *Step) Beat() error { return s.manager.BeatStep(s.taskID, s.stepID) }

func (s *Step) Finish() error            { return s.manager.FinishStep(s.taskID, s.stepID) }
func (s *Step) Fail(reason string) error { return s.manager.FailStep(s.taskID, s.stepID, reason) }

func (m *Manager) StartStep(taskID string, spec StepSpec) (*Step, error) {
	if err := validateStepSpec(spec); err != nil {
		return nil, err
	}
	witness, err := m.originWitnessRecord(Claim{
		TaskID: taskID,
		StepID: spec.StepID,
		State:  StateRunning,
		Refs:   cloneEvidenceRefs(spec.EvidenceRefs),
	})
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	task, ok := m.tasks[taskID]
	if !ok {
		return nil, fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	if task.state != StateRunning {
		return nil, fmt.Errorf("taskmgr: task %q is %s", taskID, task.state)
	}
	if _, exists := task.steps[spec.StepID]; exists {
		return nil, fmt.Errorf("taskmgr: step %q already exists on task %q", spec.StepID, taskID)
	}
	now := m.clock()
	if spec.Unit == "" && spec.Total > 0 {
		spec.Unit = "work"
	}
	task.steps[spec.StepID] = &stepState{
		spec:     cloneStepSpec(spec),
		state:    StateRunning,
		started:  now,
		start:    m.sampleAt(now),
		progress: progressState{total: spec.Total, unit: spec.Unit},
		witness:  witness,
	}
	task.stepOrder = append(task.stepOrder, spec.StepID)
	return &Step{manager: m, taskID: taskID, stepID: spec.StepID}, nil
}

func (m *Manager) SetTaskProgress(taskID string, done, total float64, unit string) error {
	return m.withTask(taskID, func(task *taskState) error {
		p, err := normalizeProgress(done, total, unit, task.progress)
		if err != nil {
			return err
		}
		task.progress = p
		beat(&task.heartbeat, m.clock())
		return nil
	})
}

func (m *Manager) SetStepProgress(taskID, stepID string, done, total float64, unit string) error {
	return m.withStep(taskID, stepID, func(step *stepState) error {
		p, err := normalizeProgress(done, total, unit, step.progress)
		if err != nil {
			return err
		}
		step.progress = p
		now := m.clock()
		beat(&step.heartbeat, now)
		if task := m.tasks[taskID]; task != nil {
			beat(&task.heartbeat, now)
		}
		return nil
	})
}

func (m *Manager) BeatTask(taskID string) error {
	return m.withTask(taskID, func(task *taskState) error {
		beat(&task.heartbeat, m.clock())
		return nil
	})
}

func (m *Manager) BeatStep(taskID, stepID string) error {
	return m.withTask(taskID, func(task *taskState) error {
		step, ok := task.steps[stepID]
		if !ok {
			return fmt.Errorf("taskmgr: step %q not found on task %q", stepID, taskID)
		}
		now := m.clock()
		beat(&step.heartbeat, now)
		beat(&task.heartbeat, now)
		return nil
	})
}

func (m *Manager) FinishTask(taskID string) error {
	return m.endTask(taskID, StateDone, "")
}

func (m *Manager) FailTask(taskID, reason string) error {
	return m.endTask(taskID, StateFailed, reason)
}

func (m *Manager) CancelTask(taskID, reason string) error {
	return m.endTask(taskID, StateCanceled, reason)
}

func (m *Manager) FinishStep(taskID, stepID string) error {
	return m.endStep(taskID, stepID, StateDone, "")
}

func (m *Manager) FailStep(taskID, stepID, reason string) error {
	return m.endStep(taskID, stepID, StateFailed, reason)
}

func (m *Manager) Snapshot() Snapshot {
	m.mu.Lock()
	defer m.mu.Unlock()

	now := m.clock()
	current := m.sampleAt(now)
	tasks := make([]TaskSnapshot, 0, len(m.order))
	for _, id := range m.order {
		if task := m.tasks[id]; task != nil {
			tasks = append(tasks, m.taskSnapshotLocked(task, now, current))
		}
	}
	return Snapshot{
		Schema:          SchemaSnapshot,
		ProcessID:       os.Getpid(),
		GoOS:            runtime.GOOS,
		GoArch:          runtime.GOARCH,
		GoVersion:       runtime.Version(),
		StartedUnixNano: m.started.UnixNano(),
		TSUnixNano:      now.UnixNano(),
		UptimeSeconds:   seconds(now.Sub(m.started)),
		Resource:        current,
		ResourceDelta:   resourceDelta(m.startResource, current),
		Tasks:           tasks,
		Concepts:        conceptUsage(tasks),
		Labels:          cloneLabels(m.labels),
	}
}

func (m *Manager) endTask(taskID string, state State, reason string) error {
	if !terminalState(state) {
		return fmt.Errorf("taskmgr: invalid terminal task state %q", state)
	}
	return m.withTask(taskID, func(task *taskState) error {
		if terminalState(task.state) {
			return nil
		}
		now := m.clock()
		sample := m.sampleAt(now)
		for _, stepID := range task.stepOrder {
			step := task.steps[stepID]
			if step != nil && step.state == StateRunning {
				step.state = state
				step.reason = reason
				step.ended = now
				step.end = sample
			}
		}
		task.state = state
		task.reason = reason
		task.ended = now
		task.end = sample
		return nil
	})
}

func (m *Manager) endStep(taskID, stepID string, state State, reason string) error {
	if !terminalState(state) {
		return fmt.Errorf("taskmgr: invalid terminal step state %q", state)
	}
	return m.withStep(taskID, stepID, func(step *stepState) error {
		if terminalState(step.state) {
			return nil
		}
		now := m.clock()
		step.state = state
		step.reason = reason
		step.ended = now
		step.end = m.sampleAt(now)
		return nil
	})
}

// withTask runs fn against the named task under the manager lock, returning a
// not-found error if the task is unknown. It centralises the lock / defer-unlock /
// lookup-or-error preamble shared by the task mutators.
func (m *Manager) withTask(taskID string, fn func(*taskState) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	task, ok := m.tasks[taskID]
	if !ok {
		return fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	return fn(task)
}

// withStep runs fn against the named step under the manager lock, returning the
// stepLocked error if the task or step is unknown. The lock-held step mutators share
// this preamble.
func (m *Manager) withStep(taskID, stepID string, fn func(*stepState) error) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	step, err := m.stepLocked(taskID, stepID)
	if err != nil {
		return err
	}
	return fn(step)
}

func (m *Manager) stepLocked(taskID, stepID string) (*stepState, error) {
	task, ok := m.tasks[taskID]
	if !ok {
		return nil, fmt.Errorf("taskmgr: task %q not found", taskID)
	}
	step, ok := task.steps[stepID]
	if !ok {
		return nil, fmt.Errorf("taskmgr: step %q not found on task %q", stepID, taskID)
	}
	return step, nil
}

func (m *Manager) taskSnapshotLocked(task *taskState, now time.Time, current ResourceSample) TaskSnapshot {
	end := now
	cur := current
	if !task.ended.IsZero() {
		end = task.ended
		cur = task.end
	}
	steps := make([]StepSnapshot, 0, len(task.stepOrder))
	currentStep := ""
	for _, id := range task.stepOrder {
		step := task.steps[id]
		if step == nil {
			continue
		}
		ss := m.stepSnapshotLocked(step, now, current)
		if ss.State == StateRunning {
			currentStep = ss.StepID
		}
		steps = append(steps, ss)
	}
	progress := progressSnapshot(task.progress)
	eta, etaAt := estimate(task.started, now, task.state, task.progress)
	liveness, beats, lastBeat, beatAge := livenessSnapshot(task.state, end, task.heartbeat, m.livenessTimeout)
	return TaskSnapshot{
		TaskID:             task.spec.TaskID,
		Title:              task.spec.Title,
		State:              task.state,
		Reason:             task.reason,
		LivenessClass:      liveness,
		BeatsSeen:          beats,
		LastBeatUnixNano:   lastBeat,
		LastBeatAgeSeconds: beatAge,
		StartedUnixNano:    task.started.UnixNano(),
		EndedUnixNano:      unixNanoOrZero(task.ended),
		RuntimeSeconds:     seconds(end.Sub(task.started)),
		Progress:           progress,
		ETASeconds:         eta,
		ETAUnixNano:        etaAt,
		CurrentStep:        currentStep,
		Resource:           ResourceWindow{Start: task.start, Current: cur, Delta: resourceDelta(task.start, cur)},
		Steps:              steps,
		Concepts:           conceptUsage([]TaskSnapshot{{Steps: steps}}),
		Labels:             cloneLabels(task.spec.Labels),
		EvidenceRefs:       cloneEvidenceRefs(task.spec.EvidenceRefs),
		Witness:            cloneWitness(task.witness),
	}
}

func (m *Manager) stepSnapshotLocked(step *stepState, now time.Time, current ResourceSample) StepSnapshot {
	end := now
	cur := current
	if !step.ended.IsZero() {
		end = step.ended
		cur = step.end
	}
	progress := progressSnapshot(step.progress)
	eta, etaAt := estimate(step.started, now, step.state, step.progress)
	liveness, beats, lastBeat, beatAge := livenessSnapshot(step.state, end, step.heartbeat, m.livenessTimeout)
	return StepSnapshot{
		StepID:             step.spec.StepID,
		Title:              step.spec.Title,
		Concept:            step.spec.Concept,
		State:              step.state,
		Reason:             step.reason,
		LivenessClass:      liveness,
		BeatsSeen:          beats,
		LastBeatUnixNano:   lastBeat,
		LastBeatAgeSeconds: beatAge,
		StartedUnixNano:    step.started.UnixNano(),
		EndedUnixNano:      unixNanoOrZero(step.ended),
		RuntimeSeconds:     seconds(end.Sub(step.started)),
		Progress:           progress,
		ETASeconds:         eta,
		ETAUnixNano:        etaAt,
		Resource:           ResourceWindow{Start: step.start, Current: cur, Delta: resourceDelta(step.start, cur)},
		Labels:             cloneLabels(step.spec.Labels),
		EvidenceRefs:       cloneEvidenceRefs(step.spec.EvidenceRefs),
		Witness:            cloneWitness(step.witness),
	}
}

func SampleRuntime(processStart, now time.Time) ResourceSample {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return ResourceSample{
		TSUnixNano:     now.UnixNano(),
		WallSeconds:    seconds(now.Sub(processStart)),
		CPUSeconds:     runtimeCPUSeconds(),
		HeapAllocBytes: ms.HeapAlloc,
		HeapInuseBytes: ms.HeapInuse,
		HeapSysBytes:   ms.HeapSys,
		SysBytes:       ms.Sys,
		Goroutines:     runtime.NumGoroutine(),
	}
}

func runtimeCPUSeconds() float64 {
	samples := []rmetrics.Sample{{Name: "/cpu/classes/total:cpu-seconds"}}
	rmetrics.Read(samples)
	if samples[0].Value.Kind() != rmetrics.KindFloat64 {
		return 0
	}
	return samples[0].Value.Float64()
}

func (m *Manager) sampleAt(now time.Time) ResourceSample {
	s := m.sampler(m.started, now)
	if s.TSUnixNano == 0 {
		s.TSUnixNano = now.UnixNano()
	}
	s.WallSeconds = seconds(now.Sub(m.started))
	return s
}

func validateTaskSpec(spec TaskSpec) error {
	if spec.TaskID == "" {
		return errors.New("taskmgr: task id is required")
	}
	if spec.Total < 0 {
		return errors.New("taskmgr: total work cannot be negative")
	}
	return nil
}

func validateStepSpec(spec StepSpec) error {
	if spec.StepID == "" {
		return errors.New("taskmgr: step id is required")
	}
	if spec.Total < 0 {
		return errors.New("taskmgr: total work cannot be negative")
	}
	return nil
}

func normalizeProgress(done, total float64, unit string, prev progressState) (progressState, error) {
	if done < 0 || total < 0 {
		return progressState{}, errors.New("taskmgr: progress cannot be negative")
	}
	if total == 0 {
		total = prev.total
	}
	if unit == "" {
		unit = prev.unit
	}
	return progressState{done: done, total: total, unit: unit}, nil
}

func progressSnapshot(p progressState) Progress {
	out := Progress{Done: p.done, Total: p.total, Unit: p.unit}
	if p.total > 0 {
		pct := 100 * p.done / p.total
		out.Percent = &pct
	}
	return out
}

func beat(h *heartbeatState, now time.Time) {
	h.beats++
	h.last = now
}

func livenessSnapshot(state State, now time.Time, h heartbeatState, timeout time.Duration) (LivenessClass, int64, int64, *float64) {
	if h.beats <= 0 || h.last.IsZero() {
		return LivenessIdle, 0, 0, nil
	}
	age := seconds(now.Sub(h.last))
	class := LivenessIdle
	if state == StateRunning {
		class = LivenessLive
		if timeout > 0 && now.Sub(h.last) > timeout {
			class = LivenessStalled
		}
	}
	return class, h.beats, h.last.UnixNano(), &age
}

func estimate(start, now time.Time, state State, p progressState) (*float64, *int64) {
	if state != StateRunning || p.total <= 0 || p.done <= 0 || p.done >= p.total {
		return nil, nil
	}
	elapsed := seconds(now.Sub(start))
	if elapsed <= 0 {
		return nil, nil
	}
	remaining := elapsed / p.done * (p.total - p.done)
	if remaining < 0 {
		return nil, nil
	}
	at := now.Add(time.Duration(remaining * float64(time.Second))).UnixNano()
	return &remaining, &at
}

func resourceDelta(start, current ResourceSample) ResourceDelta {
	return ResourceDelta{
		WallSeconds:    current.WallSeconds - start.WallSeconds,
		CPUSeconds:     current.CPUSeconds - start.CPUSeconds,
		HeapAllocBytes: signedDelta(current.HeapAllocBytes, start.HeapAllocBytes),
		HeapInuseBytes: signedDelta(current.HeapInuseBytes, start.HeapInuseBytes),
		HeapSysBytes:   signedDelta(current.HeapSysBytes, start.HeapSysBytes),
		SysBytes:       signedDelta(current.SysBytes, start.SysBytes),
		Goroutines:     current.Goroutines - start.Goroutines,
	}
}

func signedDelta(current, start uint64) int64 {
	if current >= start {
		return int64(current - start)
	}
	return -int64(start - current)
}

func conceptUsage(tasks []TaskSnapshot) []ConceptUsage {
	byConcept := map[string]*ConceptUsage{}
	for _, task := range tasks {
		for _, step := range task.Steps {
			if step.Concept == "" {
				continue
			}
			cu := byConcept[step.Concept]
			if cu == nil {
				cu = &ConceptUsage{Concept: step.Concept}
				byConcept[step.Concept] = cu
			}
			cu.Steps++
			if step.State == StateRunning {
				cu.RunningSteps++
			}
			cu.RuntimeSeconds += step.RuntimeSeconds
			cu.CPUSeconds += step.Resource.Delta.CPUSeconds
		}
	}
	names := make([]string, 0, len(byConcept))
	for name := range byConcept {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]ConceptUsage, 0, len(names))
	for _, name := range names {
		out = append(out, *byConcept[name])
	}
	return out
}

func terminalState(state State) bool {
	return state == StateDone || state == StateFailed || state == StateCanceled
}

func seconds(d time.Duration) float64 {
	if d < 0 {
		return 0
	}
	return d.Seconds()
}

func unixNanoOrZero(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixNano()
}

func cloneTaskSpec(spec TaskSpec) TaskSpec {
	spec.Labels = cloneLabels(spec.Labels)
	spec.EvidenceRefs = cloneEvidenceRefs(spec.EvidenceRefs)
	return spec
}

func cloneStepSpec(spec StepSpec) StepSpec {
	spec.Labels = cloneLabels(spec.Labels)
	spec.EvidenceRefs = cloneEvidenceRefs(spec.EvidenceRefs)
	return spec
}

func cloneLabels(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
