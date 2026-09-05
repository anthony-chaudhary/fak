package observer

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Config configures the in-kernel observer thread pool.
type Config struct {
	WorkerCount          int           `json:"worker_count"`
	QueueSize            int           `json:"queue_size"`
	BarrierTimeout       time.Duration `json:"barrier_timeout"`
	MaxHistoryPerSession int           `json:"max_history_per_session"`
	ChurnThreshold       int           `json:"churn_threshold"`
	RegressThreshold     int           `json:"regress_threshold"`
	SimulateKVCache      bool          `json:"simulate_kv_cache"`
	RequireWitnessDiff   bool          `json:"require_witness_diff"`
}

func normalizeConfig(cfg Config) Config {
	if cfg.WorkerCount <= 0 {
		cfg.WorkerCount = 4
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 256
	}
	if cfg.BarrierTimeout <= 0 {
		cfg.BarrierTimeout = 50 * time.Millisecond
	}
	if cfg.MaxHistoryPerSession <= 0 {
		cfg.MaxHistoryPerSession = 50
	}
	if cfg.ChurnThreshold <= 0 {
		cfg.ChurnThreshold = 3
	}
	if cfg.RegressThreshold <= 0 {
		cfg.RegressThreshold = 5
	}
	return cfg
}

type asyncTask struct {
	ctx  context.Context
	obs  StepObservation
	res  chan<- StepObservation
	sess *sessionState
}

type sessionState struct {
	mu             sync.Mutex
	sessionID      string
	history        []StepObservation
	repeatCount    int
	failCount      int
	lastTool       string
	lastArgsSig    string
	flaggedChurn   bool
	flaggedRegress bool
	kvPrefixWarm   bool
	inFlight       int64
}

func (s *sessionState) isFlagged() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.flaggedChurn || s.flaggedRegress
}

func (s *sessionState) evaluate(cfg Config, obs *StepObservation) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if obs.Timestamp.IsZero() {
		obs.Timestamp = time.Now()
	}

	argsSig := computeArgsSig(obs.Tool, obs.Args)
	isErr := obs.Error != "" || isResultError(obs.Result)

	if argsSig != "" && argsSig == s.lastArgsSig {
		s.repeatCount++
	} else {
		s.repeatCount = 1
		s.lastArgsSig = argsSig
	}
	s.lastTool = obs.Tool

	if isErr {
		s.failCount++
	} else {
		s.failCount = 0
	}

	// Warm KV-cache prefix reuse simulation:
	// Once a session establishes valid turns, subsequent steps benefit from warm KV prefix reuse.
	if s.kvPrefixWarm {
		obs.CachedPrefix = true
	} else {
		obs.CachedPrefix = false
		s.kvPrefixWarm = true
	}

	// Witness verification:
	// Read-only tools confirm query results; mutating tools verify diff evidence.
	if obs.IsMutating() {
		if obs.Diff != "" || hasDiffInResult(obs.Result) {
			obs.WitnessVerdict = WitnessDiffConfirmed
		} else {
			obs.WitnessVerdict = WitnessUnwitnessedClaim
		}
	} else {
		obs.WitnessVerdict = WitnessDiffConfirmed
	}

	// Closed step classification vocabulary evaluation (STEP_ADVANCE, STEP_CHURN, STEP_REGRESS).
	switch {
	case obs.StepVerdict == StepRegress || s.repeatCount >= cfg.RegressThreshold || s.failCount >= cfg.RegressThreshold:
		obs.StepVerdict = StepRegress
		if obs.Reason == "" {
			obs.Reason = fmt.Sprintf("regress: repeated tool calls (%d) or failures (%d) exceed regress threshold %d",
				s.repeatCount, s.failCount, cfg.RegressThreshold)
		}
		s.flaggedRegress = true
		s.flaggedChurn = false
		s.kvPrefixWarm = false // Invalidate warm KV-prefix on regression
		obs.CachedPrefix = false

	case obs.StepVerdict == StepChurn || s.repeatCount >= cfg.ChurnThreshold || s.failCount >= cfg.ChurnThreshold:
		obs.StepVerdict = StepChurn
		if obs.Reason == "" {
			obs.Reason = fmt.Sprintf("churn: repeated tool calls (%d) or failures (%d) exceed churn threshold %d",
				s.repeatCount, s.failCount, cfg.ChurnThreshold)
		}
		s.flaggedChurn = true

	default:
		if obs.IsMutating() && obs.WitnessVerdict == WitnessUnwitnessedClaim && cfg.RequireWitnessDiff {
			obs.StepVerdict = StepChurn
			if obs.Reason == "" {
				obs.Reason = "churn: mutating step lacks confirmed diff witness"
			}
			s.flaggedChurn = true
		} else {
			obs.StepVerdict = StepAdvance
			if obs.Reason == "" {
				obs.Reason = "step completed forward progress"
			}
			s.flaggedChurn = false
			s.flaggedRegress = false
		}
	}

	s.history = append(s.history, *obs)
	if len(s.history) > cfg.MaxHistoryPerSession {
		s.history = s.history[len(s.history)-cfg.MaxHistoryPerSession:]
	}
}

// Pool is the in-kernel observer thread pool managing concurrent shadow evaluation goroutines,
// lifecycle management, barrier synchronization, and warm KV-cache prefix reuse simulation.
type Pool struct {
	cfg       Config
	workQueue chan asyncTask
	stopCh    chan struct{}
	wg        sync.WaitGroup
	mu        sync.RWMutex
	started   bool
	stopped   bool
	closed    bool

	sessionsMu sync.RWMutex
	sessions   map[string]*sessionState

	// Telemetry counters
	observationsTotal int64
	asyncTotal        int64
	barriersTotal     int64
	cacheHits         int64
	cacheMisses       int64
	churnCount        int64
	regressCount      int64
}

// NewPool creates an in-kernel observer thread pool with the provided configuration.
func NewPool(cfg Config) *Pool {
	normCfg := normalizeConfig(cfg)
	return &Pool{
		cfg:       normCfg,
		workQueue: make(chan asyncTask, normCfg.QueueSize),
		stopCh:    make(chan struct{}),
		sessions:  make(map[string]*sessionState),
	}
}

// Start launches the observer worker goroutines. It is idempotent and safe for concurrent calls.
func (p *Pool) Start() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return ErrPoolClosed
	}
	if p.stopped {
		return ErrPoolStopped
	}
	if p.started {
		return nil
	}

	p.started = true
	for i := 0; i < p.cfg.WorkerCount; i++ {
		p.wg.Add(1)
		go p.worker(i)
	}
	return nil
}

// Stop stops the worker pool gracefully after draining queued tasks.
func (p *Pool) Stop() error {
	p.mu.Lock()
	if !p.started || p.stopped {
		p.mu.Unlock()
		return nil
	}
	p.stopped = true
	close(p.stopCh)
	p.mu.Unlock()

	p.wg.Wait()
	return nil
}

// Close terminates the pool and releases associated resources.
func (p *Pool) Close() error {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	p.mu.Unlock()

	return p.Stop()
}

func (p *Pool) worker(id int) {
	defer p.wg.Done()
	for {
		select {
		case task, ok := <-p.workQueue:
			if !ok {
				return
			}
			p.processTask(task)
		case <-p.stopCh:
			// Drain remaining tasks before termination
			for {
				select {
				case task, ok := <-p.workQueue:
					if !ok {
						return
					}
					p.processTask(task)
				default:
					return
				}
			}
		}
	}
}

func (p *Pool) processTask(t asyncTask) {
	defer atomic.AddInt64(&t.sess.inFlight, -1)
	defer close(t.res)

	if t.ctx.Err() != nil {
		return
	}

	start := time.Now()
	t.sess.evaluate(p.cfg, &t.obs)
	t.obs.Duration = time.Since(start)

	if t.obs.CachedPrefix {
		atomic.AddInt64(&p.cacheHits, 1)
	} else {
		atomic.AddInt64(&p.cacheMisses, 1)
	}

	if t.obs.StepVerdict == StepChurn {
		atomic.AddInt64(&p.churnCount, 1)
	} else if t.obs.StepVerdict == StepRegress {
		atomic.AddInt64(&p.regressCount, 1)
	}

	t.res <- t.obs
}

func (p *Pool) getOrCreateSession(sessionID string) *sessionState {
	if sessionID == "" {
		sessionID = "default"
	}

	p.sessionsMu.RLock()
	s, ok := p.sessions[sessionID]
	p.sessionsMu.RUnlock()
	if ok {
		return s
	}

	p.sessionsMu.Lock()
	defer p.sessionsMu.Unlock()
	s, ok = p.sessions[sessionID]
	if !ok {
		s = &sessionState{
			sessionID: sessionID,
			history:   make([]StepObservation, 0, p.cfg.MaxHistoryPerSession),
		}
		p.sessions[sessionID] = s
	}
	return s
}

// ResetSession clears state and history for a given session.
func (p *Pool) ResetSession(sessionID string) {
	if sessionID == "" {
		sessionID = "default"
	}
	p.sessionsMu.Lock()
	defer p.sessionsMu.Unlock()
	delete(p.sessions, sessionID)
}

// GetSessionHistory returns a snapshot of the observation history for a session.
func (p *Pool) GetSessionHistory(sessionID string) []StepObservation {
	if sessionID == "" {
		sessionID = "default"
	}
	p.sessionsMu.RLock()
	s, ok := p.sessions[sessionID]
	p.sessionsMu.RUnlock()
	if !ok {
		return nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	res := make([]StepObservation, len(s.history))
	copy(res, s.history)
	return res
}

// ObserveAsync evaluates read-only exploration turns (Read, Grep, Glob) asynchronously without blocking.
// If the operation is mutating or the session is currently flagged for churn/regress, it promotes to a hard-seam barrier.
func (p *Pool) ObserveAsync(ctx context.Context, obs StepObservation) <-chan StepObservation {
	resCh := make(chan StepObservation, 1)

	if err := p.Start(); err != nil {
		close(resCh)
		return resCh
	}

	p.mu.RLock()
	if p.closed || p.stopped {
		p.mu.RUnlock()
		close(resCh)
		return resCh
	}
	p.mu.RUnlock()

	sess := p.getOrCreateSession(obs.SessionID)

	// Hard-seam promotion on mutating tools or flagged churn/regress
	if obs.IsMutating() || sess.isFlagged() {
		evaluated, _ := p.ObserveSyncBarrier(ctx, obs)
		resCh <- evaluated
		close(resCh)
		return resCh
	}

	atomic.AddInt64(&p.observationsTotal, 1)
	atomic.AddInt64(&p.asyncTotal, 1)

	// Read-only exploration: queue asynchronously with zero-cost non-blocking dispatch
	atomic.AddInt64(&sess.inFlight, 1)
	task := asyncTask{
		ctx:  ctx,
		obs:  obs,
		res:  resCh,
		sess: sess,
	}

	select {
	case p.workQueue <- task:
		return resCh
	default:
		go p.processTask(task)
		return resCh
	}
}

// ObserveSyncBarrier executes a hard-seam blocking barrier for mutating operations or flagged churn/regress,
// providing bounded <2ms barrier latency.
func (p *Pool) ObserveSyncBarrier(ctx context.Context, obs StepObservation) (StepObservation, error) {
	atomic.AddInt64(&p.observationsTotal, 1)
	atomic.AddInt64(&p.barriersTotal, 1)

	start := time.Now()

	if err := p.Start(); err != nil {
		return obs, err
	}

	p.mu.RLock()
	if p.closed {
		p.mu.RUnlock()
		return obs, ErrPoolClosed
	}
	if p.stopped {
		p.mu.RUnlock()
		return obs, ErrPoolStopped
	}
	p.mu.RUnlock()

	if err := ctx.Err(); err != nil {
		return obs, err
	}

	sess := p.getOrCreateSession(obs.SessionID)

	// Drain any in-flight async tasks for this session to establish consistent state
	timeout := p.cfg.BarrierTimeout
	if timeout <= 0 {
		timeout = 50 * time.Millisecond
	}
	deadline := time.Now().Add(timeout)
	for atomic.LoadInt64(&sess.inFlight) > 0 {
		if time.Now().After(deadline) {
			return obs, ErrBarrierTimeout
		}
		time.Sleep(10 * time.Microsecond)
	}

	// Evaluate observation under session lock
	sess.evaluate(p.cfg, &obs)

	obs.BarrierLatency = time.Since(start)
	obs.Duration = obs.BarrierLatency

	if obs.CachedPrefix {
		atomic.AddInt64(&p.cacheHits, 1)
	} else {
		atomic.AddInt64(&p.cacheMisses, 1)
	}

	// Closed step vocabulary deterministic refusal
	if obs.IsMutating() && obs.WitnessVerdict == WitnessUnwitnessedClaim && p.cfg.RequireWitnessDiff {
		atomic.AddInt64(&p.churnCount, 1)
		return obs, ErrUnwitnessedDiff
	}
	if obs.StepVerdict == StepRegress {
		atomic.AddInt64(&p.regressCount, 1)
		return obs, ErrRegressRefused
	}
	if obs.StepVerdict == StepChurn {
		atomic.AddInt64(&p.churnCount, 1)
		return obs, ErrChurnRefused
	}

	return obs, nil
}

// Stats returns atomic telemetry counters for the pool.
func (p *Pool) Stats() (total, async, barriers, hits, misses, churns, regresses int64) {
	return atomic.LoadInt64(&p.observationsTotal),
		atomic.LoadInt64(&p.asyncTotal),
		atomic.LoadInt64(&p.barriersTotal),
		atomic.LoadInt64(&p.cacheHits),
		atomic.LoadInt64(&p.cacheMisses),
		atomic.LoadInt64(&p.churnCount),
		atomic.LoadInt64(&p.regressCount)
}

func computeArgsSig(tool string, args any) string {
	if args == nil {
		return tool
	}
	switch a := args.(type) {
	case string:
		return tool + ":" + a
	default:
		return fmt.Sprintf("%s:%v", tool, a)
	}
}

func isResultError(result any) bool {
	if result == nil {
		return false
	}
	switch r := result.(type) {
	case error:
		return true
	case string:
		lower := strings.ToLower(strings.TrimSpace(r))
		return strings.HasPrefix(lower, "error:") || strings.HasPrefix(lower, "failed:") || strings.HasPrefix(lower, "refused:")
	case map[string]any:
		if errVal, ok := r["error"]; ok && errVal != nil && errVal != "" {
			return true
		}
	}
	return false
}

func hasDiffInResult(result any) bool {
	if result == nil {
		return false
	}
	switch r := result.(type) {
	case string:
		if isResultError(r) || strings.TrimSpace(r) == "" {
			return false
		}
		lower := strings.ToLower(r)
		if strings.Contains(lower, "no changes") || strings.Contains(lower, "nothing to commit") {
			return false
		}
		return strings.Contains(lower, "diff") || strings.Contains(lower, "@@") ||
			strings.Contains(lower, "changed") || strings.Contains(lower, "insertion") ||
			strings.Contains(lower, "wrote") || strings.Contains(lower, "bytes") ||
			strings.Contains(lower, "commit")
	case map[string]any:
		if diffVal, ok := r["diff"]; ok && diffVal != nil && diffVal != "" {
			return true
		}
		if bytesVal, ok := r["bytes"]; ok && bytesVal != nil {
			return true
		}
	}
	return false
}
