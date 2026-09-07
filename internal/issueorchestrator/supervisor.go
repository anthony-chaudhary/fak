package issueorchestrator

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// Default supervisor thresholds.
const (
	DefaultSupervisorInitialDeadline = 15 * time.Minute
	DefaultSupervisorExtendIncrement = 10 * time.Minute
	DefaultSupervisorMaxDeadline     = 60 * time.Minute
	DefaultSupervisorSilenceWarn     = 10 * time.Minute
	DefaultSupervisorSilenceReap     = 25 * time.Minute
)

// SupervisorVerdict describes the outcome of evaluating a worker process.
type SupervisorVerdict string

const (
	SupervisorActive      SupervisorVerdict = "SUPERVISOR_ACTIVE"
	SupervisorExtended    SupervisorVerdict = "SUPERVISOR_EXTENDED"
	SupervisorWarnSilence SupervisorVerdict = "SUPERVISOR_WARN_SILENCE"
	SupervisorReapDead    SupervisorVerdict = "SUPERVISOR_REAP_DEAD"
	SupervisorReapWedged  SupervisorVerdict = "SUPERVISOR_REAP_WEDGED"

	// Uppercase aliases matching the specification tokens.
	SUPERVISOR_ACTIVE       = SupervisorActive
	SUPERVISOR_EXTENDED     = SupervisorExtended
	SUPERVISOR_WARN_SILENCE = SupervisorWarnSilence
	SUPERVISOR_REAP_DEAD    = SupervisorReapDead
	SUPERVISOR_REAP_WEDGED  = SupervisorReapWedged
)

// HeartbeatResult captures multi-channel health and progress signals sampled by the supervisor.
type HeartbeatResult struct {
	Timestamp     time.Time `json:"timestamp"`
	LogAppended   bool      `json:"log_appended"`
	LogBytes      int64     `json:"log_bytes"`
	LogMTime      time.Time `json:"log_mtime,omitempty"`
	WorktreeDelta bool      `json:"worktree_delta"`
	WorktreeMTime time.Time `json:"worktree_mtime,omitempty"`
	LeaseUpdated  bool      `json:"lease_updated"`
	LeaseMTime    time.Time `json:"lease_mtime,omitempty"`
	ChildActive   bool      `json:"child_active"`
	ProcessAlive  bool      `json:"process_alive"`
	AnyProgress   bool      `json:"any_progress"`
}

// WorkerSupervisorConfig configures the adaptive process supervisor.
type WorkerSupervisorConfig struct {
	PID             int
	WorktreeDir     string
	LogFile         string
	InitialDeadline time.Duration // default 15m
	ExtendIncrement time.Duration // default 10m
	MaxDeadline     time.Duration // default 60m
	SilenceWarn     time.Duration // default 10m
	SilenceReap     time.Duration // default 25m
	LivenessProbe   func(pid int) bool
	ChildProbe      func(pid int) bool
	NowFunc         func() time.Time
	ReaperFunc      func(pid int) error
}

// WorkerSupervisor tracks a worker process, senses multi-channel heartbeats,
// dynamically extends deadlines upon witnessed progress, emits soft advisory
// warnings on silence, and reaps verifiably dead or wedged processes.
type WorkerSupervisor struct {
	mu sync.RWMutex

	cfg WorkerSupervisorConfig

	startTime       time.Time
	currentDeadline time.Time
	lastHeartbeat   time.Time
	lastVerdict     SupervisorVerdict

	lastLogSize       int64
	lastLogMTime      time.Time
	lastWorktreeMTime time.Time
	lastWorktreeSize  int64
	lastWorktreeCount int
	lastLeaseMTime    time.Time
	lastLeaseSize     int64

	warnedSilence bool
	reaped        bool
	warnings      []string
}

// defaultReaper terminates the target process using standard os.Process Kill.
func defaultReaper(pid int) error {
	if pid <= 0 {
		return nil
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		return err
	}
	return proc.Kill()
}

// formatDuration formats time.Duration as compact string e.g. "10m", "25m", "60m".
func formatDuration(d time.Duration) string {
	if d >= time.Hour && d%time.Hour == 0 {
		return fmt.Sprintf("%dh", int(d.Hours()))
	}
	if d >= time.Minute && d%time.Minute == 0 {
		return fmt.Sprintf("%dm", int(d.Minutes()))
	}
	if d >= time.Second && d%time.Second == 0 {
		return fmt.Sprintf("%ds", int(d.Seconds()))
	}
	return d.String()
}

// scanWorktree scans root for file count, cumulative size, and latest modification time.
func scanWorktree(root string) (time.Time, int64, int, error) {
	if root == "" {
		return time.Time{}, 0, 0, nil
	}
	var (
		latest    time.Time
		totalSize int64
		count     int
	)
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		count++
		totalSize += info.Size()
		mt := info.ModTime()
		if mt.After(latest) {
			latest = mt
		}
		return nil
	})
	return latest, totalSize, count, err
}

// scanLeases checks lease.json and .dos/lane-journal.jsonl for updates.
func scanLeases(worktreeDir string) (time.Time, int64) {
	if worktreeDir == "" {
		return time.Time{}, 0
	}
	var (
		latest    time.Time
		totalSize int64
	)
	candidates := []string{
		filepath.Join(worktreeDir, "lease.json"),
		filepath.Join(worktreeDir, ".dos", "lane-journal.jsonl"),
		filepath.Join(worktreeDir, "lane-journal.jsonl"),
		filepath.Join(worktreeDir, ".dos", "lease.json"),
	}
	for _, p := range candidates {
		if fi, err := os.Stat(p); err == nil {
			totalSize += fi.Size()
			if fi.ModTime().After(latest) {
				latest = fi.ModTime()
			}
		}
	}
	return latest, totalSize
}

// NewWorkerSupervisor initializes a WorkerSupervisor with sane defaults.
func NewWorkerSupervisor(cfg WorkerSupervisorConfig) *WorkerSupervisor {
	if cfg.InitialDeadline <= 0 {
		cfg.InitialDeadline = DefaultSupervisorInitialDeadline
	}
	if cfg.ExtendIncrement <= 0 {
		cfg.ExtendIncrement = DefaultSupervisorExtendIncrement
	}
	if cfg.MaxDeadline <= 0 {
		cfg.MaxDeadline = DefaultSupervisorMaxDeadline
	}
	if cfg.SilenceWarn <= 0 {
		cfg.SilenceWarn = DefaultSupervisorSilenceWarn
	}
	if cfg.SilenceReap <= 0 {
		cfg.SilenceReap = DefaultSupervisorSilenceReap
	}
	if cfg.LivenessProbe == nil {
		cfg.LivenessProbe = processalive.Check
	}
	if cfg.NowFunc == nil {
		cfg.NowFunc = time.Now
	}
	if cfg.ReaperFunc == nil {
		cfg.ReaperFunc = defaultReaper
	}

	now := cfg.NowFunc()
	s := &WorkerSupervisor{
		cfg:             cfg,
		startTime:       now,
		currentDeadline: now.Add(cfg.InitialDeadline),
		lastHeartbeat:   now,
		lastVerdict:     SupervisorActive,
		warnings:        make([]string, 0),
	}

	// Capture initial baseline channel states so only new mutations trigger progress
	if cfg.LogFile != "" {
		if fi, err := os.Stat(cfg.LogFile); err == nil {
			s.lastLogSize = fi.Size()
			s.lastLogMTime = fi.ModTime()
		}
	}
	if cfg.WorktreeDir != "" {
		mt, size, count, _ := scanWorktree(cfg.WorktreeDir)
		s.lastWorktreeMTime = mt
		s.lastWorktreeSize = size
		s.lastWorktreeCount = count
	}
	s.lastLeaseMTime, s.lastLeaseSize = scanLeases(cfg.WorktreeDir)

	return s
}

func (s *WorkerSupervisor) nowLocked() time.Time {
	if s.cfg.NowFunc != nil {
		return s.cfg.NowFunc()
	}
	return time.Now()
}

func (s *WorkerSupervisor) reapLocked() error {
	if s.reaped {
		return nil
	}
	s.reaped = true
	if s.cfg.ReaperFunc != nil {
		return s.cfg.ReaperFunc(s.cfg.PID)
	}
	return defaultReaper(s.cfg.PID)
}

func (s *WorkerSupervisor) sampleHeartbeatLocked(now time.Time) (HeartbeatResult, error) {
	res := HeartbeatResult{
		Timestamp: now,
	}

	// 1. Log modification times / file growth
	if s.cfg.LogFile != "" {
		if fi, err := os.Stat(s.cfg.LogFile); err == nil {
			res.LogBytes = fi.Size()
			res.LogMTime = fi.ModTime()
			if fi.Size() > s.lastLogSize || fi.ModTime().After(s.lastLogMTime) {
				res.LogAppended = true
				s.lastLogSize = fi.Size()
				s.lastLogMTime = fi.ModTime()
			}
		}
	}

	// 2. Working-tree file modifications (mtime deltas & size/count changes)
	if s.cfg.WorktreeDir != "" {
		mt, size, count, _ := scanWorktree(s.cfg.WorktreeDir)
		res.WorktreeMTime = mt
		if mt.After(s.lastWorktreeMTime) || size != s.lastWorktreeSize || count != s.lastWorktreeCount {
			res.WorktreeDelta = true
			s.lastWorktreeMTime = mt
			s.lastWorktreeSize = size
			s.lastWorktreeCount = count
		}
	}

	// 3. lease.json or .dos/lane-journal.jsonl updates
	leaseMt, leaseSize := scanLeases(s.cfg.WorktreeDir)
	res.LeaseMTime = leaseMt
	if leaseMt.After(s.lastLeaseMTime) || leaseSize != s.lastLeaseSize {
		res.LeaseUpdated = true
		s.lastLeaseMTime = leaseMt
		s.lastLeaseSize = leaseSize
	}

	// 4. Child compiler/test process activity or process liveness
	if s.cfg.LivenessProbe != nil {
		res.ProcessAlive = s.cfg.LivenessProbe(s.cfg.PID)
	} else {
		res.ProcessAlive = processalive.Check(s.cfg.PID)
	}

	if s.cfg.ChildProbe != nil {
		res.ChildActive = s.cfg.ChildProbe(s.cfg.PID)
	}

	res.AnyProgress = res.LogAppended || res.WorktreeDelta || res.LeaseUpdated || res.ChildActive
	return res, nil
}

// SampleHeartbeat samples multi-channel heartbeats and updates internal telemetry.
func (s *WorkerSupervisor) SampleHeartbeat() (HeartbeatResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.nowLocked()
	res, err := s.sampleHeartbeatLocked(now)
	if res.AnyProgress {
		s.lastHeartbeat = now
		s.warnedSilence = false
	}
	return res, err
}

// EvaluateStatus evaluates the current status of the worker process, dynamically
// extending deadlines when progress is observed, logging soft advisory warnings
// on silence, and reaping dead or wedged processes.
func (s *WorkerSupervisor) EvaluateStatus() SupervisorVerdict {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := s.nowLocked()

	// 1. If already reaped, return cached terminal verdict
	if s.reaped {
		return s.lastVerdict
	}

	// 2. Fail-safe check: Process liveness probe
	alive := false
	if s.cfg.LivenessProbe != nil {
		alive = s.cfg.LivenessProbe(s.cfg.PID)
	} else {
		alive = processalive.Check(s.cfg.PID)
	}

	if !alive {
		_ = s.reapLocked()
		s.lastVerdict = SupervisorReapDead
		return SupervisorReapDead
	}

	// 3. Multi-channel heartbeat sample
	hb, _ := s.sampleHeartbeatLocked(now)

	// 4. Progress observed -> update lastHeartbeat, reset silence warning, extend deadline if needed
	if hb.AnyProgress {
		s.lastHeartbeat = now
		s.warnedSilence = false

		maxDeadline := s.startTime.Add(s.cfg.MaxDeadline)
		// Dynamic adaptive deadline extension: extend by +10m when progress occurs near/past deadline
		if now.Add(s.cfg.ExtendIncrement).After(s.currentDeadline) || now.After(s.currentDeadline) || now.Sub(s.startTime) >= s.cfg.InitialDeadline {
			if s.currentDeadline.Before(maxDeadline) {
				newDeadline := s.currentDeadline.Add(s.cfg.ExtendIncrement)
				if newDeadline.After(maxDeadline) {
					newDeadline = maxDeadline
				}
				s.currentDeadline = newDeadline
			}
		}

		if s.currentDeadline.After(s.startTime.Add(s.cfg.InitialDeadline)) {
			s.lastVerdict = SupervisorExtended
			return SupervisorExtended
		}
		s.lastVerdict = SupervisorActive
		return SupervisorActive
	}

	// 5. Silence / Lack of progress evaluation
	silence := now.Sub(s.lastHeartbeat)

	// Fail-safe reaping: silent past extended grace limit (> 25m with zero child activity)
	if silence >= s.cfg.SilenceReap && !hb.ChildActive {
		_ = s.reapLocked()
		s.lastVerdict = SupervisorReapWedged
		return SupervisorReapWedged
	}

	// Hard ceiling: elapsed duration exceeds MaxDeadline (e.g. 60m)
	if now.After(s.startTime.Add(s.cfg.MaxDeadline)) {
		_ = s.reapLocked()
		s.lastVerdict = SupervisorReapWedged
		return SupervisorReapWedged
	}

	// Soft advisory warnings: silence > 10m
	if silence >= s.cfg.SilenceWarn {
		msg := fmt.Sprintf("WARN [supervisor]: worker PID %d silent for %s; monitoring activity before reaping", s.cfg.PID, formatDuration(s.cfg.SilenceWarn))
		if !s.warnedSilence {
			log.Print(msg)
			s.warnedSilence = true
		}
		s.warnings = append(s.warnings, msg)
		s.lastVerdict = SupervisorWarnSilence
		return SupervisorWarnSilence
	}

	if s.currentDeadline.After(s.startTime.Add(s.cfg.InitialDeadline)) {
		s.lastVerdict = SupervisorExtended
		return SupervisorExtended
	}

	s.lastVerdict = SupervisorActive
	return SupervisorActive
}

// CurrentDeadline returns the current effective deadline for the supervised process.
func (s *WorkerSupervisor) CurrentDeadline() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentDeadline
}

// LastHeartbeat returns the timestamp of the last observed heartbeat/progress.
func (s *WorkerSupervisor) LastHeartbeat() time.Time {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastHeartbeat
}

// LastVerdict returns the most recent evaluation verdict.
func (s *WorkerSupervisor) LastVerdict() SupervisorVerdict {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.lastVerdict
}

// Reaped reports whether the process was terminated by the supervisor.
func (s *WorkerSupervisor) Reaped() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.reaped
}

// Warnings returns all soft advisory warnings emitted by this supervisor.
func (s *WorkerSupervisor) Warnings() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	res := make([]string, len(s.warnings))
	copy(res, s.warnings)
	return res
}

// ExtendDeadline explicitly extends the deadline by duration d (up to MaxDeadline).
func (s *WorkerSupervisor) ExtendDeadline(d time.Duration) time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	maxDeadline := s.startTime.Add(s.cfg.MaxDeadline)
	newDeadline := s.currentDeadline.Add(d)
	if newDeadline.After(maxDeadline) {
		newDeadline = maxDeadline
	}
	s.currentDeadline = newDeadline
	return s.currentDeadline
}

// SetChildProbe overrides or configures the child process activity probe.
func (s *WorkerSupervisor) SetChildProbe(fn func(pid int) bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cfg.ChildProbe = fn
}

// Reap manually triggers process reaping.
func (s *WorkerSupervisor) Reap() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.reapLocked()
}

// Config returns a copy of the supervisor's configuration.
func (s *WorkerSupervisor) Config() WorkerSupervisorConfig {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}
