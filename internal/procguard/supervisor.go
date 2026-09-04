package procguard

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// Common process supervisor errors.
var (
	ErrInvalidPID       = errors.New("procguard: invalid pid")
	ErrEmptySession     = errors.New("procguard: empty session id")
	ErrSupervisorClosed = errors.New("procguard: supervisor is closed")
	ErrPortConflict     = errors.New("procguard: port already allocated to another session")
	ErrInvalidPort      = errors.New("procguard: invalid port number")
)

// TrackedProcess records the execution metadata and lifecycle state of a supervised process.
type TrackedProcess struct {
	PID       int       `json:"pid"`
	SessionID string    `json:"session_id"`
	PGID      int       `json:"pgid,omitempty"`
	StartTime time.Time `json:"start_time"`
	Deadline  time.Time `json:"deadline,omitempty"`
	Cmdline   string    `json:"cmdline,omitempty"`
	Children  []int     `json:"children,omitempty"`
	Ports     []int     `json:"ports,omitempty"`
	TimedOut  bool      `json:"timed_out,omitempty"`
}

// SessionRecord aggregates all process and resource tracking state for a session.
type SessionRecord struct {
	SessionID string
	RootPIDs  map[int]bool
	ChildPIDs map[int]bool
	Ports     map[int]bool
	Deadline  time.Time
	CreatedAt time.Time
	TimedOut  bool
}

// SupervisorOption configures a ProcessSupervisor.
type SupervisorOption func(*ProcessSupervisor)

// WithTickInterval sets the watchdog inspection cadence for deadline and orphan checks.
func WithTickInterval(d time.Duration) SupervisorOption {
	return func(s *ProcessSupervisor) {
		if d > 0 {
			s.tickInterval = d
		}
	}
}

// WithDefaultDeadline sets a default execution duration for newly tracked sessions.
func WithDefaultDeadline(d time.Duration) SupervisorOption {
	return func(s *ProcessSupervisor) {
		s.defaultDeadline = d
	}
}

// WithKillFunc overrides the process termination function (defaults to KillPID).
func WithKillFunc(fn func(pid int) (bool, string)) SupervisorOption {
	return func(s *ProcessSupervisor) {
		s.killFunc = fn
	}
}

// WithAliveFunc overrides the liveness probe function (defaults to processalive.Check).
func WithAliveFunc(fn func(pid int) bool) SupervisorOption {
	return func(s *ProcessSupervisor) {
		s.aliveFunc = fn
	}
}

// WithDescendantFunc overrides the descendant discovery walk (defaults to osDescendantPIDs).
func WithDescendantFunc(fn func(pid int) ([]int, string)) SupervisorOption {
	return func(s *ProcessSupervisor) {
		s.descendantFunc = fn
	}
}

// WithOnTimeout registers an advisory callback invoked when a session reaches its deadline.
func WithOnTimeout(fn func(sessionID string, pids []int)) SupervisorOption {
	return func(s *ProcessSupervisor) {
		s.onTimeout = fn
	}
}

// ProcessSupervisor governs tool process trees, enforces execution deadlines,
// reaps orphaned descendant subtrees, and maintains port isolation under high volume.
type ProcessSupervisor struct {
	mu              sync.RWMutex
	sessions        map[string]*SessionRecord
	processes       map[int]*TrackedProcess
	ports           map[int]string // port -> sessionID
	tickInterval    time.Duration
	defaultDeadline time.Duration
	killFunc        func(pid int) (bool, string)
	aliveFunc       func(pid int) bool
	descendantFunc  func(pid int) ([]int, string)
	onTimeout       func(sessionID string, pids []int)
	stopCh          chan struct{}
	doneCh          chan struct{}
	closed          bool
}

// NewProcessSupervisor initializes a new ProcessSupervisor and starts its background watchdog.
func NewProcessSupervisor(opts ...SupervisorOption) *ProcessSupervisor {
	s := &ProcessSupervisor{
		sessions:       make(map[string]*SessionRecord),
		processes:      make(map[int]*TrackedProcess),
		ports:          make(map[int]string),
		tickInterval:   50 * time.Millisecond,
		killFunc:       KillPID,
		aliveFunc:      processalive.Check,
		descendantFunc: osDescendantPIDs,
		stopCh:         make(chan struct{}),
		doneCh:         make(chan struct{}),
	}
	for _, opt := range opts {
		opt(s)
	}
	go s.watchdogLoop()
	return s
}

func (s *ProcessSupervisor) watchdogLoop() {
	defer close(s.doneCh)
	ticker := time.NewTicker(s.tickInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.stopCh:
			return
		case <-ticker.C:
			s.checkDeadlinesAndOrphans()
		}
	}
}

func (s *ProcessSupervisor) checkDeadlinesAndOrphans() {
	s.mu.Lock()
	now := time.Now()
	var expiredSessions []string
	var orphanSessions []string

	for id, sess := range s.sessions {
		if !sess.Deadline.IsZero() && now.After(sess.Deadline) {
			sess.TimedOut = true
			expiredSessions = append(expiredSessions, id)
			continue
		}

		allRootsDead := true
		for root := range sess.RootPIDs {
			if s.aliveFunc != nil && s.aliveFunc(root) {
				allRootsDead = false
				break
			}
		}
		if allRootsDead && len(sess.RootPIDs) > 0 {
			hasLivingChildren := false
			for child := range sess.ChildPIDs {
				if s.aliveFunc != nil && s.aliveFunc(child) {
					hasLivingChildren = true
					break
				}
			}
			if hasLivingChildren {
				orphanSessions = append(orphanSessions, id)
			}
		}
	}
	s.mu.Unlock()

	for _, id := range expiredSessions {
		reaped, _ := s.ReapSession(id)
		if s.onTimeout != nil {
			s.onTimeout(id, reaped)
		}
	}

	for _, id := range orphanSessions {
		_, _ = s.ReapSession(id)
	}
}

// TrackProcess registers a root PID under sessionID.
func (s *ProcessSupervisor) TrackProcess(sessionID string, pid int) {
	s.trackProcess(sessionID, pid, time.Time{})
}

// TrackProcessWithDeadline registers a root PID under sessionID with an explicit execution deadline.
func (s *ProcessSupervisor) TrackProcessWithDeadline(sessionID string, pid int, deadline time.Time) {
	s.trackProcess(sessionID, pid, deadline)
}

// TrackProcessTimeout registers a root PID under sessionID with a timeout duration.
func (s *ProcessSupervisor) TrackProcessTimeout(sessionID string, pid int, timeout time.Duration) {
	var deadline time.Time
	if timeout > 0 {
		deadline = time.Now().Add(timeout)
	}
	s.trackProcess(sessionID, pid, deadline)
}

func (s *ProcessSupervisor) trackProcess(sessionID string, pid int, deadline time.Time) {
	if sessionID == "" || pid <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}

	session, exists := s.sessions[sessionID]
	if !exists {
		session = &SessionRecord{
			SessionID: sessionID,
			RootPIDs:  make(map[int]bool),
			ChildPIDs: make(map[int]bool),
			Ports:     make(map[int]bool),
			CreatedAt: time.Now(),
		}
		if s.defaultDeadline > 0 {
			session.Deadline = time.Now().Add(s.defaultDeadline)
		}
		s.sessions[sessionID] = session
	}

	if !deadline.IsZero() {
		session.Deadline = deadline
	} else if !session.Deadline.IsZero() {
		deadline = session.Deadline
	}

	session.RootPIDs[pid] = true

	pgid := getPGID(pid)
	tracked := &TrackedProcess{
		PID:       pid,
		SessionID: sessionID,
		PGID:      pgid,
		StartTime: time.Now(),
		Deadline:  deadline,
	}

	if s.descendantFunc != nil {
		if children, _ := s.descendantFunc(pid); len(children) > 0 {
			tracked.Children = append([]int(nil), children...)
			for _, ch := range children {
				session.ChildPIDs[ch] = true
			}
		}
	}

	s.processes[pid] = tracked
}

// TrackChildPID associates a child or grandchild PID directly with an existing session.
func (s *ProcessSupervisor) TrackChildPID(sessionID string, childPID int) {
	if sessionID == "" || childPID <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}

	session, exists := s.sessions[sessionID]
	if !exists {
		session = &SessionRecord{
			SessionID: sessionID,
			RootPIDs:  make(map[int]bool),
			ChildPIDs: make(map[int]bool),
			Ports:     make(map[int]bool),
			CreatedAt: time.Now(),
		}
		s.sessions[sessionID] = session
	}
	session.ChildPIDs[childPID] = true

	pgid := getPGID(childPID)
	s.processes[childPID] = &TrackedProcess{
		PID:       childPID,
		SessionID: sessionID,
		PGID:      pgid,
		StartTime: time.Now(),
		Deadline:  session.Deadline,
	}
}

// TrackPort registers port as actively allocated to sessionID, enforcing port isolation.
func (s *ProcessSupervisor) TrackPort(sessionID string, port int) error {
	if sessionID == "" {
		return ErrEmptySession
	}
	if port <= 0 || port > 65535 {
		return ErrInvalidPort
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrSupervisorClosed
	}

	if existingSession, taken := s.ports[port]; taken {
		if existingSession != sessionID {
			return fmt.Errorf("%w: port %d is held by session %s", ErrPortConflict, port, existingSession)
		}
		return nil
	}

	session, exists := s.sessions[sessionID]
	if !exists {
		session = &SessionRecord{
			SessionID: sessionID,
			RootPIDs:  make(map[int]bool),
			ChildPIDs: make(map[int]bool),
			Ports:     make(map[int]bool),
			CreatedAt: time.Now(),
		}
		s.sessions[sessionID] = session
	}

	s.ports[port] = sessionID
	session.Ports[port] = true
	return nil
}

// ReleasePort releases a previously tracked port reservation for sessionID.
func (s *ProcessSupervisor) ReleasePort(sessionID string, port int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if owner, ok := s.ports[port]; ok && owner == sessionID {
		delete(s.ports, port)
	}
	if session, ok := s.sessions[sessionID]; ok {
		delete(session.Ports, port)
	}
}

// SessionPorts returns all ports currently allocated to sessionID.
func (s *ProcessSupervisor) SessionPorts(sessionID string) []int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	session, ok := s.sessions[sessionID]
	if !ok || len(session.Ports) == 0 {
		return nil
	}
	ports := make([]int, 0, len(session.Ports))
	for p := range session.Ports {
		ports = append(ports, p)
	}
	sort.Ints(ports)
	return ports
}

// ReapSession terminates all processes, process groups, and descendants associated with sessionID,
// releases its port allocations, and clears it from the supervisor table.
func (s *ProcessSupervisor) ReapSession(sessionID string) ([]int, error) {
	s.mu.Lock()
	sess, ok := s.sessions[sessionID]
	if !ok {
		s.mu.Unlock()
		return nil, nil
	}

	candidatePIDs := make(map[int]bool)
	pgids := make(map[int]bool)

	for pid := range sess.RootPIDs {
		candidatePIDs[pid] = true
		if proc, exists := s.processes[pid]; exists && proc.PGID > 1 {
			pgids[proc.PGID] = true
		}
	}
	for pid := range sess.ChildPIDs {
		candidatePIDs[pid] = true
	}

	if s.descendantFunc != nil {
		for root := range sess.RootPIDs {
			if descendants, _ := s.descendantFunc(root); len(descendants) > 0 {
				for _, d := range descendants {
					candidatePIDs[d] = true
				}
			}
		}
	}

	for port := range sess.Ports {
		delete(s.ports, port)
	}

	delete(s.sessions, sessionID)
	for pid := range candidatePIDs {
		delete(s.processes, pid)
	}
	s.mu.Unlock()

	for pgid := range pgids {
		_ = killProcessGroup(pgid)
	}

	var reaped []int
	var errs []error

	for pid := range candidatePIDs {
		if pid <= 0 {
			continue
		}

		if s.aliveFunc != nil && !s.aliveFunc(pid) {
			reaped = append(reaped, pid)
			continue
		}

		if s.killFunc != nil {
			ok, detail := s.killFunc(pid)
			reapChildZombie(pid)
			if !ok && s.aliveFunc != nil && s.aliveFunc(pid) {
				errs = append(errs, fmt.Errorf("failed to reap pid %d: %s", pid, detail))
			} else {
				reaped = append(reaped, pid)
			}
		}
	}

	sort.Ints(reaped)
	if len(errs) > 0 {
		return reaped, errors.Join(errs...)
	}
	return reaped, nil
}

// ActiveProcesses returns a snapshot of all currently active/alive tracked processes.
func (s *ProcessSupervisor) ActiveProcesses() []TrackedProcess {
	s.mu.Lock()
	defer s.mu.Unlock()

	var active []TrackedProcess
	for pid, proc := range s.processes {
		if s.aliveFunc != nil && !s.aliveFunc(pid) {
			if sess, ok := s.sessions[proc.SessionID]; ok {
				delete(sess.RootPIDs, pid)
				delete(sess.ChildPIDs, pid)
			}
			delete(s.processes, pid)
			continue
		}

		cp := *proc
		if len(proc.Children) > 0 {
			cp.Children = append([]int(nil), proc.Children...)
		}
		if sess, ok := s.sessions[proc.SessionID]; ok && len(sess.Ports) > 0 {
			cp.Ports = make([]int, 0, len(sess.Ports))
			for p := range sess.Ports {
				cp.Ports = append(cp.Ports, p)
			}
			sort.Ints(cp.Ports)
		}
		active = append(active, cp)
	}

	sort.Slice(active, func(i, j int) bool {
		return active[i].PID < active[j].PID
	})
	return active
}

// hasCommandContext reports whether cmd was initialized with a non-nil context.
func hasCommandContext(cmd *exec.Cmd) bool {
	if cmd == nil {
		return false
	}
	val := reflect.ValueOf(cmd)
	if val.Kind() == reflect.Pointer && !val.IsNil() {
		f := val.Elem().FieldByName("ctx")
		if f.IsValid() && !f.IsNil() {
			return true
		}
	}
	return false
}

// ConfigureCommand attaches headless execution attributes (null stdin, process group,
// popup suppression) and wires tree cancellation on the command if created with CommandContext.
func (s *ProcessSupervisor) ConfigureCommand(cmd *exec.Cmd, sessionID string) {
	if cmd.Stdin == nil {
		if devNull, err := os.Open(os.DevNull); err == nil {
			cmd.Stdin = devNull
		}
	}
	configureSysProcAttr(cmd)

	if hasCommandContext(cmd) {
		cmd.Cancel = func() error {
			if cmd.Process == nil {
				return nil
			}
			_, err := s.ReapSession(sessionID)
			return err
		}
	}
}

// ExecuteCommand starts and supervises cmd, tracking it under sessionID, waiting for
// completion or context cancellation, and ensuring any orphaned descendants are reaped.
func (s *ProcessSupervisor) ExecuteCommand(ctx context.Context, sessionID string, cmd *exec.Cmd) error {
	s.ConfigureCommand(cmd, sessionID)

	if err := cmd.Start(); err != nil {
		return err
	}

	s.TrackProcess(sessionID, cmd.Process.Pid)

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	select {
	case <-ctx.Done():
		_, _ = s.ReapSession(sessionID)
		return ctx.Err()
	case err := <-done:
		_, _ = s.ReapSession(sessionID)
		return err
	}
}

// Close stops the watchdog goroutine and reaps any remaining sessions.
func (s *ProcessSupervisor) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.stopCh)
	s.mu.Unlock()

	<-s.doneCh

	s.mu.Lock()
	sessionIDs := make([]string, 0, len(s.sessions))
	for id := range s.sessions {
		sessionIDs = append(sessionIDs, id)
	}
	s.mu.Unlock()

	for _, id := range sessionIDs {
		_, _ = s.ReapSession(id)
	}
	return nil
}
