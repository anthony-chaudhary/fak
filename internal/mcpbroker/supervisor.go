package mcpbroker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// MCPProcessCrashCode is the constant token representing an MCP process crash verdict.
const MCPProcessCrashCode = "MCP_PROCESS_CRASH"

// ErrMCPProcessCrash is returned when an MCP server process has crashed, terminated
// unexpectedly, or exceeded its allowable restart limit.
var ErrMCPProcessCrash = errors.New(MCPProcessCrashCode)

// IsProcessCrash returns true if err indicates an MCP process crash or termination.
func IsProcessCrash(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, ErrMCPProcessCrash) || strings.Contains(err.Error(), MCPProcessCrashCode)
}

// ProcessSupervisor manages the lifecycle, JSON-RPC communication, health checks,
// backoff restarts, and graceful termination of an external MCP server subprocess.
type ProcessSupervisor struct {
	cfg    ServerConfig
	broker *Broker

	mu        sync.RWMutex
	cmd       *exec.Cmd
	transport *StdioTransport
	cancelFn  context.CancelFunc
	procDone  chan struct{}

	running     bool
	stopping    bool
	crashed     bool
	restarts    int
	maxRestarts int

	discoveredTools []MCPTool
	lastErr         error
	parentScope     context.Context
}

// NewProcessSupervisor initializes a new supervisor for the specified server configuration.
func NewProcessSupervisor(cfg ServerConfig, broker *Broker) *ProcessSupervisor {
	maxR := 3
	if cfg.MaxRestarts != nil {
		maxR = *cfg.MaxRestarts
	}
	return &ProcessSupervisor{
		cfg:         cfg,
		broker:      broker,
		maxRestarts: maxR,
	}
}

// PID returns the process ID of the running subprocess, or 0 if not running.
func (s *ProcessSupervisor) PID() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.cmd != nil && s.cmd.Process != nil {
		return s.cmd.Process.Pid
	}
	return 0
}

// SetMaxRestarts configures the maximum restart attempts for unexpected crashes.
func (s *ProcessSupervisor) SetMaxRestarts(n int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.maxRestarts = n
}

// Start launches the MCP server subprocess, completes the JSON-RPC 2.0 handshake,
// queries available tools, registers them into the broker with namespacing, and
// starts background process health supervision.
func (s *ProcessSupervisor) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.running {
		return errors.New("mcpbroker: supervisor already running")
	}

	s.stopping = false
	s.crashed = false
	s.restarts = 0
	s.lastErr = nil
	s.parentScope = ctx

	return s.startProcessLocked(ctx)
}

// startProcessLocked creates the command, starts transport, performs handshake,
// discovers tools, and begins monitoring. Must be called with s.mu held.
func (s *ProcessSupervisor) startProcessLocked(ctx context.Context) error {
	pctx, cancel := context.WithCancel(context.Background())

	cmd := exec.CommandContext(pctx, s.cfg.Command, s.cfg.Args...)
	if len(s.cfg.Env) > 0 {
		cmd.Env = append(os.Environ(), s.cfg.Env...)
	}

	transport, err := NewStdioTransport(cmd)
	if err != nil {
		cancel()
		return err
	}

	if err := transport.Start(); err != nil {
		cancel()
		return err
	}

	// Perform handshake with server timeout if configured
	hctx := ctx
	if s.cfg.Timeout > 0 {
		var cancelH context.CancelFunc
		hctx, cancelH = context.WithTimeout(ctx, s.cfg.Timeout)
		defer cancelH()
	}

	if err := transport.Handshake(hctx); err != nil {
		_ = transport.Close()
		_ = killProcessTree(cmd)
		cancel()
		return fmt.Errorf("mcpbroker: handshake failed: %w", err)
	}

	tools, err := transport.ListTools(hctx)
	if err != nil {
		_ = transport.Close()
		_ = killProcessTree(cmd)
		cancel()
		return fmt.Errorf("mcpbroker: tool discovery failed: %w", err)
	}

	procDone := make(chan struct{})
	s.cmd = cmd
	s.transport = transport
	s.cancelFn = cancel
	s.procDone = procDone
	s.discoveredTools = tools
	s.running = true
	s.crashed = false
	s.lastErr = nil

	// Register discovered tools into the broker with namespacing
	if s.broker != nil {
		handler := func(callCtx context.Context, req CallRequest) (*CallResponse, error) {
			_, rawTool, ok := ParseNamespacedTool(req.Tool)
			if !ok {
				rawTool = req.Tool
			}
			resp, err := s.CallTool(callCtx, rawTool, req.Arguments)
			if resp != nil {
				resp.Tool = req.Tool
			}
			return resp, err
		}
		_, _ = s.broker.RegisterServerTools(s.cfg.ID, tools, handler)
	}

	if s.parentScope != nil && s.parentScope.Done() != nil {
		go func(scope context.Context, done chan struct{}) {
			select {
			case <-scope.Done():
				_ = s.Stop()
			case <-done:
			}
		}(s.parentScope, procDone)
	}

	go s.monitorProcess(cmd, transport, procDone)

	return nil
}

// monitorProcess monitors the subprocess until exit and performs backoff restarts
// on unexpected crashes up to maxRestarts times.
func (s *ProcessSupervisor) monitorProcess(cmd *exec.Cmd, transport *StdioTransport, procDone chan struct{}) {
	defer close(procDone)
	waitErr := cmd.Wait()

	s.mu.Lock()
	defer s.mu.Unlock()

	// If deliberately stopped, no recovery needed
	if s.stopping {
		return
	}

	// If s.cmd has already been replaced by another instance, exit
	if s.cmd != cmd {
		return
	}

	s.running = false
	_ = transport.Close()

	// Handle crash with bounded backoff restart
	for s.restarts < s.maxRestarts {
		s.restarts++
		backoff := time.Duration(s.restarts*50) * time.Millisecond

		s.mu.Unlock()
		time.Sleep(backoff)
		s.mu.Lock()

		if s.stopping {
			return
		}

		scope := s.parentScope
		if scope == nil {
			scope = context.Background()
		}
		if scope.Err() != nil {
			return
		}

		ctx, cancel := context.WithTimeout(scope, 10*time.Second)
		err := s.startProcessLocked(ctx)
		cancel()
		if err == nil {
			return // Successfully restarted
		}
	}

	// Exhausted restart attempts: mark process as crashed
	s.crashed = true
	s.running = false
	if waitErr != nil {
		s.lastErr = fmt.Errorf("%w: %v", ErrMCPProcessCrash, waitErr)
	} else {
		s.lastErr = ErrMCPProcessCrash
	}
}

// CallTool invokes a tool on the supervised MCP server. If the process is crashed,
// it immediately returns ErrMCPProcessCrash.
func (s *ProcessSupervisor) CallTool(ctx context.Context, toolName string, args json.RawMessage) (*CallResponse, error) {
	s.mu.RLock()
	if s.crashed {
		err := s.lastErr
		if err == nil {
			err = ErrMCPProcessCrash
		}
		s.mu.RUnlock()
		return &CallResponse{
			Tool:         toolName,
			ServerID:     s.cfg.ID,
			IsError:      true,
			ErrorMessage: err.Error(),
		}, err
	}
	if !s.running || s.transport == nil {
		s.mu.RUnlock()
		return &CallResponse{
			Tool:         toolName,
			ServerID:     s.cfg.ID,
			IsError:      true,
			ErrorMessage: ErrMCPProcessCrash.Error(),
		}, ErrMCPProcessCrash
	}
	tr := s.transport
	s.mu.RUnlock()

	resp, err := tr.CallTool(ctx, toolName, args)
	if err != nil {
		if errors.Is(err, ErrMCPProcessCrash) || IsProcessCrash(err) {
			s.mu.Lock()
			s.crashed = true
			s.lastErr = ErrMCPProcessCrash
			s.mu.Unlock()
			return &CallResponse{
				Tool:         toolName,
				ServerID:     s.cfg.ID,
				IsError:      true,
				ErrorMessage: ErrMCPProcessCrash.Error(),
			}, ErrMCPProcessCrash
		}
		return resp, err
	}

	if resp != nil {
		resp.ServerID = s.cfg.ID
	}
	return resp, nil
}

// Ping checks process responsiveness via JSON-RPC ping. Returns ErrMCPProcessCrash
// if the process is crashed or fails to respond.
func (s *ProcessSupervisor) Ping(ctx context.Context) error {
	s.mu.RLock()
	if s.crashed {
		s.mu.RUnlock()
		return ErrMCPProcessCrash
	}
	if !s.running || s.transport == nil {
		s.mu.RUnlock()
		return ErrMCPProcessCrash
	}
	tr := s.transport
	s.mu.RUnlock()

	if err := tr.Ping(ctx); err != nil {
		s.mu.Lock()
		s.crashed = true
		s.lastErr = ErrMCPProcessCrash
		s.mu.Unlock()
		return ErrMCPProcessCrash
	}
	return nil
}

// Stop initiates graceful shutdown: sends cancellation, attempts graceful process
// termination with a 3-second grace timeout, and forces SIGKILL if necessary.
func (s *ProcessSupervisor) Stop() error {
	s.mu.Lock()
	if s.stopping || (!s.running && !s.crashed) {
		s.mu.Unlock()
		return nil
	}

	s.stopping = true
	s.running = false
	cmd := s.cmd
	cancelFn := s.cancelFn
	tr := s.transport
	procDone := s.procDone
	s.mu.Unlock()

	if cancelFn != nil {
		cancelFn()
	}

	if tr != nil {
		_ = tr.Close()
	}

	if cmd != nil && cmd.Process != nil {
		_ = terminateProcessTree(cmd, 3*time.Second)
	}

	if procDone != nil {
		select {
		case <-procDone:
		case <-time.After(3 * time.Second):
		}
	}

	return nil
}

// Restarts returns the count of unexpected crashes recovered by the supervisor.
func (s *ProcessSupervisor) Restarts() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.restarts
}

// IsCrashed returns true if the process is in a crashed, non-recoverable state.
func (s *ProcessSupervisor) IsCrashed() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.crashed
}

// IsRunning returns true if the process is currently active and healthy.
func (s *ProcessSupervisor) IsRunning() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.running && !s.crashed
}

// DiscoveredTools returns the list of tools discovered from the server.
func (s *ProcessSupervisor) DiscoveredTools() []MCPTool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cp := make([]MCPTool, len(s.discoveredTools))
	copy(cp, s.discoveredTools)
	return cp
}

// Stderr returns captured stderr bytes from the child process.
func (s *ProcessSupervisor) Stderr() []byte {
	s.mu.RLock()
	tr := s.transport
	s.mu.RUnlock()
	if tr == nil {
		return nil
	}
	return tr.Stderr()
}
