// Package extensionfault supervises optional extension subprocesses behind bounded
// startup/call deadlines and a per-extension circuit breaker.
package extensionfault

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

var (
	ErrQuarantined = errors.New("extension quarantined")
	ErrUnavailable = errors.New("extension unavailable")
)

type Spec struct {
	Name           string
	Command        []string
	Required       bool
	StartupTimeout time.Duration
	CallTimeout    time.Duration
	MaxRestarts    int
}

type Status struct {
	Name        string
	Running     bool
	Quarantined bool
	Failures    int
	Restarts    int
	PID         int
	LastError   string
}

type Supervisor struct {
	mu         sync.Mutex
	extensions map[string]*extension
}

type extension struct {
	spec        Spec
	mu          sync.Mutex
	cmd         *exec.Cmd
	stdin       io.WriteCloser
	stdout      *bufio.Reader
	failures    int
	restarts    int
	quarantined bool
	pid         int
	lastErr     string
}

type frame struct {
	Type    string `json:"type"`
	Payload string `json:"payload,omitempty"`
	PID     int    `json:"pid,omitempty"`
}

func New(specs ...Spec) (*Supervisor, error) {
	s := &Supervisor{extensions: make(map[string]*extension, len(specs))}
	for _, spec := range specs {
		if spec.Name == "" || len(spec.Command) == 0 {
			return nil, fmt.Errorf("invalid extension spec")
		}
		if _, ok := s.extensions[spec.Name]; ok {
			return nil, fmt.Errorf("duplicate extension %q", spec.Name)
		}
		if spec.StartupTimeout <= 0 {
			spec.StartupTimeout = time.Second
		}
		if spec.CallTimeout <= 0 {
			spec.CallTimeout = time.Second
		}
		if spec.MaxRestarts < 0 {
			spec.MaxRestarts = 0
		}
		s.extensions[spec.Name] = &extension{spec: spec}
	}
	return s, nil
}

// Call invokes one extension. A failure affects only this extension; sibling
// processes and the supervisor remain available. MaxRestarts is the number of
// replacement processes attempted after the initial process fails.
func (s *Supervisor) Call(ctx context.Context, name, payload string) (string, error) {
	s.mu.Lock()
	e := s.extensions[name]
	s.mu.Unlock()
	if e == nil {
		return "", fmt.Errorf("%w: %s", ErrUnavailable, name)
	}
	return e.call(ctx, payload)
}

func (e *extension) call(ctx context.Context, payload string) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.quarantined {
		return "", fmt.Errorf("%w: %s", ErrQuarantined, e.spec.Name)
	}
	attempts := e.spec.MaxRestarts + 1
	var err error
	for i := 0; i < attempts; i++ {
		if e.cmd == nil {
			if i > 0 || e.failures > 0 {
				e.restarts++
			}
			if err = e.start(ctx); err != nil {
				e.recordFailure(err)
				continue
			}
		}
		var out string
		out, err = e.exchange(ctx, payload)
		if err == nil {
			e.failures = 0
			e.lastErr = ""
			return out, nil
		}
		e.stop()
		e.recordFailure(err)
	}
	e.quarantined = true
	return "", fmt.Errorf("%w: %s: %v", ErrQuarantined, e.spec.Name, err)
}

func (e *extension) start(ctx context.Context) error {
	cmd := exec.Command(e.spec.Command[0], e.spec.Command[1:]...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return err
	}
	e.cmd, e.stdin, e.stdout, e.pid = cmd, stdin, bufio.NewReader(stdout), cmd.Process.Pid
	f, err := readFrame(ctx, e.stdout, e.spec.StartupTimeout)
	if err != nil {
		e.stop()
		return fmt.Errorf("startup: %w", err)
	}
	if f.Type != "ready" {
		e.stop()
		return fmt.Errorf("startup: malformed frame type %q", f.Type)
	}
	return nil
}

func (e *extension) exchange(ctx context.Context, payload string) (string, error) {
	b, _ := json.Marshal(frame{Type: "call", Payload: payload})
	if _, err := e.stdin.Write(append(b, '\n')); err != nil {
		return "", err
	}
	f, err := readFrame(ctx, e.stdout, e.spec.CallTimeout)
	if err != nil {
		return "", fmt.Errorf("call: %w", err)
	}
	if f.Type != "result" {
		return "", fmt.Errorf("call: malformed frame type %q", f.Type)
	}
	return f.Payload, nil
}

func readFrame(ctx context.Context, r *bufio.Reader, timeout time.Duration) (frame, error) {
	type result struct {
		f   frame
		err error
	}
	ch := make(chan result, 1)
	go func() {
		line, err := r.ReadBytes('\n')
		if err != nil {
			ch <- result{err: err}
			return
		}
		var f frame
		err = json.Unmarshal(line, &f)
		ch <- result{f: f, err: err}
	}()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case x := <-ch:
		return x.f, x.err
	case <-ctx.Done():
		return frame{}, ctx.Err()
	case <-timer.C:
		return frame{}, context.DeadlineExceeded
	}
}

func (e *extension) recordFailure(err error) { e.failures++; e.lastErr = err.Error() }
func (e *extension) stop() {
	if e.cmd == nil {
		return
	}
	_ = e.stdin.Close()
	if e.cmd.Process != nil {
		_ = e.cmd.Process.Kill()
	}
	_ = e.cmd.Wait()
	e.cmd, e.stdin, e.stdout, e.pid = nil, nil, nil, 0
}

func (s *Supervisor) Status(name string) (Status, bool) {
	s.mu.Lock()
	e := s.extensions[name]
	s.mu.Unlock()
	if e == nil {
		return Status{}, false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return Status{Name: name, Running: e.cmd != nil, Quarantined: e.quarantined, Failures: e.failures, Restarts: e.restarts, PID: e.pid, LastError: e.lastErr}, true
}

func (s *Supervisor) Close() error {
	s.mu.Lock()
	list := make([]*extension, 0, len(s.extensions))
	for _, e := range s.extensions {
		list = append(list, e)
	}
	s.mu.Unlock()
	for _, e := range list {
		e.mu.Lock()
		e.stop()
		e.mu.Unlock()
	}
	return nil
}
