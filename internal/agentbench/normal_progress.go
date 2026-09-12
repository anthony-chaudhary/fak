package agentbench

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type normalProgress struct {
	mu              sync.Mutex
	path            string
	started         time.Time
	deadline        time.Time
	phase           string
	output          io.Writer
	replayCompleted int
	tasksPlanned    int
	tasksCompleted  int
	tasksAccepted   int
	err             error
	cancel          context.CancelFunc
	stop            chan struct{}
	done            chan struct{}
	once            sync.Once
}

func startNormalProgress(ctx context.Context, path string, started, deadline time.Time, output io.Writer, cancel context.CancelFunc) (*normalProgress, error) {
	if output == nil {
		output = io.Discard
	}
	p := &normalProgress{path: path, started: started, deadline: deadline, output: output, cancel: cancel, stop: make(chan struct{}), done: make(chan struct{})}
	go p.loop(ctx)
	return p, nil
}

func (p *normalProgress) Writer() io.Writer { return normalProgressWriter{p: p} }

type normalProgressWriter struct{ p *normalProgress }

func (w normalProgressWriter) Write(body []byte) (int, error) {
	w.p.mu.Lock()
	defer w.p.mu.Unlock()
	return w.p.output.Write(body)
}

func (p *normalProgress) SetCounts(replayCompleted, tasksPlanned, tasksCompleted, tasksAccepted int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.replayCompleted, p.tasksPlanned, p.tasksCompleted, p.tasksAccepted = replayCompleted, tasksPlanned, tasksCompleted, tasksAccepted
}

func (p *normalProgress) SetPhase(phase string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phase = phase
	return p.writeLocked()
}

func (p *normalProgress) loop(ctx context.Context) {
	defer close(p.done)
	ticker := time.NewTicker(progressInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			p.mu.Lock()
			if p.err == nil {
				p.err = p.writeLocked()
			}
			failed := p.err != nil
			p.mu.Unlock()
			if failed {
				p.cancel()
				return
			}
		case <-p.stop:
			return
		case <-ctx.Done():
			return
		}
	}
}

func (p *normalProgress) writeLocked() error {
	now := time.Now().UTC()
	checkpoint := struct {
		Schema          string    `json:"schema"`
		Status          string    `json:"status"`
		Overall         string    `json:"overall_verdict"`
		Phase           string    `json:"phase"`
		StartedAt       time.Time `json:"started_at"`
		UpdatedAt       time.Time `json:"updated_at"`
		DeadlineAt      time.Time `json:"deadline_at"`
		ElapsedMS       int64     `json:"elapsed_ms"`
		ReplayPlanned   int       `json:"replay_planned"`
		ReplayCompleted int       `json:"replay_completed"`
		TasksPlanned    int       `json:"tasks_planned"`
		TasksCompleted  int       `json:"tasks_completed"`
		TasksAccepted   int       `json:"tasks_accepted"`
		Unknowns        []string  `json:"unknowns"`
	}{"fak.agentbench.normal-run.v1", "in_progress", "INCOMPLETE", p.phase, p.started, now, p.deadline, now.Sub(p.started).Milliseconds(), 188, p.replayCompleted, p.tasksPlanned, p.tasksCompleted, p.tasksAccepted, []string{"backend queue/prefill/decode spans", "GPU occupancy, cache residency, memory, thermals, power"}}
	if _, err := fmt.Fprintf(p.output, "agentbench normal phase=%s age=%s deadline=%s replay=%d/188 tasks=%d/%d accepted=%d unknown=backend-spans,gpu-cache-memory\n", p.phase, now.Sub(p.started).Round(time.Second), p.deadline.Format(time.RFC3339), p.replayCompleted, p.tasksCompleted, p.tasksPlanned, p.tasksAccepted); err != nil {
		return err
	}
	return writeNormalProgressJSON(p.path, checkpoint)
}

func writeNormalProgressJSON(path string, value any) error {
	temp, err := os.CreateTemp(filepath.Dir(path), ".normal-progress-*")
	if err != nil {
		return err
	}
	name := temp.Name()
	keep := false
	defer func() {
		if !keep {
			_ = os.Remove(name)
		}
	}()
	if err := json.NewEncoder(temp).Encode(value); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		_ = temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return err
	}
	keep = true
	return nil
}

func (p *normalProgress) Close() error {
	p.once.Do(func() { close(p.stop); <-p.done })
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.err
}

func (p *normalProgress) Phase() string { p.mu.Lock(); defer p.mu.Unlock(); return p.phase }
