package trajectory

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultTokenDestinationRefreshInterval bounds transcript scans and atomic
	// snapshot replacements in a live process. Kernel events only mark the
	// projection dirty; a burst cannot schedule more than one scan per interval.
	DefaultTokenDestinationRefreshInterval = 30 * time.Second
	// DefaultTokenDestinationWindow keeps the published mix recent and bounded.
	// Snapshot mtime is therefore publication freshness, not a fresh-looking fold
	// over arbitrarily old transcript history.
	DefaultTokenDestinationWindow = 24 * time.Hour
)

// TokenDestinationPublisherStats measures the live producer's real overhead and
// freshness. Durations cover both transcript audit and durable atomic publication.
type TokenDestinationPublisherStats struct {
	Attempts        uint64
	Published       uint64
	Failures        uint64
	LastDurationNS  int64
	MaxDurationNS   int64
	LastPublishedAt time.Time
	LastError       string
}

type tokenDestinationPublisherConfig struct {
	Path     string
	Interval time.Duration
	Audit    func() (AuditResult, error)
}

type tokenDestinationPublisher struct {
	path     string
	interval time.Duration
	audit    func() (AuditResult, error)
	notify   chan struct{}
	stop     chan struct{}
	done     chan struct{}
	stopOnce sync.Once
	statsMu  sync.Mutex
	stats    TokenDestinationPublisherStats
}

// StartTokenDestinationPublisher attaches the live recorder to the recorded-file
// seam consumed by fak info. The initial audit runs asynchronously; until it
// succeeds the file remains absent and the consumer reports unavailable honestly.
func (r *Recorder) StartTokenDestinationPublisher(path string, interval time.Duration) error {
	return r.startTokenDestinationPublisher(tokenDestinationPublisherConfig{
		Path: path, Interval: interval,
		Audit: func() (AuditResult, error) {
			return RunAudit(AuditOptions{
				Sources: DefaultAuditSources(), Since: DefaultTokenDestinationWindow, Now: time.Now().UTC(),
			})
		},
	})
}

func (r *Recorder) startTokenDestinationPublisher(config tokenDestinationPublisherConfig) error {
	path := strings.TrimSpace(config.Path)
	if path == "" {
		return errors.New("trajectory token-destination publisher: path is required")
	}
	if config.Interval <= 0 {
		return errors.New("trajectory token-destination publisher: interval must be positive")
	}
	if config.Audit == nil {
		return errors.New("trajectory token-destination publisher: audit function is required")
	}
	p := &tokenDestinationPublisher{
		path: path, interval: config.Interval, audit: config.Audit,
		notify: make(chan struct{}, 1), stop: make(chan struct{}), done: make(chan struct{}),
	}
	r.publisherMu.Lock()
	defer r.publisherMu.Unlock()
	if r.publisher != nil {
		if r.publisher.path == path && r.publisher.interval == config.Interval {
			return nil
		}
		return fmt.Errorf("trajectory token-destination publisher already configured for %s", r.publisher.path)
	}
	r.publisher = p
	go p.run()
	return nil
}

func (r *Recorder) notifyTokenDestinationPublisher() {
	r.publisherMu.Lock()
	p := r.publisher
	r.publisherMu.Unlock()
	if p == nil {
		return
	}
	select {
	case p.notify <- struct{}{}:
	default:
	}
}

// StopTokenDestinationPublisher stops the recorder-owned background producer.
// It is idempotent and primarily gives tests/embedded users a leak-free lifecycle;
// normal guard and gateway processes retain the producer until process exit.
func (r *Recorder) StopTokenDestinationPublisher() {
	r.publisherMu.Lock()
	p := r.publisher
	r.publisher = nil
	r.publisherMu.Unlock()
	if p == nil {
		return
	}
	p.stopOnce.Do(func() { close(p.stop) })
	<-p.done
}

// TokenDestinationPublisherStats returns a race-free producer measurement.
func (r *Recorder) TokenDestinationPublisherStats() TokenDestinationPublisherStats {
	r.publisherMu.Lock()
	p := r.publisher
	r.publisherMu.Unlock()
	if p == nil {
		return TokenDestinationPublisherStats{}
	}
	p.statsMu.Lock()
	defer p.statsMu.Unlock()
	return p.stats
}

func (p *tokenDestinationPublisher) run() {
	defer close(p.done)
	p.publish() // startup: publish a real bounded fold, or remain unavailable.
	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()
	dirty := false
	for {
		select {
		case <-p.notify:
			dirty = true
		case <-ticker.C:
			if dirty {
				dirty = false
				p.publish()
			}
		case <-p.stop:
			return
		}
	}
}

func (p *tokenDestinationPublisher) publish() {
	started := time.Now()
	result, err := p.audit()
	if err == nil {
		err = writeTokenDestinationSummaryAtomic(p.path, result.Summary, os.Rename)
	}
	duration := time.Since(started)
	p.statsMu.Lock()
	p.stats.Attempts++
	p.stats.LastDurationNS = duration.Nanoseconds()
	if p.stats.LastDurationNS > p.stats.MaxDurationNS {
		p.stats.MaxDurationNS = p.stats.LastDurationNS
	}
	if err != nil {
		p.stats.Failures++
		p.stats.LastError = err.Error()
	} else {
		p.stats.Published++
		p.stats.LastError = ""
		if info, statErr := os.Stat(p.path); statErr == nil {
			p.stats.LastPublishedAt = info.ModTime().UTC()
		} else {
			p.stats.LastPublishedAt = time.Now().UTC()
		}
	}
	p.statsMu.Unlock()
}

func writeTokenDestinationSummaryAtomic(path string, summary AuditSummaryRow, rename func(string, string) error) error {
	if rename == nil {
		return errors.New("trajectory token-destination publisher: rename function is required")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.Marshal(summary)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".token-destination-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return rename(tmpName, path)
}
