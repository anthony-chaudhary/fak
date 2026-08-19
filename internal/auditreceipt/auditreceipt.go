// Package auditreceipt exports bounded, privacy-screened organization audit receipts.
package auditreceipt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/privacy"
)

const Schema = "fak-org-audit-receipt/1"

type Receipt struct {
	Schema     string           `json:"schema"`
	DeviceID   string           `json:"device_id"`
	Tool       string           `json:"tool"`
	Verdict    string           `json:"verdict"`
	Reason     string           `json:"reason,omitempty"`
	ObservedAt string           `json:"observed_at"`
	UsageDelta map[string]int64 `json:"usage_delta"`
}

type Config struct {
	Endpoint, DeviceID, BufferPath string
	Capacity                       int
	MaxBuffered                    int
	Timeout                        time.Duration
	RedactFields                   []string
}

type Stats struct {
	Accepted, Exported, Buffered, Dropped, Failed uint64
	QueueDepth                                    int
}

type Exporter struct {
	endpoint, deviceID, path                      string
	maxBuffered                                   int
	client                                        *http.Client
	policy                                        privacy.Policy
	queue                                         chan Receipt
	done                                          chan struct{}
	closeOnce                                     sync.Once
	accepted, exported, buffered, dropped, failed atomic.Uint64
}

func New(cfg Config) (*Exporter, error) {
	if strings.TrimSpace(cfg.Endpoint) == "" {
		return nil, nil
	}
	u, err := url.Parse(cfg.Endpoint)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, errors.New("audit endpoint must be absolute http/https")
	}
	if strings.TrimSpace(cfg.DeviceID) == "" {
		return nil, errors.New("audit device identity is required")
	}
	if cfg.Capacity == 0 {
		cfg.Capacity = 256
	}
	if cfg.Capacity < 1 || cfg.Capacity > 65536 {
		return nil, errors.New("audit capacity must be 1..65536")
	}
	if cfg.Timeout == 0 {
		cfg.Timeout = 2 * time.Second
	}
	if cfg.Timeout < time.Nanosecond || cfg.Timeout > 5*time.Second {
		return nil, errors.New("audit timeout must be within 1ns..5s")
	}
	if cfg.MaxBuffered == 0 {
		cfg.MaxBuffered = 10000
	}
	if cfg.MaxBuffered < 1 || cfg.MaxBuffered > 1000000 {
		return nil, errors.New("audit buffer capacity must be 1..1000000")
	}
	if cfg.BufferPath == "" {
		return nil, errors.New("audit buffer path is required")
	}
	p := privacy.DefaultPolicy()
	p.LocalOnly = false
	p.Export = privacy.SinkPolicy{Enabled: true, RedactFields: append([]string(nil), cfg.RedactFields...)}
	e := &Exporter{endpoint: u.String(), deviceID: cfg.DeviceID, path: cfg.BufferPath, maxBuffered: cfg.MaxBuffered, client: &http.Client{Timeout: cfg.Timeout}, policy: p, queue: make(chan Receipt, cfg.Capacity), done: make(chan struct{})}
	go e.run()
	return e, nil
}

func (e *Exporter) Emit(tool, verdict, reason string, usage map[string]int64) bool {
	if e == nil {
		return false
	}
	tool = bounded(tool, 256)
	verdict = bounded(verdict, 64)
	reason = bounded(reason, 256)
	r := Receipt{Schema: Schema, DeviceID: e.deviceID, Tool: tool, Verdict: verdict, Reason: reason, ObservedAt: time.Now().UTC().Format(time.RFC3339Nano), UsageDelta: usage}
	select {
	case e.queue <- r:
		e.accepted.Add(1)
		return true
	default:
		e.dropped.Add(1)
		return false
	}
}
func (e *Exporter) Stats() Stats {
	if e == nil {
		return Stats{}
	}
	return Stats{e.accepted.Load(), e.exported.Load(), e.buffered.Load(), e.dropped.Load(), e.failed.Load(), len(e.queue)}
}
func (e *Exporter) Close(ctx context.Context) error {
	if e == nil {
		return nil
	}
	e.closeOnce.Do(func() { close(e.queue) })
	select {
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (e *Exporter) run() {
	defer close(e.done)
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case r, ok := <-e.queue:
			if !ok {
				e.replay()
				return
			}
			if err := e.send(r); err != nil {
				e.failed.Add(1)
				if e.append(r) == nil {
					e.buffered.Add(1)
				} else {
					e.dropped.Add(1)
				}
			} else {
				e.exported.Add(1)
				e.replay()
			}
		case <-tick.C:
			e.replay()
		}
	}
}
func (e *Exporter) screened(r Receipt) ([]byte, error) {
	raw, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	d, err := e.policy.Evaluate(privacy.SinkExport, raw, time.Now())
	if err != nil {
		return nil, err
	}
	if d.Receipt.Action == privacy.ActionDeny {
		return nil, errors.New("audit export denied")
	}
	return d.Payload, nil
}
func (e *Exporter) send(r Receipt) error {
	b, err := e.screened(r)
	if err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, e.endpoint, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("audit status %d", resp.StatusCode)
	}
	return nil
}
func (e *Exporter) append(r Receipt) error {
	if current, err := os.ReadFile(e.path); err == nil && bytes.Count(current, []byte{'\n'}) >= e.maxBuffered {
		return errors.New("audit buffer capacity reached")
	}
	b, err := e.screened(r)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(e.path), 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(e.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(b, '\n'))
	return err
}
func (e *Exporter) replay() {
	b, err := os.ReadFile(e.path)
	if err != nil {
		return
	}
	lines := bytes.Split(bytes.TrimSpace(b), []byte{'\n'})
	if len(lines) == 0 {
		return
	}
	remain := make([][]byte, 0, len(lines))
	for i, line := range lines {
		var r Receipt
		if json.Unmarshal(line, &r) != nil {
			continue
		}
		if e.send(r) != nil {
			remain = append(remain, lines[i:]...)
			break
		}
		e.exported.Add(1)
	}
	tmp := e.path + ".tmp"
	if len(remain) == 0 {
		_ = os.Remove(e.path)
		return
	}
	out := bytes.Join(remain, []byte{'\n'})
	out = append(out, '\n')
	if os.WriteFile(tmp, out, 0600) == nil {
		_ = os.Rename(tmp, e.path)
	}
}

func bounded(s string, n int) string {
	if len(s) <= n {
		return s
	}
	s = s[:n]
	for !utf8.ValidString(s) {
		s = s[:len(s)-1]
	}
	return s
}
