package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/privacy"
)

type otlpSpan struct {
	TraceID, SpanID, ParentSpanID, Name, Route, Method string
	Status                                             int
	Start, End                                         time.Time
}
type otlpExporter struct {
	privacy                                     privacy.Policy
	endpoint                                    string
	client                                      *http.Client
	queue                                       chan otlpSpan
	done                                        chan struct{}
	closeOnce                                   sync.Once
	closedMu                                    sync.RWMutex
	closed                                      bool
	accepted, exported, dropped, failed, denied uint64
}
type otlpStats struct {
	Accepted, Exported, Dropped, Failed, Denied uint64
	QueueDepth                                  int
}

func newOTLPExporter(endpoint string, capacity int, timeout time.Duration) (*otlpExporter, error) {
	endpoint = strings.TrimSpace(endpoint)
	if endpoint == "" {
		return nil, nil
	}
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
		return nil, fmt.Errorf("OTLP endpoint must be absolute http/https")
	}
	if !strings.HasSuffix(u.Path, "/v1/traces") {
		u.Path = strings.TrimRight(u.Path, "/") + "/v1/traces"
	}
	if capacity < 1 || capacity > 65536 {
		return nil, fmt.Errorf("OTLP queue capacity must be 1..65536")
	}
	if timeout <= 0 || timeout > 5*time.Second {
		return nil, fmt.Errorf("OTLP timeout must be within 1ns..5s")
	}
	policy := privacy.DefaultPolicy()
	policy.LocalOnly = false
	policy.Telemetry.Enabled = true
	e := &otlpExporter{privacy: policy, endpoint: u.String(), client: &http.Client{Timeout: timeout}, queue: make(chan otlpSpan, capacity), done: make(chan struct{})}
	go e.run()
	return e, nil
}
func (e *otlpExporter) enqueue(span otlpSpan) {
	if e == nil {
		return
	}
	e.closedMu.RLock()
	defer e.closedMu.RUnlock()
	if e.closed {
		atomic.AddUint64(&e.dropped, 1)
		return
	}
	payload, _ := json.Marshal(span)
	decision, err := e.privacy.Evaluate(privacy.SinkTelemetry, payload, time.Now())
	if err != nil || decision.Receipt.Action == privacy.ActionDeny {
		atomic.AddUint64(&e.denied, 1)
		return
	}
	select {
	case e.queue <- span:
		atomic.AddUint64(&e.accepted, 1)
	default:
		atomic.AddUint64(&e.dropped, 1)
	}
}
func (e *otlpExporter) run() {
	defer close(e.done)
	for span := range e.queue {
		if err := e.export(span); err != nil {
			atomic.AddUint64(&e.failed, 1)
		} else {
			atomic.AddUint64(&e.exported, 1)
		}
	}
}
func (e *otlpExporter) export(s otlpSpan) error {
	payload := map[string]any{"resourceSpans": []any{map[string]any{"resource": map[string]any{"attributes": []any{map[string]any{"key": "service.name", "value": map[string]any{"stringValue": "fak-gateway"}}}}, "scopeSpans": []any{map[string]any{"scope": map[string]any{"name": "github.com/anthony-chaudhary/fak/internal/gateway"}, "spans": []any{map[string]any{"traceId": s.TraceID, "spanId": s.SpanID, "parentSpanId": s.ParentSpanID, "name": s.Name, "kind": 2, "startTimeUnixNano": fmt.Sprint(s.Start.UnixNano()), "endTimeUnixNano": fmt.Sprint(s.End.UnixNano()), "attributes": []any{otlpAttr("http.request.method", s.Method), otlpAttr("http.route", s.Route), otlpIntAttr("http.response.status_code", s.Status)}, "status": map[string]any{"code": map[bool]int{true: 2, false: 1}[s.Status >= 500]}}}}}}}}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, e.endpoint, bytes.NewReader(body))
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
		return fmt.Errorf("OTLP status %d", resp.StatusCode)
	}
	return nil
}
func otlpAttr(k, v string) map[string]any {
	return map[string]any{"key": k, "value": map[string]any{"stringValue": v}}
}
func otlpIntAttr(k string, v int) map[string]any {
	return map[string]any{"key": k, "value": map[string]any{"intValue": fmt.Sprint(v)}}
}
func (e *otlpExporter) stats() otlpStats {
	if e == nil {
		return otlpStats{}
	}
	return otlpStats{Accepted: atomic.LoadUint64(&e.accepted), Exported: atomic.LoadUint64(&e.exported), Dropped: atomic.LoadUint64(&e.dropped), Failed: atomic.LoadUint64(&e.failed), Denied: atomic.LoadUint64(&e.denied), QueueDepth: len(e.queue)}
}
func (e *otlpExporter) close(ctx context.Context) error {
	if e == nil {
		return nil
	}
	e.closeOnce.Do(func() {
		e.closedMu.Lock()
		e.closed = true
		close(e.queue)
		e.closedMu.Unlock()
	})
	select {
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
