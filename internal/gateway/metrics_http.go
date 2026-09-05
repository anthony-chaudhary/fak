package gateway

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func (m *gatewayMetrics) observeHTTP(route, method string, status int, dur time.Duration) {
	if m == nil {
		return
	}
	key := httpMetricKey{route: route, method: method, status: itoa(uint64(status))}
	m.mu.Lock()
	counter := m.http[key]
	if counter == nil {
		counter = newLatencyCounter()
		m.http[key] = counter
	}
	counter.observe(dur.Seconds())
	m.mu.Unlock()
}

func (m *gatewayMetrics) observeOperation(operation string, v WireVerdict, err error, dur time.Duration) {
	if m == nil {
		return
	}
	key := operationMetricKey{
		operation:      operation,
		verdict:        v.Kind,
		reason:         v.Reason,
		refusalSubtype: v.RefusalSubtype,
	}
	if err != nil || key.verdict == "" {
		key.verdict = "ERROR"
	}
	key.disposition = v.Disposition
	key.by = v.By
	m.mu.Lock()
	counter := m.operations[key]
	if counter == nil {
		counter = newLatencyCounter()
		m.operations[key] = counter
	}
	counter.observe(dur.Seconds())
	m.mu.Unlock()
}

type httpMetricSnapshot struct {
	key httpMetricKey
	val latencySnapshot
}

type operationMetricSnapshot struct {
	key operationMetricKey
	val latencySnapshot
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	_, _ = w.Write([]byte(s.renderMetrics()))
}

type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int64
}

// WriteHeader records the FIRST status code written (later calls are ignored) and
// forwards it to the wrapped ResponseWriter, so the metrics middleware can label the
// request by the status actually sent.
func (r *statusRecorder) WriteHeader(status int) {
	if r.status != 0 {
		return
	}
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

// Write forwards the body to the wrapped ResponseWriter (defaulting the status to 200
// on the first write) and accumulates the byte count for the request log.
func (r *statusRecorder) Write(p []byte) (int, error) {
	if r.status == 0 {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(p)
	r.bytes += int64(n)
	return n, err
}

// Flush defaults the status to 200 if unset and flushes the wrapped ResponseWriter
// when it implements http.Flusher, preserving streaming (SSE) behavior through the
// recorder.
func (r *statusRecorder) Flush() {
	if r.status == 0 {
		r.WriteHeader(http.StatusOK)
	}
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *Server) withMetrics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		atomic.AddInt64(&s.metrics.inflight, 1)
		defer atomic.AddInt64(&s.metrics.inflight, -1)
		route := routeForMetrics(r.URL.Path)
		// Register the request as live so scrapes can see it WHILE it runs, not
		// only after it completes into the latency histogram. route is derived from
		// the path up front (routeForMetrics is pure), matching the completion-time
		// label below.
		if route != "/metrics" {
			liveID := s.metrics.beginInflight(route, start)
			defer s.metrics.endInflight(liveID)
		}
		traceID := ensureHTTPTrace(s, w, r)
		rec := &statusRecorder{ResponseWriter: w}
		// Record metrics + the request log for EVERY outcome, panic included, and contain a
		// downstream handler panic HERE — the outermost fak-owned wrapper — instead of letting
		// it unwind into net/http. net/http's own recovery writes a full goroutine stack to the
		// server ErrorLog, which under `fak guard` is the wrapped agent's controlling TTY (#2772);
		// worse, the unwind would skip the accounting below, so the failed turn would never reach
		// observeHTTP or the request log and the panic would be invisible at /metrics (#2773/#2775).
		// We convert it to a structured 500 envelope + one line + a counted turn; a served-turn panic
		// also marks /healthz (#2336, served_failure.go). http.ErrAbortHandler is re-raised untouched.
		defer func() {
			if current := requestTraceID(r); current != "" {
				traceID = current
			}
			if p := recover(); p != nil {
				if p == http.ErrAbortHandler {
					panic(p)
				}
				// Pin the faulting handler HERE, while the panicking frames are
				// still on the stack (recover has not yet returned control to the
				// deferring frame), so the log names the exact site instead of the
				// coarse route alone (#2775). Captured as a single compact frame,
				// never the multi-line goroutine dump net/http would leak (#2772).
				origin := panicOriginFrame()
				s.noteServedPanic(r.URL.Path, p)
				if rec.status == 0 {
					writeRecoveredPanicErr(rec, p)
				}
				s.logRecoveredPanic(r, route, traceID, origin, p)
			}
			status := rec.status
			if status == 0 {
				status = http.StatusOK
			}
			dur := time.Since(start)
			s.metrics.observeHTTP(route, r.Method, status, dur)
			if ctx, err := parseTraceparent(r.Header.Get(traceparentHeader)); err == nil {
				s.otlp.enqueue(otlpSpan{TraceID: ctx.TraceID, SpanID: ctx.ParentID, Name: r.Method + " " + route, Route: route, Method: r.Method, Status: status, Start: start, End: start.Add(dur)})
			}
			s.logHTTPRequest(r, route, status, dur, rec.bytes, traceID)
		}()
		next.ServeHTTP(rec, r)
	})
}

func (s *Server) logHTTPRequest(r *http.Request, route string, status int, dur time.Duration, bytes int64, traceID string) {
	if s.logf == nil {
		return
	}
	ev := map[string]any{
		"event":       "gateway_http_request",
		"method":      r.Method,
		"route":       route,
		"path":        r.URL.Path,
		"status":      status,
		"duration_ms": float64(dur.Microseconds()) / 1000.0,
		"bytes":       bytes,
	}
	if traceID != "" {
		ev["trace_id"] = traceID
	}
	if ua := r.Header.Get("User-Agent"); ua != "" {
		ev["user_agent"] = ua
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
}

// logRecoveredPanic emits the one structured event that lets a recovered handler
// panic self-identify in the log stream (#2775 step 3): route + method + path + the
// panic value + trace id + the faulting frame, in the same JSON shape as
// gateway_http_request so a scraper can aggregate panics by route/trace/origin
// instead of parsing prose. The "origin" field is the whole point of #2775 — the
// route alone is too coarse to pin the handler when several handlers share a route,
// so origin carries the exact "pkg.Func (file:line)" the panic was raised at. The
// "msg" field keeps the human-readable "recovered handler panic" marker that log
// tails and the #2773 containment test grep for, so this stays a drop-in for the
// prior prose line. Called only from withMetrics' recover, i.e. at most once per
// served turn.
func (s *Server) logRecoveredPanic(r *http.Request, route, traceID, origin string, p any) {
	if s.logf == nil {
		return
	}
	ev := map[string]any{
		"event":  "gateway_recovered_panic",
		"msg":    "recovered handler panic",
		"method": r.Method,
		"route":  route,
		"path":   r.URL.Path,
		"panic":  fmt.Sprintf("%v", p),
	}
	if origin != "" {
		ev["origin"] = origin
	}
	if traceID != "" {
		ev["trace_id"] = traceID
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
}

// panicOriginFrame walks the calling goroutine's stack from inside withMetrics'
// deferred recover and returns the frame where the panic was actually raised —
// the first non-runtime frame after runtime.gopanic (which also covers runtime-
// raised panics that dispatch through sigpanic/panicmem, e.g. a nil dereference
// or a nil-map write). It returns "pkg.Func (file:line)" with the package path
// and absolute build path trimmed off, and — critically — with no newline or
// "goroutine …" header, so the recovered-panic log stays a single short field:
// #2775 wants the faulting handler pinned, while #2772 forbids re-leaking the
// multi-line goroutine dump net/http would otherwise write to the guarded
// child's controlling TTY. Returns "" when no origin can be resolved (e.g. the
// panicking frames have already been unwound), and the caller then omits the
// field rather than logging a misleading placeholder. Must be called
// synchronously within the recover, before it returns to the deferring frame.
func panicOriginFrame() string {
	var pcs [64]uintptr
	// skip=0 keeps every frame walkable regardless of inlining; the loop below
	// discards this helper and the recover closure by only accepting a frame
	// that appears AFTER runtime.gopanic.
	n := runtime.Callers(0, pcs[:])
	if n == 0 {
		return ""
	}
	frames := runtime.CallersFrames(pcs[:n])
	sawPanic := false
	for {
		fr, more := frames.Next()
		if strings.HasPrefix(fr.Function, "runtime.") {
			if fr.Function == "runtime.gopanic" {
				sawPanic = true
			}
			if !more {
				break
			}
			continue
		}
		if sawPanic && fr.Function != "" {
			fn := fr.Function
			if i := strings.LastIndex(fn, "/"); i >= 0 {
				fn = fn[i+1:]
			}
			file := fr.File
			if i := strings.LastIndex(file, "/"); i >= 0 {
				file = file[i+1:]
			}
			return fmt.Sprintf("%s (%s:%d)", fn, file, fr.Line)
		}
		if !more {
			break
		}
	}
	return ""
}

func (s *Server) logGatewayOperation(operation, traceID, tool string, v WireVerdict, opErr error, dur time.Duration) {
	if s == nil {
		return
	}
	if operation == "adjudicate" && s.orgAudit != nil {
		verdict := v.Kind
		if opErr != nil {
			verdict = "ERROR"
		}
		s.orgAudit.Emit(tool, verdict, v.Reason, map[string]int64{})
	}
	if s.logf == nil {
		return
	}
	verdict := v.Kind
	if opErr != nil && verdict == "" {
		verdict = "ERROR"
	}
	ev := map[string]any{
		"event":       "gateway_operation",
		"operation":   operation,
		"tool":        tool,
		"trace_id":    strings.TrimSpace(traceID),
		"verdict":     verdict,
		"duration_ms": float64(dur.Microseconds()) / 1000.0,
	}
	if v.Reason != "" {
		ev["reason"] = v.Reason
	}
	if v.By != "" {
		ev["by"] = v.By
	}
	if v.Disposition != "" {
		ev["disposition"] = v.Disposition
	}
	if opErr != nil {
		ev["error"] = opErr.Error()
	}
	b, err := json.Marshal(ev)
	if err != nil {
		return
	}
	s.logf("%s", b)
}

const traceHeader = "X-Trace-Id"

func ensureHTTPTrace(s *Server, w http.ResponseWriter, r *http.Request) string {
	var ctx traceContext
	var traceID string
	if parsed, err := parseTraceparent(r.Header.Get(traceparentHeader)); err == nil {
		ctx = newTraceContext(parsed.TraceID, parsed.Flags)
		traceID = ctx.TraceID
	} else {
		if strings.TrimSpace(r.Header.Get(traceparentHeader)) != "" && s != nil {
			atomic.AddUint64(&s.traceparentInvalid, 1)
		}
		traceID = requestTraceID(r)
		if traceID == "" && s != nil {
			traceID = s.traceFor("")
		}
		ctx = newTraceContext(w3cTraceID(traceID), 0)
	}
	if traceID == "" {
		traceID = ctx.TraceID
	}
	r.Header.Set(traceHeader, traceID)
	r.Header.Set(traceparentHeader, ctx.String())
	*r = *r.WithContext(agent.WithTraceContext(r.Context(), ctx.String(), r.Header.Get("tracestate")))
	w.Header().Set(traceHeader, traceID)
	w.Header().Set(traceparentHeader, ctx.String())
	return traceID
}

func w3cTraceID(traceID string) string {
	traceID = strings.TrimSpace(traceID)
	if len(traceID) != 32 || allZeroHex(traceID) {
		return ""
	}
	if _, err := hex.DecodeString(traceID); err != nil {
		return ""
	}
	return strings.ToLower(traceID)
}

func requestTraceID(r *http.Request) string {
	if r == nil {
		return ""
	}
	return strings.TrimSpace(r.Header.Get(traceHeader))
}

func (s *Server) useHTTPTrace(w http.ResponseWriter, r *http.Request, preferred string) string {
	traceID := strings.TrimSpace(preferred)
	if traceID == "" {
		traceID = requestTraceID(r)
	}
	if traceID == "" && s != nil {
		traceID = s.traceFor("")
	}
	if traceID != "" {
		r.Header.Set(traceHeader, traceID)
		w.Header().Set(traceHeader, traceID)
	}
	return traceID
}

func routeForMetrics(path string) string {
	switch path {
	case "/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens",
		"/v1/fak/syscall", "/v1/fak/adjudicate", "/v1/fak/admit",
		"/v1/fak/changes", "/v1/fak/session/changes", "/v1/fak/revoke", "/v1/fak/policy/reload",
		"/v1/fak/route/reload", "/v1/fak/trace/reset", "/v1/models", "/mcp", "/healthz", "/metrics",
		"/debug/vars":
		return path
	default:
		if strings.HasPrefix(path, "/v1/fak/") {
			return "/v1/fak/*"
		}
		if strings.HasPrefix(path, "/v1beta/") {
			return "/v1beta/*"
		}
		return "other"
	}
}
