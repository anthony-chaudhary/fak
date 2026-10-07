package gateway

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// PerfRecordSchema versions one per-request serving-performance row.
const PerfRecordSchema = perfledger.Schema

// PerfRecord is one served turn's TTFT / prefill / decode / e2e / cache row.
type PerfRecord = perfledger.Record

func (loc servingLocality) perfLabel() string {
	switch loc {
	case localitySelfHosted:
		return perfledger.LocalitySelfHosted
	case localityVendor:
		return perfledger.LocalityVendor
	default:
		return perfledger.LocalityUnknown
	}
}

// perfDetail is what a served turn knows beyond its token/latency axes: the model
// that served it and, on a native turn, its own engine decode anatomy plus the
// cache tier its reused prompt was restored from and the restore outcome.
//
// queueWait/queueKnown are the admission-scheduler wait the request paid before
// its planner call started; queueKnown=false means no scheduler gate (queue_ms
// stays unknown), and queueKnown with a zero wait is a measured immediate admit.
type perfDetail struct {
	model  string
	engine *perfledger.Engine
	// upstreamDraft / upstreamAccepted are a proxied upstream's own speculative
	// counts from its llama.cpp-shaped timings.
	upstreamDraft, upstreamAccepted int
	cacheTier, cacheRestore         string
	queueWait                       time.Duration
	queueKnown                      bool
}

// withQueue stamps an admission lease's scheduler wait onto the detail
// (nil-safe: no lease or no scheduler leaves the queue axis unknown).
func (d perfDetail) withQueue(lease *AdmissionLease) perfDetail {
	if wait, ok := lease.QueueWait(); ok {
		d.queueWait, d.queueKnown = wait, true
	}
	return d
}

type perfQueueCtxKey struct{}

// withPerfQueue stamps the admission lease's wait onto ctx so the buffered
// planner path (complete) can annotate its perf row without a signature change.
func withPerfQueue(ctx context.Context, lease *AdmissionLease) context.Context {
	wait, ok := lease.QueueWait()
	if !ok {
		return ctx
	}
	return context.WithValue(ctx, perfQueueCtxKey{}, wait)
}

// withQueueFromContext folds a wait stamped by withPerfQueue into the detail.
func (d perfDetail) withQueueFromContext(ctx context.Context) perfDetail {
	if ctx == nil {
		return d
	}
	if wait, ok := ctx.Value(perfQueueCtxKey{}).(time.Duration); ok {
		d.queueWait, d.queueKnown = wait, true
	}
	return d
}

// perfDetailFromCompletion lifts the planner-reported model and native decode
// summary off a Completion. A non-native completion yields no engine anatomy.
func perfDetailFromCompletion(comp *agent.Completion) perfDetail {
	if comp == nil {
		return perfDetail{}
	}
	d := perfDetail{model: comp.Model}
	if nd := comp.NativeDecode; nd != nil {
		e := &perfledger.Engine{Path: nd.Path, CohortSize: nd.CohortSize}
		if sp := nd.Speculative; sp != nil {
			e.SpecRounds, e.SpecDraftTokens, e.SpecAcceptedTokens = sp.Rounds, sp.DraftTokens, sp.AcceptedTokens
		}
		d.engine = e
		d.cacheTier, d.cacheRestore = nd.CacheTier, nd.CacheRestore
	} else if t := comp.Timings; t != nil && t.DraftN > 0 {
		d.upstreamDraft = t.DraftN
		d.upstreamAccepted = min(max(t.DraftNAccepted, 0), t.DraftN)
	}
	return d
}

// perfIdentity is the server identity stamped on every perf row. Backend is
// resolved only for a direct in-kernel planner; a proxy or dual planner leaves it
// empty rather than naming a backend that may not have served the turn.
func (s *Server) perfIdentity() perfledger.Identity {
	id := perfledger.Identity{Planner: plannerKind(s.planner), Version: perfBuildVersion()}
	if ikp, ok := s.planner.(*agent.InKernelPlanner); ok {
		id.Backend, _ = ikp.ExecutionIdentity()
	}
	if host, err := os.Hostname(); err == nil {
		id.Host = host
	}
	return id
}

// perfBuildVersion is the app version, with the VCS revision appended for an
// unstamped dev build so two dev binaries on one host stay distinguishable.
func perfBuildVersion() string {
	v := appversion.Current()
	if v != "dev" {
		return v
	}
	st := binstamp.Self()
	if len(st.Revision) < 12 {
		return v
	}
	v += "+" + st.Revision[:12]
	if st.Dirty {
		v += "-dirty"
	}
	return v
}

// setPerfIdentity installs the identity every later row carries; defaultModel
// fills a row whose turn reported no model of its own.
func (m *gatewayMetrics) setPerfIdentity(defaultModel string, id perfledger.Identity) {
	if m != nil {
		m.perfServedBy.Store(&perfledger.ServedBy{Model: strings.TrimSpace(defaultModel), Identity: id})
	}
}

// recordPerf folds one served turn into the bounded ring and hands it to the
// durable sink. Offer is a non-blocking channel send, so the served turn never
// waits on disk.
func (m *gatewayMetrics) recordPerf(loc servingLocality, promptTok, complTok, cachedTok int, finishReason string, dur, ttft time.Duration, detail perfDetail) {
	if m == nil {
		return
	}
	rec := perfledger.NewRecord(time.Now(), finishReason, loc.perfLabel(), promptTok, complTok, cachedTok, dur, ttft)
	if ttft <= 0 && m.pastWriteDeadline(dur) {
		// A buffered turn writes its body only after the planner returns; past the
		// server's WriteTimeout that write is refused, so the client got nothing.
		rec = perfledger.NewFailureRecord(time.Now(), loc.perfLabel(), perfledger.ErrorClientWriteTimeout, 0, dur, 0)
	}
	rec.Model = strings.TrimSpace(detail.model)
	rec.Engine = detail.engine
	rec.UpstreamSpecDraftTokens, rec.UpstreamSpecAcceptedTokens = detail.upstreamDraft, detail.upstreamAccepted
	rec.CacheTier = detail.cacheTier
	rec.CacheRestore = detail.cacheRestore
	if detail.queueKnown {
		rec = rec.WithQueue(detail.queueWait)
	}
	m.commitPerf(rec)
}

// recordPerfFailure emits the perf row for a turn that failed before completing.
// It deliberately leaves the fak_gateway_inference_* counters alone: a turn that
// produced no tokens is not a generation, but its latency is still part of what
// the client saw, so dropping it would hide the slow tail.
func (m *gatewayMetrics) recordPerfFailure(loc servingLocality, errClass string, status int, dur, ttft time.Duration) {
	if m == nil {
		return
	}
	m.commitPerf(perfledger.NewFailureRecord(time.Now(), loc.perfLabel(), errClass, status, dur, ttft))
}

func (m *gatewayMetrics) commitPerf(rec perfledger.Record) {
	if sb := m.perfServedBy.Load(); sb != nil {
		rec.Identity = sb.Identity
		if rec.Model == "" {
			rec.Model = sb.Model
		}
	}
	m.perfMu.Lock()
	m.appendPerfLocked(rec)
	m.perfMu.Unlock()
	if sink := m.perfSink.Load(); sink != nil {
		sink.Offer(rec)
	}
}

// recordFailedTurn records the perf row for a served turn whose planner call
// failed. committed reports that response bytes were already on the wire, so the
// client saw 200 and then an in-stream error.
func (s *Server) recordFailedTurn(ctx context.Context, loc servingLocality, err error, began time.Time, ttft time.Duration, committed bool) {
	if s == nil || s.metrics == nil || err == nil {
		return
	}
	class := perfErrorClass(ctx, err)
	if class == perfledger.ErrorStall && !committed && ttft <= 0 {
		// A streamed upstream that went idle before the client saw any byte is, to
		// the client, the same first-token timeout the buffered watchdog reports.
		class = perfledger.ErrorFirstTokenTimeout
	}
	elapsed := time.Since(began)
	status := http.StatusOK
	switch {
	case !committed && s.metrics.pastWriteDeadline(elapsed):
		// The error response would be written after the server's write deadline: the
		// client saw no status, whatever the upstream failure was.
		class, status = perfledger.ErrorClientWriteTimeout, 0
	case committed:
	case class == perfledger.ErrorClientCanceled:
		status = statusClientClosedRequest
	default:
		status, _, _ = upstreamErrorStatus(err)
	}
	s.metrics.recordPerfFailure(loc, class, status, elapsed, ttft)
}

func (m *gatewayMetrics) setHTTPWriteTimeout(d time.Duration) {
	if m != nil && d > 0 {
		m.httpWriteTimeout.Store(int64(d))
	}
}

// pastWriteDeadline reports that a turn which took dur (measured from the planner
// call, so never longer than the handler) outlived the server's WriteTimeout, which
// runs from the end of the request headers. Streams lift that deadline once their
// SSE header is committed (clearStreamWriteDeadline); callers only ask for turns
// that wrote nothing yet.
func (m *gatewayMetrics) pastWriteDeadline(dur time.Duration) bool {
	if m == nil {
		return false
	}
	wt := time.Duration(m.httpWriteTimeout.Load())
	return wt > 0 && dur > wt
}

// statusClientClosedRequest is the de-facto (nginx) status for a request the
// client abandoned before a response was written.
const statusClientClosedRequest = 499

func perfErrorClass(ctx context.Context, err error) string {
	var stalled *agent.UpstreamStalledError
	var unreachable *agent.UpstreamUnreachableError
	var upstreamStatus *agent.UpstreamStatusError
	// Typed upstream failures win over a canceled ctx: the gateway may cancel the
	// turn's context itself while unwinding a stall, which is not a client cancel.
	switch {
	case errors.Is(err, context.Canceled):
		return perfledger.ErrorClientCanceled
	case agent.IsFirstTokenStall(err):
		return perfledger.ErrorFirstTokenTimeout
	case errors.As(err, &stalled):
		return perfledger.ErrorStall
	case errors.Is(err, context.DeadlineExceeded):
		return perfledger.ErrorDeadline
	case errors.As(err, &unreachable):
		return perfledger.ErrorUpstreamUnreachable
	case errors.As(err, &upstreamStatus):
		return perfledger.ErrorUpstreamStatus
	case ctx != nil && errors.Is(ctx.Err(), context.Canceled):
		return perfledger.ErrorClientCanceled
	default:
		return perfledger.ErrorUpstream
	}
}

func (m *gatewayMetrics) appendPerfLocked(recs ...perfledger.Record) {
	m.perfRecords = append(m.perfRecords, recs...)
	if len(m.perfRecords) > perfledger.RingCap {
		trimmed := make([]perfledger.Record, perfledger.RingCap)
		copy(trimmed, m.perfRecords[len(m.perfRecords)-perfledger.RingCap:])
		m.perfRecords = trimmed
		m.perfRecordsDropped = true
	}
}

func (m *gatewayMetrics) perfRecordsSnapshot() ([]perfledger.Record, bool, uint64) {
	if m == nil {
		return nil, false, 0
	}
	m.perfMu.Lock()
	out := make([]perfledger.Record, len(m.perfRecords))
	copy(out, m.perfRecords)
	capped := m.perfRecordsDropped
	m.perfMu.Unlock()
	return out, capped, m.perfSink.Load().Dropped()
}

// SetPerfLedger installs the durable perf sink and seeds the in-memory ring
// with prior history (oldest first) so the read survives a restart. seedCapped
// reports that the seed was itself a bounded tail of a longer ledger. A nil sink
// leaves the ring memory-only.
func (s *Server) SetPerfLedger(sink *perfledger.Writer, seed []perfledger.Record, seedCapped bool) {
	if s == nil || s.metrics == nil {
		return
	}
	m := s.metrics
	m.perfMu.Lock()
	if len(seed) > 0 {
		prior := m.perfRecords
		m.perfRecords = nil
		m.appendPerfLocked(seed...)
		m.appendPerfLocked(prior...)
	}
	if seedCapped {
		m.perfRecordsDropped = true
	}
	m.perfMu.Unlock()
	m.perfSink.Store(sink)
}

// PerfReport folds the last n retained per-request perf rows.
func (s *Server) PerfReport(n int) perfledger.Report {
	if s == nil || s.metrics == nil {
		return perfledger.BuildReport(nil, n, false, 0)
	}
	recs, capped, dropped := s.metrics.perfRecordsSnapshot()
	rep := perfledger.BuildReport(recs, n, capped, dropped)
	rep.Source = "live"
	return rep
}

// handleFakPerfRecent serves GET /v1/fak/perf/recent: the last N served turns'
// TTFT / prefill / decode / e2e / cache rows plus the summary fold.
//
//	?n=<int>        rows to fold (default 50, max perfledger.RingCap)
//	?format=compact one text line instead of JSON
func (s *Server) handleFakPerfRecent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		w.Header().Set("Allow", "GET, HEAD")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	n := perfledger.DefaultRecent
	if raw := strings.TrimSpace(q.Get("n")); raw != "" {
		v, err := strconv.Atoi(raw)
		if err != nil || v <= 0 {
			http.Error(w, "n must be a positive integer", http.StatusBadRequest)
			return
		}
		n = v
	}
	rep := s.PerfReport(n)
	switch q.Get("format") {
	case "", "json":
		writeJSON(w, http.StatusOK, rep)
	case "compact":
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		_, _ = w.Write([]byte(perfledger.RenderCompact(rep) + "\n"))
	default:
		http.Error(w, "format must be json or compact", http.StatusBadRequest)
	}
}
