package gateway

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
// that served it and, on a native turn, its own engine decode anatomy.
type perfDetail struct {
	model  string
	engine *perfledger.Engine
	// upstreamDraft / upstreamAccepted are a proxied upstream's own speculative
	// counts from its llama.cpp-shaped timings.
	upstreamDraft, upstreamAccepted int
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
	} else if t := comp.Timings; t != nil && t.DraftN > 0 {
		d.upstreamDraft = t.DraftN
		d.upstreamAccepted = min(max(t.DraftNAccepted, 0), t.DraftN)
	}
	return d
}

// recordPerf folds one served turn into the bounded ring and hands it to the
// durable sink. Offer is a non-blocking channel send, so the served turn never
// waits on disk.
func (m *gatewayMetrics) recordPerf(loc servingLocality, promptTok, complTok, cachedTok int, finishReason string, dur, ttft time.Duration, detail perfDetail) {
	if m == nil {
		return
	}
	rec := perfledger.NewRecord(time.Now(), finishReason, loc.perfLabel(), promptTok, complTok, cachedTok, dur, ttft)
	rec.Model = detail.model
	rec.Engine = detail.engine
	rec.UpstreamSpecDraftTokens, rec.UpstreamSpecAcceptedTokens = detail.upstreamDraft, detail.upstreamAccepted
	m.perfMu.Lock()
	m.appendPerfLocked(rec)
	m.perfMu.Unlock()
	if sink := m.perfSink.Load(); sink != nil {
		sink.Offer(rec)
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
